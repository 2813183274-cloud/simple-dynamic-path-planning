"""P3-05 orchestration, separately sealed from the unchanged P3-04 training implementation."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.dwa import DWA,DWAConfig
from envs import DynamicPathPlanningEnv
from envs.risk_reward import RiskRewardWrapper
from envs.scenario_dataset import load_dataset,save_dataset,dataset_hash
from experiment_utils import atomic_write_json,sha256_file,utc_now
from round_three_utils import load_protocol,verify_implementation,verify_round_three_model,SEAL
from scripts.evaluate_round_three import evaluate_policy
from scripts.evaluate import wilson_interval,draw_full_subplot,draw_closest_subplot
from scripts.round_two_experiment import stratified,geometric_hash
from scripts.summarize_round_two import paired_interval
from scripts.test_round_one import replay_witness

OUT=ROOT/"results/round3/final_experiment"
ARMS=("base","instant","predictive")
SOURCES=("scripts/round_three_experiment.py","scripts/round_two_experiment.py",
         "scripts/summarize_round_two.py","scripts/test_round_one.py","configs/round1/protocol.json")


def source_hashes():
    return {p:sha256_file(ROOT/p) for p in SOURCES}


def summarize_frame(f):
    result={"n":len(f),"success_count":int(f.success.sum()),"success_rate":float(f.success.mean()),
        "success_wilson_95":wilson_interval(int(f.success.sum()),len(f)),
        "mean_base_reward":float(f.base_return.mean()),"mean_training_reward":float(f.training_return.mean()),
        "mean_risk_penalty":float(f.risk_return.mean()),
        "mean_min_static_clearance":float(f.min_static_clearance.mean()),
        "mean_min_dynamic_clearance":float(f.min_dynamic_clearance.mean()),
        "worst_static_clearance":float(f.min_static_clearance.min()),
        "worst_dynamic_clearance":float(f.min_dynamic_clearance.min()),
        "successful_path_efficiency":None if not f.success.any() else float(f.loc[f.success,"path_efficiency"].mean()),
        "successful_navigation_time":None if not f.success.any() else float(f.loc[f.success,"navigation_time"].mean()),
        "full_speed_fraction_after_2s":float(f.full_speed_fraction_after_2s.mean()),
        "clipped_mean_fraction":None if f.clipped_mean_fraction.isna().all() else float(f.clipped_mean_fraction.mean()),
        "high_risk_mean_speed_episode_average":None if f.high_risk_mean_speed.isna().all() else float(f.high_risk_mean_speed.mean()),
        "risk_exposed_episodes":int(f.risk_onset_step.notna().sum()),
        "risk_no_response_episodes":int(f.risk_no_braking_response.sum()),
        "risk_no_response_fraction_exposed":None if not f.risk_onset_step.notna().any() else float(f.risk_no_braking_response.sum()/f.risk_onset_step.notna().sum()),
        "braking_delay_responders_only":None if f.braking_delay_seconds.isna().all() else float(f.braking_delay_seconds.mean())}
    for key,mask in {"static_collision":f.reason.str.startswith("static_collision"),
                     "dynamic_collision":f.reason.eq("dynamic_collision"),"timeout":f.reason.eq("timeout"),
                     "out_of_bounds":f.reason.eq("out_of_bounds"),"collision":f.collision}.items():
        result[key+"_count"]=int(mask.sum());result[key+"_rate"]=float(mask.mean())
    return result


def seal_methods():
    protocol=load_protocol();verify_implementation()
    if (OUT/"method_seal.json").exists():raise FileExistsError("Methods already sealed")
    models={}
    for arm in ARMS:
        for seed in (1,2,3):
            run=ROOT/f"runs/round3-{arm}-seed{seed}"
            path=run/"models/best/best_model.zip"
            _,config,metadata=verify_round_three_model(path)
            selected=json.loads((run/"models/best/selection_metrics.json").read_text())
            models[f"{arm}-seed{seed}"]={"path":str(path),"arm":arm,"seed":seed,
                "sha256":sha256_file(path),"best_timesteps":selected["timesteps"],
                "best_updates":selected["policy_updates"],"training_steps":metadata["final_num_timesteps"],
                "validation_success_rate":selected["success_rate"]}
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_write_json(OUT/"method_seal.json",{"frozen_at":utc_now(),"models":models,
        "implementation_seal_sha256":sha256_file(SEAL),"orchestration_sources":source_hashes(),
        "dwa_config":protocol["dwa"]["config"],"statistics":protocol["statistics"],
        "final_tests":protocol["final_tests"],"method_changes_after_final_test":False})
    # Geometry only: final outcomes from round two are never used to choose parameters/models.
    used=set()
    prior=list((ROOT/"configs").rglob("*.json"))+list((ROOT/"results/round2").glob("sealed_*.json"))
    for file in prior:
        data=json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data,dict):used.update(geometric_hash(s) for s in data.get("scenarios",[]))
    seals={}
    for split in ("same_distribution","challenge"):
        settings=protocol["final_tests"][split]
        data=stratified(settings["seed"],settings["count"],split=="challenge")
        data["dataset_name"]=f"round3-final-{split}-{settings['seed']}"
        hashes={geometric_hash(s) for s in data["scenarios"]}
        if len(hashes)!=settings["count"] or hashes&used:raise ValueError("Duplicate final geometry: pause, do not change seeds")
        for scene in data["scenarios"]:replay_witness(scene)
        used.update(hashes)
        file=OUT/f"sealed_{split}.json";save_dataset(data,file)
        seals[split]={"file_sha256":sha256_file(file),"dataset_hash":dataset_hash(data),
                      "n":len(hashes),"witnesses_passed":len(hashes)}
    atomic_write_json(OUT/"dataset_seal.json",seals)
    print("Nine models + DWA frozen; 360 new scenes unique and witnessed; final inference not yet run.",flush=True)


def verify_seals():
    verify_implementation()
    method=json.loads((OUT/"method_seal.json").read_text())
    if method["orchestration_sources"]!=source_hashes() or method["implementation_seal_sha256"]!=sha256_file(SEAL):
        raise ValueError("Sealed implementation/orchestration changed")
    datasets=json.loads((OUT/"dataset_seal.json").read_text())
    for split,record in datasets.items():
        if sha256_file(OUT/f"sealed_{split}.json")!=record["file_sha256"]:raise ValueError("Final dataset changed")
    for model in method["models"].values():
        verify_round_three_model(model["path"])
        if sha256_file(model["path"])!=model["sha256"]:raise ValueError("Model changed")
    return method,datasets


def verify_complete(folder):
    record=json.loads((folder/"complete.json").read_text())
    if record["method_seal_sha256"]!=sha256_file(OUT/"method_seal.json"):raise ValueError("Wrong method seal")
    for path,digest in record["files"].items():
        if sha256_file(folder/path)!=digest:raise ValueError("Completed output changed")


def failure_plots(frame,steps,scenes,folder):
    failures=frame.loc[~frame.success,"scenario_id"].tolist()[:3]
    if not failures:return
    fig,axes=plt.subplots(3,2,figsize=(14,15),constrained_layout=True)
    for row,i in enumerate(failures):
        scene=scenes[i];s=steps[steps.scenario_index==i]
        agent=np.vstack([scene["start_position"],s[["agent_x","agent_y"]].to_numpy()])
        target=np.vstack([scene["dynamic_obstacle"]["position"],s[["target_x","target_y"]].to_numpy()])
        distances=np.linalg.norm(agent-target,axis=1);closest=int(distances.argmin())
        record=frame[frame.scenario_id==i].iloc[0]
        h={"scenario":scene,"agent_path":agent,"dynamic_path":target,"closest_step":closest,
           "headings":np.r_[scene["start_heading"],s.agent_heading],
           "linear_velocities":np.r_[0,s.actual_v],"angular_velocities":np.r_[0,s.actual_w],
           "dynamic_velocities":np.tile(scene["dynamic_obstacle"]["velocity"],(len(agent),1)),
           "record":{"scenario_id":i,"scenario_type":scene["scenario_type"],"difficulty":scene["difficulty"],
                     "termination_reason":record.reason,"min_dynamic_center_distance":float(distances.min()),
                     "min_dynamic_clearance":float(distances.min()-4),"closest_dynamic_time":closest*.2}}
        draw_full_subplot(axes[row,0],h);draw_closest_subplot(axes[row,1],h)
    for row in range(len(failures),3):axes[row,0].axis("off");axes[row,1].axis("off")
    fig.suptitle(f"First failures by ID: {failures} | full / closest encounter")
    fig.savefig(folder/"first_failures.png",dpi=130);plt.close(fig)


def audit_trajectories(frame,steps,scenes):
    for rec in frame.itertuples():
        s=steps[steps.scenario_index==rec.scenario_id];scene=scenes[rec.scenario_id]
        assert len(s)==rec.steps
        agent=np.vstack([scene["start_position"],s[["agent_x","agent_y"]].to_numpy()])
        target=np.vstack([scene["dynamic_obstacle"]["position"],s[["target_x","target_y"]].to_numpy()])
        v=np.r_[0,s.actual_v];w=np.r_[0,s.actual_w];h=np.r_[scene["start_heading"],s.agent_heading]
        assert np.max(np.abs(np.diff(v)))<=.300001 and np.max(np.abs(np.diff(w)))<=math.pi/20+1e-6
        assert v.min()>=-1e-6 and v.max()<=3.000001 and np.max(np.abs(w))<=math.pi/4+1e-6
        np.testing.assert_allclose(np.diff(agent,axis=0),.2*v[1:,None]*np.column_stack([np.cos(h[:-1]),np.sin(h[:-1])]),atol=1e-6)
        np.testing.assert_allclose((np.diff(h)+math.pi)%(2*math.pi)-math.pi,.2*w[1:],atol=1e-6)
        np.testing.assert_allclose(np.diff(target,axis=0),np.tile(np.array(scene["dynamic_obstacle"]["velocity"])*.2,(len(s),1)),atol=1e-6)
        static=min(float((np.linalg.norm(agent-o["position"],axis=1)-5).min()) for o in scene["static_obstacles"])
        dynamic=float((np.linalg.norm(agent-target,axis=1)-4).min())
        np.testing.assert_allclose([static,dynamic],[rec.min_static_clearance,rec.min_dynamic_clearance],atol=1e-6)
        np.testing.assert_allclose(s.base_reward+s.risk_penalty,s.training_reward,atol=1e-6)
        assert s.risk_penalty.iloc[-1]==0
        if rec.success:assert static>0 and dynamic>0 and np.linalg.norm(agent[-1]-scene["goal_position"])<=3.000001


def mechanism_summary(frame,steps):
    result={"weighting":"Step distributions are pooled within outcome; episode summaries are episode-weighted."}
    for outcome,episodes in (("all",frame),("success",frame[frame.success]),("failure",frame[~frame.success])):
        if episodes.empty:
            result[outcome]={"episodes":0};continue
        s=steps[steps.scenario_index.isin(episodes.scenario_id)]
        high=s[s.risk_predictive>=.5]
        record={"episodes":len(episodes),"steps":len(s),"high_risk_steps":len(high),"episode_metrics":summarize_frame(episodes)}
        for field in ("actual_v","delta_v","actual_w"):
            record["high_risk_"+field]={"mean":None if high.empty else float(high[field].mean()),
                "quantiles_0_25_50_75_100":None if high.empty else high[field].quantile([0,.25,.5,.75,1]).tolist()}
        for field in ("reward_progress","reward_static_safety","reward_dynamic_safety","reward_time","reward_terminal","base_reward","risk_penalty"):
            record[field]={"total":float(s[field].sum()),"mean_episode_total":float(s[field].sum()/len(episodes)),
                           "nonzero_steps":int((s[field]!=0).sum())}
        result[outcome]=record
    return result


def final_evaluation():
    methods,datasets=verify_seals()
    for split in ("same_distribution","challenge"):
        file=OUT/f"sealed_{split}.json";data=load_dataset(file)
        for name,item in {**methods["models"],"DWA":None}.items():
            dest=OUT/"final"/split/name
            if (dest/"complete.json").exists():verify_complete(dest);continue
            dest.mkdir(parents=True,exist_ok=False)
            policy=DWA(DWAConfig(**methods["dwa_config"])) if item is None else PPO.load(item["path"],device="cpu")
            env=RiskRewardWrapper(DynamicPathPlanningEnv(scenario_file=file),"base" if item is None else item["arm"])
            try:_,frame=evaluate_policy(policy,env,len(data["scenarios"]),dest/"steps.csv.gz")
            finally:env.close()
            steps=pd.read_csv(dest/"steps.csv.gz")
            audit_trajectories(frame,steps,data["scenarios"])
            frame.to_csv(dest/"episodes.csv",index=False)
            summary=summarize_frame(frame)
            summary.update(dataset_hash=datasets[split]["dataset_hash"],method=name,split=split,
                           model_sha256=None if item is None else item["sha256"],trajectory_audit="passed")
            atomic_write_json(dest/"summary.json",summary)
            atomic_write_json(dest/"mechanism.json",mechanism_summary(frame,steps))
            atomic_write_json(dest/"by_type.json",[{"type":kind,**summarize_frame(g)} for kind,g in frame.groupby("scenario_type")])
            failure_plots(frame,steps,data["scenarios"],dest)
            atomic_write_json(dest/"complete.json",{"completed_at":utc_now(),
                "method_seal_sha256":sha256_file(OUT/"method_seal.json"),
                "files":{p.name:sha256_file(p) for p in dest.iterdir() if p.is_file()}})
            print(split,name,summary["success_count"],"/",summary["n"],flush=True)


def summarize():
    methods,_=verify_seals();rows=[];stats=[];pairs=[];failures=[];types=[]
    for split in ("same_distribution","challenge"):
        frames={}
        for name in [*methods["models"],"DWA"]:
            folder=OUT/"final"/split/name;verify_complete(folder)
            s=json.loads((folder/"summary.json").read_text());rows.append({k:v for k,v in s.items() if not isinstance(v,(dict,list))})
            f=pd.read_csv(folder/"episodes.csv").sort_values("scenario_id");frames[name]=f
            for reason,g in f[~f.success].groupby("reason"):
                failures.append({"split":split,"method":name,"reason":reason,"count":len(g),"ids":" ".join(map(str,g.scenario_id))})
            for kind,g in f.groupby("scenario_type"):
                types.append({"split":split,"method":name,"type":kind,**summarize_frame(g)})
        anchor=frames["DWA"]
        for f in frames.values():
            assert np.array_equal(f.scenario_id,anchor.scenario_id) and np.array_equal(f.scenario_type,anchor.scenario_type)
        matrices={arm:np.stack([frames[f"{arm}-seed{s}"].success.to_numpy(dtype=float) for s in (1,2,3)]) for arm in ARMS}
        for arm in ARMS:
            subset=[r for r in rows if r["split"]==split and r["method"].startswith(arm+"-")]
            for metric in ("success_rate","collision_rate","static_collision_rate","dynamic_collision_rate","timeout_rate",
                           "successful_path_efficiency","successful_navigation_time","mean_base_reward","mean_risk_penalty",
                           "full_speed_fraction_after_2s","clipped_mean_fraction"):
                v=np.array([r[metric] for r in subset],dtype=float);mean=float(v.mean());sd=float(v.std(ddof=1));margin=4.3026527299*sd/np.sqrt(3)
                low,high=mean-margin,mean+margin
                if metric.endswith("rate") or "fraction" in metric or "efficiency" in metric:low=max(0.,low);high=min(1.,high)
                stats.append({"split":split,"arm":arm,"metric":metric,"mean":mean,"sample_sd":sd,"seed_t_low":low,"seed_t_high":high})
        for left,right in (("predictive","instant"),("predictive","base"),("instant","base"),("predictive","DWA")):
            other=anchor.success.to_numpy(dtype=float)[None,:] if right=="DWA" else matrices[right]
            pairs.append({"split":split,"comparison":f"{left} minus {right}","primary":split=="same_distribution" and right=="instant",
                          **paired_interval(matrices[left]-other,anchor.scenario_type,10000,5821)})
            for kind in sorted(anchor.scenario_type.unique()):
                mask=anchor.scenario_type.eq(kind).to_numpy()
                pairs.append({"split":split,"scenario_type":kind,"comparison":f"{left} minus {right}","primary":False,
                              **paired_interval((matrices[left]-other)[:,mask],anchor.scenario_type[mask],10000,5821)})
    pd.DataFrame(rows).to_csv(OUT/"per_seed_metrics.csv",index=False)
    pd.DataFrame(stats).to_csv(OUT/"seed_statistics.csv",index=False)
    pd.DataFrame(types).to_csv(OUT/"by_type.csv",index=False)
    pd.DataFrame(failures).to_csv(OUT/"failure_catalog.csv",index=False)
    atomic_write_json(OUT/"paired_comparisons.json",pairs)
    lines=["# P3-05 final results","","No parameter search. Three seeds per PPO arm; DWA is a single deterministic baseline.",
           "","| Split | Method | Success | Static collision | Dynamic collision | Timeout |","|---|---|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['split']} | {r['method']} | {r['success_count']}/{r['n']} | {r['static_collision_count']} | {r['dynamic_collision_count']} | {r['timeout_count']} |")
    lines += ["","## Seed means and sample SD",""]
    for s in stats:
        if s["metric"]=="success_rate":lines.append(f"- {s['split']} {s['arm']}: {s['mean']*100:.2f}% +/- {s['sample_sd']*100:.2f} pp.")
    lines += ["","## Paired hierarchical bootstrap (95%)",""]
    for p in pairs:
        if "scenario_type" in p:continue
        lo,hi=p["bootstrap_95"];lines.append(f"- {p['split']}, {p['comparison']}: {p['mean_difference']*100:.2f} pp [{lo*100:.2f}, {hi*100:.2f}]"+(" (PRIMARY)" if p["primary"] else ""))
    lines += ["","Only three seeds: uncertain intervals, secondary comparisons not multiplicity adjusted.",
              "Success-conditional efficiency/time uses different surviving subsets. Do not compare shaped returns across arms.",
              "Braking delay excludes censored nonresponders; their counts are reported separately.",
              "All completed trajectories passed kinematic, clearance and reward consistency checks."]
    (OUT/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,split in zip(axes,("same_distribution","challenge")):
        groups=[[r["success_rate"]*100 for r in rows if r["split"]==split and r["method"].startswith(a+"-")] for a in ARMS]
        dwa=next(r["success_rate"]*100 for r in rows if r["split"]==split and r["method"]=="DWA")
        ax.bar(range(4),[np.mean(x) for x in groups]+[dwa],color=["#777777","#4477aa","#ee8833","#228833"],alpha=.65)
        for i,g in enumerate(groups):ax.scatter(i+np.linspace(-.12,.12,3),g,c="black",s=25,zorder=3)
        ax.set_xticks(range(4),[*ARMS,"DWA"]);ax.set_ylim(0,105);ax.set_ylabel("Success (%)");ax.set_title(split.replace("_"," "))
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("P3-05 | seed means and individual seeds | sealed final evaluation")
    fig.savefig(OUT/"success_comparison.png",dpi=160);plt.close(fig)
    atomic_write_json(OUT/"artifact_manifest.json",{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"})
    print("\n".join(lines))


def smoke():
    p=load_protocol();verify_implementation()
    with tempfile.TemporaryDirectory() as temp:
        data=load_dataset(ROOT/"configs/round1/typical_cases.json");env=RiskRewardWrapper(DynamicPathPlanningEnv(scenario_file=ROOT/"configs/round1/typical_cases.json"),"predictive")
        try:_,frame=evaluate_policy(DWA(DWAConfig(**p["dwa"]["config"])),env,3,Path(temp)/"steps.csv.gz")
        finally:env.close()
        audit_trajectories(frame,pd.read_csv(Path(temp)/"steps.csv.gz"),data["scenarios"])
        assert summarize_frame(frame)["n"]==3
        assert mechanism_summary(frame,pd.read_csv(Path(temp)/"steps.csv.gz"))["all"]["episodes"]==3
        # Render smoke on a copied display record only, not a manufactured experiment result.
        display=frame.copy();display.loc[0,"success"]=False
        failure_plots(display,pd.read_csv(Path(temp)/"steps.csv.gz"),data["scenarios"],Path(temp))
        assert paired_interval(np.zeros((3,4)),["a","a","b","b"],100,5821)["bootstrap_95"]==[0,0]
    print("P3-05 orchestration smoke passed on three development scenes; no final scenes read/generated.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["smoke","seal","evaluate","summarize"])
    {"smoke":smoke,"seal":seal_methods,"evaluate":final_evaluation,"summarize":summarize}[parser.parse_args().mode]()
