"""P4-04/P4-05: freeze methods, create held-out data, evaluate and summarize."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.dwa import DWA, DWAConfig
from envs.scenario_dataset import load_dataset, save_dataset, dataset_hash
from experiment_utils import atomic_write_json, sha256_file, utc_now
from round_four_policy import ARMS, LatentActionPPO
from round_four_utils import SEAL, load_protocol, make_env, verify_implementation, verify_model
from scripts.evaluate import wilson_interval, draw_full_subplot, draw_closest_subplot
from scripts.evaluate_round_four import evaluate_policy
from scripts.round_two_experiment import stratified, geometric_hash
from scripts.summarize_round_two import paired_interval
from scripts.test_round_one import replay_witness

OUT = ROOT / "results/round4/final_experiment"
SOURCES = ("scripts/round_four_experiment.py", "scripts/round_two_experiment.py",
           "scripts/summarize_round_two.py", "scripts/test_round_one.py",
           "configs/round1/protocol.json")


def source_hashes():
    return {path: sha256_file(ROOT/path) for path in SOURCES}


def summarize_frame(frame):
    success = frame[frame.success]
    result = dict(n=len(frame), success_count=int(frame.success.sum()),
        success_rate=float(frame.success.mean()),
        success_wilson_95=wilson_interval(int(frame.success.sum()), len(frame)),
        mean_base_reward=float(frame.base_return.mean()),
        mean_min_static_clearance=float(frame.min_static_clearance.mean()),
        mean_min_dynamic_clearance=float(frame.min_dynamic_clearance.mean()),
        worst_static_clearance=float(frame.min_static_clearance.min()),
        worst_dynamic_clearance=float(frame.min_dynamic_clearance.min()),
        successful_path_efficiency=None if success.empty else float(success.path_efficiency.mean()),
        successful_navigation_time=None if success.empty else float(success.navigation_time.mean()),
        full_speed_fraction_after_2s=float(frame.full_speed_fraction_after_2s.fillna(0).mean()),
        near_upper_action_fraction_after_2s=float(frame.near_upper_action_fraction_after_2s.fillna(0).mean()),
        high_risk_mean_speed_episode_average=None if frame.high_risk_mean_speed.isna().all() else float(frame.high_risk_mean_speed.mean()),
        high_risk_braking_fraction_episode_average=None if frame.high_risk_braking_fraction.isna().all() else float(frame.high_risk_braking_fraction.mean()),
        risk_exposed_episodes=int(frame.risk_onset_step.notna().sum()),
        risk_no_response_episodes=int(frame.risk_no_braking_response.sum()),
        risk_no_response_fraction_exposed=None if not frame.risk_onset_step.notna().any() else float(frame.risk_no_braking_response.sum()/frame.risk_onset_step.notna().sum()),
        braking_delay_responders_only=None if frame.braking_delay_seconds.isna().all() else float(frame.braking_delay_seconds.mean()),
        timeout_with_stagnation_count=int(frame.timeout_with_stagnation.sum()))
    masks = {"static_collision": frame.reason.str.startswith("static_collision"),
             "dynamic_collision": frame.reason.eq("dynamic_collision"),
             "timeout": frame.reason.eq("timeout"),
             "out_of_bounds": frame.reason.eq("out_of_bounds"),
             "collision": frame.collision}
    for name, mask in masks.items():
        result[name+"_count"] = int(mask.sum())
        result[name+"_rate"] = float(mask.mean())
    return result


def freeze_methods_and_data():
    p = load_protocol(); verify_implementation()
    if (OUT/"method_seal.json").exists():
        raise FileExistsError("P4 methods already sealed")
    models = {}
    for arm in ARMS:
        for seed in (1,2,3):
            path = ROOT/f"runs/round4-{arm}-seed{seed}/models/best/best_model.zip"
            _, config, meta = verify_model(path)
            selected = json.loads((path.parent/"selection_metrics.json").read_text(encoding="utf-8"))
            models[f"{arm}-seed{seed}"] = dict(path=str(path), arm=arm, seed=seed,
                sha256=sha256_file(path), best_timesteps=selected["timesteps"],
                best_updates=selected["policy_updates"], training_steps=meta["final_num_timesteps"],
                validation_success_rate=selected["success_rate"], action_schema=config["action_schema"])
    selection = json.loads((ROOT/p["dwa"]["selection"]).read_text(encoding="utf-8"))
    if selection["selected"]["id"] != p["dwa"]["candidate"]:
        raise ValueError("DWA selection does not match protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT/"method_seal.json", dict(frozen_at=utc_now(), models=models,
        implementation_seal_sha256=sha256_file(SEAL), orchestration_sources=source_hashes(),
        dwa_config=selection["selected"]["config"], statistics=p["statistics"],
        final_tests=p["final_tests"], method_changes_after_final_test=False))

    # Geometry-only deduplication: no prior outcomes are inspected for design choices.
    used = set(); scanned = []
    candidates = list((ROOT/"configs").rglob("*.json")) + list((ROOT/"results").rglob("sealed_*.json"))
    for file in candidates:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        scenes = data.get("scenarios", []) if isinstance(data, dict) else []
        if scenes:
            used.update(geometric_hash(scene) for scene in scenes)
            scanned.append(str(file.relative_to(ROOT)))
    seals = {}
    for split in ("same_distribution", "challenge"):
        settings = p["final_tests"][split]
        data = stratified(settings["seed"], settings["count"], split=="challenge")
        data["dataset_name"] = f"round4-final-{split}-{settings['seed']}"
        hashes = {geometric_hash(scene) for scene in data["scenarios"]}
        if len(hashes) != settings["count"] or hashes & used:
            raise ValueError("Duplicate P4 final geometry; pause without changing seeds")
        for scene in data["scenarios"]:
            replay_witness(scene)
        used.update(hashes)
        file = OUT/f"sealed_{split}.json"; save_dataset(data,file)
        seals[split] = dict(file_sha256=sha256_file(file), dataset_hash=dataset_hash(data),
                            n=len(hashes), witnesses_passed=len(hashes))
    atomic_write_json(OUT/"dataset_seal.json", dict(datasets=seals,
        prior_geometry_files_scanned=sorted(scanned), final_sets_mutually_disjoint=True))
    print("Six P4 models + DWA frozen; 360 new scenes unique and witnessed; no final inference yet.", flush=True)


def verify_seals():
    verify_implementation()
    methods = json.loads((OUT/"method_seal.json").read_text(encoding="utf-8"))
    if methods["orchestration_sources"] != source_hashes() or methods["implementation_seal_sha256"] != sha256_file(SEAL):
        raise ValueError("P4 implementation/orchestration changed after freeze")
    data = json.loads((OUT/"dataset_seal.json").read_text(encoding="utf-8"))
    for split, record in data["datasets"].items():
        file = OUT/f"sealed_{split}.json"
        if sha256_file(file) != record["file_sha256"] or dataset_hash(load_dataset(file)) != record["dataset_hash"]:
            raise ValueError("P4 final dataset changed")
    for model in methods["models"].values():
        verify_model(model["path"])
        if sha256_file(model["path"]) != model["sha256"]:
            raise ValueError("P4 model changed")
    return methods, data["datasets"]


def verify_complete(folder):
    record = json.loads((folder/"complete.json").read_text(encoding="utf-8"))
    if record["method_seal_sha256"] != sha256_file(OUT/"method_seal.json"):
        raise ValueError("Wrong P4 method seal")
    for relative, digest in record["files"].items():
        if sha256_file(folder/relative) != digest:
            raise ValueError("Completed P4 output changed")


def audit_trajectories(frame, steps, scenes):
    for rec in frame.itertuples():
        trace = steps[steps.scenario_index == rec.Index]
        scene = scenes[rec.Index]
        if len(trace) != rec.steps:
            raise ValueError("P4 step count mismatch")
        agent = np.vstack([scene["start_position"], trace[["agent_x","agent_y"]].to_numpy()])
        target = np.vstack([scene["dynamic_obstacle"]["position"], trace[["target_x","target_y"]].to_numpy()])
        velocity = np.r_[0, trace.actual_v]; yaw = np.r_[0, trace.actual_w]
        heading = np.r_[scene["start_heading"], trace.agent_heading]
        if np.max(np.abs(np.diff(velocity))) > .300001 or np.max(np.abs(np.diff(yaw))) > math.pi/20+1e-6:
            raise ValueError("P4 acceleration contract failed")
        np.testing.assert_allclose(np.diff(agent,axis=0), .2*velocity[1:,None]*np.column_stack([np.cos(heading[:-1]),np.sin(heading[:-1])]), atol=1e-6)
        np.testing.assert_allclose((np.diff(heading)+math.pi)%(2*math.pi)-math.pi, .2*yaw[1:], atol=1e-6)
        np.testing.assert_allclose(np.diff(target,axis=0), np.tile(np.asarray(scene["dynamic_obstacle"]["velocity"])*.2,(len(trace),1)), atol=1e-6)
        static = min(float((np.linalg.norm(agent-o["position"],axis=1)-5).min()) for o in scene["static_obstacles"])
        dynamic = float((np.linalg.norm(agent-target,axis=1)-4).min())
        np.testing.assert_allclose([static,dynamic],[rec.min_static_clearance,rec.min_dynamic_clearance],atol=1e-6)
        np.testing.assert_allclose(trace.base_reward,trace.training_reward,atol=1e-7)
        if not (trace.risk_penalty == 0).all():
            raise ValueError("P4 unexpectedly changed reward")


def mechanism_summary(frame, steps):
    result = {"weighting":"Step metrics are pooled within outcome; episode metrics are episode-weighted."}
    for outcome, episodes in (("all",frame),("success",frame[frame.success]),("failure",frame[~frame.success])):
        trace = steps[steps.scenario_index.isin(episodes.index)]
        warm = trace[trace.step_count > 10]
        high = warm[warm.pre_action_q4 >= .5]
        item = dict(episodes=len(episodes), steps=len(trace), high_risk_steps=len(high),
                    episode_metrics=None if episodes.empty else summarize_frame(episodes))
        for field in ("actual_v","delta_v","actual_w","executed_speed_action","executed_turn_action"):
            item["high_risk_"+field] = None if high.empty else dict(mean=float(high[field].mean()),
                quantiles=high[field].quantile([0,.25,.5,.75,1]).tolist())
        item["high_risk_braking_fraction"] = None if high.empty else float((high.delta_v<=-.1+1e-9).mean())
        item["near_upper_action_fraction_after_2s"] = None if warm.empty else float((warm.executed_speed_action>=.99).mean())
        if "latent_speed" in trace:
            item["latent_outside_fraction_after_2s"] = None if warm.empty else float((warm.latent_speed.abs()>1).mean())
            item["latent_speed_mean_after_2s"] = None if warm.empty else float(warm.latent_speed.mean())
            item["latent_speed_std_after_2s"] = None if warm.empty else float(warm.latent_speed.std())
        result[outcome] = item
    return result


def failure_plots(frame, steps, scenes, folder):
    failures = frame.index[~frame.success].tolist()[:3]
    if not failures:
        return
    fig, axes = plt.subplots(3,2,figsize=(14,15),constrained_layout=True)
    for row,index in enumerate(failures):
        scene=scenes[index]; trace=steps[steps.scenario_index==index]
        agent=np.vstack([scene["start_position"],trace[["agent_x","agent_y"]].to_numpy()])
        target=np.vstack([scene["dynamic_obstacle"]["position"],trace[["target_x","target_y"]].to_numpy()])
        distances=np.linalg.norm(agent-target,axis=1); closest=int(distances.argmin()); rec=frame.loc[index]
        history={"scenario":scene,"agent_path":agent,"dynamic_path":target,"closest_step":closest,
            "headings":np.r_[scene["start_heading"],trace.agent_heading],
            "linear_velocities":np.r_[0,trace.actual_v],"angular_velocities":np.r_[0,trace.actual_w],
            "dynamic_velocities":np.tile(scene["dynamic_obstacle"]["velocity"],(len(agent),1)),
            "record":{"scenario_id":index,"scenario_type":scene["scenario_type"],"difficulty":scene["difficulty"],
                "termination_reason":rec.reason,"min_dynamic_center_distance":float(distances.min()),
                "min_dynamic_clearance":float(distances.min()-4),"closest_dynamic_time":closest*.2}}
        draw_full_subplot(axes[row,0],history); draw_closest_subplot(axes[row,1],history)
    for row in range(len(failures),3):
        axes[row,0].axis("off"); axes[row,1].axis("off")
    fig.suptitle(f"P4 first failures by ID: {failures}")
    fig.savefig(folder/"first_failures.png",dpi=130); plt.close(fig)


def final_evaluation():
    methods, datasets = verify_seals()
    for split in ("same_distribution","challenge"):
        file=OUT/f"sealed_{split}.json"; data=load_dataset(file)
        for name,item in {**methods["models"],"DWA":None}.items():
            folder=OUT/"final"/split/name
            if (folder/"complete.json").exists():
                verify_complete(folder); continue
            folder.mkdir(parents=True,exist_ok=False)
            if item is None:
                policy=DWA(DWAConfig(**methods["dwa_config"]))
            else:
                policy=LatentActionPPO.load(item["path"],device="cpu")
                if policy.arm != item["arm"]:
                    raise ValueError("P4 loaded model arm mismatch")
            env=make_env(file)
            try:
                _, frame=evaluate_policy(policy,env,len(data["scenarios"]),folder/"steps.csv.gz")
            finally:
                env.close()
            frame.index.name="scenario_index"
            steps=pd.read_csv(folder/"steps.csv.gz")
            audit_trajectories(frame,steps,data["scenarios"])
            frame.to_csv(folder/"episodes.csv")
            summary=summarize_frame(frame)
            summary.update(dataset_hash=datasets[split]["dataset_hash"],method=name,split=split,
                model_sha256=None if item is None else item["sha256"],trajectory_audit="passed")
            atomic_write_json(folder/"summary.json",summary)
            atomic_write_json(folder/"mechanism.json",mechanism_summary(frame,steps))
            atomic_write_json(folder/"by_type.json",[{"type":kind,**summarize_frame(group)} for kind,group in frame.groupby("scenario_type")])
            failure_plots(frame,steps,data["scenarios"],folder)
            atomic_write_json(folder/"complete.json",dict(completed_at=utc_now(),
                method_seal_sha256=sha256_file(OUT/"method_seal.json"),
                files={path.name:sha256_file(path) for path in folder.iterdir() if path.is_file()}))
            print(split,name,summary["success_count"],"/",summary["n"],flush=True)


def training_summary():
    rows=[]
    for arm in ARMS:
        for seed in (1,2,3):
            file=ROOT/f"runs/round4-{arm}-seed{seed}/logs/training_steps.csv.gz"
            trace=pd.read_csv(file)
            if len(trace)!=200704 or not np.array_equal(trace.global_step,np.arange(1,200705)):
                raise ValueError("P4 training trace incomplete")
            warm=trace[trace.step_count>10]
            for phase,lo,hi in (("early",0,50000),("middle",50000,150000),("late",150000,200704)):
                part=warm[(warm.global_step>lo)&(warm.global_step<=hi)]
                high=part[part.pre_action_q4>=.5]
                rows.append(dict(arm=arm,seed=seed,phase=phase,steps=len(part),high_risk_steps=len(high),
                    latent_outside_fraction=float((part.latent_speed.abs()>1).mean()),
                    near_upper_action_fraction=float((part.executed_speed_action>=.99).mean()),
                    full_speed_fraction=float((part.actual_v>=2.99).mean()),
                    braking_fraction=float((part.delta_v<=-.1+1e-9).mean()),
                    high_risk_braking_fraction=None if high.empty else float((high.delta_v<=-.1+1e-9).mean()),
                    mean_latent_speed=float(part.latent_speed.mean()),mean_executed_speed_action=float(part.executed_speed_action.mean())))
    pd.DataFrame(rows).to_csv(OUT/"training_action_summary.csv",index=False)


def summarize():
    methods,_=verify_seals(); rows=[]; stats=[]; pairs=[]; failures=[]; types=[]
    for split in ("same_distribution","challenge"):
        frames={}
        for name in [*methods["models"],"DWA"]:
            folder=OUT/"final"/split/name; verify_complete(folder)
            summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            rows.append({k:v for k,v in summary.items() if not isinstance(v,(dict,list))})
            frame=pd.read_csv(folder/"episodes.csv",index_col="scenario_index").sort_index(); frames[name]=frame
            for reason,group in frame[~frame.success].groupby("reason"):
                failures.append(dict(split=split,method=name,reason=reason,count=len(group),ids=" ".join(map(str,group.index))))
            for kind,group in frame.groupby("scenario_type"):
                types.append(dict(split=split,method=name,type=kind,**summarize_frame(group)))
        anchor=frames["DWA"]
        for frame in frames.values():
            if not np.array_equal(frame.index,anchor.index) or not np.array_equal(frame.scenario_type,anchor.scenario_type):
                raise ValueError("P4 paired scene order mismatch")
        matrices={arm:np.stack([frames[f"{arm}-seed{seed}"].success.to_numpy(float) for seed in (1,2,3)]) for arm in ARMS}
        for arm in ARMS:
            subset=[row for row in rows if row["split"]==split and row["method"].startswith(arm+"-")]
            for metric in ("success_rate","collision_rate","static_collision_rate","dynamic_collision_rate","timeout_rate",
                           "successful_path_efficiency","successful_navigation_time","mean_base_reward",
                           "full_speed_fraction_after_2s","near_upper_action_fraction_after_2s",
                           "high_risk_braking_fraction_episode_average","risk_no_response_fraction_exposed"):
                values=np.array([row[metric] for row in subset],float)
                mean=float(np.nanmean(values)); sd=float(np.nanstd(values,ddof=1)); margin=4.3026527299*sd/np.sqrt(3)
                low,high=mean-margin,mean+margin
                if metric.endswith("rate") or "fraction" in metric or "efficiency" in metric:
                    low=max(0.,low); high=min(1.,high)
                stats.append(dict(split=split,arm=arm,metric=metric,mean=mean,sample_sd=sd,seed_t_low=low,seed_t_high=high))
        other=anchor.success.to_numpy(float)[None,:]
        comparisons=(("speed_tanh","clip",matrices["clip"]),("speed_tanh","DWA",other),("clip","DWA",other))
        for left,right,right_values in comparisons:
            difference=matrices[left]-right_values
            pairs.append(dict(split=split,comparison=f"{left} minus {right}",primary=split=="same_distribution" and right=="clip",
                              **paired_interval(difference,anchor.scenario_type,10000,6821)))
            for kind in sorted(anchor.scenario_type.unique()):
                mask=anchor.scenario_type.eq(kind).to_numpy()
                pairs.append(dict(split=split,scenario_type=kind,comparison=f"{left} minus {right}",primary=False,
                    **paired_interval(difference[:,mask],anchor.scenario_type[mask],10000,6821)))
    pd.DataFrame(rows).to_csv(OUT/"per_seed_metrics.csv",index=False)
    pd.DataFrame(stats).to_csv(OUT/"seed_statistics.csv",index=False)
    pd.DataFrame(types).to_csv(OUT/"by_type.csv",index=False)
    pd.DataFrame(failures).to_csv(OUT/"failure_catalog.csv",index=False)
    atomic_write_json(OUT/"paired_comparisons.json",pairs)
    training_summary()
    lines=["# P4 final results","","Pre-registered speed mapping comparison; no post-test tuning.","",
           "| Split | Method | Success | Static collision | Dynamic collision | Timeout | Stagnant timeout |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['method']} | {row['success_count']}/{row['n']} | {row['static_collision_count']} | {row['dynamic_collision_count']} | {row['timeout_count']} | {row['timeout_with_stagnation_count']} |")
    lines += ["","## Seed success means",""]
    for item in stats:
        if item["metric"]=="success_rate":
            lines.append(f"- {item['split']} {item['arm']}: {item['mean']*100:.2f}% +/- {item['sample_sd']*100:.2f} pp (seed sample SD).")
    lines += ["","## Paired hierarchical bootstrap (95%)",""]
    for item in pairs:
        if "scenario_type" in item: continue
        lo,hi=item["bootstrap_95"]
        lines.append(f"- {item['split']}, {item['comparison']}: {item['mean_difference']*100:.2f} pp [{lo*100:.2f}, {hi*100:.2f}]"+(" (PRIMARY)" if item["primary"] else ""))
    lines += ["","Only three training seeds; secondary comparisons are exploratory and not multiplicity-adjusted.",
              "Success-conditional efficiency/time compares different surviving episode sets.",
              "Braking delay is responder-only; censored nonresponses are separately reported.",
              "All completed trajectories passed motion, clearance, and unchanged-reward audits."]
    (OUT/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,split in zip(axes,("same_distribution","challenge")):
        groups=[[row["success_rate"]*100 for row in rows if row["split"]==split and row["method"].startswith(arm+"-")] for arm in ARMS]
        dwa=next(row["success_rate"]*100 for row in rows if row["split"]==split and row["method"]=="DWA")
        ax.bar(range(3),[np.mean(group) for group in groups]+[dwa],color=["#777777","#ee8833","#228833"],alpha=.7)
        for i,group in enumerate(groups): ax.scatter(i+np.linspace(-.1,.1,3),group,c="black",s=24,zorder=3)
        ax.set_xticks(range(3),["clip","speed_tanh","DWA"]); ax.set_ylim(0,105); ax.set_ylabel("Success (%)"); ax.set_title(split.replace("_"," ")); ax.grid(axis="y",alpha=.2)
    fig.suptitle("P4 sealed final evaluation")
    fig.savefig(OUT/"success_comparison.png",dpi=160); plt.close(fig)
    atomic_write_json(OUT/"artifact_manifest.json",{str(path.relative_to(OUT)):sha256_file(path) for path in OUT.rglob("*") if path.is_file() and path.name!="artifact_manifest.json"})
    print("\n".join(lines))


def smoke():
    load_protocol(); verify_implementation()
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp); env=make_env(ROOT/"configs/round1/typical_cases.json")
        try:
            _,frame=evaluate_policy(DWA(DWAConfig()),env,3,path/"steps.csv.gz")
        finally: env.close()
        frame.index.name="scenario_index"; steps=pd.read_csv(path/"steps.csv.gz")
        audit_trajectories(frame,steps,load_dataset(ROOT/"configs/round1/typical_cases.json")["scenarios"])
        assert summarize_frame(frame)["n"]==3
        assert mechanism_summary(frame,steps)["all"]["episodes"]==3
        failure_plots(frame.assign(success=False),steps,load_dataset(ROOT/"configs/round1/typical_cases.json")["scenarios"],path)
        assert paired_interval(np.zeros((3,4)),["a","a","b","b"],100,6821)["bootstrap_95"]==[0,0]
    print("P4 orchestration smoke passed on development data; no final data generated.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("smoke","seal","evaluate","summarize"))
    {"smoke":smoke,"seal":freeze_methods_and_data,"evaluate":final_evaluation,"summarize":summarize}[parser.parse_args().mode]()
