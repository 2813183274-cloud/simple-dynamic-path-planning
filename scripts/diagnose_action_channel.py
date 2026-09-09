"""P4-01: frozen-policy action and braking diagnosis on development data only."""
import argparse
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.risk_reward import risk_metrics
from envs.scenario_dataset import load_dataset
from experiment_utils import atomic_write_json, sha256_file, utc_now
from round_three_utils import implementation_manifest, verify_implementation, verify_round_three_model

OUT = ROOT / "results/round4/action_diagnosis"


def probability_below(threshold, mean, std):
    return .5 * math.erfc((mean - threshold) / (std * math.sqrt(2)))


def distribution_record(mean, std, v, rng):
    samples = rng.normal(mean, std, 64)
    clipped = np.clip(samples, -1, 1)
    targets = (clipped + 1) * 1.5
    executed = np.clip(targets, max(0., v-.3), min(3., v+.3))
    return {
        "mean": mean, "std": std, "variance": std**2,
        "p_upper_clip": 1-probability_below(1., mean, std),
        "p_brake_0_1": probability_below(2*(v-.1)/3-1, mean, std) if v>=.1 else 0.,
        "sample_mean": float(samples.mean()), "sample_clipped_mean": float(clipped.mean()),
        "sample_target_v_mean": float(targets.mean()),
        "sample_actual_v_mean": float(executed.mean()),
        "sample_brake_fraction": float((executed <= v-.1+1e-9).mean()),
    }


def brake_branch(original):
    """Maximum braking and target zero yaw rate until stopped; not goal-navigation policy."""
    env = copy.deepcopy(original)
    distance = 0.; previous = env.agent_position.copy(); initial_v = env.current_linear_velocity
    trace = []; reason = "running"
    try:
        for _ in range(20):
            v = env.current_linear_velocity
            _, _, done, truncated, info = env.step(np.array([-1., 0.], dtype=np.float32))
            np.testing.assert_allclose(env.current_linear_velocity, max(0., v-.3), atol=1e-9)
            distance += float(np.linalg.norm(env.agent_position-previous)); previous = env.agent_position.copy()
            trace.append({"x":float(previous[0]),"y":float(previous[1]),"v":env.current_linear_velocity,
                          "w":env.current_angular_velocity,"reason":info["termination_reason"]})
            reason = info["termination_reason"]
            if done or truncated or env.current_linear_velocity <= 1e-8: break
        stopped = env.current_linear_velocity <= 1e-8
        return {"initial_v":initial_v,"stopped":stopped,"reason":reason,
                "collision":reason.startswith("static_collision") or reason=="dynamic_collision",
                "safe_stop":stopped and not done and not truncated,
                "distance":distance,"duration":len(trace)*.2,"trace":trace}
    finally:
        env.close()


def test_helpers():
    assert abs(probability_below(0, 0, 1)-.5)<1e-12
    assert probability_below(-10, 0, 1)<1e-20
    scene = load_dataset(ROOT/"configs/round1/typical_cases.json")["scenarios"][0]
    env = DynamicPathPlanningEnv(stage="D", randomize_scenario=False)
    env.set_scenario(scene);env.reset()
    env.agent_position=np.array([10.,10.]);env.agent_heading=0.
    env.static_obstacles=np.array([[60.,60.],[80.,80.]])
    env.dynamic_position=np.array([70.,20.]);env.current_linear_velocity=3.
    result=brake_branch(env)
    assert result["safe_stop"]
    np.testing.assert_allclose(result["distance"],2.7,atol=1e-8)
    assert env.current_linear_velocity==3. and np.array_equal(env.agent_position,[10.,10.])
    env.close()


def training_history(run, name):
    columns=["global_step","step_count","speed_action","actual_v","delta_v","risk_predictive","termination_reason"]
    f=pd.read_csv(run/"logs/training_steps.csv.gz",usecols=columns)
    assert len(f)==200704 and np.array_equal(f.global_step,np.arange(1,200705))
    episode=(f.step_count==1).cumsum()
    endings=f.groupby(episode).termination_reason.last()
    f["outcome"]=episode.map(endings)
    rows=[]
    for phase,(low,high) in {"early":(0,50000),"middle":(50000,150000),"late":(150000,200704)}.items():
        for outcome in ("all","success","failure"):
            g=f[(f.global_step>low)&(f.global_step<=high)&(f.step_count>10)]
            if outcome=="success":g=g[g.outcome=="success"]
            if outcome=="failure":g=g[~g.outcome.isin(["success","running"])]
            risk=g[g.risk_predictive>=.5]
            rows.append({"method":name,"phase":phase,"outcome":outcome,"n":len(g),
                "speed_upper_clip_fraction":None if g.empty else float((g.speed_action>=1).mean()),
                "brake_fraction":None if g.empty else float((g.delta_v<=-.1+1e-9).mean()),
                "high_risk_steps":len(risk),
                "high_risk_brake_fraction":None if risk.empty else float((risk.delta_v<=-.1+1e-9).mean())})
    return rows


def diagnose(output):
    test_helpers();verify_implementation()
    sources=implementation_manifest();sources[str(Path(__file__).relative_to(ROOT))]=sha256_file(Path(__file__))
    datasets={k:ROOT/f"configs/round1/{k}.json" for k in ("validation_D","common_encounters")}
    data={k:load_dataset(p) for k,p in datasets.items()}
    assert all(d["dataset_split"] in ("validation","diagnostic") and d["environment_stage"]=="D" for d in data.values())
    identities={}
    for arm in ("base","instant","predictive"):
        for seed in (1,2,3):
            name=f"{arm}-seed{seed}";path=ROOT/f"runs/round3-{name}/models/best/best_model.zip"
            run,_,_=verify_round_three_model(path)
            identities[name]={"path":str(path),"sha256":sha256_file(path),"run":str(run),
                "training_steps_sha256":sha256_file(run/"logs/training_steps.csv.gz")}
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(output/"protocol.json",{"created_at":utc_now(),"purpose":"exploratory development diagnosis, no optimizer updates",
        "models":identities,"sources":sources,"datasets":{k:{"path":str(p),"sha256":sha256_file(p)} for k,p in datasets.items()},
        "sampling_seed":6401,"samples_per_state":64,
        "branch_triggers":"First pre-action q4>=0.5 or static center<7, after 2s, independently",
        "branch_action":[-1,0],"branch_end":"stop, original environment termination, or 20 steps",
        "scope":"Safe stop is only a local braking witness, not successful navigation. Sampling on best-policy states is not training-time exploration."})
    episodes=[];branches=[];summaries=[];history=[]
    for name,item in identities.items():
        policy=PPO.load(item["path"],device="cpu");rng=np.random.default_rng(6401)
        history.extend(training_history(Path(item["run"]),name))
        for dataset,path in datasets.items():
            rows=[];env=DynamicPathPlanningEnv(scenario_file=path)
            try:
                for i in range(len(data[dataset]["scenarios"])):
                    obs,_=env.reset(options={"scenario_index":i});seen=set();local_branches=[];local=[]
                    while True:
                        v=env.current_linear_velocity
                        q=risk_metrics(obs,4).risk
                        centers=np.linalg.norm(env.static_obstacles-env.agent_position,axis=1)
                        if env.step_count>=10:
                            for trigger,active in (("predicted_risk",q>=.5),("static_penalty",centers.min()<7)):
                                if active and trigger not in seen:
                                    seen.add(trigger)
                                    local_branches.append({"method":name,"dataset":dataset,"scenario_id":i,"trigger":trigger,
                                        "trigger_step":env.step_count,"clearance":float(centers.min()-5),**brake_branch(env)})
                        tensor,_=policy.policy.obs_to_tensor(obs)
                        with torch.no_grad():
                            dist=policy.policy.get_distribution(tensor).distribution
                            means=dist.mean.cpu().numpy()[0];std=dist.stddev.cpu().numpy()[0]
                        action=np.clip(means,-1,1)
                        target=float((action[0]+1)*1.5)
                        obs,reward,done,trunc,info=env.step(action)
                        expected=np.clip(target,max(0,v-.3),min(3,v+.3))
                        np.testing.assert_allclose(env.current_linear_velocity,expected,atol=1e-6)
                        row={"scenario_id":i,"step":env.step_count,"v_before":v,"v_after":env.current_linear_velocity,
                            "target_v":target,"action":float(action[0]),"turn_action":float(action[1]),
                            "q_before":q,"static_clearance_before":float(centers.min()-5),
                            "delta_v":env.current_linear_velocity-v,**distribution_record(float(means[0]),float(std[0]),v,rng)}
                        local.append(row)
                        if done or trunc:break
                    for row in local:row["success"]=bool(info["is_success"]);row["reason"]=info["termination_reason"]
                    for branch in local_branches:
                        branch["original_reason"]=info["termination_reason"]
                        branch["original_steps_to_end"]=env.step_count-branch["trigger_step"]
                    branches.extend(local_branches);rows.extend(local)
                    episodes.append({"method":name,"dataset":dataset,"scenario_id":i,"reason":info["termination_reason"],"steps":env.step_count})
            finally:env.close()
            f=pd.DataFrame(rows)
            f.to_csv(output/f"{dataset}-{name}-steps.csv.gz",index=False,compression="gzip")
            for outcome in ("all","success","failure"):
                g=f if outcome=="all" else f[f.success==(outcome=="success")]
                for zone in ("steady","high_risk","static_near"):
                    z=g[g.step>10]
                    if zone=="high_risk":z=z[z.q_before>=.5]
                    if zone=="static_near":z=z[z.static_clearance_before<2]
                    summaries.append({"method":name,"dataset":dataset,"outcome":outcome,"zone":zone,"n":len(z),
                        **{k:None if z.empty else float(z[k].mean()) for k in ("mean","std","p_upper_clip","p_brake_0_1","sample_brake_fraction","v_after","delta_v")},
                        "deterministic_upper_clip_fraction":None if z.empty else float((z.action>=1).mean()),
                        "deterministic_brake_fraction":None if z.empty else float((z.delta_v<=-.1+1e-9).mean())})
            print(name,dataset,len(data[dataset]["scenarios"]),"episodes",flush=True)
    pd.DataFrame(episodes).to_csv(output/"episodes.csv",index=False)
    pd.DataFrame(summaries).to_csv(output/"action_summary.csv",index=False)
    pd.DataFrame(history).to_csv(output/"training_exploration.csv",index=False)
    atomic_write_json(output/"braking_branches.json",branches)
    pd.DataFrame([{k:v for k,v in b.items() if k!="trace"} for b in branches]).to_csv(output/"braking_branches.csv",index=False)
    for p,digest in sources.items():assert sha256_file(ROOT/p)==digest
    for item in identities.values():
        assert sha256_file(item["path"])==item["sha256"]
        assert sha256_file(Path(item["run"])/"logs/training_steps.csv.gz")==item["training_steps_sha256"]
    atomic_write_json(output/"verification.json",{"episodes":len(episodes),"braking_branches":len(branches),"model_and_source_hashes_unchanged":True,
        "motion_mapping_checks":"passed for every original and branch step","analytic_full_speed_stop_distance_m":2.7,
        "helper_tests":"passed","optimizer_updates":0})
    atomic_write_json(output/"artifact_manifest.json",{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--test-only",action="store_true");args=parser.parse_args()
    if args.test_only:test_helpers();print("Action diagnostic helper tests passed")
    else:diagnose(args.output)
