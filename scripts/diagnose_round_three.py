"""P3-02: observation/action/reward diagnostics on development data only; no training."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from envs import DynamicPathPlanningEnv
from baselines.dwa import DWA,DWAConfig
from envs.scenario_dataset import load_dataset,dataset_hash
from experiment_utils import atomic_write_json,sha256_file,source_manifest,verify_model_identity

OUT=ROOT/"results/round3/development-diagnosis"


def main():
    if OUT.exists():raise FileExistsError("Do not overwrite an existing diagnosis")
    sources=source_manifest(ROOT)
    datasets={name:ROOT/f"configs/round1/{name}.json" for name in ("validation_D","common_encounters")}
    data={name:load_dataset(path) for name,path in datasets.items()}
    assert all(d["dataset_split"] in ("validation","diagnostic") for d in data.values())
    models={}
    for stage in ("C","D"):
        for seed in (1,2,3):
            run=ROOT/"runs"/(f"round1-{stage}-seed1" if seed==1 else f"round2-{stage}-seed{seed}")
            path=run/"models/best/best_model.zip"
            verify_model_identity(path,ROOT,allow_archived_source=True)
            models[f"{stage}-seed{seed}"]={"path":str(path),"sha256":sha256_file(path)}
    selected=json.loads((ROOT/"results/round2/validation_tuning/selection.json").read_text())
    config=selected["selected"]["config"]
    assert selected["source_sha256"]==sha256_file(ROOT/"baselines/dwa.py")
    OUT.mkdir(parents=True)
    atomic_write_json(OUT/"protocol.json",{"purpose":"exploratory development diagnosis, not final performance or causal proof",
        "models":models,"datasets":{k:{"path":str(datasets[k]),"hash":dataset_hash(v),"split":v["dataset_split"]} for k,v in data.items()},
        "core_sources":sources,"script_sha256":sha256_file(Path(__file__)),"dwa_config":config,
        "thresholds":{"full_speed":2.99,"action_clip":1.,"turn_proxy_rad_s":.1,"warmup_seconds":2.},
        "policy":"No parameter updates, no new baseline tuning, no final-test data read"})
    all_episodes=[];summaries=[]
    for name,item in {**models,"DWA":None}.items():
        policy=DWA(DWAConfig(**config)) if item is None else PPO.load(item["path"],device="cpu")
        for dataset,file in datasets.items():
            env=DynamicPathPlanningEnv(scenario_file=file)
            rows=[];episodes=[]
            for i in range(len(data[dataset]["scenarios"])):
                obs,_=env.reset(options={"scenario_index":i});steps=[]
                while True:
                    raw=np.nan
                    action,_=policy.predict(obs,deterministic=True)
                    if item is not None:
                        tensor,_=policy.policy.obs_to_tensor(obs)
                        with torch.no_grad():means=policy.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]
                        np.testing.assert_allclose(action,np.clip(means,-1,1),atol=1e-6)
                        raw=float(means[0])
                    obs,reward,done,trunc,info=env.step(action)
                    distances=np.linalg.norm(env.static_obstacles-env.agent_position,axis=1)
                    row={"dataset":dataset,"method":name,"scenario_id":i,"step":env.step_count,
                        "raw_speed_mean":raw,"speed_action":float(action[0]),"turn_action":float(action[1]),
                        "v":env.current_linear_velocity,"w":env.current_angular_velocity,
                        "static1_center":float(distances[0]),"static2_center":float(distances[1]),
                        "dynamic_center":float(np.linalg.norm(env.dynamic_position-env.agent_position)),"reward":reward,
                        **{k:v for k,v in info.items() if k.startswith("reward_")}}
                    assert np.isclose(sum(row[k] for k in row if k.startswith("reward_")),reward)
                    steps.append(row)
                    if done or trunc:break
                frame=pd.DataFrame(steps);steady=frame[frame.step>10]
                reason=info["termination_reason"]
                column={"static_collision_1":"static1_center","static_collision_2":"static2_center","dynamic_collision":"dynamic_center"}.get(reason)
                entry=frame[frame[column]<(8 if column=="dynamic_center" else 7)] if column else frame.iloc[:0]
                lead=None if entry.empty else float((frame.step.iloc[-1]-entry.step.iloc[0])*.2)
                preterminal=frame.iloc[:-1]
                active=preterminal[(preterminal.reward_static_safety<0)|(preterminal.reward_dynamic_safety<0)]
                episode={"dataset":dataset,"method":name,"scenario_id":i,"success":bool(info["is_success"]),
                    "reason":reason,"reward":float(frame.reward.sum()),"steps":len(frame),
                    "full_speed_fraction_after_2s":float((steady.v>=2.99).mean()),
                    "clipped_speed_fraction":None if item is None else float((frame.raw_speed_mean>=1).mean()),
                    "raw_speed_mean_min":None if item is None else float(frame.raw_speed_mean.min()),
                    "raw_speed_mean_max":None if item is None else float(frame.raw_speed_mean.max()),
                    "safety_entry_to_collision_s":lead,
                    "abs_w_at_safety_entry":None if entry.empty else abs(float(entry.w.iloc[0])),
                    "safety_active_nonterminal_steps":len(active),
                    "safety_active_positive_reward_steps":int((active.reward>0).sum()),
                    **{k+"_sum":float(frame[k].sum()) for k in frame if k.startswith("reward_")}}
                episodes.append(episode);rows.extend(steps)
            env.close();all_episodes.extend(episodes)
            frame=pd.DataFrame(episodes);steps=pd.DataFrame(rows)
            steps.to_csv(OUT/f"{dataset}-{name}-steps.csv.gz",index=False,compression="gzip")
            failures=frame[~frame.success];active=steps[(steps.reward_terminal==0)&((steps.reward_static_safety<0)|(steps.reward_dynamic_safety<0))]
            summary={"dataset":dataset,"method":name,"successes":int(frame.success.sum()),"n":len(frame),
                "reasons":frame.reason.value_counts().to_dict(),
                "mean_episode_full_speed_fraction_after_2s":float(frame.full_speed_fraction_after_2s.mean()),
                "failure_mean_full_speed_fraction_after_2s":None if failures.empty else float(failures.full_speed_fraction_after_2s.mean()),
                "raw_speed_mean_min":None if item is None else float(steps.raw_speed_mean.min()),
                "raw_speed_mean_max":None if item is None else float(steps.raw_speed_mean.max()),
                "speed_action_clipped_fraction":None if item is None else float((steps.raw_speed_mean>=1).mean()),
                "safety_active_steps":len(active),"safety_active_positive_reward_fraction":None if active.empty else float((active.reward>0).mean()),
                "safety_active_mean_progress":None if active.empty else float(active.reward_progress.mean()),
                "safety_active_mean_safety_penalty":None if active.empty else float((active.reward_static_safety+active.reward_dynamic_safety).mean()),
                "collision_safety_entry_lead_median_s":None if failures.empty else float(failures.safety_entry_to_collision_s.median()),
                "collision_abs_w_at_entry_median":None if failures.empty else float(failures.abs_w_at_safety_entry.median())}
            summaries.append(summary);print(dataset,name,summary["successes"],"/",len(frame),flush=True)
    assert source_manifest(ROOT)==sources
    for item in models.values():assert sha256_file(item["path"])==item["sha256"]
    pd.DataFrame(all_episodes).to_csv(OUT/"episodes.csv",index=False)
    atomic_write_json(OUT/"summary.json",summaries)
    atomic_write_json(OUT/"artifact_manifest.json",{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.iterdir() if p.is_file()})


if __name__=="__main__":main()
