"""Validation-only baseline selection, protocol sealing, and one-shot final evaluation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.dwa import DWA,DWAConfig
from envs import DynamicPathPlanningEnv
from envs.encounters import CONFIG,sample
from envs.round_one_dataset import generate,validate
from envs.scenario_dataset import scenario_hash,dataset_hash,load_dataset,save_dataset
from experiment_utils import atomic_write_json,sha256_file,source_manifest,verify_model_identity,utc_now
from scripts.evaluate import run_scenario,build_summary,build_type_metrics,draw_full_subplot,draw_closest_subplot

OUT=ROOT/"results/round2"


class TimedPolicy:
    def __init__(self,policy):self.policy=policy;self.times=[]
    def predict(self,observation,deterministic=True):
        start=time.perf_counter()
        result=self.policy.predict(observation,deterministic=deterministic)
        self.times.append(time.perf_counter()-start)
        return result


def evaluate(policy,file,output,observation_stage="D",save_examples=False):
    data=load_dataset(file)
    output.mkdir(parents=True,exist_ok=False)
    env=DynamicPathPlanningEnv(scenario_file=file);env.observation_stage=observation_stage
    timed=TimedPolicy(policy);records=[];histories=[];traces={}
    # Warm-up excluded from latency; no environment step and no learned parameter changes.
    obs,_=env.reset(options={"scenario_index":0});policy.predict(obs,deterministic=True)
    for i in range(len(data["scenarios"])):
        before=len(timed.times)
        record,h=run_scenario(timed,env,i)
        times=timed.times[before:]
        record.update(mean_inference_ms=float(np.mean(times)*1000),
            p95_inference_ms=float(np.quantile(times,.95)*1000),
            total_heading_variation=float(np.abs(h["angular_velocities"]).sum()*.2),
            full_speed_fraction=float(np.mean(h["linear_velocities"]>=2.99)))
        records.append(record)
        # Save all replayable trajectories; render first 3 failures, not curated successes.
        for key in ("agent_path","dynamic_path","headings","linear_velocities","angular_velocities"):
            traces[f"{i}_{key}"]=h[key]
        if not record["success"] and len(histories)<3:histories.append(h)
    frame=pd.DataFrame(records)
    summary=build_summary(frame)
    summary.update(mean_inference_ms=float(np.mean(timed.times)*1000),
        p95_inference_ms=float(np.quantile(timed.times,.95)*1000),
        worst_static_clearance=float(frame.min_static_clearance.min()),
        worst_dynamic_clearance=float(frame.min_dynamic_clearance.min()),
        dataset_hash=dataset_hash(data),dataset_split=data["dataset_split"],
        observation_stage=observation_stage,environment_stage=env.stage,
        timing_note="CPU sequential per-policy evaluation; wall-clock timing is hardware/load dependent")
    frame.to_csv(output/"episodes.csv",index=False)
    build_type_metrics(frame).to_csv(output/"by_type.csv",index=False)
    atomic_write_json(output/"by_risk.json",[{"risk":name,**build_summary(g)} for name,g in frame.groupby("risk_level")])
    atomic_write_json(output/"summary.json",summary)
    np.savez_compressed(output/"trajectories.npz",**traces)
    if save_examples and histories:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(14,15),constrained_layout=True)
        for row,h in enumerate(histories):
            draw_full_subplot(axes[row,0],h);draw_closest_subplot(axes[row,1],h)
        for row in range(len(histories),3):
            axes[row,0].axis("off");axes[row,1].axis("off")
        ids=",".join(str(h["record"]["scenario_id"]) for h in histories)
        fig.suptitle(f"First failures by scenario ID: {ids} (full / closest encounter)")
        fig.savefig(output/"first_failures.png",dpi=130);plt.close(fig)
    env.close()
    return summary


def tune():
    output=OUT/"validation_tuning"
    output.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for horizon in (3.,5.):
        for weight in (.3,1.,3.):
            config=DWAConfig(horizon=horizon,clearance_weight=weight)
            label=f"h{horizon:g}-c{weight:g}"
            summary=evaluate(DWA(config),ROOT/"configs/round1/validation_D.json",output/label)
            candidates.append({"id":label,"config":asdict(config),"summary":summary})
            print(label,summary["success_count"],summary["collision_count"],flush=True)
    def key(item):
        s=item["summary"]
        return (s["success_rate"],-s["collision_rate"],s["successful_average_path_efficiency"] or 0,s["average_reward"])
    best=max(candidates,key=key)
    atomic_write_json(output/"selection.json",{"selected":best,"candidates":candidates,
        "selected_at":utc_now(),"selection_data":"validation_D only", "source_sha256":sha256_file(ROOT/"baselines/dwa.py")})


def run_directory(stage,seed):
    return ROOT/"runs"/(f"round1-{stage}-seed1" if seed==1 else f"round2-{stage}-seed{seed}")


def definitions():
    result={}
    for stage in ("C","D"):
        for seed in (1,2,3):result[f"{stage}-seed{seed}"]=run_directory(stage,seed)
    for stage in ("A","B"):result[f"{stage}-seed1"]=run_directory(stage,1)
    result["round0-seed1"]=ROOT/"runs/round0-baseline-seed1"
    return result


def stratified(seed,count,challenge):
    # Exact 25% encounter-type strata; within each, preserve D's conditional risk/layout law.
    if count<=0 or count%4:raise ValueError("Count must be a positive multiple of four")
    data=generate("D",seed,1,"test",challenge)
    rng=np.random.default_rng(seed);stats=Counter();scenes=[]
    for kind in CONFIG["types"]:
        for _ in range(count//4):
            risk=str(rng.choice(CONFIG["risks"],p=CONFIG["risk_probabilities"]))
            layout=str(rng.choice(CONFIG["layouts"],p=CONFIG["layout_probabilities"]))
            if layout=="blocking":risk=str(rng.choice(["time_separated","receding"],p=[.625,.375]))
            if layout=="mixed":risk="conflict"
            s=sample(rng,"D",stats,requested=(kind,risk,layout),challenge=challenge)
            s["scenario_id"]=len(scenes);s["scenario_hash"]=scenario_hash(s);scenes.append(s)
    data.update(scenarios=scenes,num_scenarios=count,sampling_report=dict(stats),
                dataset_name=f"round2-final-{'challenge' if challenge else 'id'}-{seed}")
    validate(data)
    return data


def geometric_hash(scene):
    value=dict(scene);value.pop("environment_stage",None)
    return scenario_hash(value)


def seal():
    if (OUT/"protocol.json").exists():raise FileExistsError("Protocol already sealed")
    selected=json.loads((OUT/"validation_tuning/selection.json").read_text())
    if selected["source_sha256"]!=sha256_file(ROOT/"baselines/dwa.py"):
        raise ValueError("Baseline changed since validation selection")
    models={}
    for name,run in definitions().items():
        path=run/"models/best/best_model.zip"
        _,config,metadata=verify_model_identity(path,ROOT,allow_archived_source=True)
        selection=json.loads((run/"models/best/selection_metrics.json").read_text())
        models[name]={"path":str(path),"sha256":sha256_file(path),"run":str(run),
            "stage":config["environment"].get("stage","legacy"),"seed":config["seed"],
            "best_timesteps":selection["timesteps"],"training_steps":metadata["final_num_timesteps"]}
    protocol={"frozen_at":utc_now(),"models":models,"dwa_config":selected["selected"]["config"],
        "dwa_source_sha256":sha256_file(ROOT/"baselines/dwa.py"),
        "experiment_source_sha256":sha256_file(Path(__file__)),"core_sources":source_manifest(ROOT),
        "statistics_source_sha256":sha256_file(ROOT/"scripts/summarize_round_two.py"),
        "same_distribution":{"seed":3901,"n":240,"type_counts":{k:60 for k in CONFIG["types"]}},
        "challenge":{"seed":3902,"n":120,"type_counts":{k:30 for k in CONFIG["types"]}},
        "primary_comparison":"C and D, each 3 seeds, and validation-selected DWA",
        "secondary_ablation":"round0/A/B seed1 only; do not infer multi-seed robustness",
        "statistics":"seed mean and sample SD; seed t intervals; per-model Wilson; paired hierarchical bootstrap over seeds and within-type scenario IDs",
        "final_test_policy":"No changes to methods, weights, hyperparameters or model selection after sealing"}
    # Write frozen method identities BEFORE generating or inspecting final scene outcomes.
    atomic_write_json(OUT/"protocol.json",protocol)
    used=set()
    for file in (ROOT/"configs").rglob("*.json"):
        data=json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data,dict):
            used.update(geometric_hash(s) for s in data.get("scenarios",[]))
    for name,seed,n,challenge in [("same_distribution",3901,240,False),("challenge",3902,120,True)]:
        data=stratified(seed,n,challenge)
        hashes={geometric_hash(s) for s in data["scenarios"]}
        if len(hashes)!=n or hashes&used:raise ValueError("Test geometry overlaps existing data")
        used.update(hashes)
        save_dataset(data,OUT/f"sealed_{name}.json")
    atomic_write_json(OUT/"dataset_seal.json",{name:{"dataset_hash":dataset_hash(load_dataset(OUT/f"sealed_{name}.json")),
        "file_sha256":sha256_file(OUT/f"sealed_{name}.json")} for name in ("same_distribution","challenge")})
    print("Methods frozen and final datasets sealed; final outcomes not yet evaluated.",flush=True)


def final_evaluation():
    protocol=json.loads((OUT/"protocol.json").read_text())
    if protocol["core_sources"]!=source_manifest(ROOT):raise ValueError("Core sources changed after seal")
    if protocol["experiment_source_sha256"]!=sha256_file(Path(__file__)):raise ValueError("Evaluator changed after seal")
    if protocol["dwa_source_sha256"]!=sha256_file(ROOT/"baselines/dwa.py"):raise ValueError("DWA changed after seal")
    seals=json.loads((OUT/"dataset_seal.json").read_text())
    for split in ("same_distribution","challenge"):
        file=OUT/f"sealed_{split}.json"
        if sha256_file(file)!=seals[split]["file_sha256"]:raise ValueError("Sealed dataset changed")
        methods={**protocol["models"],"DWA":None}
        for name,item in methods.items():
            dest=OUT/"final"/split/name
            # Exact completed results may be resumed after interruption, never silently overwritten.
            if (dest/"complete.json").exists():
                complete=json.loads((dest/"complete.json").read_text())
                if complete["protocol_sha256"]!=sha256_file(OUT/"protocol.json"):
                    raise ValueError("Completed evaluation has a different protocol")
                for path,digest in complete["artifacts"].items():
                    if sha256_file(dest/path)!=digest:raise ValueError("Completed artifact changed")
                continue
            if item is None:policy=DWA(DWAConfig(**protocol["dwa_config"]));stage="D"
            else:
                if sha256_file(item["path"])!=item["sha256"]:raise ValueError("Sealed model changed")
                policy=PPO.load(item["path"],device="cpu");stage=item["stage"]
            s=evaluate(policy,file,dest,observation_stage=stage,save_examples=True)
            atomic_write_json(dest/"complete.json",{"method":name,"dataset_hash":s["dataset_hash"],
                "model":item,"dwa_config":protocol["dwa_config"] if item is None else None,"completed_at":utc_now(),
                "protocol_sha256":sha256_file(OUT/"protocol.json"),
                "artifacts":{str(p.relative_to(dest)):sha256_file(p) for p in dest.rglob("*") if p.is_file()}})
            print(split,name,s["success_count"],"/",s["total_scenarios"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["tune","seal","evaluate"]);args=p.parse_args()
    {"tune":tune,"seal":seal,"evaluate":final_evaluation}[args.mode]()
