import copy
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from envs.encounters import CONFIG, sample
from envs.round_one_dataset import generate, validate
from envs.scenario_dataset import save_dataset, scenario_hash, dataset_hash
from experiment_utils import atomic_write_json


def main():
    output=ROOT/"configs/round1"
    output.mkdir(parents=True,exist_ok=True)
    if (output/"protocol.json").exists() and "--rebuild-draft" not in sys.argv:
        raise FileExistsError("Protocol already exists; version new datasets rather than overwrite")
    datasets={}
    for stage in "ABCD":
        d=generate(stage,3101,60)
        validate(d); save_dataset(d,output/f"validation_{stage}.json"); datasets[f"validation_{stage}"]=d
    for name,stage,seed,challenge in [("common_fixed","A",3201,False),
        ("common_encounters","D",3202,False),("challenge_development","D",3203,True)]:
        d=generate(stage,seed,60,"diagnostic",challenge)
        validate(d);save_dataset(d,output/f"{name}.json");datasets[name]=d
    d=generate("D",3301,12,"diagnostic")
    rng=np.random.default_rng(3302)
    d["scenarios"]=[]
    for kind in CONFIG["types"]:
        for risk in CONFIG["risks"]:
            s=sample(rng,requested=(kind,risk,"clear"))
            s["scenario_id"]=len(d["scenarios"]);s["scenario_hash"]=scenario_hash(s)
            d["scenarios"].append(s)
    d["sampling_report"]={"note":"stratified illustrative cases, not population evaluation"}
    validate(d);save_dataset(d,output/"typical_cases.json");datasets["typical_cases"]=d
    stats=Counter();rng=np.random.default_rng(3401)
    coverage=[sample(rng,stats=stats) for _ in range(1000)]
    atomic_write_json(output/"coverage.json",{"samples":1000,"stats":dict(stats),
        "rejection_rate":1-1000/stats["attempts"],
        "target_speed_minmax":[min(s["generation"]["target_speed"] for s in coverage),max(s["generation"]["target_speed"] for s in coverage)],
        "unique_geometry_count":len({scenario_hash(s) for s in coverage}),
        "witness_min_clearance":min(s["generation"]["witness"]["minimum_clearance"] for s in coverage)})
    atomic_write_json(output/"protocol.json",{"version":"round1-v1","config":CONFIG,
        "datasets":{name:dataset_hash(d) for name,d in datasets.items()},
        "final_test":{"status":"reserved_not_generated_or_evaluated","seed":3901,"count":240,
            "stage":"D","same_distribution":True,"seal_before_final_evaluation":True},
        "final_challenge":{"status":"reserved_not_generated_or_evaluated","seed":3902,"count":120,
            "target_speed_range":[1.55,1.8],"overtaking_target_speed_range":[.3,.55],
            "meeting_time_range":[9,12],"overtaking_meeting_time_range":[20,26],"crossing_angle_degrees":[35,50]},
        "split_rules":"Independent seeds AND geometry-overlap checks. A/B are deliberately paired ablations, not independent splits."})
    print("Generated A-D validation, shared diagnostics, typical cases, coverage and sealed-test protocol.")


if __name__=="__main__":main()
