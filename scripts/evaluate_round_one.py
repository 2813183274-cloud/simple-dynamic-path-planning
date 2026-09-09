"""Stage ablations on shared DEVELOPMENT sets; no final test evaluation."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_utils import atomic_write_json, sha256_file, utc_now


def evaluate(run, dataset, output, cross=False, plots=False):
    command=[sys.executable,str(ROOT/"scripts/evaluate.py"),"--run-dir",str(run),
        "--scenario-file",str(dataset),"--output-dir",str(output),"--allow-non-test-split"]
    if cross:command.append("--cross-stage-diagnostic")
    if not plots:command.append("--no-plots")
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+".log")).open("w",encoding="utf-8") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    return json.loads((output/"summary_metrics.json").read_text(encoding="utf-8"))


def main():
    output=ROOT/"results/round1"
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for stage in ["round0","A","B","C","D"]:
        run=ROOT/"runs"/("round0-baseline-seed1" if stage=="round0" else f"round1-{stage}-seed1")
        metadata=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
        if metadata["status"]!="completed":raise RuntimeError(f"Run not completed: {run}")
        selection=json.loads((run/"models/best/selection_metrics.json").read_text(encoding="utf-8"))
        if stage!="round0":
            evaluate(run,ROOT/f"configs/round1/validation_{stage}.json",output/stage/"validation",plots=stage=="D")
        for name in ("common_fixed","common_encounters"):
            s=evaluate(run,ROOT/f"configs/round1/{name}.json",output/stage/name,cross=True)
            rows.append({"stage":stage,"dataset":name,"best_steps":selection["timesteps"],
                "success_count":s["success_count"],"total":s["total_scenarios"],
                "collision_count":s["collision_count"],"dynamic_collisions":s["dynamic_collision_count"],
                "out_of_bounds":s["out_of_bounds_count"],"timeouts":s["timeout_count"],
                "successful_path_efficiency":s["successful_average_path_efficiency"],
                "model_sha256":s["_provenance"]["model_sha256"]})
        print("Completed development evaluation:",stage,flush=True)
    run=ROOT/"runs/round1-D-seed1"
    evaluate(run,ROOT/"configs/round1/validation_D.json",output/"D/validation_replay")
    assert (output/"D/validation/episode_results.csv").read_bytes()==(output/"D/validation_replay/episode_results.csv").read_bytes()
    evaluate(run,ROOT/"configs/round1/challenge_development.json",output/"D/challenge_development")
    evaluate(run,ROOT/"configs/round1/typical_cases.json",output/"D/typical_cases",plots=True)
    with (output/"paired.log").open("w",encoding="utf-8") as stream:
        subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_counterfactual.py"),"--run-dir",str(run),
            "--scenario-file",str(ROOT/"configs/round1/typical_cases.json"),"--output-dir",str(output/"D/paired")],
            check=True,stdout=stream,stderr=subprocess.STDOUT)
    frame=pd.DataFrame(rows);frame.to_csv(output/"stage_comparison.csv",index=False)
    d=pd.read_csv(output/"D/paired/counterfactual_results.csv")
    response=[]
    for name,g in d.groupby("variant"):
        valid=g[g.intervention_geometry_issue=="none"]
        response.append({"variant":name,"n":len(g),"valid_geometry_n":len(valid),
            "success_count":int(g.success.sum()),
            "valid_action_response_count":int((valid.matched_time_action_rms_difference>1e-4).sum()),
            "mean_matched_time_path_rms":float(g.matched_time_path_rms_difference.mean()),
            "mean_min_dynamic_clearance":float(g.min_dynamic_clearance.mean())})
    atomic_write_json(output/"paired_response.json",response)
    atomic_write_json(output/"acceptance.json",{"completed_at":utc_now(),"status":"passed",
        "scope":"round1 stage A-D implementation and single-seed ablations, not a performance guarantee",
        "D_validation_replay_byte_identical":True,"final_test_evaluated":False,
        "training_budget_per_stage":200000,"training_seed":1,"common_protocol_rows":rows})
    files={}
    for folder in [output,ROOT/"configs/round1"]+[ROOT/f"runs/round1-{s}-seed1" for s in "ABCD"]:
        for p in folder.rglob("*"):
            if p.is_file() and p.name!="artifact_manifest.json":files[str(p.relative_to(ROOT))]=sha256_file(p)
    atomic_write_json(output/"artifact_manifest.json",files)
    atomic_write_json(ROOT/"runs/latest_run.json",{"run_id":run.name,"run_dir":str(run),"updated_at":utc_now()})
    print(frame.to_string(index=False))


if __name__=="__main__":main()
