"""Post-evaluation integrity/kinematics audit and descriptive failure-speed checks.

No inference, training, parameter selection or modification of sealed artifacts.
"""
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_utils import atomic_write_json,sha256_file,source_manifest
from envs.scenario_dataset import load_dataset
from scripts.round_two_experiment import geometric_hash


def main():
    out=ROOT/"results/round2"
    protocol=json.loads((out/"protocol.json").read_text())
    assert source_manifest(ROOT)==protocol["core_sources"]
    for relative,key in [("baselines/dwa.py","dwa_source_sha256"),
                         ("scripts/round_two_experiment.py","experiment_source_sha256"),
                         ("scripts/summarize_round_two.py","statistics_source_sha256")]:
        assert sha256_file(ROOT/relative)==protocol[key]
    old_files=0
    for manifest in [ROOT/"results/round1/artifact_manifest.json",ROOT/"runs/round0-baseline-seed1/round_zero/artifact_manifest.json"]:
        base=ROOT if manifest.parent.name=="round1" else manifest.parent.parent
        for relative,digest in json.loads(manifest.read_text()).items():
            assert sha256_file(base/relative)==digest,relative
            old_files+=1
    for model in protocol["models"].values():
        assert sha256_file(model["path"])==model["sha256"]
        assert model["training_steps"]==200704
    used=set()
    for file in (ROOT/"configs").rglob("*.json"):
        data=json.loads(file.read_text(encoding="utf-8"))
        if isinstance(data,dict):used.update(geometric_hash(s) for s in data.get("scenarios",[]))
    seals=json.loads((out/"dataset_seal.json").read_text());episodes=0;checks=[]
    for split in ("same_distribution","challenge"):
        file=out/f"sealed_{split}.json"
        assert sha256_file(file)==seals[split]["file_sha256"]
        scenes=load_dataset(file)["scenarios"]
        hashes={geometric_hash(s) for s in scenes}
        assert len(hashes)==len(scenes) and not used&hashes
        used.update(hashes)
        for name in [*protocol["models"],"DWA"]:
            folder=out/"final"/split/name
            complete=json.loads((folder/"complete.json").read_text())
            assert complete["protocol_sha256"]==sha256_file(out/"protocol.json")
            for relative,digest in complete["artifacts"].items():assert sha256_file(folder/relative)==digest
            frame=pd.read_csv(folder/"episodes.csv")
            assert len(frame)==len(scenes) and frame.scenario_id.tolist()==list(range(len(scenes)))
            speed_fractions=[]
            with np.load(folder/"trajectories.npz") as traces:
                for record in frame.itertuples():
                    i=record.scenario_id;scene=scenes[i]
                    agent=traces[f"{i}_agent_path"];target=traces[f"{i}_dynamic_path"]
                    h=traces[f"{i}_headings"];v=traces[f"{i}_linear_velocities"];w=traces[f"{i}_angular_velocities"]
                    assert len(v)==record.steps+1 and np.isfinite(agent).all()
                    assert v.min()>=-1e-6 and v.max()<=3+1e-6 and np.max(np.abs(w))<=math.pi/4+1e-6
                    assert np.max(np.abs(np.diff(v)))<=.3+1e-6
                    assert np.max(np.abs(np.diff(w)))<=math.pi/20+1e-6
                    np.testing.assert_allclose(np.diff(agent,axis=0),.2*v[1:,None]*np.column_stack([np.cos(h[:-1]),np.sin(h[:-1])]),atol=1e-6)
                    delta=(np.diff(h)+math.pi)%(2*math.pi)-math.pi
                    np.testing.assert_allclose(delta,.2*w[1:],atol=1e-6)
                    expected=np.asarray(scene["dynamic_obstacle"]["velocity"])*.2
                    np.testing.assert_allclose(np.diff(target,axis=0),np.tile(expected,(len(v)-1,1)),atol=1e-6)
                    dynamic=np.linalg.norm(agent-target,axis=1)-4
                    static=np.min([np.linalg.norm(agent-o["position"],axis=1)-5 for o in scene["static_obstacles"]],axis=0)
                    np.testing.assert_allclose([static.min(),dynamic.min()],[record.min_static_clearance,record.min_dynamic_clearance],atol=1e-6)
                    if record.success:assert static.min()>0 and dynamic.min()>0 and record.final_goal_distance<=3
                    if not record.success and len(v)>11:speed_fractions.append(float(np.mean(v[11:]>=2.99)))
                    episodes+=1
            checks.append({"split":split,"method":name,"episodes":len(frame),
                "failed_episode_mean_full_speed_fraction_after_2s":float(np.mean(speed_fractions)) if speed_fractions else None})
    result={"status":"passed","episodes_audited":episodes,"unique_final_scenes":360,
        "old_artifacts_unchanged":old_files,"sealed_models_unchanged":len(protocol["models"]),
        "checks":"source, model, output, split overlap, trajectory kinematics, target motion, clearances, success endpoints",
        "exploratory_failure_speed":checks,"audit_source_sha256":sha256_file(Path(__file__))}
    atomic_write_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
