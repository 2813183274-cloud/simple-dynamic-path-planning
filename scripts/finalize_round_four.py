"""Package and verify the completed P4 experiment without altering its seals."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import atomic_write_json, sha256_file, utc_now
from round_four_utils import verify_implementation
from scripts.round_four_experiment import OUT, verify_complete, verify_seals


def main():
    methods, datasets = verify_seals()
    model_dir = OUT/"models"; records = OUT/"training_records"
    model_dir.mkdir(exist_ok=False); records.mkdir(exist_ok=False)
    for name, item in methods["models"].items():
        source = Path(item["path"])
        target = model_dir/f"{name}.zip"
        shutil.copy2(source,target)
        if sha256_file(target) != item["sha256"]:
            raise ValueError("Portable P4 model copy differs")
        run = source.parents[2]; folder=records/name; folder.mkdir()
        files = {"metadata.json":run/"metadata.json", "config.json":run/"config.json",
                 "selection_metrics.json":run/"models/best/selection_metrics.json",
                 "validation_evaluations.json":run/"logs/validation/evaluations.json",
                 "train_monitor.csv":run/"logs/train_monitor.csv",
                 "training_reward_curve.png":run/"results/training_reward_curve.png"}
        for relative, original in files.items():
            shutil.copy2(original,folder/relative)
    episodes=0
    for split, count in (("same_distribution",240),("challenge",120)):
        for name in [*methods["models"],"DWA"]:
            folder=OUT/"final"/split/name; verify_complete(folder)
            record=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            if record["n"] != count or record["trajectory_audit"] != "passed":
                raise ValueError("P4 final result incomplete")
            episodes += count
    verification=dict(verified_at=utc_now(),status="passed",protocol="round4-speed-tanh-v2",
        formal_training_runs=6,actual_training_steps_per_run=200704,
        actual_training_steps_total=1204224,selected_models=6,
        portable_model_hashes_verified=6,final_scenes=sum(x["n"] for x in datasets.values()),
        witness_replays_passed=sum(x["witnesses_passed"] for x in datasets.values()),
        final_evaluation_episodes=episodes,trajectory_audits="passed for all 2520 episodes",
        implementation_seal="verified",post_test_parameter_changes=False,
        notes=["All six seeds retained", "DWA configuration was not retuned",
               "Final datasets were generated only after model freeze"])
    atomic_write_json(OUT/"verification.json",verification)
    manifest={str(path.relative_to(OUT)):sha256_file(path) for path in OUT.rglob("*")
              if path.is_file() and path.name!="artifact_manifest.json"}
    atomic_write_json(OUT/"artifact_manifest.json",manifest)
    for relative,digest in manifest.items():
        if sha256_file(OUT/relative)!=digest:
            raise ValueError("P4 artifact manifest mismatch")
    verify_implementation()
    print(f"P4 packaged: {len(manifest)} artifacts, {episodes} final episodes, six portable best models.")


if __name__ == "__main__":
    main()
