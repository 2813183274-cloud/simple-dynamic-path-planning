"""Freeze a completed run and verify deterministic validation replay (never test split)."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import atomic_write_json, sha256_file, source_manifest, verify_model_identity, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    model = run / "models/best/best_model.zip"
    _, config, metadata = verify_model_identity(model, ROOT)
    output = run / "round_zero"
    output.mkdir(exist_ok=False)
    # Snapshot legacy evidence without claiming its logs describe its model.
    inventory = []
    for folder in (ROOT / "models", ROOT / "logs"):
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                inventory.append({"path": str(file.relative_to(ROOT)), "sha256": sha256_file(file),
                                  "bytes": file.stat().st_size, "training_identity": "unverified"})
    atomic_write_json(output / "legacy_inventory.json", inventory)
    validation = run / "snapshot/validation_scenarios.json"
    for label in ("validation", "validation_replay"):
        command = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--run-dir", str(run),
                   "--scenario-file", str(validation), "--allow-non-test-split",
                   "--output-dir", str(output / label)]
        if label.endswith("replay"):
            command.append("--no-plots")
        subprocess.run(command, check=True)
    first = output / "validation/episode_results.csv"
    replay = output / "validation_replay/episode_results.csv"
    if first.read_bytes() != replay.read_bytes():
        raise RuntimeError("Deterministic validation replay differs")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_counterfactual.py"),
        "--run-dir", str(run), "--scenario-file", str(validation),
        "--output-dir", str(output / "diagnostics")], check=True)
    selection = json.loads((run / "models/best/selection_metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run / "logs/validation/evaluations.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "validation/summary_metrics.json").read_text(encoding="utf-8"))
    assert selection["model_sha256"] == sha256_file(model)
    assert summary["success_count"] == selection["success_count"]
    assert summary["collision_count"] == selection["collision_count"]
    assert history[-1]["timesteps"] == metadata["final_num_timesteps"]
    shutil.copy2(run / "results/training_reward_curve.png", output / "training_reward_curve.png")
    report = {"status": "passed", "created_at": utc_now(), "run_id": config["run_id"],
        "training_seed": config["seed"], "requested_timesteps": config["requested_additional_timesteps"],
        "actual_timesteps": metadata["final_num_timesteps"],
        "best_timesteps": selection["timesteps"], "best_policy_updates": selection["policy_updates"],
        "best_model_sha256": sha256_file(model), "validation_dataset_hash": config["validation_dataset_hash"],
        "validation_success_count": summary["success_count"],
        "validation_collision_count": summary["collision_count"],
        "validation_episode_csv_sha256": sha256_file(first), "replay_byte_identical": True,
        "source_manifest": source_manifest(ROOT), "final_test_evaluated": False,
        "limitations": ["Single seed, fixed 200000-step budget, not convergence proof",
            "Validation is used for model selection, not independent generalization evidence",
            "Legacy training provenance cannot be reconstructed from filenames",
            "Distant dynamic diagnostic is not true obstacle absence",
            "Safe resume restores model/optimizer, not exact RNG/environment replay"]}
    atomic_write_json(output / "acceptance.json", report)
    files = {str(p.relative_to(run)): sha256_file(p) for p in sorted(run.rglob("*"))
             if p.is_file() and p.name != "artifact_manifest.json"}
    atomic_write_json(output / "artifact_manifest.json", files)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
