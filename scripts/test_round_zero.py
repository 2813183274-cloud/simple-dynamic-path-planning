"""Regression checks for final-update selection and fail-closed provenance."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO
from envs import DynamicPathPlanningEnv
from envs.scenario_dataset import dataset_hash, load_dataset
from experiment_utils import atomic_write_json, source_manifest, sha256_file, verify_model_identity
from scripts.train import FixedScenarioBestModelCallback


def main():
    validation = ROOT / "configs/validation_scenarios_30.json"
    with tempfile.TemporaryDirectory() as tmp:
        run = (Path(tmp) / "probe").resolve()
        env = DynamicPathPlanningEnv()
        model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=1, seed=0, device="cpu")
        callback = FixedScenarioBestModelCallback(validation, dataset_hash(load_dataset(validation)),
            32, run / "models/best", run / "logs/validation", "probe", verbose=0)
        model.learn(32, callback=callback)
        assert [r["timesteps"] for r in callback.history] == [32, 32]
        assert [r["policy_updates"] for r in callback.history] == [0, 1]
        final = run / "models/final_model.zip"
        model.save(final)
        config = {"run_id": "probe", "source_manifest": source_manifest(ROOT)}
        metadata = {"run_id": "probe", "status": "completed", "final_model_sha256": sha256_file(final)}
        atomic_write_json(run / "config.json", config)
        atomic_write_json(run / "metadata.json", metadata)
        assert verify_model_identity(final, ROOT)[0] == run
        metadata["final_model_sha256"] = "tampered"
        atomic_write_json(run / "metadata.json", metadata)
        try:
            verify_model_identity(final, ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered model accepted")
        metadata["final_model_sha256"] = sha256_file(final)
        atomic_write_json(run / "metadata.json", metadata)
        config["source_manifest"] = {}
        atomic_write_json(run / "config.json", config)
        try:
            verify_model_identity(final, ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError("changed source accepted")
        env.close()
    print("Round-zero checks passed: final update evaluated, model hash/source verified.")


if __name__ == "__main__":
    main()
