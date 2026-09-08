"""Common-base-reward evaluation and development-only CLI for round three."""
import argparse
import csv
import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.risk_reward import RiskRewardWrapper
from envs.scenario_dataset import load_dataset, dataset_hash
from experiment_utils import atomic_write_json, sha256_file
from round_three_utils import verify_round_three_model


def selection_key(record):
    return (float(record["success_rate"]), -float(record["collision_rate"]),
            float(record["mean_successful_path_efficiency"] or 0), float(record["mean_base_reward"]))


class StepLog:
    def __init__(self, path):
        self.stream = None
        self.writer = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = gzip.open(path, "xt", encoding="utf-8", newline="")

    def write(self, row):
        if self.stream is None:
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        if self.stream is not None:
            self.stream.close()


def evaluate_policy(policy, env, count, step_log=None):
    """Policy is deterministic; env has no updates to learned parameters."""
    if count <= 0:
        raise ValueError("Evaluation requires at least one scenario")
    records = []
    log = StepLog(step_log)
    try:
        for index in range(count):
            obs, _ = env.reset(options={"scenario_index": index})
            base = env.unwrapped
            direct = max(0., float(np.linalg.norm(base.goal_position-base.start_position))-base.goal_threshold)
            full = []; clipped = []; high_risk_v = []; risk_start = None; brake_delay = None
            for _ in range(base.max_steps):
                action, _ = policy.predict(obs, deterministic=True)
                raw_mean = None
                if hasattr(policy, "policy"):
                    tensor, _ = policy.policy.obs_to_tensor(obs)
                    with torch.no_grad():
                        mean = policy.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]
                    np.testing.assert_allclose(action, np.clip(mean, -1, 1), atol=1e-6)
                    raw_mean = float(mean[0]); clipped.append(raw_mean >= 1)
                obs, _, terminated, truncated, info = env.step(action)
                step = info["step_count"]
                if step > 10:
                    full.append(info["actual_v"] >= 2.99)
                if step >= 10 and info["risk_predictive"] >= .5 and risk_start is None:
                    risk_start = step
                if risk_start is not None and step > risk_start and brake_delay is None and info["delta_v"] <= -.1 + 1e-9:
                    brake_delay = (step-risk_start)*base.dt
                if info["risk_predictive"] >= .5:
                    high_risk_v.append(info["actual_v"])
                log.write({"scenario_index": index, "raw_speed_mean": raw_mean, **info})
                if terminated or truncated:
                    break
            records.append({"scenario_id": info["scenario_id"], "scenario_type": info["scenario_type"],
                            "success": info["is_success"], "reason": info["termination_reason"],
                            "collision": info["collision_type"] != "none", "steps": step,
                            "path_length": info["path_length"], "navigation_time": step*base.dt,
                            "path_efficiency": direct/info["path_length"] if info["is_success"] and info["path_length"] > 0 else None,
                            "min_static_clearance": info["min_static_distance"]-5,
                            "min_dynamic_clearance": info["min_dynamic_distance"]-4,
                            "base_return": info["episode_base_return"], "risk_return": info["episode_risk_return"],
                            "training_return": info["episode_training_return"],
                            "full_speed_fraction_after_2s": float(np.mean(full)) if full else None,
                            "clipped_mean_fraction": float(np.mean(clipped)) if clipped else None,
                            "high_risk_mean_speed": float(np.mean(high_risk_v)) if high_risk_v else None,
                            "risk_onset_step": risk_start, "braking_delay_seconds": brake_delay,
                            "risk_no_braking_response": risk_start is not None and brake_delay is None})
    finally:
        log.close()
    frame = pd.DataFrame(records)
    success = frame[frame.success]
    summary = {"num_scenarios": count, "success_count": int(frame.success.sum()),
               "collision_count": int(frame.collision.sum()), "success_rate": float(frame.success.mean()),
               "collision_rate": float(frame.collision.mean()),
               "mean_successful_path_efficiency": float(success.path_efficiency.mean()) if len(success) else 0.,
               "mean_base_reward": float(frame.base_return.mean()),
               "mean_risk_penalty": float(frame.risk_return.mean()),
               "mean_training_reward": float(frame.training_return.mean())}
    return summary, frame


def main():
    parser = argparse.ArgumentParser(description="Development evaluation only; final-test runner belongs to P3-05")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenario-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    data = load_dataset(args.scenario_file)
    if data["dataset_split"] not in ("validation", "diagnostic") or data.get("environment_stage") != "D":
        parser.error("Only D-stage development data permitted here; not final tests")
    model_path = args.run_dir / "models/best/best_model.zip"
    run, config, _ = verify_round_three_model(model_path)
    if run != args.run_dir.resolve():
        raise ValueError("Run mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = RiskRewardWrapper(DynamicPathPlanningEnv(scenario_file=args.scenario_file), config["arm"])
    try:
        summary, frame = evaluate_policy(PPO.load(model_path, device="cpu"), env, len(data["scenarios"]),
                                         args.output_dir / "steps.csv.gz")
    finally:
        env.close()
    frame.to_csv(args.output_dir / "episodes.csv", index=False)
    summary.update(model_sha256=sha256_file(model_path), dataset_hash=dataset_hash(data), arm=config["arm"],
                   selection_reward="base_reward", split=data["dataset_split"])
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
