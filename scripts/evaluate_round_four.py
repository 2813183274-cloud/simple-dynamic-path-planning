"""P4 executable-action evaluator; development CLI, shared final-evaluation API."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.scenario_dataset import load_dataset, dataset_hash
from experiment_utils import atomic_write_json, sha256_file
from round_four_policy import LatentActionPPO
from round_four_utils import make_env, verify_model
from scripts.evaluate_round_three import StepLog, selection_key


class ResponseTracker:
    def __init__(self, dt=.2):
        self.dt = dt
        self.onset = None
        self.delay = None
        self.low_run = 0
        self.max_low_run = 0

    def update(self, info):
        # step_count is post-step, so pre-action t >= 2 starts at step 11.
        step = info["step_count"]
        if step > 10 and info["pre_action_q4"] >= .5 and self.onset is None:
            self.onset = step
        if self.onset is not None and self.delay is None and info["delta_v"] <= -.1+1e-9:
            self.delay = (step-self.onset)*self.dt
        self.low_run = self.low_run+1 if info["actual_v"] < .1 else 0
        self.max_low_run = max(self.low_run, self.max_low_run)


def evaluate_policy(model, env, count, step_log=None):
    if count <= 0:
        raise ValueError("At least one scenario required")
    rows = []
    log = StepLog(step_log)
    try:
        for index in range(count):
            obs, _ = env.reset(options={"scenario_index": index})
            base = env.unwrapped
            direct = max(0., float(np.linalg.norm(base.goal_position-base.start_position))-base.goal_threshold)
            tracker = ResponseTracker(base.dt)
            full = []; near = []; high = []; brakes = []
            for _ in range(base.max_steps):
                if isinstance(model, LatentActionPPO):
                    action, details = model.action_details(obs, deterministic=True)
                else:  # DWA: no latent distribution, still executable actions.
                    action, _ = model.predict(obs, deterministic=True)
                    details = {"executed_speed_action": float(action[0]), "executed_turn_action": float(action[1])}
                obs, _, done, truncated, info = env.step(action)
                tracker.update(info)
                if info["step_count"] > 10:
                    full.append(info["actual_v"] >= 2.99)
                    near.append(action[0] >= .99)
                    if info["pre_action_q4"] >= .5:
                        high.append(info["actual_v"])
                        brakes.append(info["delta_v"] <= -.1+1e-9)
                log.write({"scenario_index": index, **details, **info})
                if done or truncated:
                    break
            rows.append(dict(scenario_id=info["scenario_id"], scenario_type=info["scenario_type"],
                success=info["is_success"], reason=info["termination_reason"],
                collision=info["collision_type"] != "none", steps=info["step_count"],
                path_length=info["path_length"], navigation_time=info["step_count"]*base.dt,
                path_efficiency=direct/info["path_length"] if info["is_success"] and info["path_length"]>0 else None,
                min_static_clearance=info["min_static_distance"]-5,
                min_dynamic_clearance=info["min_dynamic_distance"]-4,
                base_return=info["episode_base_return"],
                full_speed_fraction_after_2s=float(np.mean(full)) if full else None,
                near_upper_action_fraction_after_2s=float(np.mean(near)) if near else None,
                high_risk_mean_speed=float(np.mean(high)) if high else None,
                high_risk_braking_fraction=float(np.mean(brakes)) if brakes else None,
                risk_onset_step=tracker.onset, braking_delay_seconds=tracker.delay,
                risk_no_braking_response=tracker.onset is not None and tracker.delay is None,
                response_status="no_risk" if tracker.onset is None else "responded" if tracker.delay is not None else "terminal_censored",
                timeout_with_stagnation=bool(truncated and tracker.max_low_run >= 25)))
    finally:
        log.close()
    frame = pd.DataFrame(rows)
    success = frame[frame.success]
    summary = dict(num_scenarios=count, success_count=int(frame.success.sum()),
        collision_count=int(frame.collision.sum()), success_rate=float(frame.success.mean()),
        collision_rate=float(frame.collision.mean()), mean_base_reward=float(frame.base_return.mean()),
        mean_successful_path_efficiency=float(success.path_efficiency.mean()) if len(success) else 0.)
    return summary, frame


def main():
    parser = argparse.ArgumentParser(description="P4 development evaluation; does not generate final datasets")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    data = load_dataset(args.scenario_file)
    if data["dataset_split"] not in ("validation", "diagnostic") or data["environment_stage"] != "D":
        parser.error("Development data only; final evaluation needs P4-04 model/data freeze")
    path = args.run_dir/"models/best/best_model.zip"
    _, config, _ = verify_model(path)
    model = LatentActionPPO.load(path, device="cpu")
    if model.arm != config["arm"]:
        raise ValueError("Saved model arm differs from certified run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = make_env(args.scenario_file)
    try:
        summary, frame = evaluate_policy(model, env, len(data["scenarios"]), args.output_dir/"steps.csv.gz")
        if args.plots:
            from scripts.evaluate import run_scenario, save_plot_sheets
            histories = []
            for i in range(len(data["scenarios"])):
                record, history = run_scenario(model, env.unwrapped, i)
                if record["termination_reason"] != frame.iloc[i]["reason"]:
                    raise ValueError("Plot replay disagrees with evaluation")
                histories.append(history)
            save_plot_sheets(histories, args.output_dir)
    finally:
        env.close()
    frame.to_csv(args.output_dir/"episodes.csv", index=False)
    summary.update(arm=model.arm, dataset_hash=dataset_hash(data), model_sha256=sha256_file(path))
    atomic_write_json(args.output_dir/"summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
