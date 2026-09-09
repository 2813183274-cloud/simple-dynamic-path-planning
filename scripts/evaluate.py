from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402
from envs.scenario_dataset import TYPE_QUOTAS, dataset_hash, load_dataset  # noqa: E402
from experiment_utils import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
    utc_now,
    source_manifest,
    dependency_versions,
    environment_signature,
    verify_model_identity,
)


def default_model_path() -> Path:
    latest_pointer = ROOT / "runs" / "latest_run.json"
    if latest_pointer.exists():
        latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
        run_dir = Path(latest["run_dir"])
        best = run_dir / "models" / "best" / "best_model.zip"
        final = run_dir / "models" / "final_model.zip"
        if best.exists() or final.exists():
            return best if best.exists() else final
    best = ROOT / "models" / "best" / "best_model.zip"
    return best if best.exists() else ROOT / "models" / "final_model.zip"


def model_from_run(run_dir: Path) -> Path:
    best = run_dir / "models" / "best" / "best_model.zip"
    final = run_dir / "models" / "final_model.zip"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f"No best or final model found in run: {run_dir}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def safe_mean(series: pd.Series) -> float | None:
    return None if series.empty else float(series.mean())


def run_scenario(model: PPO, env: DynamicPathPlanningEnv, index: int) -> tuple[dict, dict]:
    observation, reset_info = env.reset(options={"scenario_index": index})
    agent_positions = [env.agent_position.copy()]
    dynamic_positions = [env.dynamic_position.copy()]
    headings = [float(env.agent_heading)]
    linear_velocities = [float(env.current_linear_velocity)]
    angular_velocities = [float(env.current_angular_velocity)]
    dynamic_velocities = [env.dynamic_velocity.copy()]
    episode_reward = 0.0
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(reward) or not np.all(np.isfinite(observation)):
            raise FloatingPointError(f"NaN/Inf in scenario {index}")
        episode_reward += float(reward)
        agent_positions.append(env.agent_position.copy())
        dynamic_positions.append(env.dynamic_position.copy())
        headings.append(float(env.agent_heading))
        linear_velocities.append(float(env.current_linear_velocity))
        angular_velocities.append(float(env.current_angular_velocity))
        dynamic_velocities.append(env.dynamic_velocity.copy())
        if terminated or truncated:
            break

    agent_path = np.asarray(agent_positions)
    dynamic_path = np.asarray(dynamic_positions)
    dynamic_centers = np.linalg.norm(agent_path - dynamic_path, axis=1)
    dynamic_clearances = dynamic_centers - env.agent_radius - env.dynamic_radius
    static_center_history = np.stack([
        np.linalg.norm(agent_path - position, axis=1) for position in env.static_obstacles
    ], axis=1)
    static_clearance_history = static_center_history - env.agent_radius - env.static_radii
    closest_step = int(np.argmin(dynamic_centers))
    record = {
        "scenario_id": reset_info["scenario_id"],
        "scenario_type": reset_info["scenario_type"],
        "difficulty": reset_info["difficulty"],
        "risk_level": env.current_scenario.get("risk_level","legacy"),
        "layout_type": env.current_scenario.get("layout_type","legacy"),
        "success": info["is_success"],
        "termination_reason": info["termination_reason"],
        "collision_type": info["collision_type"],
        "episode_reward": episode_reward,
        "steps": info["step_count"],
        "navigation_time": info["step_count"] * env.dt,
        "path_length": info["path_length"],
        "direct_start_goal_distance": max(0.0, float(np.linalg.norm(
            env.goal_position - env.start_position
        )) - env.goal_threshold),
        "final_goal_distance": info["goal_distance"],
        "min_static_center_distance": float(np.min(static_center_history)),
        "min_static_clearance": float(np.min(static_clearance_history)),
        "min_dynamic_center_distance": float(dynamic_centers[closest_step]),
        "min_dynamic_clearance": float(dynamic_clearances[closest_step]),
        "closest_dynamic_step": closest_step,
        "closest_dynamic_time": closest_step * env.dt,
    }
    record["path_efficiency"] = (
        record["direct_start_goal_distance"] / record["path_length"]
        if record["success"] and record["path_length"] > 0.0 else None
    )
    history = {
        "scenario": env.current_scenario,
        "agent_path": agent_path,
        "dynamic_path": dynamic_path,
        "headings": np.asarray(headings),
        "linear_velocities": np.asarray(linear_velocities),
        "angular_velocities": np.asarray(angular_velocities),
        "dynamic_velocities": np.asarray(dynamic_velocities),
        "dynamic_centers": dynamic_centers,
        "dynamic_clearances": dynamic_clearances,
        "closest_step": closest_step,
        "record": record,
    }
    return record, history


def draw_scene_base(ax, scenario: dict) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle(scenario["goal_position"], 3.0, color="green", alpha=0.18))
    ax.plot(*scenario["start_position"], "bo", markersize=3)
    ax.plot(*scenario["goal_position"], "g*", markersize=7)
    for obstacle in scenario["static_obstacles"]:
        ax.add_patch(plt.Circle(obstacle["position"], obstacle["radius"], color="gray", alpha=0.75))
    ax.tick_params(labelsize=6)


def draw_full_subplot(ax, history: dict) -> None:
    scenario, record = history["scenario"], history["record"]
    agent, dynamic = history["agent_path"], history["dynamic_path"]
    closest = history["closest_step"]
    draw_scene_base(ax, scenario)
    ax.plot(agent[:, 0], agent[:, 1], "b-", linewidth=1)
    ax.plot(dynamic[:, 0], dynamic[:, 1], "--", color="orange", linewidth=1)
    radius = scenario["dynamic_obstacle"]["radius"]
    ax.add_patch(plt.Circle(dynamic[0], radius, fill=False, edgecolor="orange", linestyle=":"))
    ax.add_patch(plt.Circle(dynamic[-1], radius, color="orange", alpha=0.35))
    ax.add_patch(plt.Circle(agent[closest], 1.0, color="blue", alpha=0.8))
    ax.add_patch(plt.Circle(dynamic[closest], radius, color="orange", alpha=0.75))
    ax.plot([agent[closest, 0], dynamic[closest, 0]],
            [agent[closest, 1], dynamic[closest, 1]], "r--", linewidth=0.8)
    ax.set_title(
        f"#{record['scenario_id']:02d} {record['scenario_type']} / {record['difficulty']}\n"
        f"{record['termination_reason']}; center={record['min_dynamic_center_distance']:.2f} m, "
        f"clear={record['min_dynamic_clearance']:.2f} m",
        fontsize=7,
    )


def draw_closest_subplot(ax, history: dict) -> None:
    scenario, record = history["scenario"], history["record"]
    agent, dynamic = history["agent_path"], history["dynamic_path"]
    closest = history["closest_step"]
    start, end = max(0, closest - 15), min(len(agent), closest + 16)
    radius = scenario["dynamic_obstacle"]["radius"]
    ax.plot(agent[start:end, 0], agent[start:end, 1], "b-", linewidth=1)
    ax.plot(dynamic[start:end, 0], dynamic[start:end, 1], "--", color="orange", linewidth=1)
    ax.add_patch(plt.Circle(agent[closest], 1.0, color="blue", alpha=0.85))
    ax.add_patch(plt.Circle(dynamic[closest], radius, color="orange", alpha=0.8))
    heading = history["headings"][closest]
    ax.arrow(agent[closest, 0], agent[closest, 1], 3.0 * math.cos(heading), 3.0 * math.sin(heading),
             width=0.12, head_width=0.8, color="navy", length_includes_head=True)
    ax.plot([agent[closest, 0], dynamic[closest, 0]],
            [agent[closest, 1], dynamic[closest, 1]], "r--", linewidth=1)
    points = np.vstack([agent[start:end], dynamic[start:end]])
    ax.set_xlim(max(0.0, points[:, 0].min() - 12.0), min(100.0, points[:, 0].max() + 12.0))
    ax.set_ylim(max(0.0, points[:, 1].min() - 12.0), min(100.0, points[:, 1].max() + 12.0))
    ax.set_aspect("equal")
    dynamic_velocity = history["dynamic_velocities"][closest]
    ax.set_title(
        f"#{record['scenario_id']:02d} step={closest}, t={record['closest_dynamic_time']:.1f}s\n"
        f"center={record['min_dynamic_center_distance']:.2f}, clear={record['min_dynamic_clearance']:.2f} m; "
        f"v={history['linear_velocities'][closest]:.2f}, ω={history['angular_velocities'][closest]:.2f}\n"
        f"dynamic velocity=({dynamic_velocity[0]:.2f}, {dynamic_velocity[1]:.2f}) m/s",
        fontsize=7,
    )
    ax.tick_params(labelsize=6)


def save_plot_sheets(histories: list[dict], results_dir: Path) -> None:
    plot_dir = results_dir / "trajectory_sheets"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scenarios_per_sheet = 3
    for first in range(0, len(histories), scenarios_per_sheet):
        batch = histories[first:first + scenarios_per_sheet]
        last = first + len(batch) - 1
        # One row per scenario: full trajectory on the left, closest encounter on the right.
        fig, axes = plt.subplots(scenarios_per_sheet, 2, figsize=(14, 15), constrained_layout=True)
        for row, history in enumerate(batch):
            draw_full_subplot(axes[row, 0], history)
            draw_closest_subplot(axes[row, 1], history)
            axes[row, 0].set_ylabel(f"Scenario {history['record']['scenario_id']:02d}\ny (m)", fontsize=8)
        for row in range(len(batch), scenarios_per_sheet):
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
        axes[0, 0].text(0.5, 1.12, "Full trajectory", transform=axes[0, 0].transAxes,
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        axes[0, 1].text(0.5, 1.12, "Closest dynamic encounter", transform=axes[0, 1].transAxes,
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
        fig.suptitle(
            f"Fixed evaluation scenarios {first:03d}-{last:03d}: full vs. closest dynamic encounter",
            fontsize=15,
        )
        # Re-check the directory before every write. This also handles mounted filesystems
        # where an empty output directory may be removed by an external cleanup process.
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = plot_dir / f"scenarios_{first:03d}_{last:03d}_comparison.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot sheet: {output_path}")


def build_summary(frame: pd.DataFrame) -> dict:
    success = frame[frame.success]
    collision = frame.collision_type != "none"
    static_collision = frame.collision_type.str.startswith("static_collision")
    dynamic_collision = frame.collision_type == "dynamic_collision"
    out_of_bounds = frame.termination_reason == "out_of_bounds"
    timeout = frame.termination_reason == "timeout"
    total = len(frame)
    success_count = int(frame.success.sum())
    collision_count = int(collision.sum())
    return {
        "total_scenarios": total,
        "success_count": success_count, "success_rate": float(frame.success.mean()),
        "success_rate_wilson_95": wilson_interval(success_count, total),
        "collision_count": collision_count, "collision_rate": float(collision.mean()),
        "collision_rate_wilson_95": wilson_interval(collision_count, total),
        "static_collision_count": int(static_collision.sum()), "static_collision_rate": float(static_collision.mean()),
        "dynamic_collision_count": int(dynamic_collision.sum()), "dynamic_collision_rate": float(dynamic_collision.mean()),
        "out_of_bounds_count": int(out_of_bounds.sum()), "out_of_bounds_rate": float(out_of_bounds.mean()),
        "timeout_count": int(timeout.sum()), "timeout_rate": float(timeout.mean()),
        "average_reward": float(frame.episode_reward.mean()),
        "average_steps": float(frame.steps.mean()),
        "average_navigation_time": float(frame.navigation_time.mean()),
        "average_path_length": float(frame.path_length.mean()),
        "successful_average_path_length": safe_mean(success.path_length),
        "successful_average_path_efficiency": safe_mean(success.path_efficiency),
        "average_final_goal_distance": float(frame.final_goal_distance.mean()),
        "average_min_static_clearance": float(frame.min_static_clearance.mean()),
        "average_min_dynamic_clearance": float(frame.min_dynamic_clearance.mean()),
    }


def build_type_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_type in sorted(frame.scenario_type.unique()):
        group = frame[frame.scenario_type == scenario_type]
        rows.append({
            "scenario_type": scenario_type,
            "scenario_count": len(group),
            "success_count": int(group.success.sum()),
            "success_rate": float(group.success.mean()),
            "success_rate_wilson_95_low": wilson_interval(int(group.success.sum()), len(group))[0],
            "success_rate_wilson_95_high": wilson_interval(int(group.success.sum()), len(group))[1],
            "static_collision_count": int(group.collision_type.str.startswith("static_collision").sum()),
            "dynamic_collision_count": int((group.collision_type == "dynamic_collision").sum()),
            "average_path_length": float(group.path_length.mean()),
            "successful_average_path_efficiency": safe_mean(group[group.success].path_efficiency),
            "average_navigation_time": float(group.navigation_time.mean()),
            "average_min_static_clearance": float(group.min_static_clearance.mean()),
            "average_min_dynamic_clearance": float(group.min_dynamic_clearance.mean()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one deterministic episode on each fixed scenario")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--scenario-file", type=Path,
                        default=ROOT / "configs" / "independent_test_scenarios_30.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-test-split", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cross-stage-diagnostic", action="store_true",
                        help="Explicit transfer diagnostic: dataset motion with model's original observation encoding")
    args = parser.parse_args()
    if args.model is not None and args.run_dir is not None:
        parser.error("use either --model or --run-dir, not both")
    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    model_path = (
        args.model.resolve() if args.model is not None
        else model_from_run(run_dir) if run_dir is not None
        else default_model_path().resolve()
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run scripts/train.py first.")
    if run_dir is None:
        for parent in model_path.parents:
            if (parent / "metadata.json").exists() and (parent / "config.json").exists():
                run_dir = parent
                break
    training_metadata = None
    if run_dir is not None:
        verified_dir, training_config, _ = verify_model_identity(model_path, ROOT,
            allow_archived_source=args.cross_stage_diagnostic)
        if verified_dir != run_dir:
            raise ValueError("Requested run does not own model")
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
        training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if training_metadata.get("reward_schema_version") != REWARD_SCHEMA_VERSION:
            raise ValueError("Run reward schema is incompatible with the current environment")
    scenario_file = args.scenario_file.resolve()
    dataset = load_dataset(scenario_file)
    dataset_stage=dataset.get("environment_stage","legacy")
    model_stage=training_config["environment"].get("stage","legacy") if training_metadata else dataset_stage
    if training_metadata and model_stage!=dataset_stage and not args.cross_stage_diagnostic:
        parser.error("model and dataset stages differ; use explicit cross-stage diagnostic")
    if args.cross_stage_diagnostic and dataset.get("dataset_split")=="test":
        parser.error("cross-stage development diagnostics may not use sealed test data")
    split = dataset.get("dataset_split", "legacy_unspecified")
    if split != "test" and not args.allow_non_test_split:
        parser.error(
            f"final evaluation requires dataset_split='test', got {split!r}; "
            "use --allow-non-test-split only for diagnostics"
        )
    if args.output_dir is not None:
        results_dir = args.output_dir.resolve()
    elif run_dir is not None:
        results_dir = run_dir / "evaluation" / scenario_file.stem
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = ROOT / "results" / "evaluations" / (
            f"{timestamp}-{model_path.stem}-{scenario_file.stem}"
        )
    if results_dir.exists() and any(results_dir.iterdir()) and not args.overwrite:
        parser.error(f"output directory is not empty: {results_dir}; choose another path")
    results_dir.mkdir(parents=True, exist_ok=True)
    env = DynamicPathPlanningEnv(randomize_scenario=False, scenario_file=scenario_file)
    env.observation_stage=model_stage
    model = PPO.load(model_path, device=args.device)
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Model observation shape {model.observation_space.shape} does not match "
            f"environment shape {env.observation_space.shape}"
        )
    records, histories = [], []
    for index in range(dataset["num_scenarios"]):
        record, history = run_scenario(model, env, index)
        records.append(record)
        histories.append(history)
        print(f"[{index + 1:02d}/30] scenario {index:02d}: {record['termination_reason']}")
    frame = pd.DataFrame(records).sort_values("scenario_id")
    summary = build_summary(frame)
    by_type = build_type_metrics(frame)
    provenance = {
        "evaluated_at": utc_now(),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "run_id": training_metadata.get("run_id") if training_metadata else None,
        "scenario_file": str(scenario_file),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_split": split,
        "dataset_seed": dataset.get("dataset_seed"),
        "dataset_hash": dataset_hash(dataset),
        "observation_schema_version": training_metadata.get("observation_schema_version") if training_metadata else OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "model_schema_verified_from_run_metadata": training_metadata is not None,
        "deterministic_policy": True,
        "source_manifest": source_manifest(ROOT),
        "dependencies": dependency_versions(),
        "environment": environment_signature(env),
        "model_num_timesteps": int(model.num_timesteps),
        "legacy_training_identity_unverified": training_metadata is None,
        "cross_stage_diagnostic": args.cross_stage_diagnostic,
        "model_stage": model_stage, "dataset_stage": dataset_stage,
    }
    summary["_provenance"] = provenance
    atomic_write_csv(frame, results_dir / "episode_results.csv")
    atomic_write_csv(by_type, results_dir / "metrics_by_scenario_type.csv")
    for column in ("risk_level","layout_type"):
        grouped=[{column:str(label),**build_summary(group)} for label,group in frame.groupby(column)]
        atomic_write_json(results_dir/f"metrics_by_{column}.json",grouped)
    atomic_write_json(results_dir / "summary_metrics.json", summary)
    atomic_write_json(results_dir / "evaluation_metadata.json", provenance)
    if not args.no_plots:
        save_plot_sheets(histories, results_dir)
    env.close()
    print(f"Evaluation outputs saved to {results_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
