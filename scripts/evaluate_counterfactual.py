from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402
from envs.scenario_dataset import dataset_hash, load_dataset  # noqa: E402
from experiment_utils import atomic_write_json, sha256_file, utc_now, source_manifest, verify_model_identity  # noqa: E402
from experiment_utils import OBSERVATION_SCHEMA_VERSION, REWARD_SCHEMA_VERSION  # noqa: E402
from scripts.evaluate import atomic_write_csv, default_model_path, model_from_run  # noqa: E402
from scripts.evaluate import save_plot_sheets


VARIANTS = ("baseline", "mirror_y", "freeze_dynamic", "reverse_dynamic", "slow_dynamic")


def mirror_y(scenario: dict) -> dict:
    result = copy.deepcopy(scenario)
    result["start_position"][1] = 100.0 - result["start_position"][1]
    result["goal_position"][1] = 100.0 - result["goal_position"][1]
    result["start_heading"] = -float(result["start_heading"])
    for obstacle in result["static_obstacles"]:
        obstacle["position"][1] = 100.0 - obstacle["position"][1]
    dynamic = result["dynamic_obstacle"]
    dynamic["position"][1] = 100.0 - dynamic["position"][1]
    dynamic["velocity"][1] = -float(dynamic["velocity"][1])
    return result


def make_variant(scenario: dict, variant: str) -> dict:
    result = copy.deepcopy(scenario)
    if variant == "baseline":
        return result
    if variant == "mirror_y":
        return mirror_y(result)
    dynamic = result["dynamic_obstacle"]
    if variant == "freeze_dynamic":
        dynamic["velocity"] = [0.0, 0.0]
    elif variant == "reverse_dynamic":
        dynamic["velocity"] = [-float(v) for v in dynamic["velocity"]]
    elif variant == "slow_dynamic":
        dynamic["velocity"] = [0.5*float(v) for v in dynamic["velocity"]]
    elif variant == "distant_dynamic":
        mean_route_y = 0.5 * (result["start_position"][1] + result["goal_position"][1])
        dynamic["position"][1] = 99.0 if mean_route_y < 50.0 else 1.0
        dynamic["velocity"] = [0.0, 0.0]
        dynamic["radius"] = 0.0
    else:
        raise ValueError(f"Unknown counterfactual variant: {variant}")
    return result


def signed_route_offset(path: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    segment = goal - start
    length = float(np.linalg.norm(segment))
    fractions = ((path - start) @ segment) / float(segment @ segment)
    offsets = (
        segment[0] * (path[:, 1] - start[1])
        - segment[1] * (path[:, 0] - start[0])
    ) / length
    middle = offsets[(fractions >= 0.15) & (fractions <= 0.85)]
    return float(np.mean(middle if middle.size else offsets))


def run_variant(model: PPO, scenario: dict, scenario_id: int, variant: str, traces=None) -> dict:
    env = DynamicPathPlanningEnv(randomize_scenario=False)
    env.set_scenario(make_variant(scenario, variant), scenario_mode=f"counterfactual_{variant}")
    observation, _ = env.reset()
    # A paired diagnostic changes motion, not the baseline observation scale.
    env.dynamic_obstacle_max_speed = float(np.linalg.norm(scenario["dynamic_obstacle"]["velocity"]))
    observation = env._get_observation()
    initial_action, _ = model.predict(observation, deterministic=True)
    actions=[]; speeds=[env.current_linear_velocity]
    dynamic_path=[env.dynamic_position.copy()]
    total_reward = 0.0
    while True:
        action, _ = model.predict(observation, deterministic=True)
        actions.append(np.asarray(action).copy())
        observation, reward, terminated, truncated, info = env.step(action)
        speeds.append(env.current_linear_velocity);dynamic_path.append(env.dynamic_position.copy())
        total_reward += float(reward)
        if terminated or truncated:
            break
    path = np.asarray(env.agent_trajectory)
    start = env.start_position.copy()
    goal = env.goal_position.copy()
    mean_offset = signed_route_offset(path, start, goal)
    route_side = "above" if mean_offset > 0.5 else "below" if mean_offset < -0.5 else "center"
    direct_distance = max(0.0, float(np.linalg.norm(goal - start)) - env.goal_threshold)
    record = {
        "scenario_id": scenario_id,
        "scenario_type": scenario["scenario_type"],
        "variant": variant,
        "success": info["is_success"],
        "termination_reason": info["termination_reason"],
        "collision_type": info["collision_type"],
        "episode_reward": total_reward,
        "steps": info["step_count"],
        "path_length": info["path_length"],
        "path_efficiency": (
            direct_distance / info["path_length"]
            if info["is_success"] and info["path_length"] > 0.0 else None
        ),
        "mean_signed_route_offset": mean_offset,
        "route_side": route_side,
        "initial_linear_action": float(initial_action[0]),
        "initial_angular_action": float(initial_action[1]),
        "min_dynamic_center_distance": info["min_dynamic_distance"],
        "min_dynamic_clearance": info["min_dynamic_distance"]-env.agent_radius-env.dynamic_radius,
        "risk_level": scenario.get("risk_level","legacy"),
        "first_turn_time": next((i*env.dt for i,a in enumerate(actions) if abs(a[1])>.1),None),
        "first_slowdown_time": next((i*env.dt for i in range(1,len(speeds)) if speeds[i]<speeds[i-1]-.01),None),
    }
    if env.stage in ("C","D"):
        from envs.encounters import geometry_error
        record["intervention_geometry_issue"]=geometry_error(make_variant(scenario,variant)) or "none"
    if traces is not None:
        traces[f"{scenario_id}_{variant}_agent"]=path
        traces[f"{scenario_id}_{variant}_dynamic"]=np.asarray(dynamic_path)
        traces[f"{scenario_id}_{variant}_actions"]=np.asarray(actions)
        traces[f"{scenario_id}_{variant}_speeds"]=np.asarray(speeds)
    env.close()
    return record


def summarize(frame: pd.DataFrame) -> dict:
    baseline = frame[frame.variant == "baseline"].set_index("scenario_id")
    variants: dict[str, dict] = {}
    for variant in VARIANTS:
        group = frame[frame.variant == variant].set_index("scenario_id")
        comparison = group.join(
            baseline[["success", "route_side", "path_length"]],
            lsuffix="", rsuffix="_baseline",
        )
        variants[variant] = {
            "scenario_count": int(len(group)),
            "invalid_intervention_geometry_count": int((group.intervention_geometry_issue!="none").sum()) if "intervention_geometry_issue" in group else None,
            "success_count": int(group.success.sum()),
            "success_rate": float(group.success.mean()),
            "collision_count": int((group.collision_type != "none").sum()),
            "average_path_length": float(group.path_length.mean()),
            "average_path_efficiency_successful": (
                None if group[group.success].empty
                else float(group[group.success].path_efficiency.mean())
            ),
            "route_side_counts": {
                str(key): int(value) for key, value in group.route_side.value_counts().items()
            },
            "route_changed_from_baseline_count": int(
                (comparison.route_side != comparison.route_side_baseline).sum()
            ),
            "success_changed_from_baseline_count": int(
                (comparison.success != comparison.success_baseline).sum()
            ),
            "mean_absolute_path_length_change": float(
                (comparison.path_length - comparison.path_length_baseline).abs().mean()
            ),
        }
    mirrored = frame[frame.variant == "mirror_y"].set_index("scenario_id")
    expected_mirror = baseline.route_side.map({"above": "below", "below": "above", "center": "center"})
    variants["mirror_y"]["expected_side_match_count"] = int(
        (mirrored.route_side == expected_mirror).sum()
    )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired counterfactual policy diagnostics")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--scenario-file", type=Path,
        default=ROOT / "configs" / "validation_scenarios_30.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
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
        raise FileNotFoundError(f"Model not found: {model_path}")
    if run_dir is None:
        for parent in model_path.parents:
            if (parent / "metadata.json").exists() and (parent / "config.json").exists():
                run_dir = parent
                break
    training_metadata = None
    if run_dir is not None:
        verified_dir, _, _ = verify_model_identity(model_path, ROOT)
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
    if dataset.get("dataset_split")=="test":
        parser.error("development diagnostics cannot use sealed test split")
    if training_metadata:
        config=json.loads((run_dir/"config.json").read_text(encoding="utf-8"))
        if config["environment"].get("stage","legacy")!=dataset.get("environment_stage","legacy"):
            parser.error("diagnostic dataset must match model stage")
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = ROOT / "results" / "diagnostics" / f"{timestamp}-{model_path.stem}"
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(model_path, device=args.device)
    probe_env = DynamicPathPlanningEnv()
    if model.observation_space.shape != probe_env.observation_space.shape:
        probe_env.close()
        raise ValueError(
            f"Model observation shape {model.observation_space.shape} does not match "
            f"environment shape {probe_env.observation_space.shape}"
        )
    probe_env.close()
    records = []; traces={}
    for scenario in dataset["scenarios"]:
        for variant in VARIANTS:
            records.append(run_variant(model, scenario, int(scenario["scenario_id"]), variant,traces))
    for record in records:
        sid=record["scenario_id"];v=record["variant"]
        a=traces[f"{sid}_{v}_agent"].copy();base=traces[f"{sid}_baseline_agent"]
        if v=="mirror_y":a[:,1]=100-a[:,1]
        count=min(len(a),len(base))
        record["matched_time_path_rms_difference"]=float(np.sqrt(np.mean(np.sum((a[:count]-base[:count])**2,axis=1))))
        acts=traces[f"{sid}_{v}_actions"].copy();bacts=traces[f"{sid}_baseline_actions"]
        if v=="mirror_y":acts[:,1]*=-1
        count=min(len(acts),len(bacts))
        record["matched_time_action_rms_difference"]=float(np.sqrt(np.mean((acts[:count]-bacts[:count])**2)))
    np.savez_compressed(output_dir/"paired_trajectories.npz",**traces)
    import matplotlib.pyplot as plt
    for first in range(0,len(dataset["scenarios"]),3):
        fig,axes=plt.subplots(3,len(VARIANTS),figsize=(20,11),squeeze=False,constrained_layout=True)
        for row,s in enumerate(dataset["scenarios"][first:first+3]):
            sid=s["scenario_id"]
            for col,v in enumerate(VARIANTS):
                ax=axes[row,col]; scene=make_variant(s,v)
                a=traces[f"{sid}_{v}_agent"];d=traces[f"{sid}_{v}_dynamic"]
                ax.plot(a[:,0],a[:,1],"b-");ax.plot(d[:,0],d[:,1],"--",color="orange")
                for o in scene["static_obstacles"]:ax.add_patch(plt.Circle(o["position"],o["radius"],color="gray"))
                for t in range(0,len(a),25):
                    ax.plot(a[t,0],a[t,1],"b.");ax.plot(d[t,0],d[t,1],".",color="orange")
                ax.set(xlim=(0,100),ylim=(0,100),aspect="equal",title=f"#{sid} {v} (dots: 5s)")
        fig.savefig(output_dir/f"paired_{first:03d}.png",dpi=120);plt.close(fig)
    frame = pd.DataFrame(records).sort_values(["scenario_id", "variant"])
    provenance = {
        "evaluated_at": utc_now(),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "run_id": training_metadata.get("run_id") if training_metadata else None,
        "scenario_file": str(scenario_file),
        "dataset_hash": dataset_hash(dataset),
        "dataset_split": dataset.get("dataset_split", "legacy_unspecified"),
        "model_schema_verified_from_run_metadata": training_metadata is not None,
        "variants": list(VARIANTS),
        "source_manifest": source_manifest(ROOT),
        "normalization": "paired baseline speed scale held constant",
        "legacy_training_identity_unverified": training_metadata is None,
        "intervention_note": "Reverse/freeze/slow interventions may violate target-static geometry; flagged per row. Compare trajectories on common time support, not unequal terminal path lengths alone.",
        "distant_dynamic_note": (
            "The fixed-size observation has no absent-obstacle encoding; this variant uses a "
            "stationary zero-radius object at the far vertical boundary as a removal proxy."
        ),
    }
    output = {"_provenance": provenance, "variants": summarize(frame)}
    atomic_write_csv(frame, output_dir / "counterfactual_results.csv")
    atomic_write_json(output_dir / "counterfactual_summary.json", output)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Counterfactual diagnostics saved to {output_dir}")


if __name__ == "__main__":
    main()
