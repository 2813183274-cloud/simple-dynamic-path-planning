from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DATASET_NAME = "fixed_scenario_set_30_v2"
DATASET_SEED = 2026
TYPE_QUOTAS = {
    "low_dynamic_risk": 5,
    "static_dominant": 5,
    "crossing_up": 6,
    "crossing_down": 6,
    "mixed_risk": 5,
    "high_risk": 3,
}


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenario_hash(scenario: dict[str, Any]) -> str:
    payload = {
        "start_position": scenario["start_position"],
        "start_heading": scenario["start_heading"],
        "goal_position": scenario["goal_position"],
        "static_obstacles": scenario["static_obstacles"],
        "dynamic_obstacle": scenario["dynamic_obstacle"],
        **({"environment_stage":scenario["environment_stage"],
            "scenario_type":scenario.get("scenario_type"),"risk_level":scenario.get("risk_level"),
            "layout_type":scenario.get("layout_type"),"generation":scenario.get("generation")}
           if "environment_stage" in scenario else {}),
    }

    def rounded(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rounded(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return round(float(value), 4)
        return value

    canonical = json.dumps(rounded(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_hash(dataset: dict[str, Any]) -> str:
    """Identify a scenario collection independently of its file path."""
    payload = {
        "schema_version": dataset.get("schema_version"),
        "dataset_name": dataset.get("dataset_name"),
        "dataset_split": dataset.get("dataset_split", "legacy_unspecified"),
        "dataset_seed": dataset.get("dataset_seed"),
        "scenario_hashes": [item["scenario_hash"] for item in dataset["scenarios"]],
        **({"environment_stage": dataset["environment_stage"],
            "generation_config": dataset.get("generation_config")} if "environment_stage" in dataset else {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def save_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_native(dataset), indent=2, ensure_ascii=False), encoding="utf-8")


def _distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def point_to_segment_distance(point, start, goal) -> float:
    point, start, goal = (np.asarray(value, dtype=float) for value in (point, start, goal))
    segment = goal - start
    fraction = float(np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def static_layout_passable(scenario: dict[str, Any], agent_radius: float = 1.0) -> bool:
    """Reject a connected inflated-obstacle barrier from the bottom to top boundary."""
    obstacles = scenario["static_obstacles"]
    count = len(obstacles)
    touches_bottom = {
        i for i, obstacle in enumerate(obstacles)
        if obstacle["position"][1] - obstacle["radius"] - agent_radius <= agent_radius
    }
    touches_top = {
        i for i, obstacle in enumerate(obstacles)
        if obstacle["position"][1] + obstacle["radius"] + agent_radius >= 100.0 - agent_radius
    }
    frontier, visited = list(touches_bottom), set(touches_bottom)
    while frontier:
        current = frontier.pop()
        if current in touches_top:
            return False
        for other in range(count):
            if other in visited:
                continue
            required = (obstacles[current]["radius"] + obstacles[other]["radius"]
                        + 2.0 * agent_radius)
            if _distance(obstacles[current]["position"], obstacles[other]["position"]) <= required:
                visited.add(other)
                frontier.append(other)
    return True


def validate_scenario_geometry(scenario: dict[str, Any]) -> None:
    start = np.asarray(scenario["start_position"], dtype=float)
    goal = np.asarray(scenario["goal_position"], dtype=float)
    if not (5.0 <= start[0] <= 15.0 and 35.0 <= start[1] <= 65.0):
        raise ValueError("Start position outside the required range")
    if not (85.0 <= goal[0] <= 95.0 and 35.0 <= goal[1] <= 65.0):
        raise ValueError("Goal position outside the required range")
    if _distance(start, goal) < 65.0:
        raise ValueError("Start-goal distance is below 65 m")
    obstacles = scenario["static_obstacles"]
    dynamic = scenario["dynamic_obstacle"]
    if len(obstacles) != 2:
        raise ValueError("Exactly two static obstacles are required")
    all_obstacles = obstacles + [dynamic]
    for obstacle in all_obstacles:
        x, y = obstacle["position"]
        radius = float(obstacle["radius"])
        if not (radius <= x <= 100.0 - radius and radius <= y <= 100.0 - radius):
            raise ValueError("Obstacle is not fully inside the map")
        if _distance(start, obstacle["position"]) <= 1.0 + radius + 5.0:
            raise ValueError("Start is too close to an obstacle")
        if _distance(goal, obstacle["position"]) <= 1.0 + radius + 5.0:
            raise ValueError("Goal is too close to an obstacle")
    for left in range(len(all_obstacles)):
        for right in range(left + 1, len(all_obstacles)):
            minimum = all_obstacles[left]["radius"] + all_obstacles[right]["radius"] + 3.0
            if _distance(all_obstacles[left]["position"], all_obstacles[right]["position"]) <= minimum:
                raise ValueError("Obstacles overlap or lack the required clearance")
    if not static_layout_passable(scenario):
        raise ValueError("Static obstacle layout blocks all top-to-bottom passages")
    velocity = dynamic["velocity"]
    if float(velocity[0]) != 0.0 or not (0.8 <= abs(float(velocity[1])) <= 1.3):
        raise ValueError("Invalid dynamic obstacle velocity")


def direct_baseline_action(env) -> np.ndarray:
    delta = env.goal_position - env.agent_position
    goal_angle = math.atan2(delta[1], delta[0])
    error = env._normalize_angle(goal_angle - env.agent_heading)
    target_omega = float(np.clip(1.5 * error, -env.omega_max, env.omega_max))
    absolute_error = abs(error)
    fraction = 0.9 if absolute_error < 0.2 else 0.6 if absolute_error < 0.6 else 0.3
    target_v = fraction * env.v_max
    return np.array([2.0 * target_v / env.v_max - 1.0, target_omega / env.omega_max], dtype=np.float32)


def simulate_baseline(scenario: dict[str, Any]) -> dict[str, Any]:
    from .dynamic_path_env import DynamicPathPlanningEnv

    env = DynamicPathPlanningEnv(randomize_scenario=False)
    env.set_scenario(scenario, scenario_mode="baseline_validation")
    _, _ = env.reset()
    min_static_clearance = math.inf
    min_dynamic_clearance = math.inf
    min_static_center = math.inf
    min_dynamic_center = math.inf
    closest_dynamic_step = 0
    info = None
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(direct_baseline_action(env))
        static_centers = np.linalg.norm(env.static_obstacles - env.agent_position, axis=1)
        static_clearances = static_centers - env.agent_radius - env.static_radii
        dynamic_center = float(np.linalg.norm(env.dynamic_position - env.agent_position))
        dynamic_clearance = dynamic_center - env.agent_radius - env.dynamic_radius
        min_static_center = min(min_static_center, float(np.min(static_centers)))
        min_static_clearance = min(min_static_clearance, float(np.min(static_clearances)))
        if dynamic_clearance < min_dynamic_clearance:
            min_dynamic_clearance = dynamic_clearance
            min_dynamic_center = dynamic_center
            closest_dynamic_step = env.step_count
        if terminated or truncated:
            break
    assert info is not None
    result = {
        "baseline_termination_reason": info["termination_reason"],
        "baseline_success": info["is_success"],
        "baseline_min_static_center_distance": min_static_center,
        "baseline_min_static_clearance": min_static_clearance,
        "baseline_min_dynamic_center_distance": min_dynamic_center,
        "baseline_min_dynamic_clearance": min_dynamic_clearance,
        "baseline_closest_dynamic_step": closest_dynamic_step,
        "baseline_steps": info["step_count"],
    }
    env.close()
    return to_native(result)


def _difficulty_from_dynamic(clearance: float) -> str:
    if clearance <= 1.0:
        return "hard"
    if clearance <= 4.0:
        return "medium"
    return "easy"


def _difficulty_from_static(clearance: float) -> str:
    if clearance <= 1.0:
        return "hard"
    if clearance <= 4.0:
        return "medium"
    return "easy"


def _candidate(rng: np.random.Generator, scenario_type: str, desired_difficulty: str | None) -> dict[str, Any]:
    start = rng.uniform([5.0, 35.0], [15.0, 65.0])
    goal = rng.uniform([85.0, 35.0], [95.0, 65.0])
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0]) + rng.uniform(-0.25, 0.25)
    heading = (heading + math.pi) % (2.0 * math.pi) - math.pi

    radii = rng.uniform(3.5, 5.0, size=2)
    x_values = [rng.uniform(28.0, 45.0), rng.uniform(58.0, 75.0)]
    y_on_line = [start[1] + (x - start[0]) / (goal[0] - start[0]) * (goal[1] - start[1])
                 for x in x_values]
    if scenario_type in {"static_dominant", "mixed_risk"}:
        close_index = int(rng.integers(0, 2))
        y_values = []
        for index in range(2):
            if index == close_index:
                offset_magnitude = radii[index] + 1.0 + rng.uniform(0.2, 3.8)
            else:
                offset_magnitude = rng.uniform(12.0, 20.0)
            y_values.append(y_on_line[index] + rng.choice([-1.0, 1.0]) * offset_magnitude)
    else:
        y_values = [line_y + rng.choice([-1.0, 1.0]) * rng.uniform(12.0, 20.0)
                    for line_y in y_on_line]
    y_values = np.clip(y_values, 28.0, 72.0)

    dynamic_x = float(rng.uniform(43.0, 57.0))
    if scenario_type == "crossing_up":
        direction = 1.0
    elif scenario_type == "crossing_down":
        direction = -1.0
    else:
        direction = rng.choice([-1.0, 1.0])
    speed = float(rng.uniform(0.8, 1.3))
    dynamic_vy = direction * speed
    dynamic_radius = float(rng.uniform(2.5, 4.0))
    path_y = start[1] + (dynamic_x - start[0]) / (goal[0] - start[0]) * (goal[1] - start[1])
    dx = dynamic_x - start[0]
    nominal_time = 1.8 + max(0.0, dx - 2.43) / 2.7

    if scenario_type == "low_dynamic_risk" or scenario_type == "static_dominant":
        desired_offset = rng.choice([-1.0, 1.0]) * rng.uniform(14.0, 22.0)
    elif desired_difficulty == "easy":
        desired_offset = rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 13.0)
    elif desired_difficulty == "medium":
        desired_offset = rng.choice([-1.0, 1.0]) * rng.uniform(5.0, 8.0)
    elif desired_difficulty == "hard" or scenario_type == "high_risk":
        desired_offset = rng.uniform(-0.7, 0.7)
    else:
        desired_offset = rng.uniform(-5.0, 5.0)
    dynamic_y = path_y + desired_offset - dynamic_vy * nominal_time
    y_bounds = (25.0, 40.0) if direction > 0 else (60.0, 75.0)
    if not y_bounds[0] <= dynamic_y <= y_bounds[1]:
        dynamic_y = float(rng.uniform(*y_bounds))

    return {
        "scenario_id": -1,
        "scenario_type": scenario_type,
        "difficulty": desired_difficulty or "unassigned",
        "start_position": start.tolist(),
        "start_heading": float(heading),
        "goal_position": goal.tolist(),
        "static_obstacles": [
            {"position": [float(x_values[i]), float(y_values[i])], "radius": float(radii[i])}
            for i in range(2)
        ],
        "dynamic_obstacle": {
            "position": [dynamic_x, float(dynamic_y)],
            "velocity": [0.0, dynamic_vy],
            "radius": dynamic_radius,
        },
    }


def _accept(scenario: dict[str, Any], validation: dict[str, Any], desired: str | None) -> bool:
    scenario_type = scenario["scenario_type"]
    dynamic = validation["baseline_min_dynamic_clearance"]
    static = validation["baseline_min_static_clearance"]
    reason = validation["baseline_termination_reason"]
    if scenario_type == "low_dynamic_risk":
        return dynamic > 8.0 and reason != "dynamic_collision"
    if scenario_type == "static_dominant":
        return static <= 4.0 and dynamic > 6.0
    if scenario_type in {"crossing_up", "crossing_down"}:
        return dynamic <= 8.0 and reason not in {"static_collision_1", "static_collision_2"} \
            and _difficulty_from_dynamic(dynamic) == desired
    if scenario_type == "mixed_risk":
        return static <= 4.0 and dynamic <= 6.0 and reason not in {"static_collision_1", "static_collision_2"}
    if scenario_type == "high_risk":
        return dynamic <= 1.0 or reason == "dynamic_collision"
    return False


def generate_dataset(
    seed: int = DATASET_SEED,
    max_attempts_per_scenario: int = 2_000,
    dataset_split: str = "unspecified",
    dataset_name: str | None = None,
) -> dict[str, Any]:
    if dataset_split not in {"validation", "test", "diagnostic", "unspecified"}:
        raise ValueError(f"Unsupported dataset split: {dataset_split}")
    rng = np.random.default_rng(seed)
    requests: list[tuple[str, str | None]] = []
    requests.extend([("low_dynamic_risk", None)] * 5)
    requests.extend([("static_dominant", None)] * 5)
    for scenario_type in ("crossing_up", "crossing_down"):
        requests.extend((scenario_type, difficulty) for difficulty in ("easy", "easy", "medium", "medium", "hard", "hard"))
    requests.extend([("mixed_risk", None)] * 5)
    requests.extend([("high_risk", "hard")] * 3)
    scenarios, hashes = [], set()
    for scenario_id, (scenario_type, desired) in enumerate(requests):
        for _ in range(max_attempts_per_scenario):
            scenario = _candidate(rng, scenario_type, desired)
            try:
                validate_scenario_geometry(scenario)
            except ValueError:
                continue
            validation = simulate_baseline(scenario)
            if not _accept(scenario, validation, desired):
                continue
            scenario["difficulty"] = (
                _difficulty_from_static(validation["baseline_min_static_clearance"])
                if scenario_type in {"low_dynamic_risk", "static_dominant"}
                else _difficulty_from_dynamic(validation["baseline_min_dynamic_clearance"])
            )
            scenario["validation"] = validation
            scenario["scenario_id"] = scenario_id
            digest = scenario_hash(scenario)
            if digest in hashes:
                continue
            scenario["scenario_hash"] = digest
            hashes.add(digest)
            scenarios.append(scenario)
            break
        else:
            raise RuntimeError(
                f"Unable to generate scenario {scenario_id} ({scenario_type}, {desired}) "
                f"after {max_attempts_per_scenario} attempts"
            )
    dataset = {
        "schema_version": 2,
        "dataset_name": dataset_name or f"{dataset_split}_scenario_set_30_seed_{seed}_v2",
        "dataset_split": dataset_split,
        "dataset_seed": int(seed),
        "num_scenarios": 30,
        "generation_config": {
            "max_attempts_per_scenario": max_attempts_per_scenario,
            "type_quotas": TYPE_QUOTAS,
            "environment": {
                "map_size": [100.0, 100.0], "dt": 0.2, "v_max": 3.0,
                "omega_max": math.pi / 4.0, "linear_acceleration_max": 1.5,
                "angular_acceleration_max": math.pi / 4.0, "agent_radius": 1.0,
                "max_steps": 600,
            },
            "baseline": {"k_omega": 1.5, "speed_fractions": [0.9, 0.6, 0.3]},
        },
        "scenarios": scenarios,
    }
    validate_dataset(dataset)
    return to_native(dataset)


def validate_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") == 3:
        from .round_one_dataset import validate
        validate(dataset)
        return
    scenarios = dataset.get("scenarios", [])
    if dataset.get("num_scenarios") != 30 or len(scenarios) != 30:
        raise ValueError("Fixed test dataset must contain exactly 30 scenarios")
    if dataset.get("schema_version", 1) >= 2 and dataset.get("dataset_split") not in {
        "validation", "test", "diagnostic", "unspecified"
    }:
        raise ValueError("Invalid or missing dataset_split")
    if [scenario["scenario_id"] for scenario in scenarios] != list(range(30)):
        raise ValueError("scenario_id must be ordered from 0 to 29")
    counts = Counter(scenario["scenario_type"] for scenario in scenarios)
    if dict(counts) != TYPE_QUOTAS:
        raise ValueError(f"Incorrect scenario quotas: {dict(counts)}")
    hashes = []
    for scenario in scenarios:
        validate_scenario_geometry(scenario)
        expected = scenario_hash(scenario)
        if scenario.get("scenario_hash") != expected:
            raise ValueError(f"Hash mismatch for scenario {scenario['scenario_id']}")
        hashes.append(expected)
        if scenario["scenario_type"] == "crossing_up" and scenario["dynamic_obstacle"]["velocity"][1] <= 0:
            raise ValueError("crossing_up has the wrong velocity direction")
        if scenario["scenario_type"] == "crossing_down" and scenario["dynamic_obstacle"]["velocity"][1] >= 0:
            raise ValueError("crossing_down has the wrong velocity direction")
        validation = scenario["validation"]
        if scenario["scenario_type"] == "low_dynamic_risk" and validation["baseline_min_dynamic_clearance"] <= 8.0:
            raise ValueError("Low-risk scenario violates its dynamic clearance threshold")
        if scenario["scenario_type"] == "high_risk" and not (
            validation["baseline_min_dynamic_clearance"] <= 1.0
            or validation["baseline_termination_reason"] == "dynamic_collision"
        ):
            raise ValueError("High-risk scenario violates its risk threshold")
        if scenario["scenario_type"] == "static_dominant" and not (
            validation["baseline_min_static_clearance"] <= 4.0
            and validation["baseline_min_dynamic_clearance"] > 6.0
        ):
            raise ValueError("Static-dominant scenario violates its risk thresholds")
        if scenario["scenario_type"] == "mixed_risk" and not (
            validation["baseline_min_static_clearance"] <= 4.0
            and validation["baseline_min_dynamic_clearance"] <= 6.0
        ):
            raise ValueError("Mixed-risk scenario violates its risk thresholds")
        if scenario["scenario_type"] in {"crossing_up", "crossing_down"}:
            if validation["baseline_min_dynamic_clearance"] > 8.0:
                raise ValueError("Crossing scenario has no meaningful dynamic encounter")
            if validation["baseline_termination_reason"] in {"static_collision_1", "static_collision_2"}:
                raise ValueError("Crossing baseline terminates on a static obstacle")
    for scenario_type in ("crossing_up", "crossing_down"):
        difficulties = Counter(
            scenario["difficulty"] for scenario in scenarios if scenario["scenario_type"] == scenario_type
        )
        if difficulties != Counter({"easy": 2, "medium": 2, "hard": 2}):
            raise ValueError(f"Incorrect difficulty quotas for {scenario_type}: {dict(difficulties)}")
    if len(set(hashes)) != 30:
        raise ValueError("Scenario hashes are not unique")


def dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for scenario_type in TYPE_QUOTAS:
        selected = [item for item in dataset["scenarios"] if item["scenario_type"] == scenario_type]
        result[scenario_type] = {
            "count": len(selected),
            "mean_baseline_min_dynamic_clearance": float(np.mean([
                item["validation"]["baseline_min_dynamic_clearance"] for item in selected
            ])),
            "baseline_collision_count": sum(
                "collision" in item["validation"]["baseline_termination_reason"] for item in selected
            ),
        }
    return result
