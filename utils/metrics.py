"""Metrics added by the public workflow without changing sealed evaluators."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.scenario_dataset import load_dataset
from experiment_utils import atomic_write_json


def nominal_initial_cpa(scene: dict, agent_speed: float) -> dict[str, float]:
    """Initial TCPA/DCPA for a straight USV course toward the goal at fixed speed."""
    start = np.asarray(scene["start_position"], dtype=float)
    goal = np.asarray(scene["goal_position"], dtype=float)
    obstacle = scene["dynamic_obstacle"]
    relative_position = np.asarray(obstacle["position"], dtype=float) - start
    course = goal - start
    course /= np.linalg.norm(course)
    relative_velocity = np.asarray(obstacle["velocity"], dtype=float) - agent_speed * course
    speed_squared = float(relative_velocity @ relative_velocity)
    tcpa = 0.0 if speed_squared <= 1e-12 else max(0.0, -float(relative_position @ relative_velocity) / speed_squared)
    dcpa = float(np.linalg.norm(relative_position + tcpa * relative_velocity))
    clearance = dcpa - 1.0 - float(obstacle["radius"])
    return {"tcpa_seconds": tcpa, "dcpa_center_distance": dcpa, "dcpa_clearance": clearance}


def write_nominal_risk_metrics(dataset_path: Path, output_dir: Path, agent_speed: float) -> Path:
    """Write dataset-level initial CPA diagnostics; these are not policy trajectory metrics."""
    dataset = load_dataset(dataset_path)
    records = [
        {"scenario_id": int(scene["scenario_id"]), **nominal_initial_cpa(scene, agent_speed)}
        for scene in dataset["scenarios"]
    ]
    value = {
        "definition": (
            "Initial nominal CPA: USV moves directly toward its goal at fixed configured maximum speed; "
            "dynamic obstacle retains its initial constant velocity. Not computed from the evaluated policy trajectory."
        ),
        "agent_speed": float(agent_speed),
        "average_tcpa_seconds": float(np.mean([row["tcpa_seconds"] for row in records])),
        "average_dcpa_center_distance": float(np.mean([row["dcpa_center_distance"] for row in records])),
        "average_dcpa_clearance": float(np.mean([row["dcpa_clearance"] for row in records])),
        "records": records,
    }
    path = output_dir / "nominal_initial_tcpa_dcpa.json"
    atomic_write_json(path, value)
    return path
