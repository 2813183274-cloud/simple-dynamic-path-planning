"""Configuration and command construction for the stable public entry points."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "main.json"


def resolve_path(value: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = resolve_path(str(path), must_exist=True)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported project config schema_version")
    for section in ("environment", "reward", "ppo", "training", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    return config


def verify_locked_environment(config: dict[str, Any]) -> None:
    """Fail if the public config drifts from the sealed implementation."""
    from envs import DynamicPathPlanningEnv

    expected = config["environment"]
    reward = config["reward"]
    env = DynamicPathPlanningEnv(stage=expected["stage"])
    try:
        actual = {
            "observation_dimensions": int(env.observation_space.shape[0]),
            "action_dimensions": int(env.action_space.shape[0]),
            "map_size": [float(env.map_width), float(env.map_height)],
            "dt": float(env.dt),
            "max_steps": int(env.max_steps),
            "agent_radius": float(env.agent_radius),
            "static_obstacle_radius": float(env.static_radii[0]),
            "dynamic_obstacle_radius": float(env.dynamic_radius),
            "goal_threshold": float(env.goal_threshold),
            "linear_velocity_max": float(env.v_max),
            "angular_velocity_max": float(env.omega_max),
            "linear_acceleration_max": float(env.linear_acceleration_max),
            "angular_acceleration_max": float(env.angular_acceleration_max),
        }
        reward_actual = {
            "progress_weight": 10.0,
            "time_penalty": -0.05,
            "goal_reward": 200.0,
            "static_collision_penalty": -120.0,
            "dynamic_collision_penalty": -150.0,
            "out_of_bounds_penalty": -100.0,
            "static_safe_center_distance": float(env.static_safe_distance),
            "dynamic_safe_center_distance": float(env.dynamic_safe_distance),
        }
    finally:
        env.close()
    if actual != {k: expected[k] for k in actual}:
        raise ValueError(f"Environment config differs from implementation: {actual}")
    if reward_actual != {k: reward[k] for k in reward_actual}:
        raise ValueError(f"Reward config differs from implementation: {reward_actual}")
    from scripts.train import PPO_CONFIG
    ppo_expected = config["ppo"]
    if PPO_CONFIG != {key: ppo_expected[key] for key in PPO_CONFIG}:
        raise ValueError(f"PPO config differs from implementation: {PPO_CONFIG}")


def run_script(script: str, arguments: list[str], *, dry_run: bool = False) -> int:
    command = [sys.executable, "-B", str(ROOT / "scripts" / script), *arguments]
    if dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    return subprocess.call(command, cwd=ROOT)
