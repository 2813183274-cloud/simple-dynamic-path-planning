from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402
from envs.scenario_dataset import TYPE_QUOTAS, load_dataset, scenario_hash, static_layout_passable  # noqa: E402


def snapshot(env):
    return {
        "start": env.start_position.copy(), "heading": env.start_heading,
        "goal": env.goal_position.copy(), "static": env.static_obstacles.copy(),
        "static_radii": env.static_radii.copy(), "dynamic": env.dynamic_position.copy(),
        "dynamic_velocity": env.dynamic_velocity.copy(), "dynamic_radius": env.dynamic_radius,
    }


def assert_same(left, right):
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", type=Path,
                        default=ROOT / "configs" / "validation_scenarios_30.json")
    args = parser.parse_args()
    raw = json.loads(args.scenario_file.read_text(encoding="utf-8"))
    dataset = load_dataset(args.scenario_file)
    assert raw == dataset
    scenarios = dataset["scenarios"]
    assert len(scenarios) == 30
    assert [item["scenario_id"] for item in scenarios] == list(range(30))
    assert Counter(item["scenario_type"] for item in scenarios) == Counter(TYPE_QUOTAS)
    assert len({item["scenario_hash"] for item in scenarios}) == 30
    assert all(item["scenario_hash"] == scenario_hash(item) for item in scenarios)
    assert all(static_layout_passable(item) for item in scenarios)
    assert all(item["dynamic_obstacle"]["velocity"][1] > 0
               for item in scenarios if item["scenario_type"] == "crossing_up")
    assert all(item["dynamic_obstacle"]["velocity"][1] < 0
               for item in scenarios if item["scenario_type"] == "crossing_down")
    assert all(item["validation"]["baseline_min_dynamic_clearance"] > 8.0
               for item in scenarios if item["scenario_type"] == "low_dynamic_risk")
    assert all(item["validation"]["baseline_min_dynamic_clearance"] <= 1.0
               or item["validation"]["baseline_termination_reason"] == "dynamic_collision"
               for item in scenarios if item["scenario_type"] == "high_risk")

    env = DynamicPathPlanningEnv(randomize_scenario=False, scenario_file=args.scenario_file)
    for index in range(30):
        observation_a, info_a = env.reset(options={"scenario_index": index})
        first = snapshot(env)
        observation_b, info_b = env.reset(options={"scenario_index": index})
        second = snapshot(env)
        np.testing.assert_array_equal(observation_a, observation_b)
        assert_same(first, second)
        assert info_a["scenario_id"] == info_b["scenario_id"] == index
        assert info_a["scenario_mode"] == info_b["scenario_mode"] == "fixed_test"
        assert env.observation_space.contains(observation_a)
        assert np.all(np.isfinite(observation_a))
    env.close()
    print("All fixed-scenario dataset checks passed.")


if __name__ == "__main__":
    main()
