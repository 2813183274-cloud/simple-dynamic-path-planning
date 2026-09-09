"""Regression checks for the config-driven public workflow; no persistent outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np
from stable_baselines3.common.monitor import Monitor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.risk_reward import RiskRewardWrapper
from envs.scenario_dataset import load_dataset
from project_config import load_config, verify_locked_environment
from scripts.train import create_model
from utils.metrics import nominal_initial_cpa, write_nominal_risk_metrics


BENCHMARK_SHA256 = "75bca781a36d768b8ca25ed8dac0bd5e1e6ad63a56c2f45ca73d5786cfd2c757"


def main() -> None:
    config = load_config(ROOT / "configs/main.json")
    verify_locked_environment(config)
    benchmark = ROOT / config["evaluation"]["benchmark"]
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == BENCHMARK_SHA256
    data = load_dataset(benchmark)
    assert data["num_scenarios"] == 30
    first_cpa = nominal_initial_cpa(data["scenarios"][0], 3.0)
    assert first_cpa["tcpa_seconds"] >= 0.0 and first_cpa["dcpa_center_distance"] >= 0.0

    # The third-round base wrapper must preserve observations, rewards and termination.
    plain = DynamicPathPlanningEnv(stage="D")
    wrapped = RiskRewardWrapper(DynamicPathPlanningEnv(stage="D"), "base")
    try:
        left, _ = plain.reset(seed=991)
        right, _ = wrapped.reset(seed=991)
        np.testing.assert_array_equal(left, right)
        for action in ([1.0, 0.0], [0.3, -0.5], [-0.2, 0.7], [0.0, 0.0]) * 5:
            left = plain.step(np.asarray(action, dtype=np.float32))
            right = wrapped.step(np.asarray(action, dtype=np.float32))
            np.testing.assert_array_equal(left[0], right[0])
            assert left[1] == right[1] and left[2:4] == right[2:4]
            assert right[4]["risk_penalty"] == 0.0
            if left[2] or left[3]:
                break
    finally:
        plain.close(); wrapped.close()

    fixed = DynamicPathPlanningEnv(scenario_file=benchmark)
    try:
        for index in range(3):
            observation, info = fixed.reset(options={"scenario_index": index})
            assert observation.shape == (16,) and info["scenario_id"] == index
            next_observation, reward, _, _, _ = fixed.step(np.zeros(2, dtype=np.float32))
            assert next_observation.shape == (16,) and np.isfinite(reward)
    finally:
        fixed.close()

    # One SB3 rollout is intentionally temporary: initialization and learning must not fail.
    with tempfile.TemporaryDirectory() as temporary:
        assert write_nominal_risk_metrics(benchmark, Path(temporary), 3.0).exists()
        training_env = Monitor(DynamicPathPlanningEnv(stage="D"))
        try:
            model = create_model(training_env, 7, Path(temporary) / "tensorboard", "cpu")
            model.learn(total_timesteps=8, progress_bar=False)
            assert model.num_timesteps == 2048
        finally:
            training_env.close()
    print("Public workflow regression passed: config, benchmark, reward, fixed scenes, PPO rollout.")


if __name__ == "__main__":
    main()
