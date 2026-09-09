from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402


def run_until_done(env, minimum_steps=200):
    obs, _ = env.reset(seed=123)
    total_steps = 0
    while total_steps < minimum_steps:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == (16,) and env.observation_space.contains(obs)
        assert np.isfinite(reward)
        total_steps += 1
        if terminated or truncated:
            obs, _ = env.reset()


def check_random_scenarios(env, samples=50):
    for seed in range(samples):
        env.reset(seed=seed)
        sx, sy = env.start_position
        gx, gy = env.goal_position
        assert 5.0 <= sx <= 15.0 and 35.0 <= sy <= 65.0
        assert 85.0 <= gx <= 95.0 and 35.0 <= gy <= 65.0
        assert np.linalg.norm(env.goal_position - env.start_position) >= 65.0
        assert 28.0 <= env.static_obstacles[0, 0] <= 45.0
        assert 58.0 <= env.static_obstacles[1, 0] <= 75.0
        assert np.all((3.5 <= env.static_radii) & (env.static_radii <= 5.0))
        assert 43.0 <= env.dynamic_position[0] <= 57.0
        assert 2.5 <= env.dynamic_radius <= 4.0
        assert env.dynamic_velocity[0] == 0.0
        assert 0.8 <= abs(env.dynamic_velocity[1]) <= 1.3
        blocks_direct_path = any(
            env._point_to_segment_distance(position, env.start_position, env.goal_position)
            <= radius + env.agent_radius
            for position, radius in zip(env.static_obstacles, env.static_radii)
        )
        assert blocks_direct_path


def main():
    env = DynamicPathPlanningEnv()
    check_env(env, warn=True)
    check_random_scenarios(env)
    obs, _ = env.reset(seed=7)
    assert obs.dtype == np.float32 and obs.shape == (16,)
    assert env.observation_space.contains(obs)
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    run_until_done(env, 200)

    env.reset()
    previous_v = env.current_linear_velocity
    previous_omega = env.current_angular_velocity
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert abs(env.current_linear_velocity - previous_v) <= env.linear_acceleration_max * env.dt + 1e-9
    assert abs(env.current_angular_velocity - previous_omega) <= env.angular_acceleration_max * env.dt + 1e-9
    previous_v = env.current_linear_velocity
    previous_omega = env.current_angular_velocity
    env.step(np.array([-1.0, -1.0], dtype=np.float32))
    assert abs(env.current_linear_velocity - previous_v) <= env.linear_acceleration_max * env.dt + 1e-9
    assert abs(env.current_angular_velocity - previous_omega) <= env.angular_acceleration_max * env.dt + 1e-9

    env.reset()
    env.agent_position = env.static_obstacles[0].copy()
    assert env._check_static_collision() == 0
    env.agent_position = env.dynamic_position.copy()
    assert env._check_dynamic_collision()

    env.reset()
    env.agent_position = env.goal_position - np.array([0.1, 0.0])
    _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert terminated and not truncated and info["is_success"]

    timeout_env = DynamicPathPlanningEnv(max_steps=1)
    timeout_env.reset()
    _, _, terminated, truncated, info = timeout_env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert not terminated and truncated and info["termination_reason"] == "timeout"

    env.reset(seed=9)
    goal_distance = env.previous_goal_distance
    static_distances = np.array([3.5, 7.0])
    dynamic_distance = 4.0
    reward, components = env._calculate_reward(
        goal_distance, static_distances, dynamic_distance, terminal_reward=0.0
    )
    assert np.isclose(components["static"], -0.5)
    assert np.isclose(components["dynamic"], -1.0)
    assert np.isclose(reward, -1.55)
    env.close()
    timeout_env.close()
    print("All environment checks passed.")


if __name__ == "__main__":
    main()
