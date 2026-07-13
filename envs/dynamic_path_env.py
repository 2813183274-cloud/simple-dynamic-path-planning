from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


class DynamicPathPlanningEnv(gym.Env):
    """Continuous 2-D path-planning environment with circular obstacles."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 600,
        linear_acceleration_max: float = 1.5,
        angular_acceleration_max: float = math.pi / 4.0,
        randomize_scenario: bool = True,
        scenario_file: str | Path | None = None,
    ):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.randomize_scenario = bool(randomize_scenario)
        self.scenario_file = Path(scenario_file).resolve() if scenario_file is not None else None
        self.fixed_scenarios: list[dict[str, Any]] | None = None
        self.current_scenario: dict[str, Any] | None = None
        self.scenario_mode = "random_train" if self.randomize_scenario else "default_fixed"
        if self.scenario_file is not None:
            from .scenario_dataset import load_dataset
            self.fixed_scenarios = load_dataset(self.scenario_file)["scenarios"]
            self.randomize_scenario = False
            self.scenario_mode = "fixed_test"
        self.map_width = 100.0
        self.map_height = 100.0
        self.map_diagonal = math.hypot(self.map_width, self.map_height)
        self.dt = 0.2
        self.v_max = 3.0
        self.omega_max = math.pi / 4.0
        if linear_acceleration_max <= 0.0 or angular_acceleration_max <= 0.0:
            raise ValueError("Acceleration limits must be positive")
        self.linear_acceleration_max = float(linear_acceleration_max)
        self.angular_acceleration_max = float(angular_acceleration_max)
        self.agent_radius = 1.0
        self.start_position = np.array([10.0, 50.0], dtype=np.float64)
        self.start_heading = 0.0
        self.goal_position = np.array([90.0, 50.0], dtype=np.float64)
        self.goal_threshold = 3.0
        self.static_obstacles = np.array([[35.0, 48.0], [68.0, 52.0]], dtype=np.float64)
        self.static_radii = np.array([4.0, 4.0], dtype=np.float64)
        self.dynamic_initial_position = np.array([50.0, 20.0], dtype=np.float64)
        self.dynamic_initial_velocity = np.array([0.0, 1.0], dtype=np.float64)
        self.dynamic_radius = 3.0
        self.dynamic_obstacle_max_speed = float(np.linalg.norm(self.dynamic_initial_velocity))
        self.static_safe_distance = 7.0
        self.dynamic_safe_distance = 8.0
        self.max_steps = int(max_steps)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)

        self.agent_position = self.start_position.copy()
        self.agent_heading = self.start_heading
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.dynamic_position = self.dynamic_initial_position.copy()
        self.dynamic_velocity = self.dynamic_initial_velocity.copy()
        self.previous_goal_distance = self._distance(self.agent_position, self.goal_position)
        self.step_count = 0
        self.path_length = 0.0
        self.min_static_distance = math.inf
        self.min_dynamic_distance = math.inf
        self.agent_trajectory: list[np.ndarray] = []
        self.dynamic_trajectory: list[np.ndarray] = []
        self._figure = None
        self._axes = None

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.fixed_scenarios is not None:
            scenario_index = int((options or {}).get("scenario_index", 0))
            if not 0 <= scenario_index < len(self.fixed_scenarios):
                raise IndexError(f"scenario_index must be in [0, {len(self.fixed_scenarios) - 1}]")
            self.set_scenario(self.fixed_scenarios[scenario_index], scenario_mode="fixed_test")
        elif self.randomize_scenario:
            self._sample_scenario()
        elif self.current_scenario is None:
            self._set_default_scenario()
        self.agent_position = self.start_position.copy()
        self.agent_heading = self.start_heading
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.dynamic_position = self.dynamic_initial_position.copy()
        self.dynamic_velocity = self.dynamic_initial_velocity.copy()
        self.previous_goal_distance = self._distance(self.agent_position, self.goal_position)
        self.step_count = 0
        self.path_length = 0.0
        static_distances = np.linalg.norm(self.static_obstacles - self.agent_position, axis=1)
        self.min_static_distance = float(np.min(static_distances))
        self.min_dynamic_distance = self._distance(self.agent_position, self.dynamic_position)
        self.agent_trajectory = [self.agent_position.copy()]
        self.dynamic_trajectory = [self.dynamic_position.copy()]
        observation = self._get_observation()
        info = self._build_info("running", False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._append_scenario_info(info)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _set_default_scenario(self) -> None:
        self.start_position = np.array([10.0, 50.0], dtype=np.float64)
        self.goal_position = np.array([90.0, 50.0], dtype=np.float64)
        self.start_heading = 0.0
        self.static_obstacles = np.array([[35.0, 48.0], [68.0, 52.0]], dtype=np.float64)
        self.static_radii = np.array([4.0, 4.0], dtype=np.float64)
        self.dynamic_initial_position = np.array([50.0, 20.0], dtype=np.float64)
        self.dynamic_initial_velocity = np.array([0.0, 1.0], dtype=np.float64)
        self.dynamic_radius = 3.0
        self.dynamic_obstacle_max_speed = 1.0
        self.current_scenario = {
            "scenario_id": -1, "scenario_type": "default", "difficulty": "unclassified"
        }
        self.scenario_mode = "default_fixed"

    def set_scenario(self, scenario: dict[str, Any], scenario_mode: str = "fixed_test") -> None:
        """Load exact scenario parameters without adding random perturbations."""
        self.start_position = np.asarray(scenario["start_position"], dtype=np.float64).copy()
        self.start_heading = self._normalize_angle(float(scenario["start_heading"]))
        self.goal_position = np.asarray(scenario["goal_position"], dtype=np.float64).copy()
        self.static_obstacles = np.asarray(
            [item["position"] for item in scenario["static_obstacles"]], dtype=np.float64
        )
        self.static_radii = np.asarray(
            [item["radius"] for item in scenario["static_obstacles"]], dtype=np.float64
        )
        dynamic = scenario["dynamic_obstacle"]
        self.dynamic_initial_position = np.asarray(dynamic["position"], dtype=np.float64).copy()
        self.dynamic_initial_velocity = np.asarray(dynamic["velocity"], dtype=np.float64).copy()
        self.dynamic_radius = float(dynamic["radius"])
        self.dynamic_obstacle_max_speed = float(np.linalg.norm(self.dynamic_initial_velocity))
        self.current_scenario = scenario
        self.scenario_mode = scenario_mode

    @staticmethod
    def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0.0:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
        projection = start + fraction * segment
        return float(np.linalg.norm(point - projection))

    def _sample_scenario(self) -> None:
        """Sample a valid encounter; all randomness comes from Gymnasium's seeded RNG."""
        clearance = 2.0
        for _ in range(2_000):
            start = self.np_random.uniform([5.0, 35.0], [15.0, 65.0]).astype(np.float64)
            goal = self.np_random.uniform([85.0, 35.0], [95.0, 65.0]).astype(np.float64)
            if self._distance(start, goal) < 65.0:
                continue
            goal_heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
            start_heading = self._normalize_angle(goal_heading + self.np_random.uniform(-0.25, 0.25))

            static_positions = np.array([
                self.np_random.uniform([28.0, 28.0], [45.0, 72.0]),
                self.np_random.uniform([58.0, 28.0], [75.0, 72.0]),
            ], dtype=np.float64)
            static_radii = self.np_random.uniform(3.5, 5.0, size=2).astype(np.float64)
            if any(self._distance(position, start) <= radius + self.agent_radius + clearance
                   or self._distance(position, goal) <= radius + self.goal_threshold + clearance
                   for position, radius in zip(static_positions, static_radii)):
                continue
            if self._distance(static_positions[0], static_positions[1]) <= sum(static_radii) + clearance:
                continue
            # A straight center-line path must collide with at least one inflated static disk.
            if not any(self._point_to_segment_distance(position, start, goal)
                       <= radius + self.agent_radius
                       for position, radius in zip(static_positions, static_radii)):
                continue

            dynamic_x = float(self.np_random.uniform(43.0, 57.0))
            dynamic_radius = float(self.np_random.uniform(2.5, 4.0))
            if bool(self.np_random.integers(0, 2)):
                dynamic_y = float(self.np_random.uniform(25.0, 40.0))
                dynamic_vy = float(self.np_random.uniform(0.8, 1.3))
            else:
                dynamic_y = float(self.np_random.uniform(60.0, 75.0))
                dynamic_vy = float(self.np_random.uniform(-1.3, -0.8))
            dynamic_position = np.array([dynamic_x, dynamic_y], dtype=np.float64)
            if any(self._distance(dynamic_position, position)
                   <= dynamic_radius + radius + clearance
                   for position, radius in zip(static_positions, static_radii)):
                continue

            # Estimate the crossing with a nominal 2 m/s USV. Requiring entry into the
            # 8 m safety corridor makes a meaningful encounter likely without forcing collision.
            nominal_time = (dynamic_x - start[0]) / 2.0
            predicted_dynamic_y = dynamic_y + dynamic_vy * nominal_time
            path_fraction = (dynamic_x - start[0]) / (goal[0] - start[0])
            path_y = start[1] + path_fraction * (goal[1] - start[1])
            if abs(predicted_dynamic_y - path_y) > self.dynamic_safe_distance:
                continue

            self.start_position = start
            self.goal_position = goal
            self.start_heading = start_heading
            self.static_obstacles = static_positions
            self.static_radii = static_radii
            self.dynamic_initial_position = dynamic_position
            self.dynamic_initial_velocity = np.array([0.0, dynamic_vy], dtype=np.float64)
            self.dynamic_radius = dynamic_radius
            self.dynamic_obstacle_max_speed = abs(dynamic_vy)
            self.current_scenario = {
                "scenario_id": -1, "scenario_type": "random_train", "difficulty": "unclassified"
            }
            self.scenario_mode = "random_train"
            return
        raise RuntimeError("Unable to sample a valid randomized scenario after 2000 attempts")

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_linear_velocity = float((action[0] + 1.0) * 0.5 * self.v_max)
        target_angular_velocity = float(action[1] * self.omega_max)
        max_linear_change = self.linear_acceleration_max * self.dt
        max_angular_change = self.angular_acceleration_max * self.dt
        self.current_linear_velocity = float(np.clip(
            target_linear_velocity,
            self.current_linear_velocity - max_linear_change,
            self.current_linear_velocity + max_linear_change,
        ))
        self.current_angular_velocity = float(np.clip(
            target_angular_velocity,
            self.current_angular_velocity - max_angular_change,
            self.current_angular_velocity + max_angular_change,
        ))

        old_position = self.agent_position.copy()
        # Apply the supplied equations: translate using theta(t), then update theta.
        self.agent_position += self.current_linear_velocity * self.dt * np.array(
            [math.cos(self.agent_heading), math.sin(self.agent_heading)]
        )
        self.agent_heading = self._normalize_angle(
            self.agent_heading + self.current_angular_velocity * self.dt
        )
        self._update_dynamic_obstacle()
        self.step_count += 1
        self.path_length += self._distance(old_position, self.agent_position)
        self.agent_trajectory.append(self.agent_position.copy())
        self.dynamic_trajectory.append(self.dynamic_position.copy())

        static_distances = np.linalg.norm(self.static_obstacles - self.agent_position, axis=1)
        dynamic_distance = self._distance(self.agent_position, self.dynamic_position)
        self.min_static_distance = min(self.min_static_distance, float(np.min(static_distances)))
        self.min_dynamic_distance = min(self.min_dynamic_distance, dynamic_distance)

        goal_distance = self._distance(self.agent_position, self.goal_position)
        success = goal_distance <= self.goal_threshold
        static_collision_index = self._check_static_collision()
        dynamic_collision = self._check_dynamic_collision()
        out_of_bounds = self._check_out_of_bounds()

        if success:
            reason, terminal_reward = "success", 200.0
        elif static_collision_index == 0:
            reason, terminal_reward = "static_collision_1", -120.0
        elif static_collision_index == 1:
            reason, terminal_reward = "static_collision_2", -120.0
        elif dynamic_collision:
            reason, terminal_reward = "dynamic_collision", -150.0
        elif out_of_bounds:
            reason, terminal_reward = "out_of_bounds", -100.0
        else:
            reason, terminal_reward = "running", 0.0

        terminated = reason != "running"
        truncated = bool(not terminated and self.step_count >= self.max_steps)
        if truncated:
            reason = "timeout"
        reward, components = self._calculate_reward(goal_distance, static_distances, dynamic_distance, terminal_reward)
        self.previous_goal_distance = goal_distance
        info = self._build_info(
            reason, success, static_collision_index is not None, dynamic_collision,
            components["progress"], components["static"], components["dynamic"],
            components["time"], terminal_reward,
        )
        self._append_scenario_info(info)
        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        def object_features(position: np.ndarray) -> list[float]:
            delta = position - self.agent_position
            distance = float(np.linalg.norm(delta)) / self.map_diagonal
            relative_angle = self._normalize_angle(math.atan2(delta[1], delta[0]) - self.agent_heading)
            return [distance, math.sin(relative_angle), math.cos(relative_angle)]

        agent_velocity = self.current_linear_velocity * np.array(
            [math.cos(self.agent_heading), math.sin(self.agent_heading)]
        )
        relative_velocity = self.dynamic_velocity - agent_velocity
        velocity_scale = self.v_max + self.dynamic_obstacle_max_speed
        values = (
            object_features(self.goal_position)
            + object_features(self.static_obstacles[0])
            + object_features(self.static_obstacles[1])
            + object_features(self.dynamic_position)
            + [float(relative_velocity[0] / velocity_scale), float(relative_velocity[1] / velocity_scale)]
            + [self.current_linear_velocity / self.v_max, self.current_angular_velocity / self.omega_max]
        )
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0).astype(np.float32)

    def _update_dynamic_obstacle(self) -> None:
        self.dynamic_position += self.dynamic_velocity * self.dt
        lower, upper = self.dynamic_radius, self.map_height - self.dynamic_radius
        # Reflect any overshoot, preserving continuous motion instead of resetting.
        if self.dynamic_position[1] > upper:
            self.dynamic_position[1] = 2.0 * upper - self.dynamic_position[1]
            self.dynamic_velocity[1] = -abs(self.dynamic_velocity[1])
        elif self.dynamic_position[1] < lower:
            self.dynamic_position[1] = 2.0 * lower - self.dynamic_position[1]
            self.dynamic_velocity[1] = abs(self.dynamic_velocity[1])

    def _calculate_reward(self, goal_distance, static_distances, dynamic_distance, terminal_reward):
        progress = 10.0 * (self.previous_goal_distance - goal_distance)
        static_penalty = float(np.sum(np.where(
            static_distances < self.static_safe_distance,
            -(self.static_safe_distance - static_distances) / self.static_safe_distance,
            0.0,
        )))
        dynamic_penalty = (
            -2.0 * (self.dynamic_safe_distance - dynamic_distance) / self.dynamic_safe_distance
            if dynamic_distance < self.dynamic_safe_distance else 0.0
        )
        time_penalty = -0.05
        reward = float(progress + static_penalty + dynamic_penalty + time_penalty + terminal_reward)
        if not math.isfinite(reward):
            raise FloatingPointError("Reward became NaN or infinite")
        return reward, {"progress": float(progress), "static": static_penalty,
                        "dynamic": float(dynamic_penalty), "time": time_penalty}

    def _check_static_collision(self) -> int | None:
        distances = np.linalg.norm(self.static_obstacles - self.agent_position, axis=1)
        hits = np.flatnonzero(distances <= self.agent_radius + self.static_radii)
        return int(hits[0]) if hits.size else None

    def _check_dynamic_collision(self) -> bool:
        return self._distance(self.agent_position, self.dynamic_position) <= self.agent_radius + self.dynamic_radius

    def _check_out_of_bounds(self) -> bool:
        x, y = self.agent_position
        return bool(x - self.agent_radius < 0.0 or x + self.agent_radius > self.map_width
                    or y - self.agent_radius < 0.0 or y + self.agent_radius > self.map_height)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _build_info(self, reason, success, static_collision, dynamic_collision,
                    reward_progress, reward_static, reward_dynamic, reward_time, reward_terminal):
        distances = np.linalg.norm(self.static_obstacles - self.agent_position, axis=1)
        return {
            "is_success": bool(success),
            "collision": bool(static_collision or dynamic_collision),
            "collision_type": reason if "collision" in reason else "none",
            "static_collision": bool(static_collision),
            "dynamic_collision": bool(dynamic_collision),
            "out_of_bounds": reason == "out_of_bounds",
            "timeout": reason == "timeout",
            "termination_reason": reason,
            "goal_distance": self._distance(self.agent_position, self.goal_position),
            "static_obstacle_1_distance": float(distances[0]),
            "static_obstacle_2_distance": float(distances[1]),
            "dynamic_obstacle_distance": self._distance(self.agent_position, self.dynamic_position),
            "min_static_distance": float(self.min_static_distance),
            "min_dynamic_distance": float(self.min_dynamic_distance),
            "path_length": float(self.path_length),
            "step_count": int(self.step_count),
            "reward_progress": float(reward_progress),
            "reward_static_safety": float(reward_static),
            "reward_dynamic_safety": float(reward_dynamic),
            "reward_time": float(reward_time),
            "reward_terminal": float(reward_terminal),
        }

    def _append_scenario_info(self, info: dict[str, Any]) -> None:
        scenario = self.current_scenario or {}
        info.update({
            "scenario_id": int(scenario.get("scenario_id", -1)),
            "scenario_type": str(scenario.get("scenario_type", "unknown")),
            "difficulty": str(scenario.get("difficulty", "unclassified")),
            "scenario_mode": self.scenario_mode,
        })

    def render(self):
        if self.render_mode is None:
            return None
        if self._figure is None:
            self._figure, self._axes = plt.subplots(figsize=(7, 7))
        ax = self._axes
        ax.clear()
        ax.set(xlim=(0, self.map_width), ylim=(0, self.map_height), aspect="equal",
               xlabel="x (m)", ylabel="y (m)", title="Dynamic path planning")
        ax.add_patch(Circle(self.goal_position, self.goal_threshold, color="green", alpha=0.2))
        ax.plot(*self.start_position, "bo", label="Start")
        ax.plot(*self.goal_position, "g*", markersize=12, label="Goal")
        for index, (position, radius) in enumerate(zip(self.static_obstacles, self.static_radii), 1):
            ax.add_patch(Circle(position, radius, color="gray", alpha=0.7, label=f"Static {index}"))
        ax.add_patch(Circle(self.dynamic_position, self.dynamic_radius, color="orange", alpha=0.8,
                            label="Dynamic"))
        if self.agent_trajectory:
            trajectory = np.asarray(self.agent_trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=1, label="Agent path")
        if self.dynamic_trajectory:
            trajectory = np.asarray(self.dynamic_trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="orange", linestyle="--", linewidth=1)
        ax.add_patch(Circle(self.agent_position, self.agent_radius, color="blue"))
        heading_end = self.agent_position + 4.0 * np.array([math.cos(self.agent_heading), math.sin(self.agent_heading)])
        ax.plot([self.agent_position[0], heading_end[0]], [self.agent_position[1], heading_end[1]], "k-")
        ax.legend(loc="upper left", fontsize=8)
        self._figure.canvas.draw()
        if self.render_mode == "human":
            plt.pause(0.001)
            return None
        rgba = np.asarray(self._figure.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._axes = None
