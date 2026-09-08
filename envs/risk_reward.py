"""Round-three additive reward wrapper; the legacy environment is never modified."""
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

ARMS = {"base": (0.0, 0.0), "instant": (0.0, 2.0), "predictive": (4.0, 2.0)}
REWARD_VERSION = "round3-predictive-reward-v1"


@dataclass(frozen=True)
class RiskResult:
    risk: float
    clearances: tuple[float, ...]
    times: tuple[float, ...]


def risk_metrics(observation, horizon: float) -> RiskResult:
    """Finite-horizon closest approach using only the shared, clipped 16-vector.

    Positions/velocities are in the current body frame. Static target velocity is
    zero; own ship holds actual forward speed and heading. No future state reads.
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (16,) or not np.isfinite(obs).all():
        raise ValueError("Expected a finite 16-D observation")
    if np.any(np.abs(obs) > 1.0) or np.any(obs[[0, 3, 6, 9, 14]] < 0):
        raise ValueError("Observation is outside the D-stage normalized bounds")
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError("Horizon must be finite and nonnegative")
    r = np.array([math.sqrt(20000) * obs[i] * np.array([obs[i+2], obs[i+1]])
                  for i in (3, 6, 9)])
    v = obs[14] * 3.0
    u = np.array([[-v, 0.0], [-v, 0.0], obs[12:14] * 4.8])
    speed2 = np.einsum("ij,ij->i", u, u)
    tau = np.zeros(3)
    moving = speed2 > 1e-12
    tau[moving] = -np.einsum("ij,ij->i", r[moving], u[moving]) / speed2[moving]
    tau = np.clip(tau, 0.0, horizon)
    clearances = np.linalg.norm(r + tau[:, None] * u, axis=1) - np.array([5., 5., 4.])
    risk = float(np.max(np.clip((2.0 - clearances) / 2.0, 0.0, 1.0)))
    return RiskResult(risk, tuple(map(float, clearances)), tuple(map(float, tau)))


class RiskRewardWrapper(gym.Wrapper):
    """Add bounded risk cost at post-step observation; terminal/truncated cost is zero."""

    def __init__(self, env, arm: str):
        if arm not in ARMS:
            raise ValueError(f"Unknown round-three arm: {arm}")
        if env.unwrapped.stage != "D" or env.observation_space.shape != (16,):
            raise ValueError("Risk wrapper requires the D-stage 16-D observation")
        super().__init__(env)
        self.arm = arm
        self.horizon, self.weight = ARMS[arm]
        self.returns = np.zeros(3)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        if (base.stage != "D" or getattr(base, "observation_stage", "D") not in ("B", "C", "D")
                or base.agent_radius != 1.0 or base.dynamic_radius != 3.0
                or not np.array_equal(base.static_radii, [4.0, 4.0])):
            raise ValueError("Risk geometry/observation contract changed")
        self.returns = np.zeros(3)
        return observation, info

    def step(self, action):
        base = self.env.unwrapped
        previous_v = base.current_linear_velocity
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        instant = risk_metrics(observation, 0.0)
        predicted = risk_metrics(observation, 4.0)
        selected = predicted if self.horizon == 4.0 else instant
        penalty = 0.0 if terminated or truncated else -self.weight * selected.risk
        reward = float(base_reward + penalty)
        self.returns += [base_reward, penalty, reward]
        info = dict(info)
        info.update(base_reward=float(base_reward), risk_penalty=float(penalty), training_reward=reward,
                    risk_instant=instant.risk, risk_predictive=predicted.risk,
                    actual_v=base.current_linear_velocity, actual_w=base.current_angular_velocity,
                    delta_v=base.current_linear_velocity-previous_v,
                    speed_action=float(np.clip(action[0], -1, 1)), turn_action=float(np.clip(action[1], -1, 1)),
                    agent_x=float(base.agent_position[0]), agent_y=float(base.agent_position[1]),
                    agent_heading=float(base.agent_heading),
                    target_x=float(base.dynamic_position[0]), target_y=float(base.dynamic_position[1]))
        for i in range(3):
            info[f"risk_current_clearance_{i}"] = instant.clearances[i]
            info[f"risk_predicted_clearance_{i}"] = predicted.clearances[i]
            info[f"risk_tcpa_{i}"] = predicted.times[i]
        # Episode totals are emitted at every step and captured at termination by Monitor.
        for key, value in zip(("episode_base_return", "episode_risk_return", "episode_training_return"), self.returns):
            info[key] = float(value)
        return observation, reward, terminated, truncated, info
