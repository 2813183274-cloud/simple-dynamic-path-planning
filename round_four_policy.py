"""P4 v2: finite executable actions, original latent Gaussian PPO objective.

collect_rollouts is a constrained adaptation of SB3 2.8.0's MIT-licensed
OnPolicyAlgorithm.collect_rollouts (DLR-RM/stable-baselines3). It preserves
timeout bootstrap/GAE/callback order; only the execution map and logs differ.
"""
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.save_util import load_from_zip_file

ARMS = ("clip", "speed_tanh")
SCHEMA = "finite-execution-box-latent-rollout-v2"


def map_action(z, arm):
    if arm not in ARMS:
        raise ValueError("Unknown P4 arm")
    z = np.asarray(z)
    if z.shape[-1:] != (2,) or not np.isfinite(z).all():
        raise ValueError("Expected finite latent two-vector(s)")
    a = np.clip(z, -1, 1).copy()
    if arm == "speed_tanh":
        a[..., 0] = np.tanh(z[..., 0])
    return a


def log_speed_jacobian(z):
    return 2 * (np.log(2.) - z - torch.nn.functional.softplus(-2*z))


def action_log(z, mean, std, a, arm):
    return {"latent_speed": float(z[0]), "latent_turn": float(z[1]),
            "latent_speed_mean": float(mean[0]), "latent_turn_mean": float(mean[1]),
            "latent_speed_std": float(std[0]), "latent_turn_std": float(std[1]),
            "executed_speed_action": float(a[0]), "executed_turn_action": float(a[1]),
            "target_v": float(1.5*(a[0]+1)), "target_w": float(np.pi/4*a[1]),
            "speed_latent_outside": bool(abs(z[0]) > 1),
            "speed_hard_clipped": bool(arm == "clip" and abs(z[0]) > 1),
            "speed_near_upper": bool(a[0] >= .99),
            "tanh_endpoint_rounding": bool(arm == "speed_tanh" and abs(a[0]) == 1)}


class LatentActionPPO(PPO):
    def __init__(self, *args, arm="clip", **kwargs):
        if arm not in ARMS:
            raise ValueError("Unknown P4 arm")
        self.arm = arm
        self.action_schema = SCHEMA
        super().__init__(*args, **kwargs)
        if kwargs.get("_init_setup_model", True):
            self._check_contract()

    def _check_contract(self):
        if (self.action_schema != SCHEMA or self.arm not in ARMS or self.use_sde
                or self.policy.squash_output or self.observation_space.shape != (16,)
                or not isinstance(self.action_space, spaces.Box)
                or self.action_space.shape != (2,)
                or not np.all(self.action_space.low == -1)
                or not np.all(self.action_space.high == 1)):
            raise ValueError("P4 action/policy interface mismatch")

    @classmethod
    def load(cls, path, *args, **kwargs):
        data, _, _ = load_from_zip_file(path, device="cpu")
        if not data or data.get("action_schema") != SCHEMA or data.get("arm") not in ARMS:
            raise ValueError("Not a P4 v2 latent-action model; legacy loading forbidden")
        for key in ("arm", "action_schema", "custom_objects"):
            if key in kwargs:
                raise ValueError("Cannot override saved P4 identity")
        model = super().load(path, *args, **kwargs)
        model._check_contract()
        return model

    def action_details(self, observation, deterministic=True):
        self._check_contract()
        self.policy.set_training_mode(False)
        tensor, vectorized = self.policy.obs_to_tensor(observation)
        with torch.no_grad():
            dist = self.policy.get_distribution(tensor)
            z = dist.get_actions(deterministic=deterministic).cpu().numpy()
            mean = dist.distribution.mean.cpu().numpy()
            std = dist.distribution.stddev.cpu().numpy()
        a = map_action(z, self.arm)
        if not vectorized:
            return a[0], action_log(z[0], mean[0], std[0], a[0], self.arm)
        return a, [action_log(x, m, s, y, self.arm) for x, m, s, y in zip(z, mean, std, a)]

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if state is not None:
            raise ValueError("P4 does not support recurrent policy state")
        return self.action_details(observation, deterministic)[0], None

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self._check_contract()
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        for _ in range(n_rollout_steps):
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                dist = self.policy.get_distribution(obs_tensor)
                means = dist.distribution.mean.cpu().numpy()
                stds = dist.distribution.stddev.cpu().numpy()
            actions = actions.cpu().numpy()
            clipped_actions = map_action(actions, self.arm)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            for i, info in enumerate(infos):
                info.update(action_log(actions[i], means[i], stds[i], clipped_actions[i], self.arm))
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            for idx, done in enumerate(dones):
                if (done and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
