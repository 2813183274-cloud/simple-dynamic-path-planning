from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model_dir, log_dir = ROOT / "models", ROOT / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    env = Monitor(DynamicPathPlanningEnv(), str(log_dir / "train_monitor.csv"))
    eval_env = Monitor(DynamicPathPlanningEnv(), str(log_dir / "eval_monitor.csv"))
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=str(log_dir), seed=args.seed,
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(model_dir / "best"), log_path=str(log_dir / "eval"),
        eval_freq=10_000, n_eval_episodes=10, deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=str(model_dir / "checkpoints"), name_prefix="ppo_dynamic_path"
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback],
                    progress_bar=True)
        model.save(model_dir / "final_model")
        print(f"Final model saved to {model_dir / 'final_model.zip'}")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
