"""Reproduce the P4 v1 dependency gate without training or model mutation."""
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv


def check():
    env = gym.Wrapper(DynamicPathPlanningEnv(stage="D"))
    env.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    try:
        PPO("MlpPolicy", env, device="cpu", seed=1)
    except AssertionError as error:
        expected = "Continuous action space must have a finite lower and upper bound"
        if str(error) != expected:
            raise
        return {"protocol": "round4-speed-tanh-v1", "sb3": stable_baselines3.__version__,
                "status": "blocked_by_dependency_action_space_contract",
                "exception": str(error), "optimizer_updates": 0,
                "training_started": False}
    else:
        raise RuntimeError("Dependency behavior changed: re-audit v1 before implementation")
    finally:
        env.close()


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
