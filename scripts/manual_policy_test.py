from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None, help="Save instead of opening a window")
    args = parser.parse_args()
    env = DynamicPathPlanningEnv(render_mode="rgb_array")
    _, _ = env.reset(seed=args.seed)
    for _ in range(env.max_steps):
        delta = env.goal_position - env.agent_position
        error = normalize_angle(math.atan2(delta[1], delta[0]) - env.agent_heading)
        action = np.array([0.5, np.clip(2.0 * error / env.omega_max, -1.0, 1.0)], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.render()
    print(f"End: {info['termination_reason']}, steps={info['step_count']}, path={info['path_length']:.2f} m")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        env._figure.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    env.close()


if __name__ == "__main__":
    main()
