"""Stable, config-driven public training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from project_config import DEFAULT_CONFIG, load_config, run_script, verify_locked_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the stage-D PPO policy from a project config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True, help="Unique output directory name under runs/")
    parser.add_argument("--timesteps", type=int, help="Explicitly override the config training budget")
    parser.add_argument("--seed", type=int, help="Explicitly override the config seed")
    parser.add_argument("--device", help="Explicitly override the config device")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the delegated command")
    args = parser.parse_args()
    config = load_config(args.config)
    verify_locked_environment(config)
    training = config["training"]
    delegated = [
        "--stage", config["environment"]["stage"],
        "--timesteps", str(args.timesteps if args.timesteps is not None else training["timesteps"]),
        "--seed", str(args.seed if args.seed is not None else training["seed"]),
        "--eval-freq", str(training["eval_freq"]),
        "--reward-smoothing-window", str(training["reward_smoothing_window"]),
        "--device", args.device or training["device"],
        "--runs-dir", str(training["runs_dir"]),
        "--run-id", args.run_id,
    ]
    return run_script("train.py", delegated, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
