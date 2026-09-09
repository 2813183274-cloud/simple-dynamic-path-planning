"""Stable, config-driven public fixed-benchmark evaluation entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from project_config import DEFAULT_CONFIG, load_config, resolve_path, run_script, verify_locked_environment


def evaluation_arguments(args, config: dict, *, plots: bool | None = None) -> list[str]:
    evaluation = config["evaluation"]
    delegated = [
        "--scenario-file", str(resolve_path(args.scenario_file or evaluation["benchmark"], must_exist=True)),
        "--output-dir", str(resolve_path(args.output_dir or evaluation["output_dir"])),
        "--device", args.device or evaluation["device"],
    ]
    if args.run_dir:
        delegated += ["--run-dir", str(resolve_path(args.run_dir, must_exist=True))]
    else:
        delegated += ["--model", str(resolve_path(args.model or evaluation["default_model"], must_exist=True))]
    render = evaluation["plots"] if plots is None else plots
    if args.no_plots or not render:
        delegated.append("--no-plots")
    if args.overwrite:
        delegated.append("--overwrite")
    if evaluation.get("allow_historical_benchmark", False):
        delegated.append("--allow-non-test-split")
    return delegated


def build_parser(description: str = "Evaluate a model on the configured fixed 30-scene benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model")
    source.add_argument("--run-dir")
    parser.add_argument("--scenario-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    verify_locked_environment(config)
    return run_script("evaluate.py", evaluation_arguments(args, config), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
