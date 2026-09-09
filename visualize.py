"""Public trajectory visualization entry point; reuses the evaluator's plots."""
from __future__ import annotations

from evaluate import build_parser, evaluation_arguments
from project_config import load_config, run_script, verify_locked_environment


def main() -> int:
    args = build_parser("Evaluate the fixed benchmark and generate trajectory sheets").parse_args()
    config = load_config(args.config)
    verify_locked_environment(config)
    args.no_plots = False
    return run_script("evaluate.py", evaluation_arguments(args, config, plots=True), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
