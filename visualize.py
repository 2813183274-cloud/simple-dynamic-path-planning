"""Public trajectory visualization entry point; reuses the evaluator's plots."""
from __future__ import annotations

from evaluate import build_parser, evaluation_arguments
from project_config import load_config, run_script, verify_locked_environment
from project_config import resolve_path
from utils.metrics import write_nominal_risk_metrics


def main() -> int:
    args = build_parser("Evaluate the fixed benchmark and generate trajectory sheets").parse_args()
    config = load_config(args.config)
    verify_locked_environment(config)
    args.no_plots = False
    result = run_script("evaluate.py", evaluation_arguments(args, config, plots=True), dry_run=args.dry_run)
    if result == 0 and not args.dry_run:
        dataset = resolve_path(args.scenario_file or config["evaluation"]["benchmark"], must_exist=True)
        output = resolve_path(args.output_dir or config["evaluation"]["output_dir"])
        write_nominal_risk_metrics(dataset, output, config["environment"]["linear_velocity_max"])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
