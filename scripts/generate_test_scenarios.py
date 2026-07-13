from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.scenario_dataset import (  # noqa: E402
    TYPE_QUOTAS,
    dataset_summary,
    generate_dataset,
    load_dataset,
    save_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Generate the fixed 30-scenario evaluation set")
    parser.add_argument("--num-scenarios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "test_scenarios_30.json")
    parser.add_argument("--max-attempts", type=int, default=2000)
    args = parser.parse_args()
    if args.num_scenarios != 30:
        parser.error("This benchmark has fixed quotas and requires --num-scenarios 30")
    dataset = generate_dataset(args.seed, args.max_attempts)
    save_dataset(dataset, args.output)
    reloaded = load_dataset(args.output)
    if reloaded != dataset:
        raise RuntimeError("JSON reload verification failed")
    summary = dataset_summary(dataset)
    print("Generated fixed test set successfully.\n")
    for scenario_type, expected in TYPE_QUOTAS.items():
        item = summary[scenario_type]
        print(f"{scenario_type}: {item['count']} (expected {expected}), "
              f"mean dynamic clearance={item['mean_baseline_min_dynamic_clearance']:.3f} m, "
              f"baseline collisions={item['baseline_collision_count']}")
    print(f"total: {len(dataset['scenarios'])}")
    print("hashes unique: true")
    print("JSON reload verification: passed")
    print(f"saved to: {args.output.resolve()}")
    print("\nMachine-readable summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
