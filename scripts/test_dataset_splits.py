from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.scenario_dataset import dataset_hash, load_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that validation and test scenarios are disjoint")
    parser.add_argument(
        "--validation-file", type=Path,
        default=ROOT / "configs" / "validation_scenarios_30.json",
    )
    parser.add_argument(
        "--test-file", type=Path,
        default=ROOT / "configs" / "independent_test_scenarios_30.json",
    )
    args = parser.parse_args()
    validation = load_dataset(args.validation_file)
    test = load_dataset(args.test_file)
    assert validation.get("dataset_split") == "validation"
    assert test.get("dataset_split") == "test"
    validation_hashes = {item["scenario_hash"] for item in validation["scenarios"]}
    test_hashes = {item["scenario_hash"] for item in test["scenarios"]}
    assert validation_hashes.isdisjoint(test_hashes)
    assert dataset_hash(validation) != dataset_hash(test)
    print("Validation and independent test datasets are valid and disjoint.")


if __name__ == "__main__":
    main()
