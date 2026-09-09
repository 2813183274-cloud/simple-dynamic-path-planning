from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import atomic_write_json, canonical_hash, validate_run_id  # noqa: E402
from envs.scenario_dataset import scenario_hash  # noqa: E402
from scripts.evaluate import wilson_interval  # noqa: E402
from scripts.evaluate_counterfactual import mirror_y  # noqa: E402


def main() -> None:
    assert validate_run_id("run-20260901_seed.0") == "run-20260901_seed.0"
    for invalid in ("", "../escape", "contains space", "x" * 81):
        try:
            validate_run_id(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid run_id was accepted: {invalid!r}")

    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    low, high = wilson_interval(15, 30)
    assert np.isclose(low, 0.33154125640533766)
    assert np.isclose(high, 0.6684587435946623)

    validation = json.loads(
        (ROOT / "configs" / "validation_scenarios_30.json").read_text(encoding="utf-8")
    )
    scenario = validation["scenarios"][0]
    assert scenario_hash(mirror_y(mirror_y(copy.deepcopy(scenario)))) == scenario_hash(scenario)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "nested" / "metadata.json"
        atomic_write_json(output, {"status": "ok"})
        assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ok"}

    print("All experiment workflow checks passed.")


if __name__ == "__main__":
    main()
