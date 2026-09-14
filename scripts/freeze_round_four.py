"""Test and seal P4 v2, never launch training or replace an existing seal."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_utils import atomic_write_json, dependency_versions, sha256_file, utc_now
from round_three_utils import runtime_parameters
from round_four_utils import SEAL, create_model, make_env, implementation_manifest, dependency_sources, load_protocol
from scripts.test_round_four import run_tests


def main():
    if SEAL.exists():
        raise FileExistsError("Refuse to overwrite P4 implementation seal")
    load_protocol()
    sources=implementation_manifest()
    report=run_tests()
    env=make_env()
    try:
        model=create_model(env,"clip",1)
        actual=runtime_parameters(model)
        assert model.num_timesteps==0 and model._n_updates==0
    finally:
        env.close()
    assert sources==implementation_manifest()
    atomic_write_json(SEAL.parent/"test_report.json",report)
    atomic_write_json(SEAL,dict(sealed_at=utc_now(),status="implementation_verified_no_training",
        sources=sources,dependencies=dependency_versions(),dependency_sources=dependency_sources(),
        runtime_parameters=actual,test_report_sha256=sha256_file(SEAL.parent/"test_report.json"),
        formal_training_started=False,final_datasets_generated=False))
    print(f"Sealed {SEAL}. No formal training.")


if __name__ == "__main__":
    main()
