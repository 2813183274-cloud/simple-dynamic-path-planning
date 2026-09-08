"""Test and seal P3-04 implementation without starting formal training."""
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.risk_reward import RiskRewardWrapper
from experiment_utils import atomic_write_json,dependency_versions,sha256_file,utc_now
from round_three_utils import SEAL,implementation_manifest,load_protocol,runtime_parameters
from scripts.test_round_three import run_tests
from scripts.train import create_model


def main():
    if SEAL.exists():raise FileExistsError("Do not overwrite an implementation seal")
    load_protocol()
    sources=implementation_manifest()
    report=run_tests()
    with tempfile.TemporaryDirectory() as temp:
        env=RiskRewardWrapper(DynamicPathPlanningEnv(stage="D"),"base")
        model=create_model(env,1,Path(temp)/"tensorboard","cpu")
        actual=runtime_parameters(model)
        assert model.num_timesteps==0 and model._n_updates==0
        env.close()
    assert sources==implementation_manifest()
    SEAL.parent.mkdir(parents=True,exist_ok=True)
    atomic_write_json(SEAL.parent/"test_report.json",report)
    atomic_write_json(SEAL,{"sealed_at":utc_now(),"status":"implementation_verified_no_training",
        "sources":sources,"dependencies":dependency_versions(),"runtime_parameters":actual,
        "test_report_sha256":sha256_file(SEAL.parent/"test_report.json"),
        "formal_training_started":False,"final_datasets_generated":False})
    print(f"Implementation sealed: {SEAL}; no training started.")


if __name__=="__main__":main()
