"""P4 v2 identities and factories; old round-three seals remain authoritative."""
import json
from pathlib import Path

import numpy as np
import stable_baselines3
import torch

from envs import DynamicPathPlanningEnv
from envs.risk_reward import RiskRewardWrapper, risk_metrics
from envs.scenario_dataset import dataset_hash, load_dataset
from experiment_utils import dependency_versions, sha256_file
from round_three_utils import implementation_manifest as previous_manifest, verify_implementation as verify_previous
from round_four_policy import ARMS, SCHEMA, LatentActionPPO
from scripts.train import PPO_CONFIG

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "configs/round4/experiment_protocol_v2.json"
SEAL = ROOT / "results/round4/implementation/implementation_seal.json"
EXTRA = ("round_four_policy.py", "round_four_utils.py", "scripts/train_round_four.py",
         "scripts/evaluate_round_four.py", "scripts/test_round_four.py", "scripts/freeze_round_four.py",
         "configs/round4/experiment_protocol.json", "configs/round4/experiment_protocol_v2.json",
         "docs/ROUND_FOUR_PROTOCOL.md", "docs/ROUND_FOUR_PROTOCOL_V2.md")


def load_protocol():
    revision = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if revision["inherits"] != "configs/round4/experiment_protocol.json":
        raise ValueError("Unexpected protocol inheritance")
    p = {**json.loads((ROOT/revision["inherits"]).read_text(encoding="utf-8")), **revision}
    if (p["protocol_version"] != "round4-speed-tanh-v2" or p["action"]["schema"] != SCHEMA
            or tuple(p["arms"]) != ARMS or p["training"]["ppo"] != PPO_CONFIG):
        raise ValueError("P4 protocol constants changed")
    validation = ROOT/p["validation"]["path"]
    data = load_dataset(validation)
    if (data["dataset_split"] != "validation" or data["environment_stage"] != "D"
            or len(data["scenarios"]) != 60 or dataset_hash(data) != p["validation"]["dataset_hash"]
            or sha256_file(validation) != p["validation"]["file_sha256"]):
        raise ValueError("P4 validation identity changed")
    if sha256_file(ROOT/p["dwa"]["selection"]) != p["dwa"]["selection_sha256"]:
        raise ValueError("DWA selection changed")
    if stable_baselines3.__version__ != "2.8.0":
        raise ValueError("Re-audit collector for a different SB3 version")
    return p


class DiagnosticEnv(RiskRewardWrapper):
    """Only logs: accepts executed a; never maps actions or changes base reward."""
    def __init__(self, env):
        super().__init__(env, "base")
        self.previous_obs = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.previous_obs = obs.copy()
        return obs, info

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("DiagnosticEnv only accepts finite executable actions in [-1,1]")
        if self.previous_obs is None:
            raise RuntimeError("Reset before step")
        pre_q4 = risk_metrics(self.previous_obs, 4.).risk
        pre_v = float(self.unwrapped.current_linear_velocity)
        obs, reward, done, truncated, info = super().step(action)
        info.update(pre_action_q4=pre_q4, pre_action_v=pre_v)
        self.previous_obs = obs.copy()
        return obs, reward, done, truncated, info


def make_env(scenario_file=None):
    return DiagnosticEnv(DynamicPathPlanningEnv(stage="D", scenario_file=scenario_file))


def create_model(env, arm, seed, log=None):
    p = PPO_CONFIG
    return LatentActionPPO("MlpPolicy", env, arm=arm, seed=seed, device="cpu",
        tensorboard_log=None if log is None else str(log), verbose=0,
        policy_kwargs=dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[64,64], vf=[64,64])),
        **{k: p[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                             "gae_lambda", "clip_range", "ent_coef")})


def implementation_manifest():
    return {**previous_manifest(), **{path: sha256_file(ROOT/path) for path in EXTRA}}


def dependency_sources():
    root = Path(stable_baselines3.__file__).parent
    paths = ("common/on_policy_algorithm.py", "common/base_class.py", "common/policies.py",
             "common/distributions.py", "common/buffers.py", "ppo/ppo.py")
    return {p: sha256_file(root/p) for p in paths}


def verify_implementation(seal_path=SEAL):
    load_protocol()
    verify_previous()
    seal_path = Path(seal_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal["sources"] != implementation_manifest() or seal["dependencies"] != dependency_versions()
            or seal["dependency_sources"] != dependency_sources()
            or sha256_file(seal_path.parent/"test_report.json") != seal["test_report_sha256"]):
        raise ValueError("P4 implementation/test/dependency seal mismatch")
    return seal


def verify_model(path):
    path = Path(path).resolve()
    run = path.parent.parent if path.name == "final_model.zip" else path.parent.parent.parent
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    meta = json.loads((run/"metadata.json").read_text(encoding="utf-8"))
    seal = verify_implementation()
    p = load_protocol()
    if (config["arm"] not in ARMS or config["seed"] not in (1,2,3)
            or config["action_schema"] != SCHEMA or run.name != f"round4-{config['arm']}-seed{config['seed']}"
            or config["run_id"] != run.name or meta["run_id"] != run.name
            or meta["status"] != "completed" or meta["final_num_timesteps"] != 200704
            or meta["config_sha256"] != sha256_file(run/"config.json")
            or config["implementation_seal_sha256"] != sha256_file(SEAL)
            or config["source_manifest"] != seal["sources"]
            or config["runtime_parameters"] != seal["runtime_parameters"]
            or config["validation_dataset_hash"] != p["validation"]["dataset_hash"]):
        raise ValueError("P4 run identity mismatch")
    candidates = {run/"models/final_model.zip": meta["final_model_sha256"],
                  run/"models/best/best_model.zip": meta["best_model_sha256"]}
    if path not in candidates or sha256_file(path) != candidates[path]:
        raise ValueError("Uncertified P4 model")
    for relative, digest in seal["sources"].items():
        if sha256_file(run/"snapshot"/relative) != digest:
            raise ValueError("P4 snapshot changed")
    if (sha256_file(run/"snapshot/implementation_seal.json") != sha256_file(SEAL)
            or sha256_file(run/"snapshot/validation_scenarios.json") != p["validation"]["file_sha256"]):
        raise ValueError("P4 validation/seal snapshot changed")
    for relative, digest in meta["evidence"].items():
        if sha256_file(run/relative) != digest:
            raise ValueError("P4 training evidence changed")
    return run, config, meta
