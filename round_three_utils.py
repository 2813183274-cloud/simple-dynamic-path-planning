"""Separate round-three identities: never weaken the old experiment source contract."""
import json
from pathlib import Path

from experiment_utils import dependency_versions, sha256_file, source_manifest

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "configs/round3/experiment_protocol.json"
SEAL = ROOT / "results/round3/implementation/implementation_seal.json"
EXTRA_SOURCES = (
    "envs/risk_reward.py", "round_three_utils.py", "scripts/train_round_three.py",
    "scripts/evaluate_round_three.py", "scripts/test_round_three.py", "scripts/freeze_round_three.py",
    "baselines/__init__.py", "baselines/dwa.py", "configs/round3/experiment_protocol.json",
    "docs/ROUND_THREE_PROTOCOL.md",
)


def implementation_manifest():
    return {**source_manifest(ROOT), **{p: sha256_file(ROOT / p) for p in EXTRA_SOURCES}}


def load_protocol():
    from envs.risk_reward import ARMS
    from envs.scenario_dataset import dataset_hash, load_dataset
    from scripts.train import PPO_CONFIG
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p["protocol_version"] != "round3-predictive-reward-v1":
        raise ValueError("Unsupported protocol")
    if p["reference_core_sources"] != source_manifest(ROOT):
        raise ValueError("The unchanged baseline source contract was modified")
    if p["training"]["ppo"] != PPO_CONFIG:
        raise ValueError("PPO parameters differ from protocol")
    if {a["name"]: (a["horizon_seconds"], a["weight"]) for a in p["arms"]} != ARMS:
        raise ValueError("Arm constants differ from protocol")
    risk = p["risk"]
    if (risk["horizon_seconds"], risk["clearance_buffer_m"], risk["weight"],
        risk["relative_speed_squared_epsilon"], risk["inflated_radii_m"], risk["relative_velocity_scale"]) != (4, 2, 2, 1e-12, [5, 5, 4], 4.8):
        raise ValueError("Risk constants differ from v1")
    data = load_dataset(ROOT / p["validation"]["path"])
    if (data["dataset_split"] != "validation" or data["environment_stage"] != "D"
            or len(data["scenarios"]) != 60 or dataset_hash(data) != p["validation"]["dataset_hash"]):
        raise ValueError("Common validation contract changed")
    if sha256_file(ROOT / "baselines/dwa.py") != p["dwa"]["source_sha256"]:
        raise ValueError("DWA changed")
    return p


def runtime_parameters(model):
    """Record explicit and SB3-default learning settings without serializing weights."""
    names = ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "vf_coef",
             "max_grad_norm", "normalize_advantage", "target_kl", "use_sde", "sde_sample_freq")
    result = {name: getattr(model, name) for name in names}
    result.update(learning_rate=float(model.lr_schedule(1.0)),
                  clip_range=float(model.clip_range(1.0)),
                  clip_range_vf=None if model.clip_range_vf is None else float(model.clip_range_vf(1.0)),
                  policy_class=type(model.policy).__name__,
                  policy_architecture=str(model.policy), optimizer_class=type(model.policy.optimizer).__name__,
                  optimizer_defaults={k: v for k, v in model.policy.optimizer.defaults.items()
                                      if isinstance(v, (int, float, str, bool, tuple)) or v is None},
                  device=str(model.device), observation_shape=list(model.observation_space.shape),
                  action_shape=list(model.action_space.shape))
    # Normalize tuples to JSON arrays for exact comparison after loading a seal.
    return json.loads(json.dumps(result))


def verify_implementation(seal_path=SEAL):
    load_protocol()
    seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    if seal["sources"] != implementation_manifest() or seal["dependencies"] != dependency_versions():
        raise ValueError("Implementation/dependencies differ from the pre-training seal")
    if sha256_file(Path(seal_path).parent / "test_report.json") != seal["test_report_sha256"]:
        raise ValueError("Implementation test report changed")
    return seal


def verify_round_three_model(model_path):
    model_path = Path(model_path).resolve()
    run = model_path.parent.parent if model_path.name == "final_model.zip" else model_path.parent.parent.parent
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    seal = verify_implementation()
    protocol = load_protocol()
    if (config["run_id"] != run.name or meta["run_id"] != run.name or meta["status"] != "completed"
            or meta["config_sha256"] != sha256_file(config_path)
            or config["implementation_seal_sha256"] != sha256_file(SEAL)
            or config["source_manifest"] != seal["sources"]):
        raise ValueError("Run identity is not certified")
    if (config["arm"] not in ("base", "instant", "predictive") or config["seed"] not in (1, 2, 3)
            or run.name != f"round3-{config['arm']}-seed{config['seed']}"
            or config["validation_dataset_hash"] != protocol["validation"]["dataset_hash"]
            or config["runtime_parameters"] != seal["runtime_parameters"]):
        raise ValueError("Run parameters differ from protocol")
    candidates = {run / "models/final_model.zip": meta["final_model_sha256"],
                  run / "models/best/best_model.zip": meta["best_model_sha256"]}
    if model_path not in candidates or sha256_file(model_path) != candidates[model_path]:
        raise ValueError("Uncertified model path/hash")
    for path, digest in config["source_manifest"].items():
        if sha256_file(run / "snapshot" / path) != digest:
            raise ValueError("Run snapshot changed")
    from envs.scenario_dataset import load_dataset, dataset_hash
    if (sha256_file(run / "snapshot/implementation_seal.json") != sha256_file(SEAL)
            or dataset_hash(load_dataset(run / "snapshot/validation_scenarios.json")) != config["validation_dataset_hash"]):
        raise ValueError("Snapshot seal/validation changed")
    for relative, key in (("models/best/selection_metrics.json", "selection_metrics_sha256"),
                          ("logs/validation/evaluations.json", "validation_history_sha256"),
                          ("logs/train_monitor.csv", "training_monitor_sha256")):
        if sha256_file(run / relative) != meta[key]:
            raise ValueError("Selected-model/training evidence changed")
    if meta["final_num_timesteps"] != 200704:
        raise ValueError("Wrong training budget")
    return run, config, meta
