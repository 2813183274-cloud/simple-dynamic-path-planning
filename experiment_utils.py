from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OBSERVATION_SCHEMA_VERSION = "center-distance-16-v1"
REWARD_SCHEMA_VERSION = "linear-center-distance-v1"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id(seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-seed{seed}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-80 characters and contain only letters, digits, '.', '_', or '-'"
        )
    return run_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


def git_revision(root: str | Path) -> dict[str, Any]:
    command = [
        "git", "-c", f"safe.directory={Path(root).resolve().as_posix()}",
        "rev-parse", "HEAD",
    ]
    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=True, timeout=5
        )
        commit = result.stdout.strip()
        dirty = subprocess.run(
            command[:-2] + ["status", "--porcelain"], cwd=root,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip() != ""
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def dependency_versions() -> dict[str, str | None]:
    packages = ["numpy", "gymnasium", "stable-baselines3", "torch", "matplotlib", "pandas"]
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_signature(env: Any) -> dict[str, Any]:
    from envs.encounters import CONFIG, schema
    return {
        "map_size": [env.map_width, env.map_height],
        "dt": env.dt,
        "v_max": env.v_max,
        "omega_max": env.omega_max,
        "linear_acceleration_max": env.linear_acceleration_max,
        "angular_acceleration_max": env.angular_acceleration_max,
        "agent_radius": env.agent_radius,
        "goal_threshold": env.goal_threshold,
        "static_safe_distance": env.static_safe_distance,
        "dynamic_safe_distance": env.dynamic_safe_distance,
        "max_steps": env.max_steps,
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "observation_schema_version": schema(env.stage),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "stage": env.stage,
        "encounter_config": CONFIG if env.stage != "legacy" else None,
    }


def source_manifest(root: Path) -> dict[str, str]:
    """Fingerprint executable experiment semantics even outside a Git checkout."""
    paths = ["experiment_utils.py", "envs/__init__.py", "envs/dynamic_path_env.py", "envs/scenario_dataset.py",
             "envs/encounters.py", "envs/round_one_dataset.py",
             "scripts/train.py", "scripts/evaluate.py", "scripts/evaluate_counterfactual.py"]
    return {name: sha256_file(root / name) for name in paths}


def verify_model_identity(model_path: Path, root: Path, *, allow_archived_source: bool = False) -> tuple[Path, dict, dict]:
    """Fail closed for legacy, modified, or semantically incompatible models."""
    model_path = model_path.resolve()
    for directory in model_path.parents:
        if (directory / "metadata.json").exists() and (directory / "config.json").exists():
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            if metadata.get("run_id") != config.get("run_id") or directory.name != config.get("run_id"):
                raise ValueError("Run identity mismatch")
            if config.get("source_manifest") != source_manifest(root):
                archived = config.get("source_manifest", {})
                if not allow_archived_source or not archived or any(
                    not (directory / "snapshot" / p).is_file() or
                    sha256_file(directory / "snapshot" / p) != digest for p,digest in archived.items()
                ):
                    raise ValueError("Source contract changed or is missing; start a new training run")
            actual = sha256_file(model_path)
            candidates = [(directory / "models/final_model.zip", metadata.get("final_model_sha256")),
                          (directory / "models/best/best_model.zip", metadata.get("best_model_sha256"))]
            if not any(model_path == path.resolve() and actual == digest for path, digest in candidates):
                raise ValueError("Model path/hash is not certified by run metadata")
            if metadata.get("status") != "completed":
                raise ValueError("Only completed runs are certified for reuse")
            return directory, config, metadata
    raise ValueError("Legacy model has no verified run identity; cannot safely resume")
