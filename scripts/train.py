from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv  # noqa: E402
from envs.scenario_dataset import dataset_hash, load_dataset  # noqa: E402
from experiment_utils import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    atomic_write_json,
    default_run_id,
    dependency_versions,
    environment_signature,
    git_revision,
    sha256_file,
    utc_now,
    validate_run_id,
    source_manifest,
    verify_model_identity,
)


PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "policy_network": [64, 64],
    "value_network": [64, 64],
    "activation": "Tanh",
}


def save_training_reward_curve(
    monitor_path: Path,
    output_path: Path,
    smoothing_window: int = 100,
) -> None:
    """Plot random-training episode rewards and a rolling convergence curve."""
    frame = pd.read_csv(monitor_path, comment="#")
    if frame.empty:
        raise RuntimeError(f"No completed training episodes found in {monitor_path}")

    frame["timesteps"] = frame["l"].cumsum()
    window = min(smoothing_window, len(frame))
    min_periods = max(1, window // 5)
    frame["smoothed_reward"] = frame["r"].rolling(
        window=window, min_periods=min_periods
    ).mean()

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(
        frame["timesteps"], frame["r"],
        color="tab:blue", alpha=0.16, linewidth=0.7, label="Episode reward",
    )
    ax.plot(
        frame["timesteps"], frame["smoothed_reward"],
        color="tab:orange", linewidth=2.2, label=f"Rolling mean ({window} episodes)",
    )
    last = frame.dropna(subset=["smoothed_reward"]).iloc[-1]
    ax.scatter(last["timesteps"], last["smoothed_reward"], color="tab:orange", s=32, zorder=3)
    ax.annotate(
        f"Final smoothed reward: {last['smoothed_reward']:.1f}",
        xy=(last["timesteps"], last["smoothed_reward"]),
        xytext=(-12, 12), textcoords="offset points", ha="right", fontsize=9,
    )
    ax.set_title("Random Training Episode Reward Convergence")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Episode reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Training reward curve saved to {output_path}")


class FixedScenarioBestModelCallback(BaseCallback):
    """Select the best model on a validation set and preserve history across safe resumes."""

    def __init__(
        self,
        validation_file: Path,
        validation_dataset_hash: str,
        eval_freq: int,
        best_model_dir: Path,
        log_dir: Path,
        run_id: str,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.validation_file = validation_file.resolve()
        self.validation_dataset_hash = validation_dataset_hash
        self.eval_freq = eval_freq
        self.best_model_dir = best_model_dir
        self.log_dir = log_dir
        self.run_id = run_id
        self.eval_env: DynamicPathPlanningEnv | None = None
        self.num_scenarios = 0
        self.best_key: tuple[float, float, float, float] | None = None
        self.history: list[dict] = []
        self.last_eval_timestep = -1

    @staticmethod
    def _selection_key(record: dict) -> tuple[float, float, float, float]:
        return (
            float(record["success_rate"]),
            -float(record["collision_rate"]),
            float(record.get("mean_successful_path_efficiency") or 0.0),
            float(record["mean_reward"]),
        )

    def _on_training_start(self) -> None:
        dataset = load_dataset(self.validation_file)
        if dataset.get("dataset_split") != "validation":
            raise ValueError(
                f"Training model selection requires a validation split, got "
                f"{dataset.get('dataset_split', 'legacy_unspecified')!r}"
            )
        self.num_scenarios = int(dataset["num_scenarios"])
        self.eval_env = DynamicPathPlanningEnv(
            randomize_scenario=False,
            scenario_file=self.validation_file,
        )
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.log_dir / "evaluations.json"
        if history_path.exists():
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError(f"Invalid validation history: {history_path}")
            self.history = loaded
            if self.history:
                if any(
                    item.get("validation_dataset_hash") != self.validation_dataset_hash
                    for item in self.history
                ):
                    raise ValueError("Cannot resume with a different validation dataset")
                best_record = max(self.history, key=self._selection_key)
                self.best_key = self._selection_key(best_record)
                self.last_eval_timestep = int(self.history[-1]["timesteps"])

    def _evaluate(self) -> None:
        if self.eval_env is None:
            raise RuntimeError("Fixed evaluation environment is not initialized")

        rewards: list[float] = []
        path_efficiencies: list[float] = []
        success_count = 0
        collision_count = 0
        for index in range(self.num_scenarios):
            observation, _ = self.eval_env.reset(options={"scenario_index": index})
            direct_distance = max(0.0, float(np.linalg.norm(
                self.eval_env.goal_position - self.eval_env.start_position
            )) - self.eval_env.goal_threshold)
            episode_reward = 0.0
            while True:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
            success_count += int(info["is_success"])
            collision_count += int(info["collision_type"] != "none")
            if info["is_success"] and info["path_length"] > 0.0:
                path_efficiencies.append(direct_distance / float(info["path_length"]))

        success_rate = success_count / self.num_scenarios
        collision_rate = collision_count / self.num_scenarios
        mean_reward = float(np.mean(rewards))
        mean_efficiency = float(np.mean(path_efficiencies)) if path_efficiencies else 0.0
        record = {
            "run_id": self.run_id,
            "timesteps": self.num_timesteps,
            "policy_updates": int(self.model._n_updates),
            "evaluated_at": utc_now(),
            "validation_file": str(self.validation_file),
            "validation_dataset_hash": self.validation_dataset_hash,
            "num_scenarios": self.num_scenarios,
            "success_count": success_count,
            "success_rate": success_rate,
            "collision_count": collision_count,
            "collision_rate": collision_rate,
            "mean_successful_path_efficiency": mean_efficiency,
            "mean_reward": mean_reward,
        }
        selection_key = self._selection_key(record)
        record["is_best"] = self.best_key is None or selection_key > self.best_key
        if record["is_best"]:
            self.best_key = selection_key
            self.model.save(self.best_model_dir / "best_model")
            model_path = self.best_model_dir / "best_model.zip"
            record["model_sha256"] = sha256_file(model_path)
            atomic_write_json(self.best_model_dir / "selection_metrics.json", record)
        self.history.append(record)
        atomic_write_json(self.log_dir / "evaluations.json", self.history)
        self.last_eval_timestep = self.num_timesteps

        if self.verbose:
            marker = " -> saved best model" if record["is_best"] else ""
            print(
                f"Validation at {self.num_timesteps} steps: "
                f"success={success_count}/{self.num_scenarios} ({success_rate:.1%}), "
                f"collisions={collision_count}/{self.num_scenarios} ({collision_rate:.1%}), "
                f"path_efficiency={mean_efficiency:.3f}, mean_reward={mean_reward:.3f}{marker}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        # on_step runs before PPO.train(); equal step counts can represent different weights.
        self._evaluate()
        if self.eval_env is not None:
            self.eval_env.close()


def create_model(env: Monitor, seed: int, tensorboard_log: Path, device: str = "auto") -> PPO:
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    return PPO(
        "MlpPolicy", env,
        learning_rate=PPO_CONFIG["learning_rate"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(tensorboard_log),
        seed=seed,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO with isolated, traceable run outputs")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["legacy","A","B","C","D"], default="D")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--reward-smoothing-window", type=int, default=100)
    parser.add_argument(
        "--validation-file", "--scenario-file", dest="validation_file", type=Path,
        default=None,
        help="Validation split used only for model selection",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be a positive integer")
    if args.eval_freq <= 0:
        parser.error("--eval-freq must be a positive integer")
    if args.reward_smoothing_window <= 0:
        parser.error("--reward-smoothing-window must be a positive integer")
    validation_file = (args.validation_file or (ROOT / "configs" / "validation_scenarios_30.json"
        if args.stage=="legacy" else ROOT / "configs" / "round1" / f"validation_{args.stage}.json")).resolve()
    if not validation_file.exists():
        parser.error(f"validation file not found: {validation_file}")
    validation_dataset = load_dataset(validation_file)
    if validation_dataset.get("dataset_split") != "validation":
        parser.error("--validation-file must declare dataset_split='validation'")
    validation_fingerprint = dataset_hash(validation_dataset)
    if validation_dataset.get("environment_stage", "legacy") != args.stage:
        parser.error("validation stage must match training stage")

    run_id = validate_run_id(args.run_id or default_run_id(args.seed))
    runs_dir = args.runs_dir.resolve()
    run_dir = runs_dir / run_id
    resume_path = args.resume.resolve() if args.resume is not None else None
    if resume_path is not None and not resume_path.exists():
        parser.error(f"resume model not found: {resume_path}")
    if run_dir.exists() and resume_path is None:
        parser.error(f"run directory already exists: {run_dir}; choose another --run-id")

    source_run = source_config = source_metadata = None
    if resume_path is not None:
        source_run, source_config, source_metadata = verify_model_identity(resume_path, ROOT)
        if source_config["validation_dataset_hash"] != validation_fingerprint:
            parser.error("resume requires the same validation dataset")
        if source_config["seed"] != args.seed or source_config["ppo"] != PPO_CONFIG:
            parser.error("resume requires matching seed and PPO configuration")
        if run_dir.exists() and (run_dir != source_run or
                resume_path != source_run / "models" / "final_model.zip"):
            parser.error("existing run can only continue from its own latest certified final model")

    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    for directory in (model_dir / "best", model_dir / "checkpoints", log_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    raw_env = DynamicPathPlanningEnv(stage=args.stage)
    signature = environment_signature(raw_env)
    if source_config is not None and source_config.get("environment") != signature:
        parser.error("resume requires identical environment stage and observation semantics")
    config = {
        "run_id": run_id,
        "created_at": utc_now(),
        "seed": args.seed,
        "requested_additional_timesteps": args.timesteps,
        "eval_freq": args.eval_freq,
        "reward_smoothing_window": args.reward_smoothing_window,
        "validation_file": str(validation_file),
        "validation_dataset_hash": validation_fingerprint,
        "resume_model": str(resume_path) if resume_path else None,
        "resume_model_sha256": sha256_file(resume_path) if resume_path else None,
        "environment": signature,
        "ppo": PPO_CONFIG,
        "source_manifest": source_manifest(ROOT),
        "parent_run_id": source_config["run_id"] if source_config else None,
        "device": args.device,
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("validation_dataset_hash") != validation_fingerprint:
            parser.error("cannot resume a run with a different validation dataset")
        if existing.get("environment") != signature:
            parser.error("cannot resume a run after the environment contract changed")
        if existing.get("eval_freq") != args.eval_freq:
            parser.error("same-run resume requires the original evaluation frequency")
        continuations = existing.setdefault("continuations", [])
        continuations.append({
            "started_at": utc_now(),
            "additional_timesteps": args.timesteps,
            "resume_model": str(resume_path),
            "resume_model_sha256": sha256_file(resume_path),
        })
        config = existing
    atomic_write_json(config_path, config)
    snapshot_dir = run_dir / "snapshot"
    if not snapshot_dir.exists():
        for relative in source_manifest(ROOT):
            destination = snapshot_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        shutil.copy2(validation_file, snapshot_dir / "validation_scenarios.json")

    metadata_path = run_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {"run_id": run_id, "created_at": utc_now()}
    )
    metadata.update({
        "status": "running",
        "updated_at": utc_now(),
        "git": git_revision(ROOT),
        "dependencies": dependency_versions(),
        "observation_schema_version": signature["observation_schema_version"],
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "source_manifest": source_manifest(ROOT),
    })
    atomic_write_json(metadata_path, metadata)

    train_monitor_path = log_dir / "train_monitor.csv"
    env = Monitor(raw_env, str(train_monitor_path), override_existing=not train_monitor_path.exists())
    if resume_path is None:
        model = create_model(env, args.seed, log_dir / "tensorboard", args.device)
        reset_num_timesteps = True
    else:
        model = PPO.load(
            resume_path, env=env, device=args.device,
            tensorboard_log=str(log_dir / "tensorboard"),
        )
        if run_dir == source_run and model.num_timesteps != source_metadata["final_num_timesteps"]:
            raise ValueError("Resume model timestep disagrees with certified metadata")
        reset_num_timesteps = False
    metadata["training_start_num_timesteps"] = model.num_timesteps
    atomic_write_json(metadata_path, metadata)

    validation_callback = FixedScenarioBestModelCallback(
        validation_file=validation_file,
        validation_dataset_hash=validation_fingerprint,
        eval_freq=args.eval_freq,
        best_model_dir=model_dir / "best",
        log_dir=log_dir / "validation",
        run_id=run_id,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_dynamic_path",
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[validation_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(model_dir / "final_model")
        final_model_path = model_dir / "final_model.zip"
        save_training_reward_curve(
            train_monitor_path,
            results_dir / "training_reward_curve.png",
            smoothing_window=args.reward_smoothing_window,
        )
        metadata.update({
            "status": "completed",
            "updated_at": utc_now(),
            "final_num_timesteps": model.num_timesteps,
            "final_model": str(final_model_path),
            "final_model_sha256": sha256_file(final_model_path),
            "training_sampling_report": dict(raw_env.generation_stats),
        })
        best_model_path = model_dir / "best" / "best_model.zip"
        if best_model_path.exists():
            metadata["best_model"] = str(best_model_path)
            metadata["best_model_sha256"] = sha256_file(best_model_path)
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(runs_dir / "latest_run.json", {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "updated_at": utc_now(),
        })
        print(f"Run completed: {run_id}")
        print(f"Final model saved to {final_model_path}")
    except BaseException:
        metadata.update({"status": "failed", "updated_at": utc_now()})
        atomic_write_json(metadata_path, metadata)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
