"""Explicit, sealed, from-scratch round-three training. No legacy latest-pointer changes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.risk_reward import ARMS, REWARD_VERSION, RiskRewardWrapper
from experiment_utils import atomic_write_json, dependency_versions, environment_signature, git_revision, sha256_file, utc_now
from round_three_utils import SEAL, load_protocol, runtime_parameters, verify_implementation, verify_round_three_model
from scripts.train import FixedScenarioBestModelCallback, create_model, save_training_reward_curve
from scripts.evaluate_round_three import StepLog, evaluate_policy, selection_key


class RoundThreeValidation(FixedScenarioBestModelCallback):
    """Retain final-update scheduling but rank only common base return."""
    _selection_key = staticmethod(selection_key)

    def __init__(self, arm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm

    def _on_training_start(self):
        if (self.log_dir / "evaluations.json").exists():
            raise ValueError("Round-three v1 is from scratch, not resumable")
        super()._on_training_start()
        self.eval_env = RiskRewardWrapper(self.eval_env, self.arm)

    def record_candidate(self, summary):
        record = {**summary, "run_id": self.run_id, "arm": self.arm,
                  "timesteps": self.num_timesteps, "policy_updates": int(self.model._n_updates),
                  "evaluated_at": utc_now(), "validation_dataset_hash": self.validation_dataset_hash,
                  "selection_reward": "base_reward"}
        key = selection_key(record)
        record["is_best"] = self.best_key is None or key > self.best_key
        if record["is_best"]:
            self.best_key = key
            self.model.save(self.best_model_dir / "best_model")
            record["model_sha256"] = sha256_file(self.best_model_dir / "best_model.zip")
            atomic_write_json(self.best_model_dir / "selection_metrics.json", record)
        self.history.append(record)
        atomic_write_json(self.log_dir / "evaluations.json", self.history)
        self.last_eval_timestep = self.num_timesteps

    def _evaluate(self):
        if self.eval_env is None:
            raise RuntimeError("Validation env not initialized")
        label = f"{self.num_timesteps}-{self.model._n_updates}"
        summary, frame = evaluate_policy(self.model, self.eval_env, self.num_scenarios,
                                         self.log_dir / f"steps-{label}.csv.gz")
        frame.to_csv(self.log_dir / f"episodes-{label}.csv", index=False)
        self.record_candidate(summary)
        if self.verbose:
            print(f"{self.run_id} validation {label}: {summary['success_count']}/{self.num_scenarios}, "
                  f"base={summary['mean_base_reward']:.3f}, shaped={summary['mean_training_reward']:.3f}", flush=True)


class TrainingStepLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.log = StepLog(path)

    def _on_step(self):
        # Single training environment as fixed by this protocol. SB3 passes terminal info before auto-reset.
        info = {k: v for k, v in self.locals["infos"][0].items() if k not in ("episode", "terminal_observation", "TimeLimit.truncated")}
        self.log.write({"global_step": self.num_timesteps, **info})
        return True

    def _on_training_end(self):
        self.log.close()


def train(arm, seed, *, run_root=None):
    protocol = load_protocol()
    seal = verify_implementation()
    if arm not in ARMS or seed not in protocol["training"]["seeds"]:
        raise ValueError("Arm/seed outside fixed protocol")
    run_id = f"round3-{arm}-seed{seed}"
    run = (ROOT / "runs" if run_root is None else Path(run_root)) / run_id
    if run.exists():
        raise FileExistsError("Never overwrite or resume a v1 formal run")
    set_random_seed(seed)
    raw = DynamicPathPlanningEnv(stage="D")
    wrapped = RiskRewardWrapper(raw, arm)
    run.mkdir(parents=True, exist_ok=False)
    env = Monitor(wrapped, str(run / "logs/train_monitor.csv"), info_keywords=(
        "episode_base_return", "episode_risk_return", "episode_training_return"))
    meta = {"run_id": run_id, "status": "running", "created_at": utc_now(),
            "git": git_revision(ROOT), "dependencies": dependency_versions()}
    atomic_write_json(run / "metadata.json", meta)
    callback = None; trace = None
    try:
        model = create_model(env, seed, run / "logs/tensorboard", "cpu")
        actual = runtime_parameters(model)
        if actual != seal["runtime_parameters"]:
            raise ValueError("Actual PPO defaults/architecture differ from implementation seal")
        config = {"run_id": run_id, "arm": arm, "seed": seed, "requested_steps": 200000,
                  "environment": environment_signature(raw), "reward_schema_version": REWARD_VERSION,
                  "selection_reward": "base_reward", "source_manifest": seal["sources"],
                  "implementation_seal_sha256": sha256_file(SEAL), "runtime_parameters": actual,
                  "validation_dataset_hash": protocol["validation"]["dataset_hash"], "from_scratch": True}
        atomic_write_json(run / "config.json", config)
        meta["config_sha256"] = sha256_file(run / "config.json")
        atomic_write_json(run / "metadata.json", meta)
        for relative in seal["sources"]:
            dest = run / "snapshot" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
        shutil.copy2(SEAL, run / "snapshot/implementation_seal.json")
        validation = ROOT / protocol["validation"]["path"]
        shutil.copy2(validation, run / "snapshot/validation_scenarios.json")
        callback = RoundThreeValidation(arm, validation, protocol["validation"]["dataset_hash"], 10000,
                                        run / "models/best", run / "logs/validation", run_id)
        trace = TrainingStepLog(run / "logs/training_steps.csv.gz")
        checkpoint = CheckpointCallback(50000, str(run / "models/checkpoints"), name_prefix="ppo")
        model.learn(200000, callback=[callback, checkpoint, trace], progress_bar=True)
        if model.num_timesteps != 200704:
            raise ValueError("Actual training steps differ from protocol")
        verify_implementation()
        model.save(run / "models/final_model")
        save_training_reward_curve(run / "logs/train_monitor.csv", run / "results/training_reward_curve.png")
        meta.update(status="completed", updated_at=utc_now(), final_num_timesteps=model.num_timesteps,
                    final_model_sha256=sha256_file(run / "models/final_model.zip"),
                    best_model_sha256=sha256_file(run / "models/best/best_model.zip"),
                    selection_metrics_sha256=sha256_file(run / "models/best/selection_metrics.json"),
                    validation_history_sha256=sha256_file(run / "logs/validation/evaluations.json"),
                    training_monitor_sha256=sha256_file(run / "logs/train_monitor.csv"),
                    training_sampling_report=dict(raw.generation_stats))
        atomic_write_json(run / "metadata.json", meta)
        verify_round_three_model(run / "models/best/best_model.zip")
    except BaseException:
        meta.update(status="failed", updated_at=utc_now())
        atomic_write_json(run / "metadata.json", meta)
        raise
    finally:
        if trace is not None:
            trace.log.close()
        if callback is not None and callback.eval_env is not None:
            callback.eval_env.close()
        env.close()
    print(f"Completed {run_id}; legacy/default latest model pointer was not changed.")


def main():
    parser = argparse.ArgumentParser(description="Round-three readiness by default; --execute explicitly starts one formal run")
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    verify_implementation()
    if args.execute:
        train(args.arm, args.seed)
    else:
        print(f"Ready: {args.arm}, seed {args.seed}. No training started; --execute is required.")


if __name__ == "__main__":
    main()
