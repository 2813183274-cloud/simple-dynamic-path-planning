"""P4 explicit from-scratch entry; no execution without --execute."""
import argparse
from pathlib import Path
import shutil
import sys

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import atomic_write_json, sha256_file, utc_now, git_revision
from round_three_utils import runtime_parameters
from round_four_policy import ARMS, SCHEMA
from round_four_utils import SEAL, load_protocol, verify_implementation, verify_model, make_env, create_model
from scripts.train import FixedScenarioBestModelCallback, save_training_reward_curve
from scripts.train_round_three import RoundThreeValidation, TrainingStepLog
from scripts.evaluate_round_four import evaluate_policy


class RoundFourValidation(RoundThreeValidation):
    """Reuse candidate ranking and final-update scheduling, not P3 action assumptions."""
    def _on_training_start(self):
        if (self.log_dir/"evaluations.json").exists():
            raise ValueError("P4 is from scratch; existing validation history forbidden")
        FixedScenarioBestModelCallback._on_training_start(self)
        self.eval_env.close()
        self.eval_env = make_env(self.validation_file)

    def _evaluate(self):
        if self.eval_env is None:
            raise RuntimeError("Validation not initialized")
        label = f"{self.num_timesteps}-{self.model._n_updates}"
        summary, frame = evaluate_policy(self.model, self.eval_env, self.num_scenarios,
                                        self.log_dir/f"steps-{label}.csv.gz")
        frame.to_csv(self.log_dir/f"episodes-{label}.csv", index=False)
        self.record_candidate(summary)
        print(f"{self.run_id} validation {label}: {summary['success_count']}/{self.num_scenarios}", flush=True)


def train(arm, seed):
    p = load_protocol(); seal = verify_implementation()
    if arm not in ARMS or seed not in p["training"]["seeds"]:
        raise ValueError("Arm/seed outside protocol")
    run = ROOT/"runs"/f"round4-{arm}-seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    env = Monitor(make_env(), str(run/"logs/train_monitor.csv"),
                  info_keywords=("episode_base_return", "episode_risk_return", "episode_training_return"))
    meta = dict(run_id=run.name, status="running", created_at=utc_now(), git=git_revision(ROOT))
    atomic_write_json(run/"metadata.json", meta)
    cb = None; trace = None
    try:
        model = create_model(env, arm, seed, run/"logs/tensorboard")
        actual = runtime_parameters(model)
        if actual != seal["runtime_parameters"]:
            raise ValueError("Runtime differs from sealed configuration")
        config = dict(run_id=run.name, arm=arm, seed=seed, action_schema=SCHEMA,
            source_manifest=seal["sources"], runtime_parameters=actual,
            implementation_seal_sha256=sha256_file(SEAL),
            validation_dataset_hash=p["validation"]["dataset_hash"], from_scratch=True,
            requested_steps=p["training"]["requested_steps_per_run"])
        atomic_write_json(run/"config.json", config)
        meta["config_sha256"] = sha256_file(run/"config.json")
        atomic_write_json(run/"metadata.json", meta)
        for relative in seal["sources"]:
            dest = run/"snapshot"/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, dest)
        shutil.copy2(SEAL, run/"snapshot/implementation_seal.json")
        validation = ROOT/p["validation"]["path"]
        shutil.copy2(validation, run/"snapshot/validation_scenarios.json")
        cb = RoundFourValidation(arm, validation, p["validation"]["dataset_hash"],
                                 p["validation"]["eval_every_steps"], run/"models/best", run/"logs/validation", run.name)
        trace = TrainingStepLog(run/"logs/training_steps.csv.gz")
        checkpoint = CheckpointCallback(50000, str(run/"models/checkpoints"), name_prefix="ppo")
        model.learn(config["requested_steps"], callback=[cb, checkpoint, trace], progress_bar=True)
        if model.num_timesteps != p["training"]["expected_actual_steps_per_run"]:
            raise ValueError("Training budget mismatch")
        verify_implementation()
        model.save(run/"models/final_model")
        save_training_reward_curve(run/"logs/train_monitor.csv", run/"results/training_reward_curve.png")
        evidence = ["models/best/selection_metrics.json", "logs/validation/evaluations.json",
                    "logs/train_monitor.csv", "logs/training_steps.csv.gz"]
        meta.update(status="completed", completed_at=utc_now(), final_num_timesteps=model.num_timesteps,
            final_model_sha256=sha256_file(run/"models/final_model.zip"),
            best_model_sha256=sha256_file(run/"models/best/best_model.zip"),
            evidence={x: sha256_file(run/x) for x in evidence},
            training_sampling_report=dict(env.unwrapped.generation_stats))
        atomic_write_json(run/"metadata.json", meta)
        verify_model(run/"models/best/best_model.zip")
    except BaseException:
        meta.update(status="failed", updated_at=utc_now())
        atomic_write_json(run/"metadata.json", meta)
        raise
    finally:
        if trace is not None:
            trace.log.close()
        if cb is not None and cb.eval_env is not None:
            cb.eval_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="P4 readiness only by default")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=[1,2,3])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    verify_implementation()
    if args.execute:
        train(args.arm, args.seed)
    else:
        print(f"Ready: {args.arm}, seed {args.seed}. No training; --execute is required.")


if __name__ == "__main__":
    main()
