"""P4 v2 regression: sampling/backward only, optimizer and train prohibited."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils import source_manifest, sha256_file
from round_three_utils import runtime_parameters, verify_implementation as verify_previous
from round_four_policy import LatentActionPPO, map_action, log_speed_jacobian
from round_four_utils import make_env, create_model, load_protocol, implementation_manifest, verify_implementation
from scripts.train import create_model as create_old
from scripts.train_round_four import RoundFourValidation
from scripts.evaluate_round_four import evaluate_policy, ResponseTracker
from scripts.evaluate_round_three import selection_key


class Record(BaseCallback):
    def __init__(self, stop=False):
        super().__init__()
        self.rows = []
        self.stop = stop

    def _on_step(self):
        self.rows.append(copy.deepcopy({key: self.locals[key] for key in
            ("actions", "clipped_actions", "rewards", "dones", "infos")}))
        return not self.stop


class Probe(gym.Env):
    """Natural termination and time-limit truncation, with true terminal obs."""
    observation_space = gym.spaces.Box(-1, 1, (16,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(self, truncated):
        self.truncated = truncated
        self.count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        return np.zeros(16, np.float32), {}

    def step(self, a):
        assert np.isfinite(a).all() and np.all(np.abs(a) <= 1)
        self.count += 1
        end = self.count == 3
        obs = np.full(16, self.count/10, np.float32)
        return obs, float(a[0]), end and not self.truncated, end and self.truncated, {}


def collect(model, n=32, stop=False):
    callback = Record(stop)
    model._setup_learn(n, callback=callback)
    buffer = RolloutBuffer(n, model.observation_space, model.action_space, device="cpu",
                          gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)
    torch.manual_seed(842)
    completed = model.collect_rollouts(model.env, callback, buffer, n)
    return buffer, callback, completed


def reject(call):
    try:
        call()
    except (ValueError, FileNotFoundError, KeyError):
        return
    raise AssertionError("Expected rejection")


def run_tests():
    # Any accidentally introduced learning step fails immediately.
    with patch.object(PPO, "train", side_effect=AssertionError("Training forbidden")), \
         patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Optimizer forbidden")):
        return _tests()


def _tests():
    p = load_protocol(); verify_previous()
    sources = implementation_manifest()
    core = source_manifest(ROOT)
    checks = []
    z = np.array([[0.,0.], [1.,2.], [-2.,-3.], [20.,-20.], [-20.,20.]], np.float32)
    for arm in ("clip", "speed_tanh"):
        a = map_action(z, arm)
        np.testing.assert_array_equal(a[:,1], np.clip(z[:,1],-1,1))
        assert a[0,0] == 0 and 1.5*(a[0,0]+1) == 1.5
        assert np.isfinite(a).all() and np.all(np.abs(a)<=1)
    assert not np.isclose(map_action(z,"speed_tanh")[1,0], np.tanh(np.tanh(1.)))
    reject(lambda: map_action([float("nan"),0], "clip"))
    reject(lambda: map_action([0,0], "unknown"))
    checks.append("finite map, zero/extreme speed, unchanged steering, no double tanh")

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for truncated in (False, True):
            old = create_old(Probe(truncated), 1, temp/"tb", "cpu")
            new = create_model(Probe(truncated), "clip", 1)
            assert runtime_parameters(old) == runtime_parameters(new)
            for key, value in old.policy.state_dict().items():
                torch.testing.assert_close(value, new.policy.state_dict()[key], rtol=0, atol=0)
            before = copy.deepcopy(new.policy.state_dict())
            b0,c0,_ = collect(old); b1,c1,_ = collect(new)
            for name in ("observations", "actions", "rewards", "episode_starts", "values", "log_probs", "returns", "advantages"):
                np.testing.assert_array_equal(getattr(b0,name), getattr(b1,name))
            for x,y in zip(c0.rows,c1.rows):
                for name in ("actions", "clipped_actions", "rewards", "dones"):
                    np.testing.assert_array_equal(x[name],y[name])
            for key,value in before.items():
                torch.testing.assert_close(value,new.policy.state_dict()[key],rtol=0,atol=0)
            assert new._n_updates == 0
            old.env.close(); new.env.close()
        checks.append("clip vs stock SB3 exact rollout/initialization/GAE equivalence for termination and timeout")

        model = create_model(Probe(True), "speed_tanh", 1)
        before = copy.deepcopy(model.policy.state_dict())
        buffer, callback, _ = collect(model)
        for row in callback.rows:
            np.testing.assert_array_equal(row["clipped_actions"], map_action(row["actions"],"speed_tanh"))
            assert row["infos"][0]["latent_speed"] == float(row["actions"][0,0])
        assert np.any(np.abs(buffer.actions)>1)  # original z, not executable a
        batch = next(buffer.get(32))
        _, logp, entropy = model.policy.evaluate_actions(batch.observations, batch.actions)
        torch.testing.assert_close(torch.exp(logp-batch.old_log_prob), torch.ones_like(logp), atol=1e-6, rtol=0)
        normal = model.policy.get_distribution(batch.observations).distribution
        torch.testing.assert_close(logp,normal.log_prob(batch.actions).sum(-1))
        torch.testing.assert_close(entropy,normal.entropy().sum(-1))
        loss = -(logp + .01*entropy).mean()
        loss.backward()
        assert all(torch.isfinite(x.grad).all() for x in model.policy.parameters() if x.grad is not None)
        assert any(x.grad is not None and torch.any(x.grad != 0) for x in model.policy.parameters())
        for key,value in before.items():
            torch.testing.assert_close(value,model.policy.state_dict()[key],rtol=0,atol=0)
        with torch.no_grad():
            model.policy.action_net.bias.add_(.1)
        _, changed, _ = model.policy.evaluate_actions(batch.observations,batch.actions)
        changed_normal = model.policy.get_distribution(batch.observations).distribution
        torch.testing.assert_close(torch.exp(changed-logp.detach()),
            torch.exp(changed_normal.log_prob(batch.actions).sum(-1)-normal.log_prob(batch.actions).sum(-1)))
        model.policy.load_state_dict(before)
        t = torch.tensor([-20.,-2.,0.,1.,20.],dtype=torch.float64,requires_grad=True)
        jac = log_speed_jacobian(t)
        n0=torch.distributions.Normal(torch.tensor(0.),torch.tensor(1.))
        n1=torch.distributions.Normal(torch.tensor(.1),torch.tensor(.9))
        torch.testing.assert_close(torch.exp(n1.log_prob(t)-n0.log_prob(t)),
                                   torch.exp((n1.log_prob(t)-jac)-(n0.log_prob(t)-jac)))
        jac.sum().backward()
        assert torch.isfinite(jac).all() and torch.isfinite(t.grad).all()
        _, stopped, finished = collect(model, stop=True)
        assert not finished and len(stopped.rows)==1
        model.env.close()
        checks.append("latent rollout log-prob/ratio, perturbed ratio, Jacobian cancellation, entropy, finite backward, callback stop")

        # Actual USV trajectory and acceleration equivalence for both mappings.
        from envs import DynamicPathPlanningEnv
        for arm in ("clip", "speed_tanh"):
            raw = DynamicPathPlanningEnv(stage="D")
            diagnostic = make_env()
            np.testing.assert_array_equal(raw.reset(seed=26)[0], diagnostic.reset(seed=26)[0])
            for z_i in np.tile(z, (8,1)):
                a=map_action(z_i,arm)
                previous=raw.current_linear_velocity
                x=raw.step(a); y=diagnostic.step(a)
                np.testing.assert_array_equal(x[0],y[0]); assert x[1:4]==y[1:4]
                assert abs(raw.current_linear_velocity-previous)<=.3+1e-9
                assert y[4]["risk_penalty"]==0
                if x[2] or x[3]: break
            reject(lambda: diagnostic.step([2.,0.]))
            raw.close(); diagnostic.close()

            env = make_env(ROOT/"configs/round1/typical_cases.json")
            model=create_model(env,arm,1)
            obs,_=env.reset(options={"scenario_index":0})
            a,_=model.predict(obs,deterministic=True)
            tensor,_=model.policy.obs_to_tensor(obs)
            mean=model.policy.get_distribution(tensor).distribution.mean.detach().numpy()[0]
            np.testing.assert_array_equal(a,map_action(mean,arm))
            model.save(temp/f"{arm}.zip")
            loaded=LatentActionPPO.load(temp/f"{arm}.zip",device="cpu")
            np.testing.assert_array_equal(a,loaded.predict(obs,deterministic=True)[0])
            # Batched GEMM may differ from the single-row path by a few float32 ULPs.
            np.testing.assert_allclose(model.predict(np.stack([obs,obs]),deterministic=True)[0],
                                       np.stack([a,a]), rtol=1e-6, atol=2e-9)
            env.unwrapped.max_steps=30
            s0,f0=evaluate_policy(model,env,1,temp/f"{arm}-a.csv.gz")
            s1,f1=evaluate_policy(loaded,env,1,temp/f"{arm}-b.csv.gz")
            assert s0==s1
            pd_assert(f0,f1)
            import gzip
            with gzip.open(temp/f"{arm}-a.csv.gz","rt") as f: trace0=f.read()
            with gzip.open(temp/f"{arm}-b.csv.gz","rt") as f: assert trace0==f.read()
            # Plot replay uses the same model.predict and raw executable environment.
            from scripts.evaluate import run_scenario
            rec,_=run_scenario(loaded,env.unwrapped,0)
            assert rec["termination_reason"]==f0.iloc[0]["reason"]
            assert np.isclose(rec["episode_reward"],s0["mean_base_reward"])
            env.close()
        old=create_old(Probe(False),1,temp/"tb","cpu")
        old.save(temp/"legacy.zip")
        reject(lambda:LatentActionPPO.load(temp/"legacy.zip"))
        reject(lambda:LatentActionPPO.load(temp/"clip.zip",arm="speed_tanh"))
        old.env.close()
        checks.append("USV base reward/motion unchanged, save/load/vectorized inference, trace and plot replay, legacy rejection")

        tracker=ResponseTracker()
        tracker.update(dict(step_count=10,pre_action_q4=1.,delta_v=-.3,actual_v=0.))
        assert tracker.onset is None
        tracker.update(dict(step_count=11,pre_action_q4=1.,delta_v=-.3,actual_v=0.))
        assert tracker.delay==0
        for i in range(12,40):
            tracker.update(dict(step_count=i,pre_action_q4=0.,delta_v=0.,actual_v=0.))
        assert tracker.max_low_run>=25
        tracker=ResponseTracker()
        tracker.update(dict(step_count=11,pre_action_q4=1.,delta_v=0.,actual_v=3.))
        assert tracker.onset==11 and tracker.delay is None

        # Reuse actual selection writer and inherited final-update scheduling.
        env=make_env()
        model=create_model(env,"clip",1)
        cb=RoundFourValidation("clip", ROOT/p["validation"]["path"],p["validation"]["dataset_hash"],
                               10000,temp/"best",temp/"validation","test",verbose=0)
        cb.init_callback(model); cb.on_training_start({}, {})
        record=dict(success_rate=.5,collision_rate=.5,mean_successful_path_efficiency=.8,mean_base_reward=1.)
        cb.record_candidate(record)
        cb.record_candidate(record)
        assert cb.history[0]["is_best"] and not cb.history[1]["is_best"]
        assert selection_key({**record,"mean_base_reward":2.})>selection_key(record)
        with patch.object(cb,"_evaluate") as evaluate:
            cb.on_training_end()
            evaluate.assert_called_once()
        cb.eval_env.close(); env.close()
        checks.append("pre-action risk timing, immediate response, censoring/stagnation, tie retention and final-update scheduling")
        # Validate seal rejection without modifying real sealed files.
        from round_four_utils import dependency_sources
        from experiment_utils import dependency_versions, atomic_write_json
        atomic_write_json(temp/"test_report.json",{"ok":True})
        fake=dict(sources=sources,dependencies=dependency_versions(),dependency_sources=dependency_sources(),
                  test_report_sha256=sha256_file(temp/"test_report.json"))
        atomic_write_json(temp/"seal.json",fake)
        verify_implementation(temp/"seal.json")
        fake["sources"]={}
        atomic_write_json(temp/"seal.json",fake)
        reject(lambda:verify_implementation(temp/"seal.json"))
        checks.append("implementation seal rejects source mismatch")
    assert core==source_manifest(ROOT) and sources==implementation_manifest()
    verify_previous()
    return dict(status="passed",checks=checks,optimizer_updates=0,formal_training_started=False,
                final_datasets_generated=False,old_core_and_round3_seal_unchanged=True)


def pd_assert(a,b):
    from pandas.testing import assert_frame_equal
    assert_frame_equal(a,b,check_exact=True)


if __name__ == "__main__":
    print(json.dumps(run_tests(),ensure_ascii=False,indent=2))
