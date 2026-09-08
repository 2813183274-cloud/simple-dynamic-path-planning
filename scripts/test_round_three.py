"""P3-04 tests: no PPO.learn, no trained weights saved, no final-test dataset reads."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.risk_reward import ARMS, RiskRewardWrapper, risk_metrics
from envs.scenario_dataset import load_dataset
from experiment_utils import sha256_file, source_manifest
from round_three_utils import load_protocol, implementation_manifest, runtime_parameters, verify_implementation, verify_round_three_model
from scripts.train import create_model
from scripts.train_round_three import RoundThreeValidation
from scripts.evaluate_round_three import evaluate_policy, selection_key
from baselines.dwa import DWA, DWAConfig


def observation(static1=(50., 50.), static2=(50., -50.), dynamic=(0., 60.), relative=(0., 0.), v=0.):
    result = np.zeros(16, dtype=np.float32)
    for i, p in zip((0, 3, 6, 9), ((80., 0.), static1, static2, dynamic)):
        result[i] = np.linalg.norm(p)/math.sqrt(20000)
        angle = math.atan2(p[1], p[0]); result[i+1:i+3] = [math.sin(angle), math.cos(angle)]
    result[12:14] = np.asarray(relative)/4.8; result[14] = v/3
    return result


def expect_error(call):
    try:
        call()
    except (ValueError, FileNotFoundError, KeyError):
        return
    raise AssertionError("Expected rejection")


def run_tests():
    protocol = load_protocol(); before = source_manifest(ROOT); checks = []
    assert risk_metrics(observation(static1=(12, 0), v=3), 4).risk == 1
    assert np.isclose(risk_metrics(observation(static1=(12, 0), v=3), 4).times[0], 4)
    assert np.isclose(risk_metrics(observation(static1=(6, 0)), 0).risk, .5, atol=1e-6)
    assert np.isclose(risk_metrics(observation(static1=(7, 0)), 0).risk, 0, atol=1e-6)
    assert risk_metrics(observation(static1=(-12, 0), v=3), 4).risk == 0
    approaching = risk_metrics(observation(dynamic=(12, 0), relative=(-3, 0)), 4)
    assert approaching.risk == 1 and np.isclose(approaching.times[2], 4)
    assert risk_metrics(observation(dynamic=(12, 0), relative=(3, 0)), 4).risk == 0
    zero = risk_metrics(observation(dynamic=(5, 0), relative=(0, 0)), 4)
    assert zero.times[2] == 0 and np.isclose(zero.risk, .5, atol=1e-6)
    expect_error(lambda: risk_metrics(np.zeros(15), 4))
    expect_error(lambda: risk_metrics(np.full(16, np.nan), 4))
    expect_error(lambda: risk_metrics(np.full(16, 1e300), 4))
    expect_error(lambda: risk_metrics(observation(), -1))
    expect_error(lambda: risk_metrics(observation(), float("inf")))
    rng = np.random.default_rng(1909)
    for _ in range(200):
        obs = observation(static1=rng.uniform(-40,40,2), dynamic=rng.uniform(-40,40,2),
                          relative=rng.uniform(-3,3,2), v=rng.uniform(0,3))
        original = obs.copy()
        for horizon in (0., 4.):
            result = risk_metrics(obs, horizon)
            assert 0 <= result.risk <= 1 and all(0 <= t <= horizon for t in result.times)
        np.testing.assert_array_equal(obs, original)
    checks.append("analytical approach/recede/zero-speed/boundary, finite bounds, pure observation function")

    # Step-by-step equivalence: base and both shaped arms have identical geometry and original reward.
    total_steps = 0
    for seed in (21,22,23):
        plain = DynamicPathPlanningEnv(stage="D")
        envs = {a: RiskRewardWrapper(DynamicPathPlanningEnv(stage="D"), a) for a in ARMS}
        obs, info = plain.reset(seed=seed)
        for env in envs.values():
            np.testing.assert_array_equal(env.reset(seed=seed)[0], obs)
        for _ in range(80):
            action = rng.uniform(-1,1,2).astype(np.float32)
            expected = plain.step(action)
            for arm, env in envs.items():
                got = env.step(action)
                np.testing.assert_array_equal(got[0], expected[0])
                assert got[2:4] == expected[2:4]
                for key, val in expected[4].items():
                    assert got[4][key] == val
                x = got[4]
                assert x["base_reward"] == expected[1]
                assert np.isclose(got[1], x["base_reward"]+x["risk_penalty"])
                assert -2 <= x["risk_penalty"] <= 0
                q = risk_metrics(got[0], ARMS[arm][0]).risk
                assert x["risk_penalty"] == (0 if got[2] or got[3] else -ARMS[arm][1]*q)
                np.testing.assert_array_equal(env.unwrapped.agent_position, plain.agent_position)
                np.testing.assert_array_equal(env.unwrapped.dynamic_position, plain.dynamic_position)
                if arm == "base": assert got[1] == expected[1]
            total_steps += 1
            if expected[2] or expected[3]: break
        plain.close()
        for env in envs.values(): env.close()
    checks.append(f"{total_steps} fixed-action transitions: original state/reward/termination and RNG unchanged in all arms")

    # Real environment terminal and timeout behavior; force geometry only in this synthetic unit test.
    for arm in ARMS:
        for kind in ("success", "collision", "timeout"):
            base = DynamicPathPlanningEnv(stage="D", max_steps=1 if kind=="timeout" else 600)
            env = RiskRewardWrapper(base, arm); env.reset(seed=55)
            if kind=="success": base.goal_position=base.agent_position.copy()
            if kind=="collision": base.static_obstacles[0]=base.agent_position.copy()
            _, reward, done, trunc, info = env.step(np.array([-1,0],dtype=np.float32))
            assert done or trunc
            assert info["risk_penalty"] == 0 and reward == info["base_reward"]
            assert np.isclose(info["episode_training_return"],info["episode_base_return"]+info["episode_risk_return"])
            env.close()
    checks.append("success/collision/timeout keep terminal reward unchanged in all arms")

    scenes = load_dataset(ROOT/"configs/round1/typical_cases.json")["scenarios"]
    original = copy.deepcopy(scenes[0]); transformed = copy.deepcopy(original)
    angle=.53; rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    for key in ("start_position","goal_position"):
        transformed[key]=(rotation@np.array(original[key])+[3,4]).tolist()
    transformed["start_heading"] += angle
    for item in transformed["static_obstacles"]+[transformed["dynamic_obstacle"]]:
        item["position"]=(rotation@np.array(item["position"])+[3,4]).tolist()
    transformed["dynamic_obstacle"]["velocity"]=(rotation@np.array(original["dynamic_obstacle"]["velocity"])).tolist()
    observations=[]
    for scene in (original,transformed):
        env=DynamicPathPlanningEnv(stage="D",randomize_scenario=False);env.set_scenario(scene);env.reset();env.current_linear_velocity=1.3
        observations.append(env._get_observation());env.close()
    np.testing.assert_allclose(observations[0],observations[1],atol=1e-6)
    assert np.isclose(risk_metrics(observations[0],4).risk,risk_metrics(observations[1],4).risk)
    checks.append("body-frame rotation and world translation invariance")

    class FakeModel:
        _n_updates=0
        def save(self,path):
            Path(str(path)+".zip").write_bytes(f"fake-{self._n_updates}".encode())
    record={"success_rate":.5,"collision_rate":.5,"mean_successful_path_efficiency":.9,
            "mean_base_reward":100.,"mean_training_reward":-999.}
    assert selection_key(record)==selection_key({**record,"mean_training_reward":999999.})
    assert selection_key({**record,"mean_base_reward":101})>selection_key(record)
    assert selection_key({**record,"mean_successful_path_efficiency":None})[2]==0
    with tempfile.TemporaryDirectory() as temporary:
        out=Path(temporary); (out/"best").mkdir();(out/"validation").mkdir()
        cb=RoundThreeValidation("predictive",ROOT/protocol["validation"]["path"],protocol["validation"]["dataset_hash"],
                                10000,out/"best",out/"validation","test",verbose=0)
        cb.model=FakeModel();cb.num_timesteps=10000
        cb.record_candidate(record)
        first=sha256_file(out/"best/best_model.zip")
        cb.model._n_updates=1
        cb.record_candidate({**record,"mean_training_reward":999999})
        assert not cb.history[-1]["is_best"] and sha256_file(out/"best/best_model.zip")==first
        cb.record_candidate({**record,"mean_base_reward":101})
        assert cb.history[-1]["is_best"]
        cb._evaluate=lambda: cb.record_candidate({**record,"mean_base_reward":102})
        cb._on_training_end()
        assert cb.history[-1]["policy_updates"]==1 and cb.history[-1]["timesteps"]==10000
        # Certification rejects wrong source/dependencies; test before any formal run exists.
        from experiment_utils import atomic_write_json,dependency_versions
        fake_seal=out/"seal.json"
        atomic_write_json(out/"test_report.json",{"status":"test-fixture"})
        atomic_write_json(fake_seal,{"sources":implementation_manifest(),"dependencies":dependency_versions(),
                                    "test_report_sha256":sha256_file(out/"test_report.json")})
        verify_implementation(fake_seal)
        altered=implementation_manifest();altered["envs/risk_reward.py"]="tampered"
        atomic_write_json(fake_seal,{"sources":altered,"dependencies":dependency_versions()})
        expect_error(lambda: verify_implementation(fake_seal))
    checks.append("common base-return ranking, exact tie retains old model, final callback always evaluates updated policy, source tamper rejected")

    # Initialize (but DO NOT train) a real PPO and replay one development episode for each arm.
    with tempfile.TemporaryDirectory() as temporary:
        init=RiskRewardWrapper(DynamicPathPlanningEnv(stage="D"),"base")
        model=create_model(init,1,Path(temporary)/"tensorboard","cpu")
        params=runtime_parameters(model); assert params["observation_shape"]==[16]
        assert model.num_timesteps==0 and model._n_updates==0
        expected=None
        for arm in ARMS:
            env=RiskRewardWrapper(DynamicPathPlanningEnv(scenario_file=ROOT/"configs/round1/typical_cases.json"),arm)
            summary,frame=evaluate_policy(model,env,1,Path(temporary)/f"{arm}.csv.gz")
            if expected is None:expected=summary["mean_base_reward"]
            assert summary["mean_base_reward"]==expected
            env.close()
        assert model.num_timesteps==0 and model._n_updates==0
        # Real callback initialization and evaluation scheduling, still without optimizer updates.
        callback=RoundThreeValidation("predictive",ROOT/protocol["validation"]["path"],
            protocol["validation"]["dataset_hash"],10000,Path(temporary)/"cb/best",Path(temporary)/"cb/log","callback-test",verbose=0)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        assert isinstance(callback.eval_env,RiskRewardWrapper)
        callback.num_scenarios=1  # Development smoke only, not a formal validation claim.
        callback.num_timesteps=10000
        assert callback._on_step()
        assert callback.history[-1]["selection_reward"]=="base_reward"
        callback.eval_env.close()
        assert model._n_updates==0
        init.close()
        # Exercise the real training startup, snapshots and failure cleanup, with learning forbidden.
        import scripts.train_round_three as trainer
        fixture=Path(temporary)/"startup_seal.json"
        seal={"sources":implementation_manifest(),"runtime_parameters":params}
        atomic_write_json(fixture,seal)
        with patch.object(trainer,"verify_implementation",return_value=seal), patch.object(trainer,"SEAL",fixture), \
                patch("stable_baselines3.PPO.learn",side_effect=RuntimeError("test: learning intentionally disabled")):
            try:
                trainer.train("base",1,run_root=Path(temporary)/"startup")
            except RuntimeError as error:
                assert str(error)=="test: learning intentionally disabled"
            else:
                raise AssertionError("Startup test attempted real training")
        startup=Path(temporary)/"startup/round3-base-seed1"
        metadata=json.loads((startup/"metadata.json").read_text())
        assert metadata["status"]=="failed"
        assert (startup/"snapshot/envs/risk_reward.py").exists()
        assert not (startup/"models/final_model.zip").exists()
        # Synthetic certification fixture: not a trained model and never leaves TemporaryDirectory.
        (startup/"models/best").mkdir(parents=True)
        (startup/"logs/validation").mkdir(parents=True)
        for relative in ("models/final_model.zip","models/best/best_model.zip"):
            (startup/relative).write_bytes(b"synthetic certification test, not PPO weights")
        atomic_write_json(startup/"models/best/selection_metrics.json",{"test":True})
        atomic_write_json(startup/"logs/validation/evaluations.json",[])
        metadata.update(status="completed",final_num_timesteps=200704,
            final_model_sha256=sha256_file(startup/"models/final_model.zip"),
            best_model_sha256=sha256_file(startup/"models/best/best_model.zip"),
            selection_metrics_sha256=sha256_file(startup/"models/best/selection_metrics.json"),
            validation_history_sha256=sha256_file(startup/"logs/validation/evaluations.json"),
            training_monitor_sha256=sha256_file(startup/"logs/train_monitor.csv"))
        atomic_write_json(startup/"metadata.json",metadata)
        import round_three_utils as identities
        with patch.object(identities,"verify_implementation",return_value=seal),patch.object(identities,"SEAL",fixture):
            assert verify_round_three_model(startup/"models/best/best_model.zip")[0]==startup.resolve()
            (startup/"models/best/best_model.zip").write_bytes(b"changed")
            expect_error(lambda: verify_round_three_model(startup/"models/best/best_model.zip"))
            expect_error(lambda: verify_round_three_model(startup/"models/uncertified.zip"))
        # Existing DWA, no tuning: all three rewards must return identical physical outcomes.
        expected=None
        for arm in ARMS:
            env=RiskRewardWrapper(DynamicPathPlanningEnv(scenario_file=ROOT/"configs/round1/typical_cases.json"),arm)
            s,frame=evaluate_policy(DWA(DWAConfig(**protocol["dwa"]["config"])),env,3)
            result=(s["success_count"],s["mean_base_reward"])
            if expected is None:expected=result
            assert result==expected
            env.close()
    checks.append("untrained PPO and existing DWA development replays; identical common returns; startup/snapshot/failure cleanup with learning disabled; zero optimizer updates")
    assert source_manifest(ROOT)==before
    return {"status":"passed","checks":checks,"formal_training_started":False,"optimizer_updates":0,
            "final_test_data_read":False,"baseline_core_sources_unchanged":True}


if __name__=="__main__":
    print(json.dumps(run_tests(),ensure_ascii=False,indent=2))
