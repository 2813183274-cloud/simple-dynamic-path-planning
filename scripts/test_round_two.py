"""Checks for observation-only baseline and round-two statistics (no final test tuning)."""
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.dwa import DWA
from envs import DynamicPathPlanningEnv


def main():
    env=DynamicPathPlanningEnv(stage="D")
    policy=DWA()
    for seed in range(5):
        obs,_=env.reset(seed=seed)
        for _ in range(15):
            goal,objects,velocity,v,w=policy.decode(obs)
            c,s=math.cos(env.agent_heading),math.sin(env.agent_heading)
            rotation=np.array([[c,s],[-s,c]])
            np.testing.assert_allclose(goal,rotation@(env.goal_position-env.agent_position),atol=1e-5)
            np.testing.assert_allclose(objects[:2],(env.static_obstacles-env.agent_position)@rotation.T,atol=1e-5)
            np.testing.assert_allclose(velocity,rotation@env.dynamic_velocity,atol=1e-6)
            action,_=policy.predict(obs)
            assert action.shape==(2,) and np.isfinite(action).all() and (np.abs(action)<=1).all()
            np.testing.assert_array_equal(action,DWA().predict(obs.copy())[0])
            assert abs((action[0]+1)*1.5-v)<=.300001
            assert abs(action[1]*math.pi/4-w)<=math.pi/20+1e-6
            obs,_,done,trunc,_=env.step(action)
            if done or trunc:break
    # An obstacle directly ahead causes braking/steering rather than a straight full-speed command.
    obs=np.zeros(16,dtype=np.float32)
    for i,d in zip((0,3,6,9),(70,7,90,100)):
        obs[i]=d/math.sqrt(20000);obs[i+2]=1
    obs[14]=1;obs[12]=-3/4.8
    action,_=policy.predict(obs)
    assert action[0]<.999 or abs(action[1])>.01
    for invalid in (np.zeros(15),np.full(16,np.nan)):
        try:policy.predict(invalid)
        except ValueError:pass
        else:raise AssertionError("Invalid observation accepted")
    env.close()
    from scripts.summarize_round_two import paired_interval
    kinds=np.array(["a","a","b","b"])
    result=paired_interval(np.zeros((3,4)),kinds,repetitions=100)
    assert result["mean_difference"]==0 and result["bootstrap_95"]==[0,0]
    result=paired_interval(np.ones((3,4)),kinds,repetitions=100)
    assert result["bootstrap_95"]==[1,1]
    from scripts.round_two_experiment import definitions, evaluate
    from experiment_utils import verify_model_identity
    import tempfile
    for run in definitions().values():
        verify_model_identity(run/"models/best/best_model.zip",ROOT,allow_archived_source=True)
    # Exercise the complete export path on development data, not sealed final outcomes.
    with tempfile.TemporaryDirectory() as directory:
        summary=evaluate(DWA(),ROOT/"configs/round1/typical_cases.json",Path(directory)/"smoke",save_examples=True)
        assert summary["total_scenarios"]==12
    print("Round-two DWA decoding, limits, deterministic observation-only action, avoidance and input tests passed.")


if __name__=="__main__":main()
