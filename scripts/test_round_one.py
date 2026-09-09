import copy
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from envs import DynamicPathPlanningEnv
from envs.encounters import sample, CONFIG, cpa
from envs.scenario_dataset import load_dataset, scenario_hash


def geometry_hash(s):
    s=copy.deepcopy(s);s.pop("environment_stage",None)
    return scenario_hash(s)


def replay_witness(scene):
    env=DynamicPathPlanningEnv(randomize_scenario=False)
    env.set_scenario(scene);env.reset()
    points=scene["generation"]["witness"]["waypoints"];k=0
    for _ in range(600):
        target=np.array(points[k])
        if np.linalg.norm(target-env.agent_position)<3 and k<len(points)-1:
            k+=1;target=np.array(points[k])
        delta=target-env.agent_position
        error=env._normalize_angle(math.atan2(delta[1],delta[0])-env.agent_heading)
        vt=2 if abs(error)<.35 else 1
        wt=np.clip(1.5*error,-math.pi/4,math.pi/4)
        _,_,done,trunc,info=env.step(np.array([2*vt/3-1,wt/(math.pi/4)],dtype=np.float32))
        if done or trunc:break
    env.close()
    assert info["is_success"], (scene["scenario_id"],info["termination_reason"])


def main():
    datasets={p.stem:load_dataset(p) for p in (ROOT/"configs/round1").glob("*.json")
              if p.stem.startswith("validation_") or p.stem in ("common_fixed","common_encounters","typical_cases","challenge_development")}
    sets={name:{geometry_hash(s) for s in d["scenarios"]} for name,d in datasets.items()}
    for a in sets:
        for b in sets:
            if a<b and {a,b}!={"validation_A","validation_B"}:
                assert not sets[a]&sets[b], (a,b,"overlap")
    assert sets["validation_A"]==sets["validation_B"]
    for name,d in datasets.items():
        if d["environment_stage"] in ("C","D"):
            for s in d["scenarios"]:replay_witness(s)
    r1=np.random.default_rng(81);r2=np.random.default_rng(81)
    for _ in range(30):assert sample(r1)==sample(r2)
    scene=datasets["validation_D"]["scenarios"][0]
    a=DynamicPathPlanningEnv(randomize_scenario=False);a.set_scenario(scene);a.reset()
    a.current_linear_velocity=1.1
    original=a._get_observation()
    rotated=copy.deepcopy(scene);angle=.63
    R=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    for key in ("start_position","goal_position"):
        rotated[key]=(R@(np.array(scene[key])-50)+50).tolist()
    rotated["start_heading"]+=angle
    for o in rotated["static_obstacles"]+[rotated["dynamic_obstacle"]]:
        o["position"]=(R@(np.array(o["position"])-50)+50).tolist()
    rotated["dynamic_obstacle"]["velocity"]=(R@scene["dynamic_obstacle"]["velocity"]).tolist()
    b=DynamicPathPlanningEnv(randomize_scenario=False);b.set_scenario(rotated);b.reset();b.current_linear_velocity=1.1
    np.testing.assert_allclose(original,b._get_observation(),atol=1e-6)
    a.dynamic_position=np.array([98.,50.]);a.dynamic_velocity=np.array([1.,0.]);a._update_dynamic_obstacle()
    np.testing.assert_allclose(a.dynamic_position,[98.2,50]);assert a.dynamic_velocity[0]==1
    a.dynamic_obstacle_max_speed=0
    np.testing.assert_allclose(a._get_observation()[12:14],
        np.array([[math.cos(a.agent_heading),math.sin(a.agent_heading)],[-math.sin(a.agent_heading),math.cos(a.agent_heading)]])@
        (a.dynamic_velocity-a.current_linear_velocity*np.array([math.cos(a.agent_heading),math.sin(a.agent_heading)]))/4.8,atol=1e-6)
    assert cpa(np.array([1.,0.]),np.zeros(2))[0] is None
    a.close();b.close()
    print("Round-one checks passed: all fixed witnesses replay, split separation, seeded diversity, body-frame rotation, fixed scale, no bounce.")


if __name__=="__main__":main()
