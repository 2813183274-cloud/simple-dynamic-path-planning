"""Data protocol for stage ablations; final test seeds are reserved, not evaluated."""
from collections import Counter
import copy
import numpy as np
from .encounters import CONFIG, sample, geometry_error, schema
from .scenario_dataset import scenario_hash


def serialize_env(env):
    return {"start_position":env.start_position.tolist(), "goal_position":env.goal_position.tolist(),
        "start_heading":env.start_heading,
        "static_obstacles":[{"position":p.tolist(),"radius":float(r)} for p,r in zip(env.static_obstacles,env.static_radii)],
        "dynamic_obstacle":{"position":env.dynamic_initial_position.tolist(),
            "velocity":env.dynamic_initial_velocity.tolist(),"radius":env.dynamic_radius}}


def generate(stage, seed, count=60, split="validation", challenge=False):
    from .dynamic_path_env import DynamicPathPlanningEnv
    rng=np.random.default_rng(seed); stats=Counter(); scenes=[]
    env=DynamicPathPlanningEnv(stage=stage)
    for i in range(count):
        if stage in ("A","B"):
            env.reset(seed=seed if i==0 else None)
            s=serialize_env(env)
            s.update(scenario_type="crossing_up" if env.dynamic_initial_velocity[1]>0 else "crossing_down",
                     risk_level="legacy_corridor",layout_type="blocking",difficulty="standard",environment_stage=stage)
        else:
            s=sample(rng,stage,stats,challenge=challenge)
        s["scenario_id"]=i; s["scenario_hash"]=scenario_hash(s); scenes.append(s)
    env.close()
    return {"schema_version":3,"dataset_name":f"round1-{stage}-{split}-{seed}",
        "dataset_split":split,"dataset_seed":seed,"num_scenarios":count,"environment_stage":stage,
        "observation_schema":schema(stage),"generation_config":copy.deepcopy(CONFIG),
        "challenge":challenge,"sampling_report":dict(stats),"scenarios":scenes}


def validate(dataset):
    stage=dataset.get("environment_stage")
    if stage not in ("A","B","C","D"): raise ValueError("Invalid stage")
    if dataset.get("generation_config")!=CONFIG: raise ValueError("Generator contract changed")
    if dataset.get("dataset_split") not in ("validation","diagnostic","test"): raise ValueError("Invalid split")
    scenes=dataset["scenarios"]
    if not scenes or len(scenes)!=dataset["num_scenarios"]: raise ValueError("Invalid count")
    hashes=[]
    for i,s in enumerate(scenes):
        if s["scenario_id"]!=i or s["environment_stage"]!=stage: raise ValueError("Scene identity mismatch")
        expected=scenario_hash(s)
        if s["scenario_hash"]!=expected: raise ValueError("Scene hash mismatch")
        hashes.append(expected)
        if len(s["static_obstacles"])!=2: raise ValueError("Expected two statics")
        if any(o["radius"]!=4 for o in s["static_obstacles"]) or s["dynamic_obstacle"]["radius"]!=3:
            raise ValueError("Fixed radius contract violated")
        if not np.isfinite(np.array(s["dynamic_obstacle"]["velocity"])).all(): raise ValueError("Nonfinite speed")
        if stage in ("C","D"):
            error=geometry_error(s)
            if error: raise ValueError(error)
            if s["scenario_type"] not in CONFIG["types"] or s["risk_level"] not in CONFIG["risks"]:
                raise ValueError("Invalid encounter label")
            if np.linalg.norm(s["dynamic_obstacle"]["velocity"])>CONFIG["target_max_speed"]+1e-9:
                raise ValueError("Target speed exceeds normalization contract")
            if s["generation"]["witness"]["minimum_clearance"]<=.5: raise ValueError("Invalid witness")
    if len(set(hashes))!=len(hashes): raise ValueError("Duplicate geometry")
