"""Versioned, type-conditioned random encounters and executable feasibility witnesses."""
from collections import Counter
import math
import numpy as np

VERSION = "encounters-v1"
STAGES = ("legacy", "A", "B", "C", "D")
CONFIG = {
    "version": VERSION, "static_radius": 4.0, "dynamic_radius": 3.0,
    "velocity_scale": 4.8, "target_max_speed": 1.8,
    "types": ["head_on", "crossing_left", "crossing_right", "overtaking"],
    "type_probabilities": [0.25, 0.25, 0.25, 0.25],
    "risks": ["conflict", "time_separated", "receding"],
    "risk_probabilities": [0.6, 0.25, 0.15],
    "layouts": ["clear", "blocking", "mixed", "two_sides"],
    "layout_probabilities": [0.3, 0.2, 0.3, 0.2],
    "layout_risk_rule": "D blocking: time_separated/receding 0.625/0.375; mixed: conflict; others: risk_probabilities",
    "target_speed_range": [0.6, 1.5], "meeting_time_range": [12.0, 26.0],
    "crossing_angle_degrees": [55.0, 125.0], "nominal_speed": 2.0,
    "nominal_acceleration": 1.5, "time_horizon": 120.0,
    "max_attempts": 1500, "exit_behavior": "continue_straight_outside_map",
    "feasibility": "executed waypoint-controller witness, dt=0.2, same speed/acceleration limits",
}


def schema(stage):
    return "body-fixedscale-center16-v2" if stage in ("B", "C", "D") else "center-distance-16-v1"


def segment_distance(p, a, b):
    v = b-a
    t = np.clip(np.dot(p-a, v)/max(float(v@v), 1e-12), 0, 1)
    return float(np.linalg.norm(p-a-t*v))


def cpa(position, relative_velocity):
    speed2 = float(relative_velocity @ relative_velocity)
    if speed2 < 1e-12:
        return None, float(np.linalg.norm(position))
    t = -float(position @ relative_velocity)/speed2
    return t, float(np.linalg.norm(position+max(0.0, t)*relative_velocity))


def witness(s):
    """Find one executable path; failure means reject, not proof of impossibility."""
    start, goal = np.array(s["start_position"]), np.array(s["goal_position"])
    e = (goal-start)/np.linalg.norm(goal-start)
    n = np.array([-e[1], e[0]])
    statics = [(float(o["position"][0]), float(o["position"][1]), o["radius"]+1)
               for o in s["static_obstacles"]]
    d = s["dynamic_obstacle"]
    for offset in (0, 16, -16, 25, -25):
        points = [start+12*e+offset*n, goal-12*e+offset*n, goal] if offset else [goal]
        if any(np.any(p < 3) or np.any(p > 97) for p in points):
            continue
        x,y = start
        h = s["start_heading"]
        v=w=0.0
        k=0
        minimum=math.inf
        for step in range(1,601):
            target=points[k]
            if math.hypot(target[0]-x,target[1]-y)<3 and k<len(points)-1:
                k+=1
                target=points[k]
            error=(math.atan2(target[1]-y,target[0]-x)-h+math.pi)%(2*math.pi)-math.pi
            vt=2.0 if abs(error)<0.35 else 1.0
            wt=max(-math.pi/4,min(math.pi/4,1.5*error))
            v=max(v-.3,min(v+.3,vt))
            w=max(w-math.pi/20,min(w+math.pi/20,wt))
            x+=v*.2*math.cos(h); y+=v*.2*math.sin(h)
            h+=w*.2
            dx=d["position"][0]+step*.2*d["velocity"][0]
            dy=d["position"][1]+step*.2*d["velocity"][1]
            clear=min([math.hypot(x-a,y-b)-r for a,b,r in statics]+[
                math.hypot(x-dx,y-dy)-d["radius"]-1, x-1,99-x,y-1,99-y])
            minimum=min(minimum,clear)
            if clear<=0.5:
                break
            if math.hypot(goal[0]-x,goal[1]-y)<=3:
                return {"offset": offset, "steps": step, "minimum_clearance": minimum,
                        "waypoints": [p.tolist() for p in points]}
    return None


def geometry_error(s):
    start, goal = np.array(s["start_position"]), np.array(s["goal_position"])
    all_obs=s["static_obstacles"]+[s["dynamic_obstacle"]]
    for o in all_obs:
        p=np.array(o["position"]); r=o["radius"]
        if np.any(p<r+1) or np.any(p>99-r): return "outside"
        if min(np.linalg.norm(p-start),np.linalg.norm(p-goal))<r+7: return "endpoint"
    for i,o in enumerate(all_obs):
        for q in all_obs[i+1:]:
            if np.linalg.norm(np.array(o["position"])-q["position"])<o["radius"]+q["radius"]+3:
                return "overlap"
    d=s["dynamic_obstacle"]; a=np.array(d["position"]); b=a+120*np.array(d["velocity"])
    for o in s["static_obstacles"]:
        if segment_distance(np.array(o["position"]),a,b)<o["radius"]+d["radius"]+1:
            return "target_static_sweep"
    return None


def sample(rng, stage="D", stats=None, requested=None, challenge=False):
    stats = stats if stats is not None else Counter()
    kind,risk,layout = requested or (
        str(rng.choice(CONFIG["types"],p=CONFIG["type_probabilities"])),
        str(rng.choice(CONFIG["risks"],p=CONFIG["risk_probabilities"])),
        "blocking" if stage=="C" else str(rng.choice(CONFIG["layouts"],p=CONFIG["layout_probabilities"])))
    if requested is None and stage=="D":
        if layout=="blocking": risk=str(rng.choice(["time_separated","receding"],p=[.625,.375]))
        if layout=="mixed": risk="conflict"
    # Keep the selected stratum on rejection: valid easy strata must not crowd out hard ones.
    for _ in range(CONFIG["max_attempts"]):
        stats["attempts"]+=1
        start=rng.uniform([7,40],[13,60]); goal=rng.uniform([87,40],[93,60])
        e=(goal-start)/np.linalg.norm(goal-start); n=np.array([-e[1],e[0]])
        h=math.atan2(e[1],e[0])
        angle = (rng.uniform(160,200) if kind=="head_on" else
                 rng.uniform(-12,12) if kind=="overtaking" else
                 rng.uniform(35,50) if challenge else rng.uniform(55,125))
        if kind=="crossing_left": angle=-angle
        speed=float(rng.uniform(.3,.55) if challenge and kind=="overtaking" else
                    rng.uniform(1.55,1.8) if challenge else rng.uniform(.6,1.5))
        theta=h+math.radians(angle); vel=speed*np.array([math.cos(theta),math.sin(theta)])
        t=float(rng.uniform(20,26) if challenge and kind=="overtaking" else
                rng.uniform(9,12) if challenge else rng.uniform(12,26))
        meet=start+(2*t-4/3)*e
        p=meet-vel*t
        if risk=="time_separated": p+=n*float(rng.choice([-1,1]))*rng.uniform(13,22)
        if risk=="receding":
            p=start-rng.uniform(1,3)*e+math.copysign(rng.uniform(20,30),float(vel@n))*n
        positions=[]
        for idx,fraction in enumerate((.32,.7)):
            if kind=="overtaking" and idx==0 and layout in ("blocking","mixed"):
                fraction=.18
            off = rng.choice([-1,1])*rng.uniform(15,23)
            blocking_index = 1 if kind=="head_on" else 0
            if layout in ("blocking","mixed") and idx==blocking_index: off=rng.uniform(-2,2)
            if layout=="two_sides": off=(1 if idx==0 else -1)*rng.uniform(10,15)
            positions.append(start+(fraction*np.linalg.norm(goal-start)+rng.uniform(-4,4))*e+off*n)
        s={"scenario_id":-1,"scenario_type":kind,"risk_level":risk,"layout_type":layout,
           "difficulty":"challenge" if challenge else "standard", "environment_stage":stage,
           "start_position":start.tolist(),"goal_position":goal.tolist(),"start_heading":h,
           "static_obstacles":[{"position":q.tolist(),"radius":4.0} for q in positions],
           "dynamic_obstacle":{"position":p.tolist(),"radius":3.0,"velocity":vel.tolist()}}
        error=geometry_error(s)
        if error: stats[error]+=1; continue
        times=np.arange(0,41,.2)
        displacement=np.where(times<4/3,.75*times**2,2*times-4/3)
        nominal=start+displacement[:,None]*e
        target=p+times[:,None]*vel
        distances=np.linalg.norm(nominal-target,axis=1)
        minimum=float(distances.min()); closest=float(times[distances.argmin()])
        if risk=="conflict" and not (minimum<3.5 and closest>6):
            stats["risk_mismatch"]+=1; continue
        if risk!="conflict" and minimum<10:
            stats["risk_mismatch"]+=1; continue
        if risk=="receding" and float((p-start)@(vel-2*e))<=0:
            stats["risk_mismatch"]+=1; continue
        certificate=witness(s)
        if certificate is None: stats["no_witness"]+=1; continue
        tcpa,dcpa=cpa(p-start,vel-2*e)
        s["generation"]={"nominal_min_center_distance":minimum,"nominal_closest_time":closest,
            "tcpa_constant_speed":tcpa,"dcpa_constant_speed":dcpa,"target_speed":speed,
            "relative_heading_degrees":math.degrees(math.atan2(vel@n,vel@e)),
            "witness":certificate,"version":VERSION}
        stats["accepted"]+=1
        stats[f"type:{kind}"]+=1; stats[f"risk:{risk}"]+=1; stats[f"layout:{layout}"]+=1
        return s
    raise RuntimeError(f"No valid scene for {kind}/{risk}/{layout}; stats={dict(stats)}")
