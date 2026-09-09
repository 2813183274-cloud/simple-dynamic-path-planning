"""DWA-style local planner with constant-velocity target prediction and braking rollout.

Based on the velocity-window/admissibility principles of Fox, Burgard & Thrun (1997).
This is a project-specific implementation, not a reproduction of their original code.
Only the same 16-D observation supplied to the PPO policy is consumed.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DWAConfig:
    horizon: float = 4.0
    clearance_weight: float = 1.0
    heading_weight: float = 0.5
    speed_weight: float = 0.1
    margin: float = 0.5
    v_samples: int = 7
    w_samples: int = 21
    dt: float = 0.2
    v_max: float = 3.0
    w_max: float = math.pi / 4
    acceleration: float = 1.5
    angular_acceleration: float = math.pi / 4


class DWA:
    def __init__(self, config=DWAConfig()):
        self.config=config
        self.emergency_steps=0

    @staticmethod
    def decode(observation):
        o=np.asarray(observation,dtype=float)
        if o.shape!=(16,) or not np.all(np.isfinite(o)):
            raise ValueError("DWA requires a finite 16-vector")
        objects=[]
        for i in (0,3,6,9):
            objects.append(math.sqrt(20000)*o[i]*np.array([o[i+2],o[i+1]]))
        v=o[14]*3.0;w=o[15]*math.pi/4
        target_velocity=o[12:14]*4.8+np.array([v,0.0])
        return objects[0],np.array(objects[1:]),target_velocity,v,w

    def predict(self, observation, deterministic=True):
        c=self.config
        goal,obstacles,target_velocity,v0,w0=self.decode(observation)
        vs=np.linspace(max(0,v0-c.acceleration*c.dt),min(c.v_max,v0+c.acceleration*c.dt),c.v_samples)
        ws=np.unique(np.r_[np.linspace(max(-c.w_max,w0-c.angular_acceleration*c.dt),
                    min(c.w_max,w0+c.angular_acceleration*c.dt),c.w_samples),
                    np.clip(0,w0-c.angular_acceleration*c.dt,w0+c.angular_acceleration*c.dt)])
        v,w=np.meshgrid(vs,ws,indexing="ij");v=v.ravel();w=w.ravel()
        x=np.zeros_like(v);y=x.copy();heading=x.copy();minimum=np.full_like(v,np.inf)
        rollout_steps=int(round(c.horizon/c.dt))
        vv=v.copy();ww=w.copy()
        radii=np.array([5.0,5.0,4.0]) # fixed obstacle radius plus 1m USV footprint
        for step in range(rollout_steps+int(math.ceil(c.v_max/c.acceleration/c.dt))+1):
            if step>=rollout_steps:
                vv=np.maximum(0,vv-c.acceleration*c.dt)
                ww=np.sign(ww)*np.maximum(0,np.abs(ww)-c.angular_acceleration*c.dt)
            x+=vv*c.dt*np.cos(heading);y+=vv*c.dt*np.sin(heading)
            heading+=ww*c.dt
            centers=obstacles.copy()
            centers[2]+=target_velocity*(step+1)*c.dt
            clearance=np.sqrt((x[:,None]-centers[:,0])**2+(y[:,None]-centers[:,1])**2)-radii
            minimum=np.minimum(minimum,clearance.min(axis=1))
            if step==rollout_steps-1:
                endpoint=np.column_stack([x,y]);end_heading=heading.copy()
        progress=(np.linalg.norm(goal)-np.linalg.norm(goal-endpoint,axis=1))/c.horizon
        angle=np.arctan2(goal[1]-endpoint[:,1],goal[0]-endpoint[:,0])-end_heading
        angle=(angle+math.pi)%(2*math.pi)-math.pi
        score=progress+c.speed_weight*v-c.heading_weight*np.abs(angle)/math.pi
        score-=c.clearance_weight/(np.maximum(minimum,0)+0.2)
        valid=minimum>c.margin
        if valid.any():
            score[~valid]=-np.inf
            chosen=int(np.argmax(score))
        else:
            self.emergency_steps+=1
            # No safe candidate in the window: decelerate maximally and maximize clearance.
            candidates=np.flatnonzero(np.isclose(v,v.min()))
            chosen=int(candidates[np.argmax(minimum[candidates])])
        return np.array([2*v[chosen]/c.v_max-1,w[chosen]/c.w_max],dtype=np.float32),None

    def specification(self):
        return asdict(self.config)
