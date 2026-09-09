"""Preregistered seed-level and paired stratified hierarchical statistics.

Bootstrap intervals are descriptive, not multiplicity-adjusted hypothesis tests.
Scenario IDs are paired across policies; three training seeds are NOT 720 independent runs.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_utils import atomic_write_json,sha256_file

OUT=ROOT/"results/round2"


def paired_interval(difference,types,repetitions=10000,seed=4821):
    difference=np.asarray(difference,dtype=float)
    types=np.asarray(types)
    if difference.ndim!=2 or difference.shape[1]!=len(types):raise ValueError("Unpaired data")
    groups=[np.flatnonzero(types==t) for t in sorted(set(types))]
    rng=np.random.default_rng(seed);values=np.empty(repetitions)
    for i in range(repetitions):
        seeds=rng.integers(difference.shape[0],size=difference.shape[0])
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values[i]=difference[np.ix_(seeds,indices)].mean()
    return {"mean_difference":float(difference.mean()),
            "bootstrap_95":np.quantile(values,[.025,.975]).tolist(),
            "bootstrap_repetitions":repetitions,"bootstrap_seed":seed}


def main():
    protocol=json.loads((OUT/"protocol.json").read_text())
    if sha256_file(Path(__file__))!=protocol["statistics_source_sha256"]:
        raise ValueError("Statistics code differs from sealed plan")
    summaries=[];types_all=[];failure_rows=[];paired=[];seed_stats=[]
    for split in ("same_distribution","challenge"):
        frames={}
        for name in [*protocol["models"],"DWA"]:
            folder=OUT/"final"/split/name
            complete=json.loads((folder/"complete.json").read_text())
            if complete["protocol_sha256"]!=sha256_file(OUT/"protocol.json"):raise ValueError("Protocol mismatch")
            for relative,digest in complete["artifacts"].items():
                if sha256_file(folder/relative)!=digest:raise ValueError("Artifact mismatch")
            summary=json.loads((folder/"summary.json").read_text())
            model=protocol["models"].get(name,{})
            summaries.append({"split":split,"method":name,"best_timesteps":model.get("best_timesteps"),
                **{k:v for k,v in summary.items() if not isinstance(v,(dict,list))}})
            f=pd.read_csv(folder/"episodes.csv").sort_values("scenario_id").reset_index(drop=True)
            frames[name]=f
            for kind,g in f.groupby("scenario_type"):
                types_all.append({"split":split,"method":name,"scenario_type":kind,"n":len(g),
                    "success_rate":float(g.success.mean()),
                    "static_collision_rate":float(g.collision_type.str.startswith("static_collision").mean()),
                    "dynamic_collision_rate":float(g.collision_type.eq("dynamic_collision").mean())})
            for reason,g in f[~f.success].groupby("termination_reason"):
                failure_rows.append({"split":split,"method":name,"reason":reason,"count":len(g),
                    "scenario_ids":" ".join(map(str,g.scenario_id.tolist()))})
        anchor=frames["DWA"]
        for f in frames.values():
            assert np.array_equal(f.scenario_id,anchor.scenario_id)
            assert np.array_equal(f.scenario_type,anchor.scenario_type)
        for stage in ("C","D"):
            group=[next(s for s in summaries if s["split"]==split and s["method"]==f"{stage}-seed{i}") for i in (1,2,3)]
            for metric in ("success_rate","collision_rate","static_collision_rate","dynamic_collision_rate",
                           "successful_average_path_efficiency","average_navigation_time","average_reward",
                           "average_min_static_clearance","average_min_dynamic_clearance","mean_inference_ms"):
                vals=np.array([s[metric] for s in group],dtype=float)
                mean=float(vals.mean());sd=float(vals.std(ddof=1));half=4.3026527299*sd/np.sqrt(3)
                low,high=mean-half,mean+half
                if metric.endswith("_rate") or metric.endswith("efficiency"):low=max(0.,low);high=min(1.,high)
                seed_stats.append({"split":split,"stage":stage,"metric":metric,"n_seeds":3,
                    "mean":mean,"sample_sd":sd,"seed_t_95_low":low,"seed_t_95_high":high})
        matrices={stage:np.stack([frames[f"{stage}-seed{i}"].success.to_numpy(dtype=float) for i in (1,2,3)]) for stage in ("C","D")}
        for left,right in (("D","C"),("C","DWA"),("D","DWA")):
            other=anchor.success.to_numpy(dtype=float)[None,:] if right=="DWA" else matrices[right]
            paired.append({"split":split,"comparison":f"{left} minus {right}",
                **paired_interval(matrices[left]-other,anchor.scenario_type)})
    pd.DataFrame(summaries).to_csv(OUT/"per_seed_metrics.csv",index=False)
    pd.DataFrame(seed_stats).to_csv(OUT/"seed_statistics.csv",index=False)
    bytype=pd.DataFrame(types_all);bytype.to_csv(OUT/"per_seed_by_type.csv",index=False)
    primary=bytype[bytype.method.str.match(r"[CD]-seed")].copy()
    primary["stage"]=primary.method.str[0]
    primary.groupby(["split","stage","scenario_type"])[["success_rate","static_collision_rate","dynamic_collision_rate"]].agg(["mean","std"]).to_csv(OUT/"seed_statistics_by_type.csv")
    pd.DataFrame(failure_rows).to_csv(OUT/"failure_catalog.csv",index=False)
    atomic_write_json(OUT/"paired_comparisons.json",paired)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,split in zip(axes,("same_distribution","challenge")):
        values=[]
        for stage in ("C","D"):
            values.append([s["success_rate"]*100 for s in summaries if s["split"]==split and s["method"].startswith(stage+"-seed")])
        dwa=next(s["success_rate"]*100 for s in summaries if s["split"]==split and s["method"]=="DWA")
        ax.bar([0,1,2],[np.mean(v) for v in values]+[dwa],color=["#4477aa","#ee8833","#228833"],alpha=.65)
        for x,v in enumerate(values):ax.scatter(x+np.linspace(-.12,.12,3),v,color="black",s=25,zorder=3)
        ax.set_xticks([0,1,2],["PPO C (3 seeds)","PPO D (3 seeds)","DWA"])
        ax.set_ylim(0,105);ax.set_ylabel("Success (%)");ax.set_title(split.replace("_"," "))
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Sealed final evaluation | bars: seed means; dots: individual seeds")
    fig.savefig(OUT/"success_comparison.png",dpi=180);plt.close(fig)
    # Each run already contains its actual random-training reward curve; no shaded SD bands.
    rows=["# Round-two sealed evaluation","","C/D: three matched training seeds; DWA: deterministic baseline.",
          "Means and sample SD below measure seed variation; episode Wilson intervals are in each summary.json.",
          "Three seeds give imprecise uncertainty estimates; no significance or convergence claim is implied.",
          "","| Split | Method | Success | Static collisions | Dynamic collisions |","|---|---|---:|---:|---:|"]
    for s in summaries:
        rows.append(f'| {s["split"]} | {s["method"]} | {s["success_count"]}/{s["total_scenarios"]} | {s["static_collision_count"]} | {s["dynamic_collision_count"]} |')
    rows+= ["","## Seed-level success statistics",""]
    for s in seed_stats:
        if s["metric"]=="success_rate":rows.append(f'- {s["split"]} {s["stage"]}: {100*s["mean"]:.2f}% ± {100*s["sample_sd"]:.2f} percentage points (sample SD); seed t 95% [{100*s["seed_t_95_low"]:.2f}, {100*s["seed_t_95_high"]:.2f}]%.')
    rows += ["","## Paired success differences",""]
    for p in paired:
        lo,hi=p["bootstrap_95"]
        rows.append(f'- {p["split"]}, {p["comparison"]}: {100*p["mean_difference"]:.2f} pp; paired hierarchical bootstrap 95% [{100*lo:.2f}, {100*hi:.2f}] pp.')
    rows += ["","A/B/round0 are single-seed supplementary ablations, not robust multi-seed evidence.",
             "All PPO policies run in the same D motion environment with the observation encoding they were trained on.",
             "DWA knows the fixed radii and kinematic constants, but consumes no hidden map, pose, future or witness information.",
             "Path efficiency is conditional on success; different successful subsets are not a paired efficiency comparison.",
             "Timing is local CPU wall-clock inference, not a real-time deployment benchmark."]
    (OUT/"SUMMARY.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    atomic_write_json(OUT/"artifact_manifest.json",{str(p.relative_to(ROOT)):sha256_file(p)
        for p in OUT.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"})
    print("\n".join(rows))


if __name__=="__main__":main()
