from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity
from .memory_architecture import MemoryKind, adult_germline_amplification
from .model import SilvaParameters, inherited_srna
from .slow_reference import P_VALUES, P_GRID, R_GRID, SCALES, scaled_reference_optimal_b, simulate_scaled

ARCHITECTURES = tuple(MemoryKind)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_scores(cycle: np.ndarray, architecture: MemoryKind, scale: float, params: SilvaParameters, repeats: int = 10):
    combos=[]
    rs=R_GRID if architecture.transmits_srna else (0.0,)
    for pb in P_GRID:
        for r in rs:
            combos.append((r,pb))
    r_arr=np.asarray([x[0] for x in combos],dtype=float)
    p_arr=np.asarray([x[1] for x in combos],dtype=float)
    n0=np.zeros(len(combos),dtype=float)
    b0=np.zeros(len(combos),dtype=float)
    mu=np.full(len(combos),params.mu,dtype=float)
    env=np.tile(np.asarray(cycle,dtype=float),repeats)
    targets={e:scaled_reference_optimal_b(e,scale,params) for e in (0.1,0.9)}
    logsum=np.zeros(len(combos),dtype=float)
    count=0
    for generation,epsilon in enumerate(env):
        bopt=targets[0.1 if epsilon<0.5 else 0.9]
        logw,nf=simulate_scaled(n0,mu,b0,p_arr,float(epsilon),bopt,params,scale)
        if generation>=len(env)-len(cycle):
            logsum+=logw
            count+=1
        n0=inherited_srna(nf,r_arr) if architecture.transmits_srna else np.zeros_like(n0)
        b0=adult_germline_amplification(b0,p_arr,bopt,params) if architecture.transmits_mechanism else np.zeros_like(b0)
    return combos,np.exp(logsum/count)


def run_fast(output_dir: Path, seed: int = 7, environment_replicates: int = 6):
    output_dir.mkdir(parents=True,exist_ok=True)
    params=SilvaParameters()
    rows=[]
    for scale in SCALES:
        for pi,target_p in enumerate(P_VALUES):
            for env_rep in range(environment_replicates):
                cycle=balanced_cycle_for_similarity(target_p,rng=np.random.default_rng(seed+pi*1000+env_rep))
                for arch in ARCHITECTURES:
                    combos,scores=batch_scores(cycle.epsilon,arch,scale,params)
                    for (r,pb),fit in zip(combos,scores):
                        rows.append({"dynamics_scale":scale,"target_p_epsilon":target_p,"environment_replicate":env_rep,"architecture":arch.value,"r_germ":r,"p_b":pb,"fitness":float(fit)})
    _write_csv(output_dir/"slow_reference_grid.csv",rows)
    grouped={}
    for row in rows:
        key=(row["dynamics_scale"],row["target_p_epsilon"],row["architecture"],row["r_germ"],row["p_b"])
        grouped.setdefault(key,[]).append(row["fitness"])
    means=[]
    for (scale,p,arch,r,pb),vals in grouped.items():
        means.append({"dynamics_scale":scale,"target_p_epsilon":p,"architecture":arch,"r_germ":r,"p_b":pb,"fitness_mean":mean(vals)})
    _write_csv(output_dir/"slow_reference_mean.csv",means)
    best=[]
    for scale in SCALES:
        for p in P_VALUES:
            for arch in ARCHITECTURES:
                cand=[x for x in means if x["dynamics_scale"]==scale and x["target_p_epsilon"]==p and x["architecture"]==arch.value]
                best.append(dict(max(cand,key=lambda x:x["fitness_mean"])))
    _write_csv(output_dir/"slow_reference_best.csv",best)
    fig,ax=plt.subplots(figsize=(9,5))
    for scale in SCALES:
        for arch in (MemoryKind.NO_MEMORY,MemoryKind.MECHANISM_MEMORY):
            sub=[x for x in best if x["dynamics_scale"]==scale and x["architecture"]==arch.value]
            ax.plot([x["target_p_epsilon"] for x in sub],[x["fitness_mean"] for x in sub],marker="o",label=f"{arch.value}, {scale}x")
    ax.set_xlabel(r"target $p_\epsilon$")
    ax.set_ylabel("best fixed-genome fitness")
    ax.set_title("0.75x dynamics sensitivity (vectorized cross-check)")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir/"slow_dynamics_mechanism.png",dpi=240); plt.close(fig)
    (output_dir/"metadata.json").write_text(json.dumps({"vectorized":True,"seed":seed,"environment_replicates":environment_replicates,"scales":SCALES},indent=2),encoding="utf-8")
