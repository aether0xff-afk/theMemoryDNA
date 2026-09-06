from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity
from .memory_architecture import MemoryKind, adult_germline_amplification
from .model import SilvaParameters, effective_amplification, inherited_srna, log_instantaneous_fitness, srna_rhs


P_VALUES = (0.11, 0.53, 0.89)
P_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
R_GRID = (0.0, 0.05, 0.11, 0.20)
SCALES = (1.0, 0.75)
ARCHITECTURES = tuple(MemoryKind)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_scaled(
    n_initial: np.ndarray,
    mu: np.ndarray,
    b_initial: np.ndarray,
    p_b: np.ndarray,
    epsilon: float,
    b_opt: float,
    params: SilvaParameters,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Silva development with b,d,m,mu all multiplied by the same scale.

    Because multiplying b and m by s leaves b*n/m unchanged, while b, d and
    mu in the RHS each gain s, the scaled Eq.1 RHS is exactly s times the
    unscaled RHS evaluated at the nominal b(t). This implements the paper's
    explicit 0.75x slow-dynamics manipulation without changing the steady state.
    Plasticity delay a and developmental duration c are unchanged.
    """

    n = np.maximum(np.asarray(n_initial, dtype=float).copy(), 0.0)
    mu = np.asarray(mu, dtype=float)
    b_initial = np.asarray(b_initial, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    substeps = max(1, int(params.rk4_substeps))
    dt = 1.0 / substeps
    total_steps = params.cell_divisions * substeps
    integral = np.zeros_like(n)

    for step in range(total_steps):
        t0 = step * dt
        w0 = log_instantaneous_fitness(n, epsilon, p_b, params)

        def rhs(x: np.ndarray, t: float) -> np.ndarray:
            b_t = effective_amplification(b_initial, p_b, b_opt, t, params.plasticity_delay_a)
            return scale * srna_rhs(x, mu, b_t, d=params.d, m=params.m)

        k1 = rhs(n, t0)
        k2 = rhs(n + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = rhs(n + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = rhs(n + dt * k3, t0 + dt)
        n_next = np.maximum(n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
        w1 = log_instantaneous_fitness(n_next, epsilon, p_b, params)
        integral += 0.5 * (w0 + w1) * dt
        n = n_next
    return integral / params.cell_divisions, n


@lru_cache(maxsize=32)
def scaled_reference_optimal_b(epsilon: float, scale: float, params: SilvaParameters = SilvaParameters()) -> float:
    grid = np.linspace(0.0, 1.0, 401)
    scores = []
    for b in grid:
        logw, _ = simulate_scaled(
            np.array([0.0]), np.array([params.mu]), np.array([b]), np.array([0.0]),
            epsilon, b, params, scale
        )
        scores.append(float(logw[0]))
    best = int(np.argmax(scores))
    lo, hi = grid[max(0,best-1)], grid[min(len(grid)-1,best+1)]
    fine = np.linspace(lo, hi, 101)
    fs = []
    for b in fine:
        logw, _ = simulate_scaled(
            np.array([0.0]), np.array([params.mu]), np.array([b]), np.array([0.0]),
            epsilon, b, params, scale
        )
        fs.append(float(logw[0]))
    return float(fine[int(np.argmax(fs))])


def lineage_score(
    cycle: np.ndarray,
    architecture: MemoryKind,
    *,
    r_germ: float,
    p_b: float,
    scale: float,
    params: SilvaParameters,
    repeats: int = 10,
) -> float:
    env = np.tile(np.asarray(cycle, dtype=float), repeats)
    n0 = np.array([0.0])
    b0 = np.array([0.0])
    p = np.array([p_b])
    mu = np.array([params.mu])
    b_targets = {e: scaled_reference_optimal_b(e, scale, params) for e in (0.1,0.9)}
    last_logs: list[float] = []

    for generation, epsilon in enumerate(env):
        b_opt = b_targets[0.1 if epsilon < 0.5 else 0.9]
        logw, nf = simulate_scaled(n0, mu, b0, p, float(epsilon), b_opt, params, scale)
        if generation >= len(env)-len(cycle):
            last_logs.append(float(logw[0]))
        n0 = inherited_srna(nf, r_germ) if architecture.transmits_srna else np.array([0.0])
        b0 = adult_germline_amplification(b0, p, b_opt, params) if architecture.transmits_mechanism else np.array([0.0])
    return float(np.exp(np.mean(last_logs)))


def run_slow_reference(output_dir: Path, *, seed: int = 7, environment_replicates: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = SilvaParameters()
    grid_rows: list[dict] = []

    for scale in SCALES:
        for p_idx, target_p in enumerate(P_VALUES):
            for env_rep in range(environment_replicates):
                cycle = balanced_cycle_for_similarity(target_p, rng=np.random.default_rng(seed + p_idx*1000 + env_rep))
                for architecture in ARCHITECTURES:
                    rs = R_GRID if architecture.transmits_srna else (0.0,)
                    for p_b in P_GRID:
                        for r in rs:
                            fit = lineage_score(cycle.epsilon, architecture, r_germ=r, p_b=p_b, scale=scale, params=params)
                            grid_rows.append({
                                "dynamics_scale": scale,
                                "target_p_epsilon": target_p,
                                "environment_replicate": env_rep,
                                "architecture": architecture.value,
                                "r_germ": r,
                                "p_b": p_b,
                                "fitness": fit,
                            })
    _write_csv(output_dir / "slow_reference_grid.csv", grid_rows)

    grouped: dict[tuple, list[float]] = {}
    for row in grid_rows:
        key=(row["dynamics_scale"],row["target_p_epsilon"],row["architecture"],row["r_germ"],row["p_b"])
        grouped.setdefault(key,[]).append(float(row["fitness"]))
    means=[]
    for key, vals in grouped.items():
        scale,pval,arch,r,pb=key
        means.append({"dynamics_scale":scale,"target_p_epsilon":pval,"architecture":arch,"r_germ":r,"p_b":pb,"fitness_mean":mean(vals)})
    _write_csv(output_dir / "slow_reference_mean.csv", means)

    best=[]
    for scale in SCALES:
        for pval in P_VALUES:
            for arch in ARCHITECTURES:
                candidates=[r for r in means if r["dynamics_scale"]==scale and r["target_p_epsilon"]==pval and r["architecture"]==arch.value]
                best.append(dict(max(candidates,key=lambda x:x["fitness_mean"])))
    _write_csv(output_dir / "slow_reference_best.csv", best)

    fig,ax=plt.subplots(figsize=(9,5))
    for scale in SCALES:
        for arch in (MemoryKind.NO_MEMORY,MemoryKind.MECHANISM_MEMORY):
            sub=[r for r in best if r["dynamics_scale"]==scale and r["architecture"]==arch.value]
            ax.plot([r["target_p_epsilon"] for r in sub],[r["fitness_mean"] for r in sub],marker="o",label=f"{arch.value}, {scale}x")
    ax.set_xlabel(r"target $p_\epsilon$")
    ax.set_ylabel("best fixed-genome fitness")
    ax.set_title("Paper-derived 0.75x dynamics sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "slow_dynamics_mechanism.png",dpi=240)
    plt.close(fig)

    (output_dir / "metadata.json").write_text(json.dumps({
        "seed":seed,"environment_replicates":environment_replicates,"scales":SCALES,
        "scaling_rule":"Eq.1 RHS multiplied by scale, exactly equivalent to multiplying b,d,m,mu by scale.",
        "paper_condition":"0.75x within-generation dynamics from Silva et al. 2021 S8 analysis"
    },indent=2),encoding="utf-8")
