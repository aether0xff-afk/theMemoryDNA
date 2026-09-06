from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity
from .memory_architecture import MemoryKind, adult_germline_amplification
from .model import SilvaParameters, inherited_srna, reference_optimal_b, simulate_development


ANCHOR_P = (0.11, 0.53, 0.89)
P_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
R_GRID = (0.0, 0.05, 0.11, 0.20)
ARCHITECTURES = tuple(MemoryKind)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_lineage_score(
    environment: np.ndarray,
    architecture: MemoryKind,
    *,
    r_germ: float,
    p_b: float,
    params: SilvaParameters,
    repeats: int = 10,
) -> float:
    """Periodic-lineage log geometric mean with the genetic baseline held fixed.

    The genetic reference is Silva strategy-A's default mu with basal b=0.
    Only the epigenetic architecture and the specified r_germ/P_b are varied.
    The supplied 20-generation environment is repeated to wash out the arbitrary
    initial epigenetic state; only the final cycle contributes to the score.
    """

    cycle = np.asarray(environment, dtype=float)
    env = np.tile(cycle, repeats)
    n_initial = np.array([0.0])
    b_initial = np.array([0.0])
    mu = np.array([params.mu])
    p = np.array([p_b])
    logfits: list[float] = []

    b_targets = {
        0.1: reference_optimal_b(0.1, params),
        0.9: reference_optimal_b(0.9, params),
    }

    for generation, epsilon in enumerate(env):
        b_opt = b_targets[0.1 if epsilon < 0.5 else 0.9]
        logfit, n_final = simulate_development(
            n_initial,
            mu,
            b_initial,
            p,
            float(epsilon),
            b_opt,
            params,
        )
        if generation >= len(env) - len(cycle):
            logfits.append(float(logfit[0]))

        if architecture.transmits_srna:
            n_initial = inherited_srna(n_final, r_germ)
        else:
            n_initial = np.array([0.0])

        if architecture.transmits_mechanism:
            b_initial = adult_germline_amplification(b_initial, p, b_opt, params)
        else:
            b_initial = np.array([0.0])

    return float(np.exp(np.mean(logfits)))


def run_reference_benchmark(output_dir: Path, *, seed: int = 7, sequence_replicates: int = 8) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = SilvaParameters()
    rows: list[dict] = []

    for p_idx, target_p in enumerate(ANCHOR_P):
        for env_rep in range(sequence_replicates):
            cycle = balanced_cycle_for_similarity(
                target_p,
                rng=np.random.default_rng(seed + p_idx * 1000 + env_rep),
            )
            for architecture in ARCHITECTURES:
                r_values = R_GRID if architecture.transmits_srna else (0.0,)
                for p_b in P_GRID:
                    for r_germ in r_values:
                        score = fixed_lineage_score(
                            cycle.epsilon,
                            architecture,
                            r_germ=r_germ,
                            p_b=p_b,
                            params=params,
                        )
                        rows.append(
                            {
                                "target_p_epsilon": target_p,
                                "realized_p_epsilon": cycle.p_epsilon,
                                "environment_replicate": env_rep,
                                "architecture": architecture.value,
                                "r_germ": r_germ,
                                "p_b": p_b,
                                "fitness": score,
                            }
                        )

    _write_csv(output_dir / "reference_grid.csv", rows)

    # Average over alternative environment realizations, then choose the best
    # paper-motivated parameter-grid point for each architecture and p_epsilon.
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["target_p_epsilon"], row["architecture"], row["r_germ"], row["p_b"])
        grouped.setdefault(key, []).append(float(row["fitness"]))

    mean_grid: list[dict] = []
    for (target_p, architecture, r_germ, p_b), values in grouped.items():
        mean_grid.append(
            {
                "target_p_epsilon": target_p,
                "architecture": architecture,
                "r_germ": r_germ,
                "p_b": p_b,
                "fitness_mean": mean(values),
                "fitness_sd_across_environments": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    _write_csv(output_dir / "reference_grid_mean.csv", mean_grid)

    best: list[dict] = []
    for target_p in ANCHOR_P:
        for architecture in ARCHITECTURES:
            candidates = [row for row in mean_grid if row["target_p_epsilon"] == target_p and row["architecture"] == architecture.value]
            winner = max(candidates, key=lambda row: float(row["fitness_mean"]))
            best.append(dict(winner))
    _write_csv(output_dir / "reference_best.csv", best)

    fig, ax = plt.subplots(figsize=(8, 5))
    for architecture in ARCHITECTURES:
        subset = [row for row in best if row["architecture"] == architecture.value]
        ax.plot(
            [row["target_p_epsilon"] for row in subset],
            [row["fitness_mean"] for row in subset],
            marker="o",
            label=architecture.value,
        )
    ax.set_xlabel(r"target $p_\epsilon$")
    ax.set_ylabel("best fixed-genome fitness")
    ax.set_title("Memory-channel effect with genetic parameters held fixed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reference_best_fitness.png", dpi=240)
    plt.close(fig)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "sequence_replicates": sequence_replicates,
                "mu": params.mu,
                "basal_b": 0.0,
                "r_grid": R_GRID,
                "p_b_grid": P_GRID,
                "note": "Genetic parameters are fixed; only the paper-motivated memory architecture and r_germ/P_b grid differ.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
