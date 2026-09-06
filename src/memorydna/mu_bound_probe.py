from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity, repeat_cycle
from .ga import GAConfig, ModelKind, run_population
from .model import SilvaParameters
from .plateau_probe import _fit_linear, _fit_saturating, _tail_summary


MU_LOWER_BOUNDS = (0.0, 1.0, 2.0, 3.0)
SWITCH_COUNTS = (11, 9, 7, 5, 4, 3, 2)
P_SWEEP = tuple(1.0 - k / 19.0 for k in SWITCH_COUNTS)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_mu_bound_probe(
    output_dir: Path,
    *,
    seed: int = 7,
    population: int = 160,
    generations: int = 500,
    replicates: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = SilvaParameters()
    per_rep: list[dict] = []

    for bound_idx, mu_low in enumerate(MU_LOWER_BOUNDS):
        cfg = GAConfig(population_size=population, mu_bounds=(mu_low, 10.0))
        for p_idx, target_p in enumerate(P_SWEEP):
            for rep in range(replicates):
                rng = np.random.default_rng(seed + 700_000 + bound_idx * 100_000 + p_idx * 1000 + rep)
                cycle = balanced_cycle_for_similarity(target_p, rng=rng)
                env = repeat_cycle(cycle, generations)
                run_seed = seed + bound_idx * 100_000 + p_idx * 1000 + rep
                history = run_population(env, ModelKind.EPI, seed=run_seed, params=params, cfg=cfg)
                summary = _tail_summary(history, window=50)
                per_rep.append(
                    {
                        "mu_lower_bound": mu_low,
                        "target_p_epsilon": target_p,
                        "realized_p_epsilon": cycle.p_epsilon,
                        "switches": cycle.changes,
                        "replicate": rep,
                        "distance_mu_to_lower_bound": float(summary["mean_mu"] - mu_low),
                        **summary,
                    }
                )

    aggregate: list[dict] = []
    for mu_low in MU_LOWER_BOUNDS:
        for target_p in P_SWEEP:
            rows = [
                r for r in per_rep
                if float(r["mu_lower_bound"]) == mu_low
                and abs(float(r["target_p_epsilon"]) - target_p) < 1e-12
            ]
            row: dict[str, float | int] = {
                "mu_lower_bound": mu_low,
                "target_p_epsilon": target_p,
                "realized_p_epsilon": mean(float(r["realized_p_epsilon"]) for r in rows),
                "replicates": len(rows),
            }
            for metric in (
                "mean_r_germ", "mean_p_b", "mean_mu", "mean_b", "mean_fitness",
                "distance_mu_to_lower_bound",
            ):
                vals = [float(r[metric]) for r in rows]
                row[f"{metric}_mean"] = mean(vals)
                row[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
            aggregate.append(row)

    _write_csv(output_dir / "mu_bound_replicates.csv", per_rep)
    _write_csv(output_dir / "mu_bound_aggregate.csv", aggregate)

    fits: dict[str, dict] = {}
    for mu_low in MU_LOWER_BOUNDS:
        rows = [r for r in per_rep if float(r["mu_lower_bound"]) == mu_low]
        x = np.asarray([float(r["realized_p_epsilon"]) for r in rows])
        y = np.asarray([float(r["mean_r_germ"]) for r in rows])
        linear = _fit_linear(x, y)
        saturating = _fit_saturating(x, y)
        fits[str(mu_low)] = {
            "linear": linear,
            "saturating": saturating,
            "delta_aic_linear_minus_saturating": float(linear["aic"] - saturating["aic"]),
        }
    with (output_dir / "mu_bound_plateau_models.json").open("w", encoding="utf-8") as f:
        json.dump(fits, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mu_low in MU_LOWER_BOUNDS:
        rows = [r for r in aggregate if float(r["mu_lower_bound"]) == mu_low]
        x = [float(r["realized_p_epsilon"]) for r in rows]
        y = [float(r["mean_r_germ_mean"]) for r in rows]
        e = [float(r["mean_r_germ_sd"]) for r in rows]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=fr"$\mu_{{min}}={mu_low:g}$")
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $r_{germ}$")
    ax.set_title(r"Is the $r_{germ}$ plateau caused by the GA lower bound on $\mu$?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mu_bound_r_sensitivity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mu_low in MU_LOWER_BOUNDS:
        rows = [r for r in aggregate if float(r["mu_lower_bound"]) == mu_low]
        x = [float(r["realized_p_epsilon"]) for r in rows]
        y = [float(r["mean_mu_mean"]) for r in rows]
        e = [float(r["mean_mu_sd"]) for r in rows]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=fr"$\mu_{{min}}={mu_low:g}$")
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $\mu$")
    ax.set_title(r"De novo production under different $\mu$ bounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mu_bound_mu_sensitivity.png", dpi=220)
    plt.close(fig)

    metadata = {
        "seed": seed,
        "population": population,
        "generations": generations,
        "replicates": replicates,
        "mu_lower_bounds": list(MU_LOWER_BOUNDS),
        "p_sweep": list(P_SWEEP),
        "purpose": "Distinguish biological/evolutionary saturation of r_germ from an artifact caused by the computational lower bound on mu.",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity of evolved r_germ plateau to the GA lower bound on mu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/mu-bound-probe"))
    args = parser.parse_args()
    run_mu_bound_probe(args.output, seed=args.seed)
    print(f"Wrote mu-bound sensitivity outputs to {args.output}")


if __name__ == "__main__":
    main()
