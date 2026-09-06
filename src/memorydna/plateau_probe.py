from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity, repeat_cycle
from .ga import GAConfig, ModelKind, run_population
from .model import SilvaParameters, inherited_srna, reference_optimal_b, simulate_development


# Exact similarities realizable by a balanced 20-generation cycle.
# Includes the three original anchors (~0.11, ~0.53, ~0.89) plus intermediate points.
SWITCH_COUNTS = (18, 17, 15, 13, 11, 9, 7, 5, 4, 3, 2)
P_SWEEP = tuple(1.0 - k / 19.0 for k in SWITCH_COUNTS)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tail_summary(history: list[dict], window: int = 50) -> dict[str, float]:
    tail = history[-min(window, len(history)) :]
    keys = ("mean_fitness", "mean_r_germ", "mean_p_b", "mean_mu", "mean_b", "mean_n_initial", "mean_n_final")
    return {k: mean(float(row[k]) for row in tail) for k in keys}


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    n = len(y)
    rss_safe = max(rss, 1e-300)
    aic = n * math.log(rss_safe / n) + 2 * 2
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "rss": rss, "aic": float(aic)}


def _fit_saturating(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    # y = a + b * min(x, threshold); threshold chosen by grid search.
    candidates = np.linspace(float(np.min(x)) + 0.05, float(np.max(x)) - 0.05, 181)
    best: dict[str, float] | None = None
    n = len(y)
    for threshold in candidates:
        z = np.minimum(x, threshold)
        X = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        rss = float(np.sum((y - pred) ** 2))
        rss_safe = max(rss, 1e-300)
        # a, b, threshold => 3 fitted parameters.
        aic = n * math.log(rss_safe / n) + 2 * 3
        row = {
            "intercept": float(beta[0]),
            "slope_before_plateau": float(beta[1]),
            "threshold": float(threshold),
            "plateau_value": float(beta[0] + beta[1] * threshold),
            "rss": rss,
            "aic": float(aic),
        }
        if best is None or row["aic"] < best["aic"]:
            best = row
    assert best is not None
    return best


def run_fine_sweep(
    output_dir: Path,
    *,
    seed: int,
    population: int = 160,
    generations: int = 500,
    replicates: int = 12,
) -> tuple[list[dict], list[dict]]:
    params = SilvaParameters()
    cfg = GAConfig(population_size=population)
    per_rep: list[dict] = []

    for p_idx, target_p in enumerate(P_SWEEP):
        for rep in range(replicates):
            env_rng = np.random.default_rng(seed + 100_000 + p_idx * 10_000 + rep)
            cycle = balanced_cycle_for_similarity(target_p, rng=env_rng)
            env = repeat_cycle(cycle, generations)
            run_seed = seed + p_idx * 1000 + rep
            history = run_population(env, ModelKind.EPI, seed=run_seed, params=params, cfg=cfg)
            summary = _tail_summary(history)
            per_rep.append(
                {
                    "target_p_epsilon": target_p,
                    "realized_p_epsilon": cycle.p_epsilon,
                    "switches": cycle.changes,
                    "replicate": rep,
                    **summary,
                }
            )

    aggregate: list[dict] = []
    for target_p in P_SWEEP:
        rows = [r for r in per_rep if abs(float(r["target_p_epsilon"]) - target_p) < 1e-12]
        agg: dict[str, float | int] = {
            "target_p_epsilon": target_p,
            "realized_p_epsilon": mean(float(r["realized_p_epsilon"]) for r in rows),
            "switches": int(round(mean(float(r["switches"]) for r in rows))),
            "replicates": len(rows),
        }
        for metric in ("mean_fitness", "mean_r_germ", "mean_p_b", "mean_mu", "mean_b", "mean_n_initial", "mean_n_final"):
            vals = [float(r[metric]) for r in rows]
            agg[f"{metric}_mean"] = mean(vals)
            agg[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
        aggregate.append(agg)

    _write_csv(output_dir / "fine_sweep_replicates.csv", per_rep)
    _write_csv(output_dir / "fine_sweep_aggregate.csv", aggregate)

    correlations: list[dict] = []
    for target_p in P_SWEEP:
        rows = [r for r in per_rep if abs(float(r["target_p_epsilon"]) - target_p) < 1e-12]
        rvals = [float(r["mean_r_germ"]) for r in rows]
        correlations.append(
            {
                "target_p_epsilon": target_p,
                "corr_r_vs_p_b": _corr(rvals, [float(r["mean_p_b"]) for r in rows]),
                "corr_r_vs_mu": _corr(rvals, [float(r["mean_mu"]) for r in rows]),
                "corr_r_vs_b": _corr(rvals, [float(r["mean_b"]) for r in rows]),
                "corr_r_vs_fitness": _corr(rvals, [float(r["mean_fitness"]) for r in rows]),
            }
        )
    _write_csv(output_dir / "fine_sweep_correlations.csv", correlations)

    x = np.asarray([float(r["realized_p_epsilon"]) for r in per_rep], dtype=float)
    y = np.asarray([float(r["mean_r_germ"]) for r in per_rep], dtype=float)
    linear = _fit_linear(x, y)
    saturating = _fit_saturating(x, y)
    model_comparison = {
        "linear": linear,
        "saturating": saturating,
        "delta_aic_linear_minus_saturating": float(linear["aic"] - saturating["aic"]),
        "interpretation": "positive delta favors the saturating model",
    }
    with (output_dir / "plateau_model.json").open("w", encoding="utf-8") as f:
        json.dump(model_comparison, f, ensure_ascii=False, indent=2)

    return per_rep, aggregate


def fixed_strategy_landscape(
    output_dir: Path,
    *,
    seed: int,
    r_points: int = 21,
    p_points: int = 21,
    environment_variants: int = 3,
    burnin_cycles: int = 6,
    score_cycles: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Map long-run fitness over fixed (r_germ, P_b) strategies.

    mu and baseline b are held at Silva's reference values. This removes genetic
    compensation by mu/b and asks whether the direct inheritance dimension itself
    becomes selectively flat as environmental similarity rises.
    """
    params = SilvaParameters()
    r_grid = np.linspace(0.0, 0.50, r_points)
    p_grid = np.linspace(0.0, 0.80, p_points)
    rr, pp = np.meshgrid(r_grid, p_grid, indexing="ij")
    r_flat = rr.ravel()
    p_flat = pp.ravel()
    count = r_flat.size
    mu = np.full(count, params.mu, dtype=float)
    b = np.zeros(count, dtype=float)
    b_targets = {0.1: reference_optimal_b(0.1, params), 0.9: reference_optimal_b(0.9, params)}

    landscape_rows: list[dict] = []
    summaries: list[dict] = []

    for p_idx, target_p in enumerate(P_SWEEP):
        score_accum = np.zeros(count, dtype=float)
        realized_ps: list[float] = []

        for variant in range(environment_variants):
            rng = np.random.default_rng(seed + 500_000 + p_idx * 1000 + variant)
            cycle = balanced_cycle_for_similarity(target_p, rng=rng)
            realized_ps.append(cycle.p_epsilon)
            total_cycles = burnin_cycles + score_cycles
            env = repeat_cycle(cycle, total_cycles * len(cycle.epsilon))
            n_initial = np.zeros(count, dtype=float)
            scored_logs: list[np.ndarray] = []
            score_start = burnin_cycles * len(cycle.epsilon)

            for generation, epsilon in enumerate(env):
                eps = float(epsilon)
                b_opt = b_targets[0.1 if eps < 0.5 else 0.9]
                logfit, n_final = simulate_development(n_initial, mu, b, p_flat, eps, b_opt, params)
                if generation >= score_start:
                    scored_logs.append(logfit)
                n_initial = inherited_srna(n_final, r_flat)

            variant_score = np.mean(np.stack(scored_logs, axis=0), axis=0)
            score_accum += variant_score

        mean_logfit = score_accum / environment_variants
        realized_p = mean(realized_ps)
        for idx in range(count):
            landscape_rows.append(
                {
                    "target_p_epsilon": target_p,
                    "realized_p_epsilon": realized_p,
                    "r_germ": float(r_flat[idx]),
                    "p_b": float(p_flat[idx]),
                    "mean_log_fitness": float(mean_logfit[idx]),
                    "relative_fitness": float(math.exp(mean_logfit[idx] - float(np.max(mean_logfit)))),
                }
            )

        matrix = mean_logfit.reshape(r_points, p_points)
        best_idx = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        best_r = float(r_grid[best_idx[0]])
        best_p = float(p_grid[best_idx[1]])
        max_score = float(matrix[best_idx])

        # Optimize P_b separately for every r, then inspect only the direct-r profile.
        r_profile = np.max(matrix, axis=1)
        r0_score = float(r_profile[0])
        relative = np.exp(r_profile - float(np.max(r_profile)))
        near = r_grid[relative >= 0.999]  # within 0.1% of optimum.
        summaries.append(
            {
                "target_p_epsilon": target_p,
                "realized_p_epsilon": realized_p,
                "best_r_germ": best_r,
                "best_p_b": best_p,
                "best_mean_log_fitness": max_score,
                "direct_r_advantage_log": float(np.max(r_profile) - r0_score),
                "direct_r_advantage_ratio": float(math.exp(np.max(r_profile) - r0_score)),
                "near_optimal_r_min": float(np.min(near)),
                "near_optimal_r_max": float(np.max(near)),
                "near_optimal_r_width": float(np.max(near) - np.min(near)),
            }
        )

    _write_csv(output_dir / "fixed_strategy_landscape.csv", landscape_rows)
    _write_csv(output_dir / "fixed_strategy_summary.csv", summaries)
    return landscape_rows, summaries


def _plot_probe(output_dir: Path, aggregate: list[dict], landscape_summary: list[dict]) -> None:
    x = np.asarray([float(r["realized_p_epsilon"]) for r in aggregate])

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.asarray([float(r["mean_r_germ_mean"]) for r in aggregate])
    e = np.asarray([float(r["mean_r_germ_sd"]) for r in aggregate])
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $r_{germ}$")
    ax.set_title("Fine sweep of direct sRNA inheritance")
    fig.tight_layout()
    fig.savefig(output_dir / "fine_sweep_r.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    yp = np.asarray([float(r["mean_p_b_mean"]) for r in aggregate])
    ep = np.asarray([float(r["mean_p_b_sd"]) for r in aggregate])
    ax.errorbar(x, yp, yerr=ep, marker="o", capsize=3, label=r"$P_b$")
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $P_b$")
    ax.set_title("Transgenerational plasticity across environmental similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "fine_sweep_pb.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ym = np.asarray([float(r["mean_mu_mean"]) for r in aggregate])
    em = np.asarray([float(r["mean_mu_sd"]) for r in aggregate])
    ax.errorbar(x, ym, yerr=em, marker="o", capsize=3)
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $\mu$")
    ax.set_title("Compensatory change in de novo sRNA production")
    fig.tight_layout()
    fig.savefig(output_dir / "fine_sweep_mu.png", dpi=220)
    plt.close(fig)

    lx = np.asarray([float(r["realized_p_epsilon"]) for r in landscape_summary])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lx, [float(r["best_r_germ"]) for r in landscape_summary], marker="o")
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel(r"fitness-optimal fixed $r_{germ}$")
    ax.set_title("Direct-inheritance optimum with genetic compensation removed")
    fig.tight_layout()
    fig.savefig(output_dir / "landscape_best_r.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lx, [float(r["near_optimal_r_width"]) for r in landscape_summary], marker="o")
    ax.set_xlabel(r"realized $p_\epsilon$")
    ax.set_ylabel("width of 99.9%-optimal r region")
    ax.set_title("Does selection on direct inheritance become flat?")
    fig.tight_layout()
    fig.savefig(output_dir / "landscape_r_flatness.png", dpi=220)
    plt.close(fig)


def run_probe(output_dir: Path, *, seed: int = 7) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_rep, aggregate = run_fine_sweep(output_dir, seed=seed)
    _, landscape_summary = fixed_strategy_landscape(output_dir, seed=seed)
    _plot_probe(output_dir, aggregate, landscape_summary)

    params = SilvaParameters()
    metadata = {
        "seed": seed,
        "p_sweep": list(P_SWEEP),
        "switch_counts": list(SWITCH_COUNTS),
        "silva_parameters": asdict(params),
        "fine_sweep": {"population": 160, "generations": 500, "replicates": 12, "tail_window": 50},
        "landscape": {
            "r_range": [0.0, 0.50],
            "p_b_range": [0.0, 0.80],
            "grid": [21, 21],
            "environment_variants": 3,
            "burnin_cycles": 6,
            "score_cycles": 4,
            "near_optimal_definition": "relative fitness >= 0.999 after optimizing P_b for each r",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe why evolved r_germ plateaus at moderate/high environmental similarity")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/plateau-probe"))
    args = parser.parse_args()
    run_probe(args.output, seed=args.seed)
    print(f"Wrote plateau-probe outputs to {args.output}")


if __name__ == "__main__":
    main()
