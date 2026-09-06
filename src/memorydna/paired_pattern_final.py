from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .dual_memory import DualMemoryConfig, MemoryArchitecture, run_dual_population
from .final_study import _matched_patterns, _summary
from .model import SilvaParameters


ARCHITECTURES = tuple(MemoryArchitecture)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_paired_pattern_audit(
    output_dir: Path,
    *,
    seed: int = 7,
    population: int = 140,
    generations: int = 500,
    replicates: int = 8,
) -> None:
    """Paired-seed audit of the final study's matched temporal-order experiment.

    The environmental cycles differ only in higher-order temporal arrangement.
    For replicate r, the exact same GA seed is used for every pattern and every
    architecture. This removes between-pattern GA seed sets as a possible
    explanation for a temporal-order effect.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    params = SilvaParameters()
    cfg = DualMemoryConfig(population_size=population, evolve_mu=True)
    patterns = _matched_patterns(seed + 30_000)

    summaries: list[dict] = []
    metadata_patterns: dict[str, dict] = {}
    for name, cycle in patterns.items():
        env = np.tile(cycle, int(np.ceil(generations / len(cycle))))[:generations]
        realized = float(1.0 - np.count_nonzero(env[1:] != env[:-1]) / (len(env) - 1))
        metadata_patterns[name] = {"cycle": cycle.tolist(), "realized_p_epsilon": realized}

    for rep in range(replicates):
        common_seed = seed + 70_000 + rep
        for name, cycle in patterns.items():
            env = np.tile(cycle, int(np.ceil(generations / len(cycle))))[:generations]
            realized = metadata_patterns[name]["realized_p_epsilon"]
            for arch in ARCHITECTURES:
                hist = run_dual_population(env, arch, seed=common_seed, params=params, cfg=cfg)
                sm = _summary(
                    hist,
                    (
                        "mean_fitness",
                        "mean_mu",
                        "mean_r_germ",
                        "mean_p_b",
                        "mean_n_initial",
                        "mean_b_inherited",
                    ),
                )
                summaries.append(
                    {
                        "replicate": rep,
                        "common_ga_seed": common_seed,
                        "pattern": name,
                        "architecture": arch.value,
                        "realized_p_epsilon": realized,
                        **{f"final_{k}": v for k, v in sm.items()},
                    }
                )

    _write_csv(output_dir / "paired_pattern_summary.csv", summaries)

    aggregate: list[dict] = []
    for name in patterns:
        for arch in ARCHITECTURES:
            subset = [r for r in summaries if r["pattern"] == name and r["architecture"] == arch.value]
            rec = {
                "pattern": name,
                "architecture": arch.value,
                "n_replicates": len(subset),
                "realized_p_epsilon": subset[0]["realized_p_epsilon"],
            }
            for metric in (
                "final_mean_fitness",
                "final_mean_mu",
                "final_mean_r_germ",
                "final_mean_p_b",
                "final_mean_n_initial",
                "final_mean_b_inherited",
            ):
                vals = [float(r[metric]) for r in subset]
                rec[f"{metric}_mean"] = mean(vals)
                rec[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
            aggregate.append(rec)
    _write_csv(output_dir / "paired_pattern_aggregate.csv", aggregate)

    # Within-architecture paired pattern contrasts. No p-value dependency is
    # required: raw paired differences and sign fractions are retained so any
    # downstream test can be reproduced exactly.
    contrast_rows: list[dict] = []
    names = list(patterns)
    for arch in ARCHITECTURES:
        subset = [r for r in summaries if r["architecture"] == arch.value]
        by = {(int(r["replicate"]), str(r["pattern"])): float(r["final_mean_fitness"]) for r in subset}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                diffs = [by[(rep, a)] - by[(rep, b)] for rep in range(replicates)]
                contrast_rows.append(
                    {
                        "architecture": arch.value,
                        "pattern_a": a,
                        "pattern_b": b,
                        "mean_paired_difference_a_minus_b": mean(diffs),
                        "sd_paired_difference": stdev(diffs) if len(diffs) > 1 else 0.0,
                        "fraction_a_better": sum(d > 0 for d in diffs) / len(diffs),
                        "n": len(diffs),
                    }
                )
    _write_csv(output_dir / "paired_pattern_contrasts.csv", contrast_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(patterns)
    x = np.arange(len(names))
    width = 0.13
    for j, arch in enumerate(ARCHITECTURES):
        ys, es = [], []
        for name in names:
            vals = [float(r["final_mean_fitness"]) for r in summaries if r["pattern"] == name and r["architecture"] == arch.value]
            ys.append(mean(vals))
            es.append(stdev(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + (j - 2.5) * width, ys, width, yerr=es, capsize=2, label=arch.value)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("final-window mean fitness")
    ax.set_title("Paired-seed matched temporal-order audit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_pattern_fitness.png", dpi=220)
    plt.close(fig)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "population": population,
                "generations": generations,
                "replicates": replicates,
                "evolve_mu": True,
                "paired_ga_seed_across_patterns": True,
                "patterns": metadata_patterns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-seed audit for the final dual-memory temporal-pattern experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population", type=int, default=140)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/final-paired-pattern"))
    args = parser.parse_args()
    run_paired_pattern_audit(
        args.output,
        seed=args.seed,
        population=args.population,
        generations=args.generations,
        replicates=args.replicates,
    )
    print(f"Wrote paired temporal-pattern audit to {args.output}")


if __name__ == "__main__":
    main()
