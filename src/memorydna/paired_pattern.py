from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .ga import GAConfig
from .memory_architecture import MemoryKind, run_memory_population
from .memory_environment import matched_pattern_cycle
from .model import SilvaParameters

PATTERNS = ("regular", "clustered", "stochastic")
ARCHITECTURES = tuple(MemoryKind)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_paired_pattern(
    output_dir: Path,
    *,
    seed: int = 7,
    population: int = 160,
    generations: int = 500,
    replicates: int = 20,
) -> None:
    """Repeat the matched-p pattern test with common GA RNG seeds across patterns.

    Pattern cycles still differ, but replicate r uses the exact same GA random
    seed for regular, clustered and stochastic conditions. This turns the
    pattern contrast into a paired robustness test and reduces variation due to
    stochastic selection/crossover/mutation histories.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    params = SilvaParameters()
    cfg = GAConfig(population_size=population)

    summaries: list[dict] = []
    pattern_meta: list[dict] = []
    cycles = {}
    for p_idx, pattern in enumerate(PATTERNS):
        cycle = matched_pattern_cycle(pattern, changes=5, seed=seed + 90_000 + p_idx)
        cycles[pattern] = cycle
        pattern_meta.append(
            {
                "pattern": pattern,
                "p_epsilon": cycle.p_epsilon,
                "changes": cycle.changes,
                "run_lengths": list(cycle.run_lengths),
                "cycle": cycle.epsilon.tolist(),
            }
        )

    metrics = (
        "mean_fitness",
        "mean_mu",
        "mean_r_germ",
        "mean_p_b",
        "mean_n_initial",
        "mean_b_initial",
        "mean_b_germ_final",
    )

    for rep in range(replicates):
        common_ga_seed = seed + 91_000 + rep
        for pattern in PATTERNS:
            cycle = cycles[pattern]
            env = np.tile(cycle.epsilon, int(np.ceil(generations / len(cycle.epsilon))))[:generations]
            for architecture in ARCHITECTURES:
                history = run_memory_population(
                    env,
                    architecture,
                    seed=common_ga_seed,
                    params=params,
                    cfg=cfg,
                )
                tail = history[-40:]
                row = {
                    "replicate": rep,
                    "common_ga_seed": common_ga_seed,
                    "pattern": pattern,
                    "architecture": architecture.value,
                    "p_epsilon": cycle.p_epsilon,
                }
                for metric in metrics:
                    row[f"final_{metric}"] = mean(float(item[metric]) for item in tail)
                summaries.append(row)

    _write_csv(output_dir / "paired_pattern_summary.csv", summaries)

    aggregate: list[dict] = []
    for architecture in ARCHITECTURES:
        for pattern in PATTERNS:
            subset = [r for r in summaries if r["architecture"] == architecture.value and r["pattern"] == pattern]
            result = {"architecture": architecture.value, "pattern": pattern, "n_replicates": len(subset)}
            for metric in metrics:
                vals = [float(r[f"final_{metric}"]) for r in subset]
                result[f"{metric}_mean"] = mean(vals)
                result[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
            aggregate.append(result)
    _write_csv(output_dir / "paired_pattern_aggregate.csv", aggregate)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(PATTERNS))
    width = 0.19
    for j, architecture in enumerate(ARCHITECTURES):
        ys, es = [], []
        for pattern in PATTERNS:
            subset = [r for r in summaries if r["architecture"] == architecture.value and r["pattern"] == pattern]
            vals = [float(r["final_mean_fitness"]) for r in subset]
            ys.append(mean(vals))
            es.append(stdev(vals))
        ax.bar(x + (j - 1.5) * width, ys, width, yerr=es, capsize=3, label=architecture.value)
    ax.set_xticks(x, PATTERNS)
    ax.set_ylabel("final mean fitness")
    ax.set_title(r"Paired-seed temporal-pattern test ($p_\epsilon$ matched)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paired_pattern_fitness.png", dpi=240)
    plt.close(fig)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "population": population,
                "generations": generations,
                "replicates": replicates,
                "paired_ga_seed_across_patterns": True,
                "patterns": pattern_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
