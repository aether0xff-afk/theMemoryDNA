from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity, regime_schedule, repeat_cycle
from .ga import GAConfig, ModelKind, run_population
from .model import SilvaParameters, reference_optimal_b


STATIC_P = (0.11, 0.53, 0.89)
MODELS = (ModelKind.GENETIC, ModelKind.PLASTIC, ModelKind.EPI, ModelKind.FIXED_EPI)


PRESETS = {
    "smoke": {"population": 24, "static_generations": 40, "switch_phase": 30, "replicates": 1},
    "quick": {"population": 80, "static_generations": 240, "switch_phase": 120, "replicates": 5},
    "full": {"population": 180, "static_generations": 600, "switch_phase": 200, "replicates": 20},
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _final_window(history: list[dict], window: int = 30) -> list[dict]:
    return history[-min(window, len(history)) :]


def _summarize_static(histories: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, int], list[dict]] = {}
    for row in histories:
        key = (str(row["model"]), float(row["target_p_epsilon"]), int(row["replicate"]))
        groups.setdefault(key, []).append(row)

    per_rep: list[dict] = []
    for (model, p, rep), rows in groups.items():
        rows = sorted(rows, key=lambda x: int(x["generation"]))
        tail = _final_window(rows)
        per_rep.append(
            {
                "model": model,
                "target_p_epsilon": p,
                "replicate": rep,
                "final_mean_fitness": mean(float(r["mean_fitness"]) for r in tail),
                "final_mean_r_germ": mean(float(r["mean_r_germ"]) for r in tail),
                "final_mean_p_b": mean(float(r["mean_p_b"]) for r in tail),
                "final_mean_mu": mean(float(r["mean_mu"]) for r in tail),
                "final_mean_b": mean(float(r["mean_b"]) for r in tail),
            }
        )
    return per_rep


def _aggregate(rows: list[dict], metric: str) -> dict[tuple[str, float], tuple[float, float]]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), float(row["target_p_epsilon"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    out = {}
    for key, values in grouped.items():
        out[key] = (mean(values), stdev(values) if len(values) > 1 else 0.0)
    return out


def _plot_static(summary: list[dict], output_dir: Path) -> None:
    agg_fit = _aggregate(summary, "final_mean_fitness")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(STATIC_P))
    width = 0.19
    for j, model in enumerate(MODELS):
        ys = [agg_fit[(model.value, p)][0] for p in STATIC_P]
        es = [agg_fit[(model.value, p)][1] for p in STATIC_P]
        ax.bar(x + (j - 1.5) * width, ys, width, yerr=es, label=model.value, capsize=3)
    ax.set_xticks(x, [str(p) for p in STATIC_P])
    ax.set_xlabel(r"target $p_\epsilon$")
    ax.set_ylabel("final-window mean fitness")
    ax.set_title("Ablation comparison across environmental similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "static_fitness.png", dpi=220)
    plt.close(fig)

    epi = [r for r in summary if r["model"] == ModelKind.EPI.value]
    agg_r = _aggregate(epi, "final_mean_r_germ")
    agg_p = _aggregate(epi, "final_mean_p_b")

    fig, ax = plt.subplots(figsize=(7, 5))
    ys = [agg_r[(ModelKind.EPI.value, p)][0] for p in STATIC_P]
    es = [agg_r[(ModelKind.EPI.value, p)][1] for p in STATIC_P]
    ax.errorbar(STATIC_P, ys, yerr=es, marker="o", capsize=4)
    ax.set_xlabel(r"target $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $r_{germ}$")
    ax.set_title("Does sRNA transmission rate evolve with environmental similarity?")
    fig.tight_layout()
    fig.savefig(output_dir / "static_evolved_r.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ys = [agg_p[(ModelKind.EPI.value, p)][0] for p in STATIC_P]
    es = [agg_p[(ModelKind.EPI.value, p)][1] for p in STATIC_P]
    ax.errorbar(STATIC_P, ys, yerr=es, marker="o", capsize=4)
    ax.set_xlabel(r"target $p_\epsilon$")
    ax.set_ylabel(r"evolved mean $P_b$")
    ax.set_title("Evolved plasticity across environmental similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "static_evolved_plasticity.png", dpi=220)
    plt.close(fig)


def _switch_recovery(rows: list[dict], phase_meta: list[dict]) -> list[dict]:
    by_model_rep: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_model_rep.setdefault((str(row["model"]), int(row["replicate"])), []).append(row)

    out: list[dict] = []
    for (model, rep), history in by_model_rep.items():
        history = sorted(history, key=lambda x: int(x["generation"]))
        fitness = np.array([float(r["mean_fitness"]) for r in history])
        for phase_idx in range(1, len(phase_meta)):
            phase = phase_meta[phase_idx]
            start, end = int(phase["start"]), int(phase["end"])
            tail_start = max(start, end - min(25, end - start))
            target = float(np.mean(fitness[tail_start:end])) * 0.90
            roll = 10
            recovered = None
            for g in range(start, end - roll + 1):
                if float(np.mean(fitness[g : g + roll])) >= target:
                    recovered = g - start
                    break
            out.append(
                {
                    "model": model,
                    "replicate": rep,
                    "switch_generation": start,
                    "from_target_p": phase_meta[phase_idx - 1]["target_p_epsilon"],
                    "to_target_p": phase["target_p_epsilon"],
                    "recovery_generations": recovered if recovered is not None else end - start,
                }
            )
    return out


def _plot_switch(rows: list[dict], phase_meta: list[dict], output_dir: Path) -> None:
    # Average trajectories across replicates.
    grouped: dict[tuple[str, int], list[float]] = {}
    grouped_r: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["generation"]))
        grouped.setdefault(key, []).append(float(row["mean_fitness"]))
        grouped_r.setdefault(key, []).append(float(row["mean_r_germ"]))

    generations = max(int(r["generation"]) for r in rows) + 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODELS:
        y = [mean(grouped[(model.value, g)]) for g in range(generations)]
        ax.plot(range(generations), y, label=model.value)
    for phase in phase_meta[1:]:
        ax.axvline(int(phase["start"]), linestyle="--", linewidth=1)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean fitness")
    ax.set_title("Response to changing environmental autocorrelation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "switch_fitness.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = [mean(grouped_r[(ModelKind.EPI.value, g)]) for g in range(generations)]
    ax.plot(range(generations), y)
    for phase in phase_meta[1:]:
        ax.axvline(int(phase["start"]), linestyle="--", linewidth=1)
    ax.set_xlabel("generation")
    ax.set_ylabel(r"mean $r_{germ}$ (Epi-GA)")
    ax.set_title("Evolution of the epigenetic transmission rate after regime shifts")
    fig.tight_layout()
    fig.savefig(output_dir / "switch_r_trace.png", dpi=220)
    plt.close(fig)


def run_suite(output_dir: Path, preset: str, seed: int) -> None:
    spec = PRESETS[preset]
    output_dir.mkdir(parents=True, exist_ok=True)

    params = SilvaParameters()
    cfg = GAConfig(population_size=int(spec["population"]))

    metadata = {
        "preset": preset,
        "seed": seed,
        "silva_parameters": params.__dict__,
        "ga_config": cfg.__dict__,
        "reference_b_wmax": {
            "epsilon_0.1": reference_optimal_b(0.1, params),
            "epsilon_0.9": reference_optimal_b(0.9, params),
        },
    }

    static_rows: list[dict] = []
    for p_idx, p_eps in enumerate(STATIC_P):
        env_rng = np.random.default_rng(seed + 10_000 + p_idx)
        cycle = balanced_cycle_for_similarity(p_eps, rng=env_rng)
        env = repeat_cycle(cycle, int(spec["static_generations"]))
        for rep in range(int(spec["replicates"])):
            paired_seed = seed + p_idx * 1000 + rep
            for model in MODELS:
                hist = run_population(env, model, seed=paired_seed, params=params, cfg=cfg)
                for row in hist:
                    row.update(
                        {
                            "target_p_epsilon": p_eps,
                            "realized_cycle_p_epsilon": cycle.p_epsilon,
                            "replicate": rep,
                        }
                    )
                static_rows.extend(hist)

    static_summary = _summarize_static(static_rows)
    _write_csv(output_dir / "static_history.csv", static_rows)
    _write_csv(output_dir / "static_summary.csv", static_summary)
    _plot_static(static_summary, output_dir)

    phase_len = int(spec["switch_phase"])
    phases = [(phase_len, 0.89), (phase_len, 0.11), (phase_len, 0.89)]
    env, phase_meta = regime_schedule(phases, seed=seed + 50_000)
    switch_rows: list[dict] = []
    for rep in range(int(spec["replicates"])):
        paired_seed = seed + 60_000 + rep
        for model in MODELS:
            hist = run_population(env, model, seed=paired_seed, params=params, cfg=cfg)
            for row in hist:
                row["replicate"] = rep
            switch_rows.extend(hist)

    recovery = _switch_recovery(switch_rows, phase_meta)
    _write_csv(output_dir / "switch_history.csv", switch_rows)
    _write_csv(output_dir / "switch_recovery.csv", recovery)
    _plot_switch(switch_rows, phase_meta, output_dir)

    metadata["switch_phases"] = phase_meta
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoryDNA epigenetic GA experiments")
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    args = parser.parse_args()
    run_suite(args.output, args.preset, args.seed)
    print(f"Wrote experiment outputs to {args.output}")


if __name__ == "__main__":
    main()
