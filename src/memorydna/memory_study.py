from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .environment import balanced_cycle_for_similarity, regime_schedule, repeat_cycle
from .ga import GAConfig
from .memory_architecture import MemoryKind, run_memory_population
from .memory_environment import matched_pattern_cycle
from .model import SilvaParameters, reference_optimal_b


ARCHITECTURES = tuple(MemoryKind)
ANCHOR_P = (0.11, 0.53, 0.89)
FINE_P = tuple(1.0 - k / 19.0 for k in (18, 17, 15, 13, 11, 9, 7, 5, 4, 3, 2))
PATTERNS = ("regular", "clustered", "stochastic")

PRESETS = {
    "smoke": {
        "population": 20,
        "anchor_generations": 40,
        "anchor_replicates": 1,
        "fine_generations": 40,
        "fine_replicates": 1,
        "pattern_generations": 40,
        "pattern_replicates": 1,
        "switch_phase": 30,
        "switch_replicates": 1,
    },
    "quick": {
        "population": 64,
        "anchor_generations": 220,
        "anchor_replicates": 5,
        "fine_generations": 180,
        "fine_replicates": 3,
        "pattern_generations": 220,
        "pattern_replicates": 5,
        "switch_phase": 100,
        "switch_replicates": 5,
    },
    "full": {
        "population": 160,
        "anchor_generations": 500,
        "anchor_replicates": 20,
        "fine_generations": 360,
        "fine_replicates": 10,
        "pattern_generations": 500,
        "pattern_replicates": 20,
        "switch_phase": 180,
        "switch_replicates": 20,
    },
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tail_summary(rows: list[dict], window: int = 40) -> dict[str, float]:
    tail = rows[-min(window, len(rows)) :]
    metrics = (
        "mean_fitness",
        "mean_mu",
        "mean_b_genetic",
        "mean_r_germ",
        "mean_p_b",
        "mean_n_initial",
        "mean_n_final",
        "mean_b_initial",
        "mean_b_germ_final",
    )
    return {metric: mean(float(row[metric]) for row in tail) for metric in metrics}


def _aggregate_replicates(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    metrics = [key for key in rows[0] if key.startswith("final_")] if rows else []
    out: list[dict] = []
    for key, group in grouped.items():
        result = {field: value for field, value in zip(group_fields, key)}
        result["n_replicates"] = len(group)
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        out.append(result)
    return out


def _architecture_summary(history: list[dict], condition_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in history:
        key = tuple(row[field] for field in ("architecture", "replicate", *condition_fields))
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, rows in grouped.items():
        rows.sort(key=lambda item: int(item["generation"]))
        summary = _tail_summary(rows)
        result = {
            "architecture": key[0],
            "replicate": key[1],
        }
        for field, value in zip(condition_fields, key[2:]):
            result[field] = value
        result.update({f"final_{metric}": value for metric, value in summary.items()})
        out.append(result)
    return out


def _plot_metric(
    summary: list[dict],
    *,
    x_field: str,
    metric: str,
    output: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for architecture in ARCHITECTURES:
        subset = [row for row in summary if row["architecture"] == architecture.value]
        xs = sorted({float(row[x_field]) for row in subset})
        ys = []
        es = []
        for x in xs:
            values = [float(row[metric]) for row in subset if float(row[x_field]) == x]
            ys.append(mean(values))
            es.append(stdev(values) if len(values) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=architecture.value)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)


def _common_metadata(component: str, preset: str, seed: int, params: SilvaParameters, cfg: GAConfig) -> dict:
    return {
        "component": component,
        "preset": preset,
        "seed": seed,
        "architectures": [item.value for item in ARCHITECTURES],
        "silva_parameters": params.__dict__,
        "ga_config": cfg.__dict__,
        "reference_b_wmax": {
            "epsilon_0.1": reference_optimal_b(0.1, params),
            "epsilon_0.9": reference_optimal_b(0.9, params),
        },
        "architecture_definition": {
            "no-memory": "Somatic Eq.4 only; no cross-generation n or b# state.",
            "state-memory": "Eq.4 + direct transcript inheritance n_child=r_germ*n_parent.",
            "mechanism-memory": "Eq.4 + inherited germline amplification state b# via Eqs.5-6 (strategy-E-like).",
            "dual-memory": "Both direct transcript inheritance and Eqs.5-6 b# inheritance.",
        },
    }


def run_anchor(output_dir: Path, preset: str, seed: int) -> None:
    spec = PRESETS[preset]
    params = SilvaParameters()
    cfg = GAConfig(population_size=int(spec["population"]))
    generations = int(spec["anchor_generations"])
    replicates = int(spec["anchor_replicates"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    cycle_meta: list[dict] = []
    for p_idx, target_p in enumerate(ANCHOR_P):
        cycle = balanced_cycle_for_similarity(target_p, rng=np.random.default_rng(seed + 10_000 + p_idx))
        env = repeat_cycle(cycle, generations)
        cycle_meta.append({"target_p": target_p, "realized_p": cycle.p_epsilon, "changes": cycle.changes})
        for rep in range(replicates):
            paired_seed = seed + p_idx * 1000 + rep
            for architecture in ARCHITECTURES:
                rows = run_memory_population(env, architecture, seed=paired_seed, params=params, cfg=cfg)
                for row in rows:
                    row.update({"target_p_epsilon": target_p, "realized_p_epsilon": cycle.p_epsilon, "replicate": rep})
                history.extend(rows)

    summary = _architecture_summary(history, ("target_p_epsilon", "realized_p_epsilon"))
    aggregate = _aggregate_replicates(summary, ("architecture", "target_p_epsilon", "realized_p_epsilon"))
    _write_csv(output_dir / "anchor_history.csv", history)
    _write_csv(output_dir / "anchor_summary.csv", summary)
    _write_csv(output_dir / "anchor_aggregate.csv", aggregate)

    _plot_metric(summary, x_field="target_p_epsilon", metric="final_mean_fitness", output=output_dir / "anchor_fitness.png", title="Memory architecture fitness", xlabel=r"target $p_\epsilon$", ylabel="final mean fitness")
    _plot_metric(summary, x_field="target_p_epsilon", metric="final_mean_r_germ", output=output_dir / "anchor_r_germ.png", title="Evolved transcript transmission", xlabel=r"target $p_\epsilon$", ylabel=r"mean $r_{germ}$")
    _plot_metric(summary, x_field="target_p_epsilon", metric="final_mean_p_b", output=output_dir / "anchor_plasticity.png", title="Evolved plasticity", xlabel=r"target $p_\epsilon$", ylabel=r"mean $P_b$")
    _plot_metric(summary, x_field="target_p_epsilon", metric="final_mean_mu", output=output_dir / "anchor_mu.png", title="De novo production versus environmental similarity", xlabel=r"target $p_\epsilon$", ylabel=r"mean $\mu$")
    _plot_metric(summary, x_field="target_p_epsilon", metric="final_mean_b_initial", output=output_dir / "anchor_b_memory.png", title="Inherited amplification state", xlabel=r"target $p_\epsilon$", ylabel=r"mean zygotic $b$")

    metadata = _common_metadata("anchor", preset, seed, params, cfg)
    metadata.update({"generations": generations, "replicates": replicates, "cycles": cycle_meta})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_fine(output_dir: Path, preset: str, seed: int) -> None:
    spec = PRESETS[preset]
    params = SilvaParameters()
    cfg = GAConfig(population_size=int(spec["population"]))
    generations = int(spec["fine_generations"])
    replicates = int(spec["fine_replicates"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    cycle_meta: list[dict] = []
    for p_idx, target_p in enumerate(FINE_P):
        cycle = balanced_cycle_for_similarity(target_p, rng=np.random.default_rng(seed + 20_000 + p_idx))
        env = repeat_cycle(cycle, generations)
        cycle_meta.append({"target_p": target_p, "realized_p": cycle.p_epsilon, "changes": cycle.changes})
        for rep in range(replicates):
            paired_seed = seed + 20_000 + p_idx * 1000 + rep
            for architecture in ARCHITECTURES:
                rows = run_memory_population(env, architecture, seed=paired_seed, params=params, cfg=cfg)
                for row in rows:
                    row.update({"target_p_epsilon": target_p, "realized_p_epsilon": cycle.p_epsilon, "replicate": rep})
                history.extend(rows)

    summary = _architecture_summary(history, ("target_p_epsilon", "realized_p_epsilon"))
    aggregate = _aggregate_replicates(summary, ("architecture", "target_p_epsilon", "realized_p_epsilon"))
    _write_csv(output_dir / "fine_history.csv", history)
    _write_csv(output_dir / "fine_summary.csv", summary)
    _write_csv(output_dir / "fine_aggregate.csv", aggregate)
    for metric, filename, ylabel in (
        ("final_mean_fitness", "fine_fitness.png", "final mean fitness"),
        ("final_mean_r_germ", "fine_r_germ.png", r"mean $r_{germ}$"),
        ("final_mean_p_b", "fine_plasticity.png", r"mean $P_b$"),
        ("final_mean_mu", "fine_mu.png", r"mean $\mu$"),
        ("final_mean_b_initial", "fine_b_memory.png", r"mean zygotic $b$")
    ):
        _plot_metric(summary, x_field="target_p_epsilon", metric=metric, output=output_dir / filename, title=f"Fine sweep: {metric}", xlabel=r"target $p_\epsilon$", ylabel=ylabel)

    metadata = _common_metadata("fine", preset, seed, params, cfg)
    metadata.update({"generations": generations, "replicates": replicates, "cycles": cycle_meta})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_patterns(output_dir: Path, preset: str, seed: int) -> None:
    spec = PRESETS[preset]
    params = SilvaParameters()
    cfg = GAConfig(population_size=int(spec["population"]))
    generations = int(spec["pattern_generations"])
    replicates = int(spec["pattern_replicates"])
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    pattern_meta: list[dict] = []
    for pattern_idx, pattern in enumerate(PATTERNS):
        cycle = matched_pattern_cycle(pattern, changes=5, seed=seed + 30_000 + pattern_idx)
        env = np.tile(cycle.epsilon, int(np.ceil(generations / len(cycle.epsilon))))[:generations]
        pattern_meta.append({"pattern": pattern, "p_epsilon": cycle.p_epsilon, "changes": cycle.changes, "run_lengths": list(cycle.run_lengths), "cycle": cycle.epsilon.tolist()})
        for rep in range(replicates):
            paired_seed = seed + 30_000 + pattern_idx * 1000 + rep
            for architecture in ARCHITECTURES:
                rows = run_memory_population(env, architecture, seed=paired_seed, params=params, cfg=cfg)
                for row in rows:
                    row.update({"pattern": pattern, "matched_p_epsilon": cycle.p_epsilon, "replicate": rep})
                history.extend(rows)

    summary = _architecture_summary(history, ("pattern", "matched_p_epsilon"))
    aggregate = _aggregate_replicates(summary, ("architecture", "pattern", "matched_p_epsilon"))
    _write_csv(output_dir / "pattern_history.csv", history)
    _write_csv(output_dir / "pattern_summary.csv", summary)
    _write_csv(output_dir / "pattern_aggregate.csv", aggregate)

    for metric, filename, ylabel in (
        ("final_mean_fitness", "pattern_fitness.png", "final mean fitness"),
        ("final_mean_r_germ", "pattern_r_germ.png", r"mean $r_{germ}$"),
        ("final_mean_p_b", "pattern_plasticity.png", r"mean $P_b$"),
        ("final_mean_mu", "pattern_mu.png", r"mean $\mu$"),
        ("final_mean_b_initial", "pattern_b_memory.png", r"mean zygotic $b$")
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(PATTERNS))
        width = 0.19
        for j, architecture in enumerate(ARCHITECTURES):
            ys, es = [], []
            for pattern in PATTERNS:
                values = [float(row[metric]) for row in summary if row["architecture"] == architecture.value and row["pattern"] == pattern]
                ys.append(mean(values))
                es.append(stdev(values) if len(values) > 1 else 0.0)
            ax.bar(x + (j - 1.5) * width, ys, width, yerr=es, capsize=3, label=architecture.value)
        ax.set_xticks(x, PATTERNS)
        ax.set_xlabel("temporal pattern (matched p_epsilon)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Matched-autocorrelation pattern test: {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=240)
        plt.close(fig)

    metadata = _common_metadata("patterns", preset, seed, params, cfg)
    metadata.update({"generations": generations, "replicates": replicates, "patterns": pattern_meta})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _switch_metrics(history: list[dict], phase_meta: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in history:
        grouped.setdefault((str(row["architecture"]), int(row["replicate"])), []).append(row)

    out: list[dict] = []
    for (architecture, rep), rows in grouped.items():
        rows.sort(key=lambda item: int(item["generation"]))
        for phase_idx, phase in enumerate(phase_meta):
            start, end = int(phase["start"]), int(phase["end"])
            tail = rows[max(start, end - 30):end]
            result = {
                "architecture": architecture,
                "replicate": rep,
                "phase": phase_idx,
                "target_p_epsilon": phase["target_p_epsilon"],
                "start_generation": start,
                "end_generation": end,
            }
            for metric in ("mean_fitness", "mean_r_germ", "mean_p_b", "mean_mu", "mean_b_initial", "mean_b_germ_final", "mean_n_initial"):
                result[f"tail_{metric}"] = mean(float(row[metric]) for row in tail)
            out.append(result)
    return out


def run_switch(output_dir: Path, preset: str, seed: int) -> None:
    spec = PRESETS[preset]
    params = SilvaParameters()
    cfg = GAConfig(population_size=int(spec["population"]))
    phase_len = int(spec["switch_phase"])
    replicates = int(spec["switch_replicates"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env, phase_meta = regime_schedule([(phase_len, 0.89), (phase_len, 0.11), (phase_len, 0.89)], seed=seed + 40_000)
    history: list[dict] = []
    for rep in range(replicates):
        paired_seed = seed + 40_000 + rep
        for architecture in ARCHITECTURES:
            rows = run_memory_population(env, architecture, seed=paired_seed, params=params, cfg=cfg)
            for row in rows:
                row["replicate"] = rep
            history.extend(rows)

    phase_summary = _switch_metrics(history, phase_meta)
    _write_csv(output_dir / "switch_history.csv", history)
    _write_csv(output_dir / "switch_phase_summary.csv", phase_summary)

    generations = len(env)
    for metric, filename, ylabel in (
        ("mean_fitness", "switch_fitness.png", "mean fitness"),
        ("mean_r_germ", "switch_r_germ.png", r"mean $r_{germ}$"),
        ("mean_p_b", "switch_plasticity.png", r"mean $P_b$"),
        ("mean_mu", "switch_mu.png", r"mean $\mu$"),
        ("mean_b_initial", "switch_b_memory.png", r"mean zygotic $b$")
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        for architecture in ARCHITECTURES:
            y = []
            for generation in range(generations):
                values = [float(row[metric]) for row in history if row["architecture"] == architecture.value and int(row["generation"]) == generation]
                y.append(mean(values))
            ax.plot(range(generations), y, label=architecture.value)
        for phase in phase_meta[1:]:
            ax.axvline(int(phase["start"]), linestyle="--", linewidth=1)
        ax.set_xlabel("generation")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Regime shift: {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=240)
        plt.close(fig)

    metadata = _common_metadata("switch", preset, seed, params, cfg)
    metadata.update({"phase_length": phase_len, "replicates": replicates, "phases": phase_meta})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_component(component: str, output_dir: Path, preset: str, seed: int) -> None:
    runners = {
        "anchor": run_anchor,
        "fine": run_fine,
        "patterns": run_patterns,
        "switch": run_switch,
    }
    if component not in runners:
        raise ValueError(f"Unknown component: {component}")
    runners[component](output_dir, preset, seed)
