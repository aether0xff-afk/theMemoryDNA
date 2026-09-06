from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

from .dual_memory import DualMemoryConfig, MemoryArchitecture, run_dual_population
from .environment import BENIGN, STRESSFUL, balanced_cycle_for_similarity, measured_similarity, regime_schedule, repeat_cycle
from .model import SilvaParameters


ARCHITECTURES = (
    MemoryArchitecture.GENETIC,
    MemoryArchitecture.SOMATIC,
    MemoryArchitecture.STATE,
    MemoryArchitecture.MECHANISM,
    MemoryArchitecture.DUAL,
    MemoryArchitecture.FULL_DUAL,
)
STATIC_P = (0.11, 0.53, 0.89)

PRESETS = {
    "smoke": {"population": 24, "generations": 40, "switch_phase": 25, "replicates": 1, "pattern_reps": 1},
    "quick": {"population": 72, "generations": 220, "switch_phase": 100, "replicates": 4, "pattern_reps": 3},
    "full": {"population": 140, "generations": 500, "switch_phase": 160, "replicates": 12, "pattern_reps": 8},
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tail(rows: list[dict], window: int = 40) -> list[dict]:
    return rows[-min(window, len(rows)) :]


def _summary(history: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
    tail = _tail(history)
    return {key: mean(float(row[key]) for row in tail) for key in keys}


def _aggregate(rows: list[dict], group_keys: tuple[str, ...], metric: str) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(float(row[metric]))
    out: list[dict] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec[f"{metric}_mean"] = mean(values)
        rec[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        rec["n"] = len(values)
        out.append(rec)
    return out


def _paired_effects(static_summary: list[dict]) -> list[dict]:
    """Paired effect sizes using identical replicate seeds, without scipy dependency."""
    by_key = {
        (str(r["mu_mode"]), float(r["target_p_epsilon"]), str(r["architecture"]), int(r["replicate"])): r
        for r in static_summary
    }
    comparisons = (
        (MemoryArchitecture.STATE.value, MemoryArchitecture.GENETIC.value),
        (MemoryArchitecture.MECHANISM.value, MemoryArchitecture.GENETIC.value),
        (MemoryArchitecture.DUAL.value, MemoryArchitecture.GENETIC.value),
        (MemoryArchitecture.FULL_DUAL.value, MemoryArchitecture.GENETIC.value),
        (MemoryArchitecture.DUAL.value, MemoryArchitecture.STATE.value),
        (MemoryArchitecture.DUAL.value, MemoryArchitecture.MECHANISM.value),
    )
    out: list[dict] = []
    modes = sorted({str(r["mu_mode"]) for r in static_summary})
    for mode in modes:
        for p in STATIC_P:
            reps = sorted({int(r["replicate"]) for r in static_summary if str(r["mu_mode"]) == mode and float(r["target_p_epsilon"]) == p})
            for a, b in comparisons:
                diffs = []
                for rep in reps:
                    ra = by_key[(mode, p, a, rep)]
                    rb = by_key[(mode, p, b, rep)]
                    diffs.append(float(ra["final_mean_fitness"]) - float(rb["final_mean_fitness"]))
                out.append(
                    {
                        "mu_mode": mode,
                        "target_p_epsilon": p,
                        "architecture_a": a,
                        "architecture_b": b,
                        "mean_paired_fitness_difference": mean(diffs),
                        "sd_paired_difference": stdev(diffs) if len(diffs) > 1 else 0.0,
                        "fraction_a_better": sum(d > 0 for d in diffs) / len(diffs),
                        "n": len(diffs),
                    }
                )
    return out


def _matched_patterns(seed: int) -> dict[str, np.ndarray]:
    """20-generation 10/10 cycles with nine within-cycle switches.

    All deterministic patterns start benign and end stressful, so repeating them
    adds the same boundary switch. They therefore have the same mean environment
    and the same first-order transition count, while differing in higher-order
    temporal arrangement/run-length structure.
    """

    def from_runs(benign_lengths: list[int], stressful_lengths: list[int]) -> np.ndarray:
        out: list[float] = []
        for a, b in zip(benign_lengths, stressful_lengths):
            out.extend([BENIGN] * a)
            out.extend([STRESSFUL] * b)
        arr = np.asarray(out, dtype=float)
        assert arr.size == 20
        assert int(np.sum(arr == BENIGN)) == 10
        assert int(np.sum(arr == STRESSFUL)) == 10
        assert int(np.count_nonzero(arr[1:] != arr[:-1])) == 9
        return arr

    rng = np.random.default_rng(seed)
    stochastic = balanced_cycle_for_similarity(0.53, generations=20, rng=rng).epsilon
    if stochastic[0] == STRESSFUL:
        stochastic = np.where(stochastic == BENIGN, STRESSFUL, BENIGN)
    # Ensure the same boundary direction as the deterministic patterns.
    if stochastic[-1] != STRESSFUL:
        stochastic = stochastic[::-1]
        if stochastic[0] == STRESSFUL:
            stochastic = np.where(stochastic == BENIGN, STRESSFUL, BENIGN)

    return {
        "periodic-2": from_runs([2, 2, 2, 2, 2], [2, 2, 2, 2, 2]),
        "early-persistent": from_runs([6, 1, 1, 1, 1], [1, 1, 1, 1, 6]),
        "late-persistent": from_runs([1, 1, 1, 1, 6], [6, 1, 1, 1, 1]),
        "stochastic-matched": stochastic,
    }


def _plot_static(summary: list[dict], output_dir: Path) -> None:
    for mode in ("fixed-mu", "evolved-mu"):
        rows = [r for r in summary if r["mu_mode"] == mode]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(STATIC_P))
        width = 0.13
        for j, arch in enumerate(ARCHITECTURES):
            ys, es = [], []
            for p in STATIC_P:
                vals = [float(r["final_mean_fitness"]) for r in rows if r["architecture"] == arch.value and float(r["target_p_epsilon"]) == p]
                ys.append(mean(vals))
                es.append(stdev(vals) if len(vals) > 1 else 0.0)
            ax.bar(x + (j - 2.5) * width, ys, width, yerr=es, label=arch.value, capsize=2)
        ax.set_xticks(x, [str(p) for p in STATIC_P])
        ax.set_xlabel(r"target $p_\epsilon$")
        ax.set_ylabel("final-window mean fitness")
        ax.set_title(f"Memory architecture comparison ({mode})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"architecture_fitness_{mode}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for arch in (MemoryArchitecture.STATE, MemoryArchitecture.DUAL, MemoryArchitecture.FULL_DUAL):
            ys = []
            for p in STATIC_P:
                vals = [float(r["final_mean_r_germ"]) for r in rows if r["architecture"] == arch.value and float(r["target_p_epsilon"]) == p]
                ys.append(mean(vals))
            ax.plot(STATIC_P, ys, marker="o", label=arch.value)
        ax.set_xlabel(r"target $p_\epsilon$")
        ax.set_ylabel(r"evolved $r_{germ}$")
        ax.set_title(f"State-memory allocation ({mode})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"state_memory_{mode}.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for arch in (MemoryArchitecture.MECHANISM, MemoryArchitecture.DUAL, MemoryArchitecture.FULL_DUAL):
            ys = []
            for p in STATIC_P:
                vals = [float(r["final_mean_b_inherited"]) for r in rows if r["architecture"] == arch.value and float(r["target_p_epsilon"]) == p]
                ys.append(mean(vals))
            ax.plot(STATIC_P, ys, marker="o", label=arch.value)
        ax.set_xlabel(r"target $p_\epsilon$")
        ax.set_ylabel(r"inherited amplification state $b^\#$")
        ax.set_title(f"Mechanism-memory state ({mode})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"mechanism_memory_{mode}.png", dpi=220)
        plt.close(fig)


def _plot_patterns(rows: list[dict], output_dir: Path) -> None:
    names = sorted({str(r["pattern"]) for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.13
    for j, arch in enumerate(ARCHITECTURES):
        ys, es = [], []
        for name in names:
            vals = [float(r["final_mean_fitness"]) for r in rows if r["pattern"] == name and r["architecture"] == arch.value]
            ys.append(mean(vals))
            es.append(stdev(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + (j - 2.5) * width, ys, width, yerr=es, label=arch.value, capsize=2)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("final-window mean fitness")
    ax.set_title("Same first-order similarity, different temporal order")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "matched_pattern_fitness.png", dpi=220)
    plt.close(fig)


def _plot_switch(rows: list[dict], phase_meta: list[dict], output_dir: Path) -> None:
    generations = max(int(r["generation"]) for r in rows) + 1
    for metric, ylabel, filename in (
        ("mean_fitness", "mean fitness", "regime_fitness.png"),
        ("mean_r_germ", r"mean $r_{germ}$", "regime_state_memory.png"),
        ("mean_b_inherited", r"mean inherited $b^\#$", "regime_mechanism_memory.png"),
        ("mean_mu", r"mean $\mu$", "regime_mu.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        for arch in ARCHITECTURES:
            ys = []
            for g in range(generations):
                vals = [float(r[metric]) for r in rows if r["architecture"] == arch.value and int(r["generation"]) == g]
                ys.append(mean(vals))
            ax.plot(range(generations), ys, label=arch.value)
        for phase in phase_meta[1:]:
            ax.axvline(int(phase["start"]), linestyle="--", linewidth=1)
        ax.set_xlabel("generation")
        ax.set_ylabel(ylabel)
        ax.set_title("Regime-shift dynamics")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def _write_markdown_report(output_dir: Path, static_summary: list[dict], pattern_summary: list[dict]) -> None:
    lines = [
        "# Final dual-memory study — machine-generated summary",
        "",
        "This file summarizes the corrected, paper-grounded final study. Interpret biological claims with the model provenance in `docs/FINAL_STUDY.md`.",
        "",
        "## Best architecture by condition",
        "",
        "| mu mode | p_epsilon | best architecture | mean fitness |",
        "|---|---:|---|---:|",
    ]
    for mode in ("fixed-mu", "evolved-mu"):
        for p in STATIC_P:
            candidates = []
            for arch in ARCHITECTURES:
                vals = [float(r["final_mean_fitness"]) for r in static_summary if r["mu_mode"] == mode and float(r["target_p_epsilon"]) == p and r["architecture"] == arch.value]
                candidates.append((mean(vals), arch.value))
            best_fit, best_arch = max(candidates)
            lines.append(f"| {mode} | {p:.2f} | {best_arch} | {best_fit:.6f} |")

    lines.extend(["", "## Matched-pattern experiment", "", "All patterns use a balanced 10/10 environment and the same within-cycle switch count; only temporal arrangement differs.", "", "| pattern | best architecture | mean fitness |", "|---|---|---:|"])
    for pattern in sorted({str(r["pattern"]) for r in pattern_summary}):
        candidates = []
        for arch in ARCHITECTURES:
            vals = [float(r["final_mean_fitness"]) for r in pattern_summary if r["pattern"] == pattern and r["architecture"] == arch.value]
            candidates.append((mean(vals), arch.value))
        best_fit, best_arch = max(candidates)
        lines.append(f"| {pattern} | {best_arch} | {best_fit:.6f} |")

    (output_dir / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_study(output_dir: Path, preset: str, seed: int) -> None:
    spec = PRESETS[preset]
    output_dir.mkdir(parents=True, exist_ok=True)
    params = SilvaParameters()
    base_cfg = DualMemoryConfig(population_size=int(spec["population"]))

    metadata = {
        "preset": preset,
        "seed": seed,
        "silva_parameters": asdict(params),
        "architectures": [a.value for a in ARCHITECTURES],
        "design_note": "r_b is intentionally absent: Silva Eqs. 5-6 inherit b# directly; P_b controls its plastic update.",
    }

    static_history: list[dict] = []
    static_summary: list[dict] = []
    for mode_idx, (mu_mode, evolve_mu) in enumerate((("fixed-mu", False), ("evolved-mu", True))):
        cfg = replace(base_cfg, evolve_mu=evolve_mu)
        for p_idx, p_eps in enumerate(STATIC_P):
            cycle = balanced_cycle_for_similarity(p_eps, rng=np.random.default_rng(seed + 10_000 + p_idx))
            env = repeat_cycle(cycle, int(spec["generations"]))
            realized = measured_similarity(env)
            for rep in range(int(spec["replicates"])):
                paired_seed = seed + mode_idx * 100_000 + p_idx * 1000 + rep
                for arch in ARCHITECTURES:
                    hist = run_dual_population(env, arch, seed=paired_seed, params=params, cfg=cfg)
                    for row in hist:
                        row.update({"mu_mode": mu_mode, "target_p_epsilon": p_eps, "realized_p_epsilon": realized, "replicate": rep})
                    static_history.extend(hist)
                    sm = _summary(hist, ("mean_fitness", "mean_mu", "mean_r_germ", "mean_p_b", "mean_n_initial", "mean_n_final", "mean_b_inherited", "mean_b_germ_final"))
                    static_summary.append({"mu_mode": mu_mode, "target_p_epsilon": p_eps, "realized_p_epsilon": realized, "architecture": arch.value, "replicate": rep, **{f"final_{k}": v for k, v in sm.items()}})

    _write_csv(output_dir / "architecture_history.csv", static_history)
    _write_csv(output_dir / "architecture_summary.csv", static_summary)
    _write_csv(output_dir / "paired_effects.csv", _paired_effects(static_summary))
    _plot_static(static_summary, output_dir)

    # Higher-order temporal-pattern experiment: evolved mu, matched first-order switch count.
    pattern_history: list[dict] = []
    pattern_summary: list[dict] = []
    patterns = _matched_patterns(seed + 30_000)
    pattern_cfg = replace(base_cfg, evolve_mu=True)
    pattern_generations = int(spec["generations"])
    pattern_meta = {}
    for pattern_idx, (name, cycle_arr) in enumerate(patterns.items()):
        reps = int(np.ceil(pattern_generations / cycle_arr.size))
        env = np.tile(cycle_arr, reps)[:pattern_generations]
        pattern_meta[name] = {"cycle": cycle_arr.tolist(), "realized_p_epsilon": measured_similarity(env)}
        for rep in range(int(spec["pattern_reps"])):
            paired_seed = seed + 40_000 + pattern_idx * 1000 + rep
            for arch in ARCHITECTURES:
                hist = run_dual_population(env, arch, seed=paired_seed, params=params, cfg=pattern_cfg)
                for row in hist:
                    row.update({"pattern": name, "realized_p_epsilon": measured_similarity(env), "replicate": rep})
                pattern_history.extend(hist)
                sm = _summary(hist, ("mean_fitness", "mean_mu", "mean_r_germ", "mean_p_b", "mean_n_initial", "mean_b_inherited"))
                pattern_summary.append({"pattern": name, "realized_p_epsilon": measured_similarity(env), "architecture": arch.value, "replicate": rep, **{f"final_{k}": v for k, v in sm.items()}})

    _write_csv(output_dir / "pattern_history.csv", pattern_history)
    _write_csv(output_dir / "pattern_summary.csv", pattern_summary)
    _plot_patterns(pattern_summary, output_dir)

    # Regime shift: evolved-mu system, all memory architectures.
    phase_len = int(spec["switch_phase"])
    phases = [(phase_len, 0.89), (phase_len, 0.11), (phase_len, 0.89)]
    env, phase_meta = regime_schedule(phases, seed=seed + 50_000)
    regime_rows: list[dict] = []
    for rep in range(int(spec["replicates"])):
        paired_seed = seed + 60_000 + rep
        for arch in ARCHITECTURES:
            hist = run_dual_population(env, arch, seed=paired_seed, params=params, cfg=pattern_cfg)
            for row in hist:
                row["replicate"] = rep
            regime_rows.extend(hist)
    _write_csv(output_dir / "regime_history.csv", regime_rows)
    _plot_switch(regime_rows, phase_meta, output_dir)

    metadata["matched_patterns"] = pattern_meta
    metadata["regime_phases"] = phase_meta
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _write_markdown_report(output_dir, static_summary, pattern_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final MemoryDNA state-vs-mechanism inheritance study")
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/final-study"))
    args = parser.parse_args()
    run_final_study(args.output, args.preset, args.seed)
    print(f"Wrote final-study outputs to {args.output}")


if __name__ == "__main__":
    main()
