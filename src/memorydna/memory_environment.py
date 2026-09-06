from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import BENIGN, STRESSFUL, EnvironmentCycle, measured_similarity


@dataclass(frozen=True)
class PatternCycle:
    name: str
    epsilon: np.ndarray
    p_epsilon: float
    changes: int
    run_lengths: tuple[int, ...]


def _compose_regular(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    out = [base] * parts
    for i in range(extra):
        out[-(i + 1)] += 1
    return out


def _compose_clustered(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    # One long run plus the shortest possible remaining runs.
    return [total - (parts - 1)] + [1] * (parts - 1)


def _compose_random(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int).tolist()


def matched_pattern_cycle(
    pattern: str,
    *,
    changes: int = 5,
    generations: int = 20,
    seed: int = 0,
    start_stress: bool = False,
) -> PatternCycle:
    """Build balanced cycles with the same p_epsilon but different run-length structure.

    The default uses 20 generations, 10 benign + 10 stressful, and exactly five
    switches. Therefore every pattern has p_epsilon = 1 - 5/19 ~= 0.737. Only
    the distribution of run lengths changes, separating first-order similarity
    from higher-order temporal patterning.
    """

    if generations % 2:
        raise ValueError("generations must be even")
    if not 1 <= changes < generations:
        raise ValueError("changes must be between 1 and generations-1")

    runs = changes + 1
    stress_runs = (runs + (1 if start_stress else 0)) // 2
    benign_runs = runs - stress_runs
    half = generations // 2
    if stress_runs > half or benign_runs > half:
        raise ValueError("requested switch count is infeasible for a balanced cycle")

    rng = np.random.default_rng(seed)
    composer = {
        "regular": lambda total, parts: _compose_regular(total, parts),
        "clustered": lambda total, parts: _compose_clustered(total, parts),
        "stochastic": lambda total, parts: _compose_random(total, parts, rng),
    }.get(pattern)
    if composer is None:
        raise ValueError("pattern must be one of: regular, clustered, stochastic")

    benign_lengths = composer(half, benign_runs)
    stress_lengths = composer(half, stress_runs)

    out: list[float] = []
    all_lengths: list[int] = []
    bi = si = 0
    stress = start_stress
    for _ in range(runs):
        if stress:
            length = stress_lengths[si]
            si += 1
            out.extend([STRESSFUL] * length)
        else:
            length = benign_lengths[bi]
            bi += 1
            out.extend([BENIGN] * length)
        all_lengths.append(length)
        stress = not stress

    sequence = np.asarray(out, dtype=float)
    realized_changes = int(np.count_nonzero(sequence[1:] != sequence[:-1]))
    return PatternCycle(
        name=pattern,
        epsilon=sequence,
        p_epsilon=measured_similarity(sequence),
        changes=realized_changes,
        run_lengths=tuple(all_lengths),
    )


def as_environment_cycle(pattern_cycle: PatternCycle) -> EnvironmentCycle:
    return EnvironmentCycle(
        epsilon=pattern_cycle.epsilon.copy(),
        p_epsilon=pattern_cycle.p_epsilon,
        changes=pattern_cycle.changes,
    )
