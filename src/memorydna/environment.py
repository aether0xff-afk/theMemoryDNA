from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BENIGN = 0.1
STRESSFUL = 0.9


@dataclass(frozen=True)
class EnvironmentCycle:
    epsilon: np.ndarray
    p_epsilon: float
    changes: int


def measured_similarity(sequence: np.ndarray) -> float:
    """Parent-offspring environmental similarity used by Silva et al.

    For G generations there are G-1 adjacent parent-offspring pairs, so
    p_epsilon = 1 - k/(G-1), where k is the number of switches.
    """

    sequence = np.asarray(sequence)
    if sequence.size < 2:
        return 1.0
    k = int(np.count_nonzero(sequence[1:] != sequence[:-1]))
    return 1.0 - k / (sequence.size - 1)


def _random_positive_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts <= 0 or parts > total:
        raise ValueError(f"Cannot compose total={total} into parts={parts} positive parts")
    if parts == 1:
        return [total]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    boundaries = np.concatenate(([0], cuts, [total]))
    return np.diff(boundaries).astype(int).tolist()


def balanced_cycle_for_similarity(
    target_p_epsilon: float,
    *,
    generations: int = 20,
    rng: np.random.Generator | None = None,
) -> EnvironmentCycle:
    """Create a 50/50 benign/stressful cycle with requested autocorrelation.

    The paper uses G=20 and p_epsilon≈0.11, 0.53, 0.89. With 19 adjacent
    transitions these correspond exactly to k=17, 9, and 2 switches after
    rounding.
    """

    if generations % 2 != 0:
        raise ValueError("Balanced cycles require an even number of generations")
    if not 0.0 <= target_p_epsilon <= 1.0:
        raise ValueError("target_p_epsilon must lie in [0, 1]")

    rng = rng or np.random.default_rng()
    half = generations // 2
    max_changes = generations - 1
    changes = int(round((1.0 - target_p_epsilon) * max_changes))
    changes = max(1, min(max_changes, changes))
    runs = changes + 1

    # Alternating runs. Their counts differ by at most one depending on start.
    start_stress = bool(rng.integers(0, 2))
    if start_stress:
        stress_runs = (runs + 1) // 2
        benign_runs = runs // 2
    else:
        benign_runs = (runs + 1) // 2
        stress_runs = runs // 2

    # A balanced 10/10 cycle can only realize run counts <= 10 for each state.
    # If a rounded target is impossible, walk to the nearest feasible k.
    while benign_runs > half or stress_runs > half:
        changes -= 1
        runs = changes + 1
        if start_stress:
            stress_runs = (runs + 1) // 2
            benign_runs = runs // 2
        else:
            benign_runs = (runs + 1) // 2
            stress_runs = runs // 2

    benign_lengths = _random_positive_composition(half, benign_runs, rng) if benign_runs else []
    stress_lengths = _random_positive_composition(half, stress_runs, rng) if stress_runs else []

    out: list[float] = []
    bi = si = 0
    state_stress = start_stress
    for _ in range(runs):
        if state_stress:
            out.extend([STRESSFUL] * stress_lengths[si])
            si += 1
        else:
            out.extend([BENIGN] * benign_lengths[bi])
            bi += 1
        state_stress = not state_stress

    sequence = np.asarray(out, dtype=float)
    return EnvironmentCycle(
        epsilon=sequence,
        p_epsilon=measured_similarity(sequence),
        changes=int(np.count_nonzero(sequence[1:] != sequence[:-1])),
    )


def repeat_cycle(cycle: EnvironmentCycle, generations: int) -> np.ndarray:
    reps = int(np.ceil(generations / cycle.epsilon.size))
    return np.tile(cycle.epsilon, reps)[:generations]


def regime_schedule(
    phases: list[tuple[int, float]],
    *,
    seed: int = 0,
    cycle_length: int = 20,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Build a multi-phase environment such as 0.89 -> 0.11 -> 0.89."""

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    metadata: list[dict[str, float | int]] = []
    offset = 0
    for length, target_p in phases:
        cycle = balanced_cycle_for_similarity(target_p, generations=cycle_length, rng=rng)
        seq = repeat_cycle(cycle, length)
        pieces.append(seq)
        metadata.append(
            {
                "start": offset,
                "end": offset + length,
                "target_p_epsilon": target_p,
                "realized_cycle_p_epsilon": cycle.p_epsilon,
                "cycle_changes": cycle.changes,
            }
        )
        offset += length
    return np.concatenate(pieces), metadata
