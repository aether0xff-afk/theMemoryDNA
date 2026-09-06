from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .ga import GAConfig
from .model import SilvaParameters, effective_amplification, inherited_srna, reference_optimal_b, simulate_development


class MemoryKind(str, Enum):
    """Cross-generational information architectures.

    All four architectures retain the same within-generation somatic plasticity
    (Silva Eq. 4). They differ only in which epigenetic state crosses generations.

    NO_MEMORY:       Silva strategy-D-like control: somatic plasticity, no cross-generation state.
    STATE_MEMORY:    direct transcript inheritance + somatic plasticity (modified D; cf. Silva S9).
    MECHANISM_MEMORY: inherited amplification state b# + somatic plasticity (strategy E-like).
    DUAL_MEMORY:     direct transcript inheritance + inherited b# state (modified E; cf. Silva S9).
    """

    NO_MEMORY = "no-memory"
    STATE_MEMORY = "state-memory"
    MECHANISM_MEMORY = "mechanism-memory"
    DUAL_MEMORY = "dual-memory"

    @property
    def transmits_srna(self) -> bool:
        return self in (MemoryKind.STATE_MEMORY, MemoryKind.DUAL_MEMORY)

    @property
    def transmits_mechanism(self) -> bool:
        return self in (MemoryKind.MECHANISM_MEMORY, MemoryKind.DUAL_MEMORY)


@dataclass
class MemoryPopulationState:
    genome: np.ndarray  # [N,4] = mu, genetically encoded basal b, r_germ, P_b
    n_initial: np.ndarray
    b_initial: np.ndarray  # zygotic amplification state used as Eq.4 starting point
    n_final: np.ndarray
    b_germ_final: np.ndarray


def _clip_genome(genome: np.ndarray, cfg: GAConfig) -> np.ndarray:
    out = genome.copy()
    for j, (lo, hi) in enumerate((cfg.mu_bounds, cfg.b_bounds, cfg.r_bounds, cfg.p_bounds)):
        out[:, j] = np.clip(out[:, j], lo, hi)
    return out


def _force_architecture(kind: MemoryKind, genome: np.ndarray) -> np.ndarray:
    out = genome.copy()
    if not kind.transmits_srna:
        out[:, 2] = 0.0
    return out


def initialize_memory_population(
    kind: MemoryKind,
    params: SilvaParameters,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> MemoryPopulationState:
    n = cfg.population_size
    genome = np.column_stack(
        [
            rng.normal(params.mu, 0.25, n),
            rng.uniform(0.0, 0.10, n),
            rng.uniform(0.0, 0.30, n),
            rng.uniform(0.0, 0.50, n),
        ]
    )
    genome = _force_architecture(kind, _clip_genome(genome, cfg))
    zeros = np.zeros(n, dtype=float)
    return MemoryPopulationState(
        genome=genome,
        n_initial=zeros.copy(),
        b_initial=genome[:, 1].copy(),
        n_final=zeros.copy(),
        b_germ_final=genome[:, 1].copy(),
    )


def adult_germline_amplification(
    b_initial: np.ndarray | float,
    p_b: np.ndarray | float,
    b_opt: float,
    params: SilvaParameters,
) -> np.ndarray:
    """Adult germline b# after Silva Eq. 6.

    For strategy E-like architectures, the zygote starts with the amplification
    state inherited from the previous adult germline. During the current
    generation, the germline state moves a fraction P_b toward b_Wmax using the
    same delayed response term as Eq. 6. The adult value is what the offspring
    inherits next generation.
    """

    return effective_amplification(
        np.asarray(b_initial, dtype=float),
        np.asarray(p_b, dtype=float),
        b_opt,
        float(params.cell_divisions),
        params.plasticity_delay_a,
    )


def _tournament_indices(
    log_fitness: np.ndarray,
    count: int,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    n = log_fitness.size
    k = max(2, cfg.tournament_size)
    candidates = rng.integers(0, n, size=(count, k))
    winner_col = np.argmax(log_fitness[candidates], axis=1)
    return candidates[np.arange(count), winner_col]


def _reproduce_memory(
    state: MemoryPopulationState,
    log_fitness: np.ndarray,
    kind: MemoryKind,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> MemoryPopulationState:
    n = cfg.population_size
    elite_count = min(max(0, cfg.elitism), n)
    elite_idx = np.argsort(log_fitness)[-elite_count:] if elite_count else np.array([], dtype=int)
    child_count = n - elite_count

    mothers = _tournament_indices(log_fitness, child_count, cfg, rng)
    fathers = _tournament_indices(log_fitness, child_count, cfg, rng)
    gm = state.genome[mothers]
    gf = state.genome[fathers]

    alpha = rng.uniform(0.0, 1.0, size=(child_count, 4))
    cross = rng.random(child_count) < cfg.crossover_rate
    children = gm.copy()
    children[cross] = alpha[cross] * gm[cross] + (1.0 - alpha[cross]) * gf[cross]

    sigmas = np.array([cfg.mu_sigma, cfg.b_sigma, cfg.r_sigma, cfg.p_sigma], dtype=float)
    mutation_mask = rng.random((child_count, 4)) < cfg.mutation_rate
    children += mutation_mask * rng.normal(0.0, sigmas, size=(child_count, 4))
    children = _force_architecture(kind, _clip_genome(children, cfg))

    if kind.transmits_srna:
        child_n = inherited_srna(state.n_final[mothers], state.genome[mothers, 2])
    else:
        child_n = np.zeros(child_count, dtype=float)

    if kind.transmits_mechanism:
        # Silva Eq. 5: the offspring zygote inherits the mother's adult germline b#.
        child_b = state.b_germ_final[mothers].copy()
    else:
        # No cross-generation b# channel: start from the offspring's genetic basal b.
        child_b = children[:, 1].copy()

    if elite_count:
        elite_genome = state.genome[elite_idx].copy()
        if kind.transmits_srna:
            elite_n = inherited_srna(state.n_final[elite_idx], state.genome[elite_idx, 2])
        else:
            elite_n = np.zeros(elite_count, dtype=float)
        if kind.transmits_mechanism:
            elite_b = state.b_germ_final[elite_idx].copy()
        else:
            elite_b = elite_genome[:, 1].copy()

        genome = np.vstack([elite_genome, children])
        n_initial = np.concatenate([elite_n, child_n])
        b_initial = np.concatenate([elite_b, child_b])
    else:
        genome = children
        n_initial = child_n
        b_initial = child_b

    return MemoryPopulationState(
        genome=genome,
        n_initial=n_initial,
        b_initial=b_initial,
        n_final=np.zeros(n, dtype=float),
        b_germ_final=b_initial.copy(),
    )


def run_memory_population(
    environment: np.ndarray,
    kind: MemoryKind,
    *,
    seed: int,
    params: SilvaParameters | None = None,
    cfg: GAConfig | None = None,
) -> list[dict[str, float | int | str]]:
    """Run an evolving population under one cross-generational memory architecture."""

    params = params or SilvaParameters()
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(seed)
    state = initialize_memory_population(kind, params, cfg, rng)

    b_targets = {
        0.1: reference_optimal_b(0.1, params),
        0.9: reference_optimal_b(0.9, params),
    }

    history: list[dict[str, float | int | str]] = []
    for generation, epsilon in enumerate(np.asarray(environment, dtype=float)):
        genome = state.genome
        mu = genome[:, 0]
        b_genetic = genome[:, 1]
        r = genome[:, 2]
        p = genome[:, 3]
        b_opt = b_targets[0.1 if epsilon < 0.5 else 0.9]

        logfit, n_final = simulate_development(
            state.n_initial,
            mu,
            state.b_initial,
            p,
            float(epsilon),
            b_opt,
            params,
        )

        if kind.transmits_mechanism:
            b_germ_final = adult_germline_amplification(state.b_initial, p, b_opt, params)
        else:
            # Eq. 5 control: no environmentally induced germline update is inherited.
            b_germ_final = b_genetic.copy()

        state.n_final = n_final
        state.b_germ_final = np.asarray(b_germ_final, dtype=float)
        raw_fitness = np.exp(np.clip(logfit, -700.0, 0.0))

        history.append(
            {
                "generation": generation,
                "architecture": kind.value,
                "epsilon": float(epsilon),
                "mean_fitness": float(np.mean(raw_fitness)),
                "geometric_mean_fitness": float(np.exp(np.mean(logfit))),
                "mean_log_fitness": float(np.mean(logfit)),
                "best_log_fitness": float(np.max(logfit)),
                "mean_mu": float(np.mean(mu)),
                "mean_b_genetic": float(np.mean(b_genetic)),
                "mean_r_germ": float(np.mean(r)),
                "mean_p_b": float(np.mean(p)),
                "mean_n_initial": float(np.mean(state.n_initial)),
                "mean_n_final": float(np.mean(n_final)),
                "mean_b_initial": float(np.mean(state.b_initial)),
                "mean_b_germ_final": float(np.mean(state.b_germ_final)),
                "b_wmax_reference": float(b_opt),
            }
        )

        state = _reproduce_memory(state, logfit, kind, cfg, rng)

    return history
