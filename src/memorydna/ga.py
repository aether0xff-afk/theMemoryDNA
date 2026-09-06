from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import (
    SilvaParameters,
    inherited_srna,
    reference_optimal_b,
    simulate_development,
)


class ModelKind(str, Enum):
    GENETIC = "ga"
    PLASTIC = "plastic-ga"
    EPI = "epi-ga"
    FIXED_EPI = "fixed-epi"


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 160
    mutation_rate: float = 0.25
    crossover_rate: float = 0.85
    tournament_size: int = 3
    elitism: int = 2

    # Genome order: mu, b, r_germ, P_b
    mu_bounds: tuple[float, float] = (2.0, 10.0)
    b_bounds: tuple[float, float] = (0.0, 0.6)
    r_bounds: tuple[float, float] = (0.0, 0.95)
    p_bounds: tuple[float, float] = (0.0, 1.0)

    mu_sigma: float = 0.20
    b_sigma: float = 0.025
    r_sigma: float = 0.035
    p_sigma: float = 0.035

    fixed_epi_r: float = 0.11
    fixed_epi_p: float = 0.80
    fixed_epi_b: float = 0.0


@dataclass
class PopulationState:
    genome: np.ndarray  # shape [N,4]: mu,b,r,p
    n_initial: np.ndarray
    n_final: np.ndarray


def _forced_genes(kind: ModelKind, genome: np.ndarray, params: SilvaParameters, cfg: GAConfig) -> np.ndarray:
    g = genome.copy()
    if kind == ModelKind.GENETIC:
        g[:, 2] = 0.0
        g[:, 3] = 0.0
    elif kind == ModelKind.PLASTIC:
        g[:, 2] = 0.0
    elif kind == ModelKind.FIXED_EPI:
        g[:, 0] = params.mu
        g[:, 1] = cfg.fixed_epi_b
        g[:, 2] = cfg.fixed_epi_r
        g[:, 3] = cfg.fixed_epi_p
    return g


def initialize_population(
    kind: ModelKind,
    params: SilvaParameters,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> PopulationState:
    n = cfg.population_size
    genome = np.column_stack(
        [
            rng.normal(params.mu, 0.25, n),
            rng.uniform(0.0, 0.10, n),
            rng.uniform(0.0, 0.30, n),
            rng.uniform(0.0, 0.50, n),
        ]
    )
    genome = _clip_genome(genome, cfg)
    genome = _forced_genes(kind, genome, params, cfg)
    zeros = np.zeros(n, dtype=float)
    return PopulationState(genome=genome, n_initial=zeros.copy(), n_final=zeros.copy())


def _clip_genome(genome: np.ndarray, cfg: GAConfig) -> np.ndarray:
    g = genome.copy()
    bounds = [cfg.mu_bounds, cfg.b_bounds, cfg.r_bounds, cfg.p_bounds]
    for j, (lo, hi) in enumerate(bounds):
        g[:, j] = np.clip(g[:, j], lo, hi)
    return g


def _tournament_indices(log_fitness: np.ndarray, count: int, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    n = log_fitness.size
    k = max(2, cfg.tournament_size)
    candidates = rng.integers(0, n, size=(count, k))
    scores = log_fitness[candidates]
    winners = np.argmax(scores, axis=1)
    return candidates[np.arange(count), winners]


def _reproduce(
    state: PopulationState,
    log_fitness: np.ndarray,
    kind: ModelKind,
    params: SilvaParameters,
    cfg: GAConfig,
    rng: np.random.Generator,
) -> PopulationState:
    n = cfg.population_size

    if kind == ModelKind.FIXED_EPI:
        # Genome is fixed, but reproductive success can still propagate the
        # maternal epigenetic state n through lineages.
        mothers = _tournament_indices(log_fitness, n, cfg, rng)
        genome = state.genome[mothers].copy()
        n_initial = inherited_srna(state.n_final[mothers], state.genome[mothers, 2])
        return PopulationState(genome=genome, n_initial=n_initial, n_final=np.zeros(n))

    elite_count = min(max(0, cfg.elitism), n)
    elite_idx = np.argsort(log_fitness)[-elite_count:] if elite_count else np.array([], dtype=int)
    child_count = n - elite_count

    mothers = _tournament_indices(log_fitness, child_count, cfg, rng)
    fathers = _tournament_indices(log_fitness, child_count, cfg, rng)

    gm = state.genome[mothers]
    gf = state.genome[fathers]
    alpha = rng.uniform(0.0, 1.0, size=(child_count, 4))
    do_cross = rng.random(child_count) < cfg.crossover_rate
    children = gm.copy()
    children[do_cross] = alpha[do_cross] * gm[do_cross] + (1.0 - alpha[do_cross]) * gf[do_cross]

    sigmas = np.array([cfg.mu_sigma, cfg.b_sigma, cfg.r_sigma, cfg.p_sigma])
    mutation_mask = rng.random((child_count, 4)) < cfg.mutation_rate
    children += mutation_mask * rng.normal(0.0, sigmas, size=(child_count, 4))
    children = _clip_genome(children, cfg)
    children = _forced_genes(kind, children, params, cfg)

    # Maternal transmission uses the mother's r_germ at the moment of transfer.
    if kind == ModelKind.EPI:
        child_n = inherited_srna(state.n_final[mothers], state.genome[mothers, 2])
    else:
        child_n = np.zeros(child_count, dtype=float)

    if elite_count:
        elite_genome = state.genome[elite_idx].copy()
        if kind == ModelKind.EPI:
            elite_n = inherited_srna(state.n_final[elite_idx], state.genome[elite_idx, 2])
        else:
            elite_n = np.zeros(elite_count, dtype=float)
        genome = np.vstack([elite_genome, children])
        n_initial = np.concatenate([elite_n, child_n])
    else:
        genome = children
        n_initial = child_n

    return PopulationState(genome=genome, n_initial=n_initial, n_final=np.zeros(n))


def run_population(
    environment: np.ndarray,
    kind: ModelKind,
    *,
    seed: int,
    params: SilvaParameters | None = None,
    cfg: GAConfig | None = None,
) -> list[dict[str, float | int | str]]:
    params = params or SilvaParameters()
    cfg = cfg or GAConfig()
    rng = np.random.default_rng(seed)
    state = initialize_population(kind, params, cfg, rng)

    b_targets = {
        0.1: reference_optimal_b(0.1, params),
        0.9: reference_optimal_b(0.9, params),
    }

    history: list[dict[str, float | int | str]] = []
    for generation, epsilon in enumerate(np.asarray(environment, dtype=float)):
        genome = state.genome
        mu = genome[:, 0]
        b = genome[:, 1]
        r = genome[:, 2]
        p = genome[:, 3]

        b_opt = b_targets[0.1 if epsilon < 0.5 else 0.9]
        logfit, n_final = simulate_development(
            state.n_initial,
            mu,
            b,
            p,
            float(epsilon),
            b_opt,
            params,
        )
        state.n_final = n_final

        # Stable raw-fitness summary. The absolute mean is useful for report
        # plots; selection itself uses log fitness and tournament ranks.
        raw_fitness = np.exp(np.clip(logfit, -700.0, 0.0))
        history.append(
            {
                "generation": generation,
                "model": kind.value,
                "epsilon": float(epsilon),
                "mean_fitness": float(np.mean(raw_fitness)),
                "geometric_mean_fitness": float(np.exp(np.mean(logfit))),
                "mean_log_fitness": float(np.mean(logfit)),
                "best_log_fitness": float(np.max(logfit)),
                "mean_mu": float(np.mean(mu)),
                "mean_b": float(np.mean(b)),
                "mean_r_germ": float(np.mean(r)),
                "mean_p_b": float(np.mean(p)),
                "std_r_germ": float(np.std(r)),
                "std_p_b": float(np.std(p)),
                "mean_n_initial": float(np.mean(state.n_initial)),
                "mean_n_final": float(np.mean(n_final)),
                "b_wmax_reference": float(b_opt),
            }
        )

        state = _reproduce(state, logfit, kind, params, cfg, rng)

    return history
