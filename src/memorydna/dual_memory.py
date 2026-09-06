from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .model import SilvaParameters, inherited_srna, log_instantaneous_fitness, reference_optimal_b, srna_rhs


class MemoryArchitecture(str, Enum):
    """Paper-grounded memory architectures used in the final study.

    genetic
        No plasticity and no non-genetic inheritance.
    somatic
        Silva strategy D: somatic plasticity only; amplification resets each generation.
    state
        Silva strategy C-like: direct maternal sRNA transmission only.
    mechanism
        Silva strategy F: germline amplification state changes plastically and is inherited.
    dual
        Direct sRNA transmission + strategy-F-like germline amplification-state inheritance.
    full_dual
        Direct sRNA transmission + Silva strategy E: both soma and germline respond plastically,
        and the germline amplification state is inherited.
    """

    GENETIC = "genetic"
    SOMATIC = "somatic"
    STATE = "state-memory"
    MECHANISM = "mechanism-memory"
    DUAL = "dual-memory"
    FULL_DUAL = "full-dual-memory"


@dataclass(frozen=True)
class DualMemoryConfig:
    population_size: int = 140
    mutation_rate: float = 0.25
    crossover_rate: float = 0.85
    tournament_size: int = 3
    elitism: int = 2

    # Genome order: mu, r_germ, P_b. Fixed amplification is held at zero so that
    # the two epigenetic channels can be isolated cleanly.
    mu_bounds: tuple[float, float] = (0.0, 10.0)
    r_bounds: tuple[float, float] = (0.0, 0.95)
    p_bounds: tuple[float, float] = (0.0, 1.0)
    mu_sigma: float = 0.20
    r_sigma: float = 0.035
    p_sigma: float = 0.035
    evolve_mu: bool = True

    # Genotype-specific b_Wmax lookup. The lookup is built once per run and
    # linearly interpolated during evolution.
    bopt_mu_grid_points: int = 25


@dataclass
class DualPopulationState:
    genome: np.ndarray  # [N,3] => mu, r_germ, P_b
    n_initial: np.ndarray
    b_inherited: np.ndarray
    n_final: np.ndarray
    b_germ_final: np.ndarray


def uses_state_memory(kind: MemoryArchitecture) -> bool:
    return kind in {MemoryArchitecture.STATE, MemoryArchitecture.DUAL, MemoryArchitecture.FULL_DUAL}


def uses_germline_memory(kind: MemoryArchitecture) -> bool:
    return kind in {MemoryArchitecture.MECHANISM, MemoryArchitecture.DUAL, MemoryArchitecture.FULL_DUAL}


def uses_somatic_plasticity(kind: MemoryArchitecture) -> bool:
    return kind in {MemoryArchitecture.SOMATIC, MemoryArchitecture.FULL_DUAL}


def _plastic_fraction(t: float, a: float) -> float:
    if np.isinf(a):
        return 1.0 if t > 0.0 else 0.0
    return 1.0 - float(np.exp(-a * t))


def amplification_toward_optimum(
    b_initial: np.ndarray,
    p_b: np.ndarray,
    b_opt: np.ndarray,
    t: float,
    a: float,
) -> np.ndarray:
    """Silva Eqs. 4 and 6 in vector form.

    b_t = b_initial + P_b (b_Wmax - b_initial) (1-exp(-a t))
    """

    return b_initial + p_b * (b_opt - b_initial) * _plastic_fraction(t, a)


def build_bopt_lookup(
    params: SilvaParameters,
    cfg: DualMemoryConfig,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Precompute b_Wmax(mu, epsilon) for the co-evolution experiment.

    Silva defines b_Wmax for a fixed biological parameter set. Because this final
    study additionally allows mu to evolve, the optimum amplification target is
    recomputed over a mu grid rather than incorrectly using the resident optimum
    at mu=6.798 for every genotype.
    """

    if cfg.evolve_mu:
        lo, hi = cfg.mu_bounds
        mu_grid = np.linspace(lo, hi, max(3, int(cfg.bopt_mu_grid_points)))
    else:
        mu_grid = np.asarray([params.mu], dtype=float)

    lookup: dict[float, np.ndarray] = {}
    for epsilon in (0.1, 0.9):
        vals = []
        for mu in mu_grid:
            p = replace(params, mu=float(mu))
            vals.append(reference_optimal_b(float(epsilon), p))
        lookup[float(epsilon)] = np.asarray(vals, dtype=float)
    return mu_grid, lookup


def interpolate_bopt(
    mu: np.ndarray,
    epsilon: float,
    mu_grid: np.ndarray,
    lookup: dict[float, np.ndarray],
) -> np.ndarray:
    key = 0.1 if epsilon < 0.5 else 0.9
    if mu_grid.size == 1:
        return np.full_like(mu, lookup[key][0], dtype=float)
    return np.interp(mu, mu_grid, lookup[key])


def simulate_dual_development(
    *,
    n_initial: np.ndarray,
    b_inherited: np.ndarray,
    mu: np.ndarray,
    r_germ: np.ndarray,
    p_b: np.ndarray,
    epsilon: float,
    b_opt: np.ndarray,
    kind: MemoryArchitecture,
    params: SilvaParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one generation with Silva strategies C/D/E/F as composable channels.

    Soma
    ----
    - genetic/state: b*=0
    - somatic: strategy D, b* moves from 0 toward current b_Wmax
    - mechanism/dual: strategy F, b* remains equal to inherited b#
    - full_dual: strategy E, b* moves from inherited b# toward current b_Wmax

    Germline
    --------
    - genetic/somatic/state: b# remains 0
    - mechanism/dual/full_dual: Eq. 6 updates inherited b# toward current b_Wmax

    Lifetime fitness is integrated from t=0 with the trapezoidal rule, matching
    the corrected Eq. 8A implementation used elsewhere in the repository.
    """

    del r_germ  # transmission happens during reproduction, not development
    n = np.maximum(np.asarray(n_initial, dtype=float).copy(), 0.0)
    b0 = np.maximum(np.asarray(b_inherited, dtype=float).copy(), 0.0)
    mu = np.asarray(mu, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    b_opt = np.asarray(b_opt, dtype=float)

    substeps = max(1, int(params.rk4_substeps))
    dt = 1.0 / substeps
    total_steps = params.cell_divisions * substeps
    logw_integral = np.zeros_like(n, dtype=float)

    def soma_b(t: float) -> np.ndarray:
        if kind == MemoryArchitecture.SOMATIC:
            zeros = np.zeros_like(b0)
            return amplification_toward_optimum(zeros, p_b, b_opt, t, params.plasticity_delay_a)
        if kind == MemoryArchitecture.FULL_DUAL:
            return amplification_toward_optimum(b0, p_b, b_opt, t, params.plasticity_delay_a)
        if kind in {MemoryArchitecture.MECHANISM, MemoryArchitecture.DUAL}:
            return b0
        return np.zeros_like(b0)

    for step in range(total_steps):
        t0 = step * dt
        logw_start = log_instantaneous_fitness(n, epsilon, p_b, params)

        def rhs(x: np.ndarray, t: float) -> np.ndarray:
            return srna_rhs(x, mu, soma_b(t), d=params.d, m=params.m)

        k1 = rhs(n, t0)
        k2 = rhs(n + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = rhs(n + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = rhs(n + dt * k3, t0 + dt)
        n_next = np.maximum(n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)

        logw_end = log_instantaneous_fitness(n_next, epsilon, p_b, params)
        logw_integral += 0.5 * (logw_start + logw_end) * dt
        n = n_next

    if uses_germline_memory(kind):
        b_germ_final = amplification_toward_optimum(
            b0, p_b, b_opt, float(params.cell_divisions), params.plasticity_delay_a
        )
    else:
        b_germ_final = np.zeros_like(b0)

    b_soma_final = soma_b(float(params.cell_divisions))
    return logw_integral / params.cell_divisions, n, b_soma_final, b_germ_final


def _clip_genome(genome: np.ndarray, cfg: DualMemoryConfig, params: SilvaParameters) -> np.ndarray:
    g = genome.copy()
    g[:, 0] = np.clip(g[:, 0], *cfg.mu_bounds)
    g[:, 1] = np.clip(g[:, 1], *cfg.r_bounds)
    g[:, 2] = np.clip(g[:, 2], *cfg.p_bounds)
    if not cfg.evolve_mu:
        g[:, 0] = params.mu
    return g


def _force_architecture(
    genome: np.ndarray,
    kind: MemoryArchitecture,
    params: SilvaParameters,
    cfg: DualMemoryConfig,
) -> np.ndarray:
    g = _clip_genome(genome, cfg, params)
    if kind in {MemoryArchitecture.GENETIC, MemoryArchitecture.SOMATIC, MemoryArchitecture.MECHANISM}:
        g[:, 1] = 0.0
    if kind in {MemoryArchitecture.GENETIC, MemoryArchitecture.STATE}:
        g[:, 2] = 0.0
    return g


def initialize_dual_population(
    kind: MemoryArchitecture,
    params: SilvaParameters,
    cfg: DualMemoryConfig,
    rng: np.random.Generator,
) -> DualPopulationState:
    n = cfg.population_size
    genome = np.column_stack(
        [
            rng.normal(params.mu, 0.25, n),
            rng.uniform(0.0, 0.30, n),
            rng.uniform(0.0, 0.50, n),
        ]
    )
    genome = _force_architecture(genome, kind, params, cfg)
    zeros = np.zeros(n, dtype=float)
    return DualPopulationState(
        genome=genome,
        n_initial=zeros.copy(),
        b_inherited=zeros.copy(),
        n_final=zeros.copy(),
        b_germ_final=zeros.copy(),
    )


def _tournament(log_fitness: np.ndarray, count: int, cfg: DualMemoryConfig, rng: np.random.Generator) -> np.ndarray:
    n = log_fitness.size
    k = max(2, cfg.tournament_size)
    candidates = rng.integers(0, n, size=(count, k))
    scores = log_fitness[candidates]
    winners = np.argmax(scores, axis=1)
    return candidates[np.arange(count), winners]


def _reproduce(
    state: DualPopulationState,
    log_fitness: np.ndarray,
    kind: MemoryArchitecture,
    params: SilvaParameters,
    cfg: DualMemoryConfig,
    rng: np.random.Generator,
) -> DualPopulationState:
    n = cfg.population_size
    elite_count = min(max(0, cfg.elitism), n)
    elite_idx = np.argsort(log_fitness)[-elite_count:] if elite_count else np.array([], dtype=int)
    child_count = n - elite_count

    mothers = _tournament(log_fitness, child_count, cfg, rng)
    fathers = _tournament(log_fitness, child_count, cfg, rng)
    gm = state.genome[mothers]
    gf = state.genome[fathers]

    alpha = rng.uniform(0.0, 1.0, size=(child_count, 3))
    do_cross = rng.random(child_count) < cfg.crossover_rate
    children = gm.copy()
    children[do_cross] = alpha[do_cross] * gm[do_cross] + (1.0 - alpha[do_cross]) * gf[do_cross]

    sigmas = np.array([cfg.mu_sigma, cfg.r_sigma, cfg.p_sigma], dtype=float)
    mutation_mask = rng.random((child_count, 3)) < cfg.mutation_rate
    children += mutation_mask * rng.normal(0.0, sigmas, size=(child_count, 3))
    children = _force_architecture(children, kind, params, cfg)

    if uses_state_memory(kind):
        child_n = inherited_srna(state.n_final[mothers], state.genome[mothers, 1])
    else:
        child_n = np.zeros(child_count, dtype=float)

    if uses_germline_memory(kind):
        child_b = state.b_germ_final[mothers].copy()
    else:
        child_b = np.zeros(child_count, dtype=float)

    if elite_count:
        elite_genome = state.genome[elite_idx].copy()
        elite_n = (
            inherited_srna(state.n_final[elite_idx], state.genome[elite_idx, 1])
            if uses_state_memory(kind)
            else np.zeros(elite_count, dtype=float)
        )
        elite_b = state.b_germ_final[elite_idx].copy() if uses_germline_memory(kind) else np.zeros(elite_count)
        genome = np.vstack([elite_genome, children])
        n_initial = np.concatenate([elite_n, child_n])
        b_inherited = np.concatenate([elite_b, child_b])
    else:
        genome = children
        n_initial = child_n
        b_inherited = child_b

    zeros = np.zeros(n, dtype=float)
    return DualPopulationState(
        genome=genome,
        n_initial=n_initial,
        b_inherited=b_inherited,
        n_final=zeros.copy(),
        b_germ_final=zeros.copy(),
    )


def run_dual_population(
    environment: np.ndarray,
    kind: MemoryArchitecture,
    *,
    seed: int,
    params: SilvaParameters | None = None,
    cfg: DualMemoryConfig | None = None,
) -> list[dict[str, float | int | str]]:
    params = params or SilvaParameters()
    cfg = cfg or DualMemoryConfig()
    rng = np.random.default_rng(seed)
    state = initialize_dual_population(kind, params, cfg, rng)
    mu_grid, bopt_lookup = build_bopt_lookup(params, cfg)

    history: list[dict[str, float | int | str]] = []
    for generation, epsilon in enumerate(np.asarray(environment, dtype=float)):
        genome = state.genome
        mu = genome[:, 0]
        r = genome[:, 1]
        p = genome[:, 2]
        b_opt = interpolate_bopt(mu, float(epsilon), mu_grid, bopt_lookup)

        logfit, n_final, b_soma_final, b_germ_final = simulate_dual_development(
            n_initial=state.n_initial,
            b_inherited=state.b_inherited,
            mu=mu,
            r_germ=r,
            p_b=p,
            epsilon=float(epsilon),
            b_opt=b_opt,
            kind=kind,
            params=params,
        )
        state.n_final = n_final
        state.b_germ_final = b_germ_final
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
                "mean_r_germ": float(np.mean(r)),
                "mean_p_b": float(np.mean(p)),
                "mean_n_initial": float(np.mean(state.n_initial)),
                "mean_n_final": float(np.mean(n_final)),
                "mean_b_inherited": float(np.mean(state.b_inherited)),
                "mean_b_soma_final": float(np.mean(b_soma_final)),
                "mean_b_germ_final": float(np.mean(b_germ_final)),
                "mean_b_wmax": float(np.mean(b_opt)),
            }
        )
        state = _reproduce(state, logfit, kind, params, cfg, rng)

    return history
