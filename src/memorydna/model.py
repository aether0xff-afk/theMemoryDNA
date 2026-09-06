from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class SilvaParameters:
    """Biological/model parameters taken from Silva, Otto & Immler (2021).

    Time is measured in cell divisions. The defaults match the paper's main
    simulations unless otherwise noted.
    """

    mu: float = 6.798
    d: float = 0.1
    m: float = 5.0
    cell_divisions: int = 20
    alpha: float = 15.0
    beta: float = 0.1
    h: float = 5.0
    c_n: float = 1.0e-5
    c_b: float = 50.0e-5
    plasticity_delay_a: float = 0.15
    rk4_substeps: int = 1


def srna_rhs(n: np.ndarray | float, mu: np.ndarray | float, b: np.ndarray | float, *, d: float, m: float):
    """Silva et al. Eq. 1.

    dn/dt = ( b/(1 + b*n/m) - d )*n + mu
    """

    n_arr = np.asarray(n, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    denom = 1.0 + (b_arr * n_arr / m)
    return ((b_arr / denom) - d) * n_arr + mu_arr


def srna_phenotype(n: np.ndarray | float, *, h: float = 5.0):
    """Silva et al. Eq. 7 sRNA phenotype term."""

    n_arr = np.asarray(n, dtype=float)
    # Exact phenotype in Eq. 7: (exp(n)-1)/(exp(n)+exp(h)-2).
    # Divide numerator and denominator by exp(n) for numerical stability.
    exp_neg_n = np.exp(-np.clip(n_arr, 0.0, 700.0))
    return (1.0 - exp_neg_n) / (1.0 + (np.exp(h) - 2.0) * exp_neg_n)


def log_instantaneous_fitness(
    n: np.ndarray | float,
    epsilon: float,
    p_b: np.ndarray | float,
    params: SilvaParameters,
):
    """Natural log of Silva et al. Eq. 7.

    Eq. 7 is evaluated in log space for numerical stability. This reproduces
    the paper's reported instantaneous optima near n≈2.85 for ε=0.1 and
    n≈7.19 for ε=0.9 under the default parameters.
    """

    n_arr = np.asarray(n, dtype=float)
    p_arr = np.asarray(p_b, dtype=float)
    phenotype = srna_phenotype(n_arr, h=params.h)
    mismatch = epsilon - phenotype
    cost = np.log1p(params.c_n * n_arr + params.c_b * p_arr)
    return -cost - (params.beta + params.alpha * epsilon) * mismatch * mismatch


def instantaneous_fitness(
    n: np.ndarray | float,
    epsilon: float,
    p_b: np.ndarray | float,
    params: SilvaParameters,
):
    return np.exp(log_instantaneous_fitness(n, epsilon, p_b, params))


def _plasticity_fraction(t: float, a: float) -> float:
    if np.isinf(a):
        return 1.0 if t > 0.0 else 0.0
    return 1.0 - float(np.exp(-a * t))


def effective_amplification(
    b_base: np.ndarray,
    p_b: np.ndarray,
    b_opt: float,
    t: float,
    a: float,
) -> np.ndarray:
    """Somatic plasticity response corresponding to Silva et al. Eq. 4.

    In this project the inherited amplification-state channel (paper Eqs. 5-6)
    is intentionally not used as a second epigenome. The epigenome is the
    transmitted sRNA abundance n itself; b_base and P_b remain genetic traits.
    """

    frac = _plasticity_fraction(t, a)
    return b_base + p_b * (b_opt - b_base) * frac


def simulate_development(
    n_initial: np.ndarray,
    mu: np.ndarray,
    b_base: np.ndarray,
    p_b: np.ndarray,
    epsilon: float,
    b_opt: float,
    params: SilvaParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one generation of development for a vectorized population.

    Returns
    -------
    log_lifetime_fitness:
        Mean log fitness over development, i.e. log of the geometric mean.
    n_final:
        Adult/germline sRNA abundance at the end of development.
    """

    n = np.maximum(np.asarray(n_initial, dtype=float).copy(), 0.0)
    mu = np.asarray(mu, dtype=float)
    b_base = np.asarray(b_base, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    substeps = max(1, int(params.rk4_substeps))
    dt = 1.0 / substeps
    total_steps = params.cell_divisions * substeps
    logw_sum = np.zeros_like(n, dtype=float)

    for step in range(total_steps):
        t0 = step * dt

        def rhs(x: np.ndarray, t: float) -> np.ndarray:
            b_t = effective_amplification(
                b_base,
                p_b,
                b_opt,
                t,
                params.plasticity_delay_a,
            )
            return srna_rhs(x, mu, b_t, d=params.d, m=params.m)

        k1 = rhs(n, t0)
        k2 = rhs(n + 0.5 * dt * k1, t0 + 0.5 * dt)
        k3 = rhs(n + 0.5 * dt * k2, t0 + 0.5 * dt)
        k4 = rhs(n + dt * k3, t0 + dt)
        n = np.maximum(n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)

        # Fitness is sampled uniformly through developmental time. The mean of
        # log fitness is the log geometric mean used by Silva et al. (Eq. 8A).
        logw_sum += log_instantaneous_fitness(n, epsilon, p_b, params)

    return logw_sum / total_steps, n


def _reference_lifetime_logfitness_for_b(b: float, epsilon: float, params: SilvaParameters) -> float:
    n0 = np.array([0.0])
    mu = np.array([params.mu])
    b_base = np.array([b])
    p_b = np.array([0.0])
    logw, _ = simulate_development(n0, mu, b_base, p_b, epsilon, b, params)
    return float(logw[0])


@lru_cache(maxsize=64)
def reference_optimal_b(
    epsilon: float,
    params: SilvaParameters = SilvaParameters(),
    b_max: float = 1.0,
    grid_points: int = 401,
) -> float:
    """Numerically obtain b_Wmax for the paper's reference genotype.

    Silva et al. define b_Wmax as the amplification rate maximizing fitness in
    the current environment. We optimize that definition numerically rather than
    hard-code a target chosen by this project.
    """

    grid = np.linspace(0.0, b_max, grid_points)
    scores = np.array(
        [_reference_lifetime_logfitness_for_b(float(b), epsilon, params) for b in grid]
    )
    best = int(np.argmax(scores))

    # Small local refinement around the best grid point; dependency-free and
    # deterministic, so the repository does not need scipy.
    lo = grid[max(0, best - 1)]
    hi = grid[min(len(grid) - 1, best + 1)]
    fine = np.linspace(lo, hi, 101)
    fine_scores = np.array(
        [_reference_lifetime_logfitness_for_b(float(b), epsilon, params) for b in fine]
    )
    return float(fine[int(np.argmax(fine_scores))])


def inherited_srna(n_final_mother: np.ndarray | float, r_germ_mother: np.ndarray | float):
    """Silva et al. direct sRNA inheritance rule: n_initial,g+1 = r_germ*n_final,g."""

    return np.maximum(
        np.asarray(n_final_mother, dtype=float) * np.asarray(r_germ_mother, dtype=float),
        0.0,
    )
