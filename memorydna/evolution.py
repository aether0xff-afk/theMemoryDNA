"""Evolutionary extension of the Silva et al. small-RNA model.

Novel layer: a finite population evolves the strength P_b of a Strategy-F-like
transgenerational plastic response by fitness-proportional selection + mutation.
The inherited amplification state acts as an epigenetic state variable.

Important distinction:
- Biological dynamics/fitness/constants are from Silva et al. (2021).
- Population size, mutation SD, number of generations, and the evolutionary
  algorithm are computational experiment choices introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .model import SilvaParams, integrate_constant_b


P_TO_SWITCHES = {0.11: 17, 0.53: 9, 0.89: 2}


@dataclass(frozen=True)
class GAConfig:
    population: int = 3000
    generations: int = 1500
    mutation_sd: float = 0.015
    replicates: int = 12
    dynamics_scale: float = 0.75
    b_max_unscaled: float = 4.0
    b_grid_points: int = 801
    pb_grid_points: int = 101
    integration_substeps: int = 8


def environmental_similarity(sequence: np.ndarray) -> float:
    switches = int(np.sum(sequence[1:] != sequence[:-1]))
    return 1.0 - switches / (len(sequence) - 1)


def make_environment_cycle(p_epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Construct the paper's G=20, 50:50 benign/stress environment cycle.

    The requested p_epsilon labels correspond to k=17,9,2 switches, giving
    exact realized similarities 0.1053, 0.5263, 0.8947, rounded in the paper
    to 0.11, 0.53, 0.89.
    """
    if p_epsilon not in P_TO_SWITCHES:
        raise ValueError(f"p_epsilon must be one of {tuple(P_TO_SWITCHES)}")
    target = P_TO_SWITCHES[p_epsilon]
    base = np.array([0.1] * 10 + [0.9] * 10, dtype=float)
    for _ in range(200_000):
        seq = rng.permutation(base)
        if int(np.sum(seq[1:] != seq[:-1])) == target:
            return seq
    raise RuntimeError("Failed to sample an environment cycle with target switches")


class FitnessLookup:
    """Fast bilinear lookup of Eq.1+Eq.7+Eq.8A fitness over (b, P_b)."""

    def __init__(self, params: SilvaParams, config: GAConfig):
        self.params = params
        self.config = config
        self.b_grid = np.linspace(0.0, config.b_max_unscaled, config.b_grid_points)
        self.pb_grid = np.linspace(0.0, 1.0, config.pb_grid_points)
        traj = integrate_constant_b(
            self.b_grid,
            params=params,
            dynamics_scale=config.dynamics_scale,
            substeps=config.integration_substeps,
        )
        self.tables = {}
        for eps in (params.eps_benign, params.eps_stress):
            tr = traj[:, :, None]
            pb = self.pb_grid[None, None, :]
            from .model import point_fitness
            w = point_fitness(tr, eps, pb, params)
            self.tables[eps] = np.exp(np.mean(np.log(np.clip(w, 1e-300, None)), axis=0))

        self.b_opt = {
            eps: float(self.b_grid[np.argmax(self.tables[eps][:, 0])])
            for eps in (params.eps_benign, params.eps_stress)
        }

    def fitness(self, b_unscaled: np.ndarray, pb: np.ndarray, epsilon: float) -> np.ndarray:
        table = self.tables[float(epsilon)]
        b = np.clip(np.asarray(b_unscaled, dtype=float), self.b_grid[0], self.b_grid[-1])
        p = np.clip(np.asarray(pb, dtype=float), 0.0, 1.0)
        xb = (b - self.b_grid[0]) / (self.b_grid[-1] - self.b_grid[0]) * (len(self.b_grid) - 1)
        xp = p * (len(self.pb_grid) - 1)
        i = np.floor(xb).astype(int)
        j = np.floor(xp).astype(int)
        i = np.clip(i, 0, len(self.b_grid) - 2)
        j = np.clip(j, 0, len(self.pb_grid) - 2)
        fb = xb - i
        fp = xp - j
        return (
            (1-fb)*(1-fp)*table[i, j]
            + fb*(1-fp)*table[i+1, j]
            + (1-fb)*fp*table[i, j+1]
            + fb*fp*table[i+1, j+1]
        )


def run_replicate(p_epsilon: float, *, seed: int, lookup: FitnessLookup, config: GAConfig):
    """Run one maternal-lineage evolutionary simulation.

    Genetic state is P_b. Epigenetic state is the inherited maternal germline
    amplification state b#. The current soma keeps inherited b, while the
    germline moves P_b of the way toward the current environment's b_Wmax with
    the paper's delayed-response factor 1-exp(-a*t). The adult germline state is
    then inherited by offspring.

    Crossover is intentionally omitted so the mother-offspring epigenetic
    lineage stays intact; selection + mutation form the evolutionary layer.
    """
    rng = np.random.default_rng(seed)
    env = make_environment_cycle(p_epsilon, rng)
    n = config.population
    pb = rng.uniform(0.0, 1.0, n)
    b_inherited = np.zeros(n, dtype=float)
    response_at_adult = 1.0 - math.exp(-lookup.params.a * lookup.params.c)
    trace = np.empty((config.generations, 6), dtype=float)

    for g in range(config.generations):
        eps = float(env[g % len(env)])
        fitness = lookup.fitness(b_inherited, pb, eps)
        b_target = lookup.b_opt[eps]
        b_germline_final = b_inherited + pb * (b_target - b_inherited) * response_at_adult

        trace[g] = (
            g, eps, float(np.mean(pb)), float(np.median(pb)),
            float(np.mean(fitness)), float(np.mean(b_inherited)),
        )

        probs = fitness / np.sum(fitness)
        mothers = rng.choice(n, size=n, replace=True, p=probs)
        pb = np.clip(pb[mothers] + rng.normal(0.0, config.mutation_sd, n), 0.0, 1.0)
        b_inherited = b_germline_final[mothers]

    return {
        "p_label": p_epsilon,
        "p_realized": environmental_similarity(env),
        "environment": env,
        "trace": trace,
        "final_pb": pb,
        "final_b": b_inherited,
    }


def run_regime_shift(*, seed: int, lookup: FitnessLookup, config: GAConfig, segment_generations: int = 500):
    """Novel stress test: high -> low -> high environmental autocorrelation."""
    rng = np.random.default_rng(seed)
    n = config.population
    pb = rng.uniform(0.0, 1.0, n)
    b_inherited = np.zeros(n, dtype=float)
    response_at_adult = 1.0 - math.exp(-lookup.params.a * lookup.params.c)
    regimes = [0.89, 0.11, 0.89]
    cycles = [make_environment_cycle(p, rng) for p in regimes]
    rows = []
    g_global = 0
    for regime_idx, (p, env) in enumerate(zip(regimes, cycles)):
        for local_g in range(segment_generations):
            eps = float(env[local_g % 20])
            fitness = lookup.fitness(b_inherited, pb, eps)
            b_target = lookup.b_opt[eps]
            b_germline_final = b_inherited + pb * (b_target - b_inherited) * response_at_adult
            rows.append((g_global, regime_idx, p, eps, float(np.mean(pb)), float(np.mean(fitness)), float(np.mean(b_inherited))))
            probs = fitness / np.sum(fitness)
            mothers = rng.choice(n, size=n, replace=True, p=probs)
            pb = np.clip(pb[mothers] + rng.normal(0.0, config.mutation_sd, n), 0.0, 1.0)
            b_inherited = b_germline_final[mothers]
            g_global += 1
    return np.asarray(rows, dtype=float)
