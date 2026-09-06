"""Silva, Otto & Immler (2021) small-RNA model primitives.

Biological equations and default constants come from:
Silva WTAF, Otto SP, Immler S. PLoS Genetics 17(5): e1009581 (2021).
DOI: 10.1371/journal.pgen.1009581

The evolutionary-population layer in evolution.py is our computational extension.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SilvaParams:
    # Defaults reported in the paper.
    mu: float = 6.798
    d: float = 0.1
    m: float = 5.0
    c: int = 20
    alpha: float = 15.0
    beta: float = 0.1
    h: float = 5.0
    Cn: float = 1e-5
    Cb_multiple: float = 50.0
    a: float = 0.15
    eps_benign: float = 0.1
    eps_stress: float = 0.9

    @property
    def Cb(self) -> float:
        return self.Cb_multiple * self.Cn


def srna_derivative(n: np.ndarray | float, b: np.ndarray | float, *, mu: float, d: float, m: float):
    """Eq. 1: dn/dt = ((b / (1 + b n / m)) - d) n + mu."""
    return ((b / (1.0 + b * n / m)) - d) * n + mu


def srna_phenotype(n: np.ndarray | float, h: float = 5.0):
    """Logistic-shaped sRNA phenotype used inside Eq. 7.

    Algebraically equivalent to (exp(n)-1)/(exp(n)+exp(h)-2), written in a
    numerically stable form.
    """
    n = np.asarray(n, dtype=float)
    exp_neg_n = np.exp(-np.clip(n, 0.0, 700.0))
    return (1.0 - exp_neg_n) / (1.0 + (math.exp(h) - 2.0) * exp_neg_n)


def point_fitness(n, epsilon: float, plasticity: float, params: SilvaParams = SilvaParams()):
    """Eq. 7 instantaneous fitness W(n_t, epsilon_g, P_b)."""
    pheno = srna_phenotype(n, params.h)
    cost = 1.0 + params.Cn * np.asarray(n, dtype=float) + params.Cb * plasticity
    match = np.exp(-(params.beta + params.alpha * epsilon) * (pheno - epsilon) ** 2)
    return match / cost


def integrate_constant_b(
    b: np.ndarray | float,
    *,
    n0: np.ndarray | float = 0.0,
    params: SilvaParams = SilvaParams(),
    dynamics_scale: float = 1.0,
    substeps: int = 10,
):
    """Integrate Eq. 1 and return n at integer cell divisions 0..c using RK4.

    The paper's slow/fast analyses multiply b, d, m and mu by the same scale.
    dynamics_scale=0.75 reproduces the slow-dynamics condition described for
    Fig. 3/S8; 1.0 is the default model.
    """
    b = np.asarray(b, dtype=float) * dynamics_scale
    n = np.broadcast_to(np.asarray(n0, dtype=float), np.broadcast(b, np.asarray(n0)).shape).copy()
    mu = params.mu * dynamics_scale
    d = params.d * dynamics_scale
    m = params.m * dynamics_scale
    dt = 1.0 / substeps
    samples = [n.copy()]

    for _cell in range(params.c):
        for _ in range(substeps):
            f = lambda x: srna_derivative(x, b, mu=mu, d=d, m=m)
            k1 = f(n)
            k2 = f(n + 0.5 * dt * k1)
            k3 = f(n + 0.5 * dt * k2)
            k4 = f(n + dt * k3)
            n = np.maximum(0.0, n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))
        samples.append(n.copy())
    return np.stack(samples, axis=0)


def life_fitness_from_trajectory(trajectory, epsilon: float, plasticity, params: SilvaParams = SilvaParams()):
    """Eq. 8A: geometric mean fitness across development."""
    w = point_fitness(trajectory, epsilon, plasticity, params)
    return np.exp(np.mean(np.log(np.clip(w, 1e-300, None)), axis=0))


def paper_sanity_value(params: SilvaParams = SilvaParams()) -> float:
    """Adult n for strategy A under paper defaults; paper reports ~58.8."""
    return float(integrate_constant_b(0.0, params=params, dynamics_scale=1.0)[-1])
