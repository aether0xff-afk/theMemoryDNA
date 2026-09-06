import numpy as np

from memorydna.model import (
    SilvaParameters,
    inherited_srna,
    log_instantaneous_fitness,
    reference_optimal_b,
    simulate_development,
)


def test_strategy_a_reproduces_reported_adult_srna_level():
    params = SilvaParameters(rk4_substeps=2)
    logw, n_final = simulate_development(
        n_initial=np.array([0.0]),
        mu=np.array([params.mu]),
        b_base=np.array([0.0]),
        p_b=np.array([0.0]),
        epsilon=0.1,
        b_opt=0.0,
        params=params,
    )
    # Silva et al. report ~58.8 at c=20, b=0, d=.1, mu=6.798.
    assert abs(float(n_final[0]) - 58.8) < 0.1
    assert np.isfinite(logw[0])


def test_eq7_pointwise_optima_match_paper_figure_description():
    params = SilvaParameters()
    n = np.linspace(0.0, 12.0, 12001)
    low = n[np.argmax(log_instantaneous_fitness(n, 0.1, 0.0, params))]
    high = n[np.argmax(log_instantaneous_fitness(n, 0.9, 0.0, params))]
    # Paper text reports ~2.85 and ~7.19 respectively.
    assert abs(low - 2.85) < 0.10
    assert abs(high - 7.19) < 0.10


def test_direct_srna_inheritance_rule():
    out = inherited_srna(np.array([100.0, 20.0]), np.array([0.1, 0.5]))
    assert np.allclose(out, [10.0, 10.0])


def test_reference_b_optimum_is_environment_dependent():
    params = SilvaParameters()
    b_benign = reference_optimal_b(0.1, params)
    b_stress = reference_optimal_b(0.9, params)
    assert b_benign <= b_stress
    assert b_stress > 0.0


def _strategy_c_longrun_logfitness(r_germ: float, params: SilvaParameters) -> float:
    """Long-run 50:50 environmental fitness for paper Strategy C."""
    n_initial = np.array([0.0])
    mu = np.array([params.mu])
    b = np.array([0.0])
    p = np.array([0.0])

    # With b=0 and fixed mu, sRNA dynamics are environment independent. Iterate
    # transmission until the zygotic sRNA state reaches its periodic fixed point.
    for _ in range(80):
        _, n_final = simulate_development(n_initial, mu, b, p, 0.1, 0.0, params)
        n_initial = inherited_srna(n_final, r_germ)

    low, _ = simulate_development(n_initial, mu, b, p, 0.1, 0.0, params)
    high, _ = simulate_development(n_initial, mu, b, p, 0.9, 0.0, params)
    return 0.5 * (float(low[0]) + float(high[0]))


def test_strategy_c_transmission_is_advantageous_and_optimal_near_point_one():
    """Regression against Silva Fig. 3 / S5: direct transmission should help.

    The paper reports a selective advantage for Strategy C at mu=6.798 and an
    optimum r_germ around 0.1 in the related two-dimensional S5 landscape.
    This test guards the early-life fitness contribution at t=0.
    """
    params = SilvaParameters(rk4_substeps=1)
    r_grid = np.linspace(0.0, 0.25, 26)
    scores = np.array([_strategy_c_longrun_logfitness(float(r), params) for r in r_grid])
    best_r = float(r_grid[int(np.argmax(scores))])
    assert scores.max() > scores[0]
    assert 0.05 <= best_r <= 0.15
