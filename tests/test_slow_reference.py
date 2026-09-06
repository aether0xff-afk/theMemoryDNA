import numpy as np

from memorydna.model import SilvaParameters, reference_optimal_b, simulate_development
from memorydna.slow_reference import simulate_scaled, scaled_reference_optimal_b


def test_scale_one_matches_primary_development_solver():
    params = SilvaParameters(rk4_substeps=2)
    n0 = np.array([3.0, 7.0])
    mu = np.array([params.mu, params.mu])
    b0 = np.array([0.0, 0.05])
    pb = np.array([0.2, 0.8])
    bopt = reference_optimal_b(0.9, params)
    ref_logw, ref_n = simulate_development(n0, mu, b0, pb, 0.9, bopt, params)
    got_logw, got_n = simulate_scaled(n0, mu, b0, pb, 0.9, bopt, params, 1.0)
    assert np.allclose(got_logw, ref_logw, rtol=1e-10, atol=1e-10)
    assert np.allclose(got_n, ref_n, rtol=1e-10, atol=1e-10)


def test_slow_scale_changes_optimal_amplification_reference():
    params = SilvaParameters()
    fast = scaled_reference_optimal_b(0.9, 1.0, params)
    slow = scaled_reference_optimal_b(0.9, 0.75, params)
    assert fast >= 0.0
    assert slow >= 0.0
