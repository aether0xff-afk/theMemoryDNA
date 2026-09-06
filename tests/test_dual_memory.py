import numpy as np

from memorydna.dual_memory import (
    DualMemoryConfig,
    MemoryArchitecture,
    amplification_toward_optimum,
    initialize_dual_population,
    run_dual_population,
    simulate_dual_development,
)
from memorydna.model import SilvaParameters


def test_eq6_amplification_update_matches_closed_form():
    b0 = np.array([0.05])
    p = np.array([0.8])
    bopt = np.array([0.25])
    out = amplification_toward_optimum(b0, p, bopt, 20.0, 0.15)
    expected = b0 + p * (bopt - b0) * (1.0 - np.exp(-0.15 * 20.0))
    assert np.allclose(out, expected)


def test_strategy_f_like_germline_memory_changes_but_not_current_soma():
    params = SilvaParameters()
    logw, n_final, b_soma_final, b_germ_final = simulate_dual_development(
        n_initial=np.array([0.0]),
        b_inherited=np.array([0.0]),
        mu=np.array([params.mu]),
        r_germ=np.array([0.0]),
        p_b=np.array([0.8]),
        epsilon=0.9,
        b_opt=np.array([0.25]),
        kind=MemoryArchitecture.MECHANISM,
        params=params,
    )
    assert np.isfinite(logw[0])
    assert n_final[0] > 0.0
    assert b_soma_final[0] == 0.0
    assert b_germ_final[0] > 0.0


def test_architecture_constraints_are_forced_at_initialization():
    params = SilvaParameters()
    cfg = DualMemoryConfig(population_size=20)
    rng = np.random.default_rng(7)

    genetic = initialize_dual_population(MemoryArchitecture.GENETIC, params, cfg, rng)
    assert np.allclose(genetic.genome[:, 1], 0.0)
    assert np.allclose(genetic.genome[:, 2], 0.0)

    state = initialize_dual_population(MemoryArchitecture.STATE, params, cfg, rng)
    assert np.allclose(state.genome[:, 2], 0.0)

    mechanism = initialize_dual_population(MemoryArchitecture.MECHANISM, params, cfg, rng)
    assert np.allclose(mechanism.genome[:, 1], 0.0)


def test_fixed_mu_mode_preserves_silva_mu():
    params = SilvaParameters()
    cfg = DualMemoryConfig(population_size=16, evolve_mu=False, bopt_mu_grid_points=3)
    env = np.array([0.1, 0.9, 0.9, 0.1], dtype=float)
    hist = run_dual_population(env, MemoryArchitecture.DUAL, seed=3, params=params, cfg=cfg)
    assert len(hist) == len(env)
    assert all(abs(float(row["mean_mu"]) - params.mu) < 1e-12 for row in hist)


def test_mechanism_memory_persists_across_generations_in_stress():
    params = SilvaParameters()
    cfg = DualMemoryConfig(population_size=24, evolve_mu=False, bopt_mu_grid_points=3)
    env = np.full(8, 0.9)
    hist = run_dual_population(env, MemoryArchitecture.MECHANISM, seed=11, params=params, cfg=cfg)
    assert float(hist[0]["mean_b_inherited"]) == 0.0
    assert float(hist[-1]["mean_b_inherited"]) > 0.0


def test_dual_memory_smoke_run_records_both_channels():
    params = SilvaParameters()
    cfg = DualMemoryConfig(population_size=18, evolve_mu=False, bopt_mu_grid_points=3)
    env = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9], dtype=float)
    hist = run_dual_population(env, MemoryArchitecture.DUAL, seed=5, params=params, cfg=cfg)
    assert len(hist) == len(env)
    assert all("mean_n_initial" in row and "mean_b_inherited" in row for row in hist)
    assert float(hist[-1]["mean_b_inherited"]) >= 0.0
