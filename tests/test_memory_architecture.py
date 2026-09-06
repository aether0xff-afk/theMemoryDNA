import numpy as np

from memorydna.ga import GAConfig
from memorydna.memory_architecture import (
    MemoryKind,
    adult_germline_amplification,
    initialize_memory_population,
    run_memory_population,
)
from memorydna.memory_environment import matched_pattern_cycle
from memorydna.model import SilvaParameters, effective_amplification, reference_optimal_b


def test_adult_germline_amplification_matches_silva_eq6():
    params = SilvaParameters()
    b0 = np.array([0.02, 0.10])
    p = np.array([0.4, 0.8])
    b_opt = reference_optimal_b(0.9, params)
    expected = effective_amplification(
        b0, p, b_opt, params.cell_divisions, params.plasticity_delay_a
    )
    got = adult_germline_amplification(b0, p, b_opt, params)
    assert np.allclose(got, expected)


def test_architecture_flags_separate_cross_generation_channels():
    assert not MemoryKind.NO_MEMORY.transmits_srna
    assert not MemoryKind.NO_MEMORY.transmits_mechanism
    assert MemoryKind.STATE_MEMORY.transmits_srna
    assert not MemoryKind.STATE_MEMORY.transmits_mechanism
    assert not MemoryKind.MECHANISM_MEMORY.transmits_srna
    assert MemoryKind.MECHANISM_MEMORY.transmits_mechanism
    assert MemoryKind.DUAL_MEMORY.transmits_srna
    assert MemoryKind.DUAL_MEMORY.transmits_mechanism


def test_non_transcript_architectures_force_r_germ_zero():
    params = SilvaParameters()
    cfg = GAConfig(population_size=16)
    rng = np.random.default_rng(7)
    for kind in (MemoryKind.NO_MEMORY, MemoryKind.MECHANISM_MEMORY):
        state = initialize_memory_population(kind, params, cfg, rng)
        assert np.allclose(state.genome[:, 2], 0.0)


def test_matched_patterns_have_same_balance_and_similarity():
    patterns = [matched_pattern_cycle(name, changes=5, seed=7) for name in ("regular", "clustered", "stochastic")]
    for cycle in patterns:
        assert np.count_nonzero(cycle.epsilon == 0.1) == 10
        assert np.count_nonzero(cycle.epsilon == 0.9) == 10
        assert cycle.changes == 5
        assert abs(cycle.p_epsilon - (1.0 - 5.0 / 19.0)) < 1e-12
    assert patterns[0].run_lengths != patterns[1].run_lengths


def test_memory_population_smoke_all_architectures():
    params = SilvaParameters()
    cfg = GAConfig(population_size=18)
    env = np.tile(np.array([0.1, 0.9]), 10)
    for i, kind in enumerate(MemoryKind):
        history = run_memory_population(env, kind, seed=100 + i, params=params, cfg=cfg)
        assert len(history) == len(env)
        last = history[-1]
        assert np.isfinite(last["mean_fitness"])
        assert np.isfinite(last["mean_b_germ_final"])
        if not kind.transmits_srna:
            assert last["mean_r_germ"] == 0.0
