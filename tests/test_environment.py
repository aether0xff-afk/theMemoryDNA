import numpy as np

from memorydna.environment import balanced_cycle_for_similarity, measured_similarity, regime_schedule


def test_paper_similarity_cycles_are_balanced_and_close_to_targets():
    for seed, target in enumerate((0.11, 0.53, 0.89)):
        cycle = balanced_cycle_for_similarity(target, rng=np.random.default_rng(seed))
        assert len(cycle.epsilon) == 20
        assert np.count_nonzero(cycle.epsilon == 0.1) == 10
        assert np.count_nonzero(cycle.epsilon == 0.9) == 10
        assert abs(cycle.p_epsilon - target) < 0.02
        assert cycle.p_epsilon == measured_similarity(cycle.epsilon)


def test_regime_schedule_length():
    env, meta = regime_schedule([(10, 0.89), (12, 0.11), (8, 0.89)], seed=1)
    assert len(env) == 30
    assert [m["start"] for m in meta] == [0, 10, 22]
    assert [m["end"] for m in meta] == [10, 22, 30]
