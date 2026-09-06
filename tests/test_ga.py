import numpy as np

from memorydna.ga import GAConfig, ModelKind, run_population
from memorydna.model import SilvaParameters


def test_epi_ga_smoke_run_and_ablation_constraints():
    env = np.array([0.1, 0.9, 0.9, 0.1, 0.1, 0.9])
    cfg = GAConfig(population_size=12, elitism=1)
    params = SilvaParameters()

    epi = run_population(env, ModelKind.EPI, seed=7, params=params, cfg=cfg)
    assert len(epi) == len(env)
    assert all(np.isfinite(row["mean_fitness"]) for row in epi)

    genetic = run_population(env, ModelKind.GENETIC, seed=7, params=params, cfg=cfg)
    assert all(row["mean_r_germ"] == 0.0 for row in genetic)
    assert all(row["mean_p_b"] == 0.0 for row in genetic)

    plastic = run_population(env, ModelKind.PLASTIC, seed=7, params=params, cfg=cfg)
    assert all(row["mean_r_germ"] == 0.0 for row in plastic)
