import numpy as np

from memorydna.plateau_probe import P_SWEEP, SWITCH_COUNTS, _fit_linear, _fit_saturating


def test_p_sweep_contains_original_anchor_similarities():
    realized = [round(p, 6) for p in P_SWEEP]
    assert round(1 - 17 / 19, 6) in realized
    assert round(1 - 9 / 19, 6) in realized
    assert round(1 - 2 / 19, 6) in realized
    assert len(P_SWEEP) == len(SWITCH_COUNTS) == 11


def test_saturating_fit_recovers_plateau_better_than_linear():
    x = np.linspace(0.05, 0.9, 60)
    y = 0.02 + 0.18 * np.minimum(x, 0.5)
    linear = _fit_linear(x, y)
    sat = _fit_saturating(x, y)
    assert sat["aic"] < linear["aic"]
    assert abs(sat["threshold"] - 0.5) < 0.08
