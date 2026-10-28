import numpy as np

from aiups import voltage as V
from aiups.interval import I


def test_pandapower_matches_exact_formula():
    P = np.array([55.0, 80.0, 105.0])
    assert np.abs(V.pandapower_check(P) - V.v_exact(P)).max() < 1e-8
    assert np.abs(V.pandapower_check(P, V.TAN_PHI_LOAD) - V.v_exact(P, P * V.TAN_PHI_LOAD)).max() < 1e-8


def test_interval_voltage_encloses_points():
    P = np.linspace(50, 110, 301)
    X = I(P[:-1], P[1:])
    Vi = V.v_exact_iv(X, V.TAN_PHI_LOAD)
    mid = 0.5 * (P[:-1] + P[1:])
    v = V.v_exact(mid, mid * V.TAN_PHI_LOAD)
    assert np.all(Vi.lo <= v) and np.all(v <= Vi.hi)


def test_lin_error_bound_dominates_samples():
    P = np.linspace(60, 100, 1001)
    err = np.abs(V.v_exact(P) - V.v_lin(P)).max()
    assert V.lin_error_bound(60, 100) >= err


def test_dvdp_enclosure_matches_finite_difference():
    P = np.linspace(60, 100, 41)
    h = 1e-4
    fd = (V.v_exact(P + h) - V.v_exact(P - h)) / (2 * h)
    D = V.dv_dp_iv(I(P - 1e-3, P + 1e-3))
    assert np.all(D.lo <= fd + 1e-9) and np.all(fd - 1e-9 <= D.hi)
