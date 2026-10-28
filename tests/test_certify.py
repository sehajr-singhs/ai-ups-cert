import numpy as np
import pytest

from aiups import config as C
from aiups.certify import Cert, analyze, certify_design, verify_invariance
from aiups.controller import Design, pg_rate
from aiups.sim import simulate
from aiups.profiles import square, time_grid


@pytest.fixture(scope="module")
def certified():
    d = Design.from_alpha(10.0, 1.0, 6.0, 0.15, mode="w")
    cert, info = certify_design(d, n_dirs=16)
    assert cert is not None, info
    return cert, info


def test_bab_proves_invariance(certified):
    cert, info = certified
    ok, faces, summ = verify_invariance(cert)
    assert ok and summ["boxes"] > 0 and all(f["status"] == "proved" for f in faces)


def test_bab_rejects_a_too_small_set(certified):
    cert, _ = certified
    small = Cert(cert.d, cert.N, cert.b * 0.4)
    ok, faces, _ = verify_invariance(small)
    assert not ok
    assert any(f["status"] == "counterexample" for f in faces)


def test_bounds_are_sound_against_vertices(certified):
    cert, _ = certified
    a = analyze(cert)
    V = cert.vertices_x()
    assert a["e_hi_MWh"] >= V[:, 1].max() and a["e_lo_MWh"] <= V[:, 1].min()
    assert a["Pb_max_MW"] >= max(C.P_MAX - V[:, 0].min(), V[:, 0].max() - C.P_MIN)


def test_simulated_trajectories_stay_in_set(certified):
    cert, _ = certified
    d = cert.d
    t = time_grid(1800.0, 0.2)
    PL = np.stack([square(t, p, du, 0.4) for p in (5, 60, 300) for du in (0.1, 0.5, 0.9)], 1)
    Pg0 = PL[0].copy(); e0 = -d.tau * (Pg0 - 80.0) / 3600
    out = simulate(lambda Pg, e, L: pg_rate(d, Pg, e, L), PL, Pg0, e0, dt=0.2, stride=5)
    H = cert.h(out["P_g"].ravel(), out["e"].ravel())
    assert H.max() <= 1e-3  # Euler error allowance only
    assert out["ramp"].max() <= d.R + 1e-9
