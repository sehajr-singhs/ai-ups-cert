import numpy as np

from aiups.controller import Design, pg_rate, pg_rate_naive_limiter
from aiups.profiles import square, time_grid
from aiups.sim import simulate


def test_naive_rate_limiter_drains_on_short_bursts():
    d = Design.from_alpha(10.0, 1.0, 6.0, 0.15)
    t = time_grid(7200.0)
    PL = square(t, 10.0, 0.2, 0.4)[:, None]
    out = simulate(pg_rate_naive_limiter(d.R), PL, PL[0], np.zeros(1), 50.0, -1.5, 1.5, stride=50)
    assert not np.isnan(out["t_empty"][0])           # battery hits empty
    assert out["ramp"][0] > 10 * d.R                 # and then the grid sees steps


def test_no_ups_ramp_is_the_load_step():
    t = time_grid(60.0)
    PL = square(t, 10.0, 0.5, 0.4)
    assert np.abs(np.diff(PL)).max() == 40.0


def test_certified_controller_keeps_soc_on_same_profile():
    d = Design.from_alpha(10.0, 1.0, 6.0, 0.15)
    t = time_grid(7200.0)
    PL = square(t, 10.0, 0.2, 0.4)[:, None]
    Pg0 = PL[0].copy(); e0 = -d.tau * (Pg0 - 80.0) / 3600
    out = simulate(lambda Pg, e, L: pg_rate(d, Pg, e, L), PL, Pg0, e0, 50.0, -1.5, 1.5, stride=50)
    assert np.isnan(out["t_empty"][0])
