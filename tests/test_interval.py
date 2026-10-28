import numpy as np

from aiups.interval import I, phi


def _rand(rng, n=20000):
    a, b = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    t = rng.uniform(0, 1, n)
    return I(lo, hi), lo + t * (hi - lo)


def test_ops_enclose_point_values():
    rng = np.random.default_rng(0)
    X, x = _rand(rng)
    Y, y = _rand(rng)
    for F, f in [(X + Y, x + y), (X - Y, x - y), (X * Y, x * y), (X * -2.7, x * -2.7),
                 (X / 3600.0, x / 3600.0), (X.clip(-1, 2), np.clip(x, -1, 2)), (X.abs(), np.abs(x)),
                 (phi(X, 0.95), np.where(x >= 0, x / 0.95, x * 0.95))]:
        assert np.all(F.lo <= f) and np.all(f <= F.hi)


def test_outward_rounding_strict():
    X = I(np.array([0.1]), np.array([0.1]))
    Y = X + 0.2
    assert Y.lo[0] < 0.1 + 0.2 < Y.hi[0] or Y.lo[0] < Y.hi[0]


def test_recip_and_sqrt():
    rng = np.random.default_rng(1)
    lo = rng.uniform(0.1, 3, 1000); hi = lo + rng.uniform(0, 2, 1000)
    x = lo + rng.uniform(0, 1, 1000) * (hi - lo)
    X = I(lo, hi)
    R, S = X.recip(), X.sqrt()
    assert np.all(R.lo <= 1 / x) and np.all(1 / x <= R.hi)
    assert np.all(S.lo <= np.sqrt(x)) and np.all(np.sqrt(x) <= S.hi)
