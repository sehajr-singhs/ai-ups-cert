"""Minimal vectorized interval arithmetic with outward rounding.

Every operation rounds its lower end toward -inf and its upper end toward
+inf with one np.nextafter step. IEEE 754 +, -, *, / and sqrt are correctly
rounded (error <= 0.5 ulp), so one ulp of outward widening makes every
enclosure sound: the true real-number result lies inside [lo, hi].
clip/min/max are exact and need no widening.
"""
import numpy as np

_NINF, _PINF = -np.inf, np.inf


def _dn(x):
    return np.nextafter(x, _NINF)


def _up(x):
    return np.nextafter(x, _PINF)


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=float)

    # -- arithmetic -------------------------------------------------------
    def __add__(self, o):
        if isinstance(o, I):
            return I(_dn(self.lo + o.lo), _up(self.hi + o.hi))
        return I(_dn(self.lo + o), _up(self.hi + o))

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-o if isinstance(o, I) else -o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, I):
            p = np.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
            return I(_dn(p.min(0)), _up(p.max(0)))
        o = float(o)
        if o >= 0:
            return I(_dn(self.lo * o), _up(self.hi * o))
        return I(_dn(self.hi * o), _up(self.lo * o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = float(o)
        if o > 0:
            return I(_dn(self.lo / o), _up(self.hi / o))
        if o < 0:
            return I(_dn(self.hi / o), _up(self.lo / o))
        raise ZeroDivisionError

    def recip(self):
        """1/x for intervals with lo > 0."""
        assert np.all(self.lo > 0)
        return I(_dn(1.0 / self.hi), _up(1.0 / self.lo))

    def div(self, o: "I"):
        return self * o.recip()

    # -- exact monotone maps ---------------------------------------------
    def clip(self, a, b):
        return I(np.clip(self.lo, a, b), np.clip(self.hi, a, b))

    def abs(self):
        lo = np.where(self.lo >= 0, self.lo, np.where(self.hi <= 0, -self.hi, 0.0))
        hi = np.maximum(np.abs(self.lo), np.abs(self.hi))
        return I(lo, hi)

    def sqrt(self):
        return I(_dn(np.sqrt(np.maximum(self.lo, 0.0))), _up(np.sqrt(self.hi)))

    def contains(self, x):
        return (self.lo <= x) & (x <= self.hi)


def phi(p: I, eta: float) -> I:
    """Interval of the (monotone increasing) storage-side power phi(p) = p/eta if p>=0 else eta*p."""
    lo = np.where(p.lo >= 0, _dn(p.lo / eta), _dn(p.lo * eta))
    hi = np.where(p.hi >= 0, _up(p.hi / eta), _up(p.hi * eta))
    return I(lo, hi)
