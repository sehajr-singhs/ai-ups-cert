"""Plant: AI campus load, battery, and state of charge.

States: P_g (MW, grid-side power set by the controller) and e (MWh, stored
energy measured from the reference e_ref = 0). Input: P_L (MW), any
measurable signal with values in [P_MIN, P_MAX]. Battery power
P_b = P_L - P_g (positive = discharge). Energy balance with one-way
efficiency eta:  de/dt = -phi(P_b)/3600,  phi(p) = p/eta (p>=0), eta*p (p<0).
"""
import numpy as np

from . import config as C


def phi(p, eta=C.ETA):
    """Storage-side power (MW) for terminal battery power p (MW). Monotone increasing."""
    return np.where(p >= 0, p / eta, p * eta)


def battery_power(P_L, P_g, P_b_max=np.inf):
    """Battery terminal power with the physical power limit |P_b| <= P_b_max."""
    return np.clip(P_L - P_g, -P_b_max, P_b_max)


def soc_rate(P_b, eta=C.ETA):
    """de/dt in MWh per second."""
    return -phi(P_b, eta) / C.SEC_PER_H
