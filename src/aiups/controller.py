"""Grid-side controller (the object we certify) and ablation controllers.

Certified law (saturated first-order low-pass with a saturated SOC loop):

    s(e)      = clip(k_e * (e_ref - e), -c, c)                 SOC restoring term
    u         = P_L + s(e)                                     filter target
    dP_g/dt   = clip((u - P_g) / tau, -R, R)                   rate-clipped low-pass
    P_b       = clip(P_L - P_g, -P_b_max, P_b_max)             battery power limit
    de/dt     = -phi(P_b) / 3600

The certificate shows the battery limit never binds on the certified set, so
there the grid draws exactly P_g, whose slope the clip bounds by R.
"""
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C
from .interval import I


@dataclass(frozen=True)
class Design:
    R: float          # MW/s ramp limit enforced by the clip
    tau: float        # s, filter time constant
    c: float          # MW, SOC loop authority
    ke: float         # MW per MWh, SOC loop gain
    e_ref: float = 0.0  # MWh, SOC reference (energies are measured from it)
    mode: str = "w"     # "w": restore the settled SOC w = e + tau (P_g - P_mid)/3600; "e": restore e itself

    @staticmethod
    def from_alpha(R_mw_per_min, alpha, c, ke_frac, mode="w"):
        R = R_mw_per_min / 60.0
        tau = alpha * C.DP / R
        e0 = tau * C.DP / C.SEC_PER_H          # MWh, natural energy scale tau*DP
        ke = c / (ke_frac * e0)
        return Design(R=R, tau=tau, c=c, ke=ke, mode=mode)

    @property
    def e_scale(self):
        return self.tau * C.DP / C.SEC_PER_H

    def to_dict(self):
        d = asdict(self)
        d["R_mw_per_min"] = self.R * 60.0
        return d


P_MID = 0.5 * (C.P_MIN + C.P_MAX)


def settled_soc(d: Design, P_g, e):
    """w = e + tau (P_g - P_mid)/3600: the SOC the battery returns to once P_g settles at P_mid
    (lossless, unclipped). For the plain "e" mode the restored quantity is e itself."""
    if d.mode == "w":
        return e + d.tau * (P_g - P_MID) / C.SEC_PER_H
    return e


def soc_term(d: Design, e, P_g=None):
    if d.mode == "w":
        e = settled_soc(d, P_g, e)
    return np.clip(d.ke * (d.e_ref - e), -d.c, d.c)


def pg_rate(d: Design, P_g, e, P_L):
    """dP_g/dt of the certified controller (numpy)."""
    return np.clip((P_L + soc_term(d, e, P_g) - P_g) / d.tau, -d.R, d.R)


def pg_rate_iv(d: Design, P_g: I, e: I, P_L: I) -> I:
    """Sound interval enclosure of dP_g/dt."""
    q = e
    if d.mode == "w":
        q = e + ((P_g - P_MID) * d.tau) / C.SEC_PER_H
    s = (((q * -1.0) + d.e_ref) * d.ke).clip(-d.c, d.c)
    return (((P_L + s) - P_g) / d.tau).clip(-d.R, d.R)


# --------------------------------------------------------------- ablations
def pg_rate_no_soc(d: Design, P_g, e, P_L):
    """Same filter with the SOC loop removed (k_e = 0)."""
    return np.clip((P_L - P_g) / d.tau, -d.R, d.R)


def pg_rate_naive_limiter(R, tau_fast=1.0):
    """Classic ramp-rate limiter: track the load as fast as the ramp limit allows, no SOC loop."""
    def f(P_g, e, P_L):
        return np.clip((P_L - P_g) / tau_fast, -R, R)
    return f
