"""PCC voltage: exact two-bus (radial feeder) solution, its linearization, and a pandapower check.

Feeder: stiff source V_s behind a series impedance Z = R + jX (no shunt), load
S = P + jQ drawn at the PCC. With |V| the PCC magnitude, the exact solution on
the normal (high-voltage) branch is

    |V|^2 = ( A + sqrt(A^2 - 4 |Z|^2 (P^2 + Q^2)) ) / 2,   A = V_s^2 - 2 (R P + X Q),

and the usual linearization is |V| ~= V_s - (R P + X Q) / V_s (all per unit).
"""
import numpy as np

from . import config as C
from .interval import I

Z_BASE = C.V_BASE_KV ** 2 / C.S_BASE_MVA
R_PU = C.LINE_R_OHM_PER_KM * C.LINE_KM / Z_BASE
X_PU = C.LINE_X_OHM_PER_KM * C.LINE_KM / Z_BASE
Z2 = R_PU ** 2 + X_PU ** 2
TAN_PHI_LOAD = np.tan(np.arccos(C.PF_LOAD_NO_UPS))


def v_exact(P_mw, Q_mvar=0.0, Vs=C.V_SOURCE_PU):
    P = np.asarray(P_mw) / C.S_BASE_MVA
    Q = np.asarray(Q_mvar) / C.S_BASE_MVA
    A = Vs ** 2 - 2 * (R_PU * P + X_PU * Q)
    return np.sqrt((A + np.sqrt(A ** 2 - 4 * Z2 * (P ** 2 + Q ** 2))) / 2)


def v_lin(P_mw, Q_mvar=0.0, Vs=C.V_SOURCE_PU):
    return Vs - (R_PU * np.asarray(P_mw) + X_PU * np.asarray(Q_mvar)) / C.S_BASE_MVA / Vs


def _parts(P: I, tanphi: float, Vs=C.V_SOURCE_PU):
    P = P / C.S_BASE_MVA
    Q = P * tanphi
    A = (P * R_PU + Q * X_PU) * -2.0 + Vs ** 2
    D = A * A - (P * P + Q * Q) * (4.0 * Z2)
    return P, Q, A, D


def v_exact_iv(P_mw: I, tanphi=0.0, Vs=C.V_SOURCE_PU) -> I:
    """Sound enclosure of the exact PCC voltage (pu) for P in an interval, Q = P tan(phi)."""
    _, _, A, D = _parts(P_mw, tanphi, Vs)
    return ((A + D.sqrt()) / 2.0).sqrt()


def v_lin_iv(P_mw: I, tanphi=0.0, Vs=C.V_SOURCE_PU) -> I:
    return ((P_mw * (R_PU + X_PU * tanphi)) / C.S_BASE_MVA) * (-1.0 / Vs) + Vs


def dv_dp_iv(P_mw: I, tanphi=0.0, Vs=C.V_SOURCE_PU) -> I:
    """Sound enclosure of dV/dP (pu per MW) with Q = P tan(phi)."""
    P, Q, A, D = _parts(P_mw, tanphi, Vs)
    Ap = -2.0 * (R_PU + X_PU * tanphi)
    # dD/dP / 2 = A A' - 4 Z2 (P + Q tanphi)
    num = A * Ap - (P + Q * tanphi) * (4.0 * Z2)
    dV2 = (num.div(D.sqrt()) + Ap) / 2.0
    V = ((A + D.sqrt()) / 2.0).sqrt()
    return (dV2.div(V * 2.0)) / C.S_BASE_MVA


def sound_range(fn_iv, P_lo, P_hi, tanphi=0.0, n=4000):
    """Cover [P_lo, P_hi] with n intervals and return the hull of fn_iv (a sound enclosure)."""
    edges = np.linspace(P_lo, P_hi, n + 1)
    edges[0], edges[-1] = P_lo, P_hi
    out = fn_iv(I(edges[:-1], edges[1:]), tanphi)
    return float(out.lo.min()), float(out.hi.max())


def lin_error_bound(P_lo, P_hi, tanphi=0.0, n=4000):
    lo, hi = sound_range(lambda P, t: v_exact_iv(P, t) - v_lin_iv(P, t), P_lo, P_hi, tanphi, n)
    return max(abs(lo), abs(hi))


def pandapower_check(P_values, tanphi=0.0):
    """AC power flow on the same radial feeder; returns PCC |V| (pu) per P."""
    import pandapower as pp
    out = []
    for P in P_values:
        net = pp.create_empty_network(sn_mva=C.S_BASE_MVA)
        b0 = pp.create_bus(net, vn_kv=C.V_BASE_KV)
        b1 = pp.create_bus(net, vn_kv=C.V_BASE_KV)
        pp.create_ext_grid(net, b0, vm_pu=C.V_SOURCE_PU)
        pp.create_line_from_parameters(net, b0, b1, length_km=C.LINE_KM, r_ohm_per_km=C.LINE_R_OHM_PER_KM,
                                       x_ohm_per_km=C.LINE_X_OHM_PER_KM, c_nf_per_km=0.0, max_i_ka=5.0)
        pp.create_load(net, b1, p_mw=float(P), q_mvar=float(P * tanphi))
        pp.runpp(net, tolerance_mva=1e-10)
        out.append(float(net.res_bus.vm_pu.iloc[1]))
    return np.array(out)
