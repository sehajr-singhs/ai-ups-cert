"""Admissible load profiles (values in [P_MIN, P_MAX], steps allowed).

These are for simulation and for searching worst cases. The certificate
does not use them: it holds for every measurable profile with values in
[P_MIN, P_MAX].
"""
import numpy as np

from . import config as C


def time_grid(T, dt=C.DT):
    return np.arange(0.0, T, dt)


def square(t, period, duty, depth, phase=0.0, high=C.P_MAX):
    """Synchronized training bursts: high for duty*period, then drop by depth*P_MAX."""
    low = max(high - depth * C.P_MAX, C.P_MIN)
    frac = np.mod(t / period + phase, 1.0)
    return np.where(frac < duty, high, low)


def checkpoint(t, every, dip_len, base=C.P_MAX, idle=C.P_MIN, first=None):
    """Full load with dips to idle for dip_len seconds every `every` seconds (checkpointing)."""
    first = every / 2 if first is None else first
    in_dip = (np.mod(t - first, every) < dip_len) & (t >= first)
    return np.where(in_dip, idle, base)


def random_walk(t, rng, sigma=1.5, hold=1.0, p_jump=0.02):
    """Random walk with occasional jumps between the limits, clipped to [P_MIN, P_MAX]."""
    n_hold = max(1, int(round(hold / (t[1] - t[0]))))
    n = int(np.ceil(len(t) / n_hold))
    x = np.empty(n)
    x[0] = rng.uniform(C.P_MIN, C.P_MAX)
    for i in range(1, n):
        if rng.random() < p_jump:
            x[i] = rng.choice([C.P_MIN, C.P_MAX])
        else:
            x[i] = np.clip(x[i - 1] + sigma * rng.standard_normal(), C.P_MIN, C.P_MAX)
    return np.repeat(x, n_hold)[: len(t)]


def family(t, rng, n_square=150, n_ckpt=30, n_rw=30):
    """Sample the three simulation families from the brief. Returns (P_L [T, N], labels)."""
    cols, labels = [], []
    for _ in range(n_square):
        per = rng.uniform(2, 30)
        duty = rng.uniform(0.2, 0.8)
        depth = rng.uniform(0.1, 0.4)
        cols.append(square(t, per, duty, depth, rng.uniform()))
        labels.append(f"square T={per:.1f}s duty={duty:.2f} depth={depth:.2f}")
    for _ in range(n_ckpt):
        every = rng.uniform(120, 900)
        dip = rng.uniform(5, 90)
        cols.append(checkpoint(t, every, dip))
        labels.append(f"checkpoint every={every:.0f}s dip={dip:.0f}s")
    for _ in range(n_rw):
        cols.append(random_walk(t, rng))
        labels.append("random walk")
    return np.stack(cols, axis=1), labels
