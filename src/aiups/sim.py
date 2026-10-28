"""Forward-Euler simulation (illustration only; the proof does not use it)."""
import numpy as np

from . import config as C
from .model import soc_rate


def simulate(rate_fn, P_L, P_g0, e0, P_b_max=np.inf, e_lo=-np.inf, e_hi=np.inf,
             dt=C.DT, stride=1, physical_soc_limits=True):
    """Simulate N profiles at once. P_L has shape [T, N].

    rate_fn(P_g, e, P_L) -> dP_g/dt. Physical limits: |P_b| <= P_b_max, and if
    physical_soc_limits the battery cannot discharge at e <= e_lo or charge at
    e >= e_hi (the grid then absorbs the difference).
    Returns a dict of recorded traces (every `stride` steps) and per-profile stats.
    """
    T, N = P_L.shape
    P_g = np.broadcast_to(np.asarray(P_g0, float), (N,)).copy()
    e = np.broadcast_to(np.asarray(e0, float), (N,)).copy()
    rec = {k: [] for k in ("P_L", "P_g", "grid", "P_b", "e")}
    e_min = e.copy(); e_max = e.copy()
    pb_abs = np.zeros(N); ramp = np.zeros(N)
    grid_prev = None
    t_empty = np.full(N, np.nan)
    for k in range(T):
        pl = P_L[k]
        pb = np.clip(pl - P_g, -P_b_max, P_b_max)
        if physical_soc_limits:
            pb = np.where(e <= e_lo, np.minimum(pb, 0.0), pb)
            pb = np.where(e >= e_hi, np.maximum(pb, 0.0), pb)
        grid = pl - pb
        if grid_prev is not None:
            ramp = np.maximum(ramp, np.abs(grid - grid_prev) / dt)
        grid_prev = grid
        newly = np.isnan(t_empty) & (e <= e_lo)
        t_empty[newly] = k * dt
        if k % stride == 0:
            rec["P_L"].append(pl.copy()); rec["P_g"].append(P_g.copy())
            rec["grid"].append(grid.copy()); rec["P_b"].append(pb.copy()); rec["e"].append(e.copy())
        e_min = np.minimum(e_min, e); e_max = np.maximum(e_max, e)
        pb_abs = np.maximum(pb_abs, np.abs(pb))
        P_g = P_g + dt * rate_fn(P_g, e, pl)
        e = e + dt * soc_rate(pb)
    out = {k: np.array(v) for k, v in rec.items()}
    out["t"] = np.arange(0, T, stride) * dt
    out.update(e_min=e_min, e_max=e_max, pb_abs=pb_abs, ramp=ramp, t_empty=t_empty)
    return out
