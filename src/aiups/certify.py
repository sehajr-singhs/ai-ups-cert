"""Sound robust-invariance certificate by interval branch-and-bound.

Certified set (a polytope in the (P_g, e) plane):

    S = { x : h_k(x) <= 0, k = 1..K },   h_k(x) = n_k . z(x) - b_k,
    z(x) = ((P_g - P_mid) / DP, e / E0),   E0 = tau * DP / 3600.

Equivalently S = {B <= 0} with B(x) = max_k h_k(x), a max-of-affine
(one hidden layer ReLU) function.

Invariance lemma (proved in RESULTS.md and the paper). Let f(x, P_L) be the
closed-loop vector field and eps > 0. If for every face k

    grad h_k . f(x, P_L) <= 0   for all P_L in [P_MIN, P_MAX] and all x in
    band_k = { 0 <= h_k(x) <= eps,  h_j(x) <= eps for j != k },

then every Caratheodory solution with x(0) in S and any measurable P_L(t) in
[P_MIN, P_MAX] stays in S for all t >= 0.

The BaB below proves the band condition on the full (P_g, e, P_L) box: a box
is discarded only if interval arithmetic (outward rounded) shows it misses the
band, or shows grad h_k . f <= 0 on all of it. Anything else is bisected. The
check fails (never silently passes) on a concrete counterexample or on the
depth or box budget.
"""
import time

import numpy as np

from . import config as C
from .controller import Design, pg_rate, pg_rate_iv, soc_term
from .interval import I, phi as phi_iv
from .model import phi

P_MID = 0.5 * (C.P_MIN + C.P_MAX)


def directions(n=C.N_DIRS):
    th = 2 * np.pi * np.arange(n) / n
    N = np.stack([np.cos(th), np.sin(th)], axis=1)
    N[np.abs(N) < 1e-15] = 0.0
    return N


# ------------------------------------------------------------------ geometry (float, for candidates and plots)
def polygon(N, b, big=50.0):
    """Vertices (counter-clockwise, z-coordinates) of {z : N z <= b} intersected with a big box."""
    P = np.array([[-big, -big], [big, -big], [big, big], [-big, big]], float)
    for n, bb in zip(N, b):
        if len(P) == 0:
            break
        out = []
        m = len(P)
        for i in range(m):
            a, c = P[i], P[(i + 1) % m]
            fa, fc = n @ a - bb, n @ c - bb
            if fa <= 0:
                out.append(a)
            if fa * fc < 0:
                out.append(a + (c - a) * (fa / (fa - fc)))
        P = np.array(out)
    return P


class Cert:
    """A design plus a polytope candidate, with float and interval evaluators."""

    def __init__(self, d: Design, N, b, eps=C.BAND_EPS):
        self.d, self.N, self.b, self.eps = d, np.asarray(N, float), np.asarray(b, float), eps
        self.E0 = d.e_scale

    # coordinates
    def z(self, Pg, e):
        return (Pg - P_MID) / C.DP, e / self.E0

    def x(self, z):
        return P_MID + C.DP * z[..., 0], self.E0 * z[..., 1]

    def vertices_x(self, b=None):
        V = polygon(self.N, self.b if b is None else b)
        Pg, e = self.x(V)
        return np.stack([Pg, e], axis=1)

    # closed-loop field with an unsaturated battery (valid wherever |P_L - P_g| <= P_b_max)
    def field(self, Pg, e, PL):
        return pg_rate(self.d, Pg, e, PL), -phi(PL - Pg) / C.SEC_PER_H

    def lie(self, k, Pg, e, PL):
        fP, fe = self.field(Pg, e, PL)
        return self.N[k, 0] * fP / C.DP + self.N[k, 1] * fe / self.E0

    def h(self, Pg, e):
        z1, z2 = self.z(Pg, e)
        return self.N[:, 0:1] * z1 + self.N[:, 1:2] * z2 - self.b[:, None]

    # interval versions
    def h_iv(self, Pg: I, e: I):
        z1 = (Pg - P_MID) / C.DP
        z2 = e / self.E0
        return [z1 * self.N[j, 0] + z2 * self.N[j, 1] - self.b[j] for j in range(len(self.b))]

    def lie_iv(self, k, Pg: I, e: I, PL: I):
        fP = pg_rate_iv(self.d, Pg, e, PL)
        fe = phi_iv(PL - Pg, C.ETA) * -1.0 / C.SEC_PER_H
        return (fP / C.DP) * self.N[k, 0] + (fe / self.E0) * self.N[k, 1]

    def domain(self, pad=0.05):
        """Bounding box of S_eps (from float vertices) padded by 0.05 in normalized units.
        The pad is about 1e13 times larger than any rounding in the vertex computation."""
        V = polygon(self.N, self.b + self.eps)
        lo = V.min(0) - pad
        hi = V.max(0) + pad
        Pg_lo, e_lo = self.x(lo)
        Pg_hi, e_hi = self.x(hi)
        return np.array([Pg_lo, e_lo, C.P_MIN]), np.array([Pg_hi, e_hi, C.P_MAX])

    def scales(self):
        return np.array([C.DP, self.E0, C.DP])

    def to_dict(self):
        return dict(design=self.d.to_dict(), normals=self.N.tolist(), offsets=self.b.tolist(),
                    band_eps=self.eps, E0_MWh=self.E0, P_mid=P_MID, DP=C.DP)


# ------------------------------------------------------------------ generic sound BaB
def bab(cert: Cert, classify, max_depth=C.BAB_MAX_DEPTH, max_boxes=C.BAB_MAX_BOXES, chunk=C.BAB_CHUNK):
    """classify(lo, hi) -> (done_mask, cex_mask, points). Returns (ok, stats)."""
    t0 = time.perf_counter()
    lo0, hi0 = cert.domain()
    sc = cert.scales()
    stack = [(lo0[None, :], hi0[None, :], np.zeros(1, int))]
    n_boxes = 0
    max_d = 0
    status, cex = "proved", None
    while stack:
        lo, hi, dep = stack.pop()
        if len(lo) > chunk:
            stack.append((lo[chunk:], hi[chunk:], dep[chunk:]))
            lo, hi, dep = lo[:chunk], hi[:chunk], dep[:chunk]
        n_boxes += len(lo)
        max_d = max(max_d, int(dep.max()))
        done, is_cex, pts = classify(lo, hi)
        if is_cex.any():
            status, cex = "counterexample", pts[is_cex][0].tolist()
            break
        rest = ~done
        if not rest.any():
            continue
        lo, hi, dep = lo[rest], hi[rest], dep[rest]
        if (dep >= max_depth).any():
            status = "depth_limit"
            break
        if n_boxes > max_boxes:
            status = "box_budget"
            break
        w = (hi - lo) / sc
        ax = w.argmax(1)
        r = np.arange(len(lo))
        mid = 0.5 * (lo[r, ax] + hi[r, ax])
        hi_a = hi.copy(); hi_a[r, ax] = mid
        lo_b = lo.copy(); lo_b[r, ax] = mid
        stack.append((np.concatenate([lo, lo_b]), np.concatenate([hi_a, hi]),
                      np.concatenate([dep + 1, dep + 1])))
    return status == "proved", dict(status=status, boxes=int(n_boxes), max_depth=int(max_d),
                                    seconds=time.perf_counter() - t0, counterexample=cex)


def _boxes_iv(lo, hi):
    return I(lo[:, 0], hi[:, 0]), I(lo[:, 1], hi[:, 1]), I(lo[:, 2], hi[:, 2])


def verify_face(cert: Cert, k, **kw):
    eps = cert.eps
    K = len(cert.b)

    def classify(lo, hi):
        Pg, e, PL = _boxes_iv(lo, hi)
        hs = cert.h_iv(Pg, e)
        out = (hs[k].hi < 0) | (hs[k].lo > eps)
        for j in range(K):
            if j != k:
                out |= hs[j].lo > eps
        L = cert.lie_iv(k, Pg, e, PL)
        done = out | (L.hi <= 0)
        m = 0.5 * (lo + hi)
        hm = cert.h(m[:, 0], m[:, 1])
        in_band = (hm[k] >= 0) & (hm[k] <= eps) & np.all(np.delete(hm, k, 0) <= eps, axis=0)
        Lm = cert.lie(k, m[:, 0], m[:, 1], m[:, 2])
        return done, (~done) & in_band & (Lm > 0), m

    return bab(cert, classify, **kw)


def verify_invariance(cert: Cert, **kw):
    """Run the band condition for every face. Returns (ok, per-face stats, summary)."""
    faces = []
    for k in range(len(cert.b)):
        ok, st = verify_face(cert, k, **kw)
        st["face"] = k
        faces.append(st)
        if not ok:
            # keep going so every failing face is reported
            pass
    summ = dict(
        proved=all(f["status"] == "proved" for f in faces),
        boxes=int(sum(f["boxes"] for f in faces)),
        max_depth=int(max(f["max_depth"] for f in faces)),
        seconds=float(sum(f["seconds"] for f in faces)),
    )
    return summ["proved"], faces, summ


def verify_bound(cert: Cert, g_iv, g_np, T, region_eps=0.0, **kw):
    """Prove g(P_g, e, P_L) <= T on {x : h_j(x) <= region_eps for all j} x [P_MIN, P_MAX]."""
    K = len(cert.b)

    def classify(lo, hi):
        Pg, e, PL = _boxes_iv(lo, hi)
        hs = cert.h_iv(Pg, e)
        out = np.zeros(len(lo), bool)
        for j in range(K):
            out |= hs[j].lo > region_eps
        g = g_iv(Pg, e, PL)
        done = out | (g.hi <= T)
        m = 0.5 * (lo + hi)
        inside = np.all(cert.h(m[:, 0], m[:, 1]) <= region_eps, axis=0)
        return done, (~done) & inside & (g_np(m[:, 0], m[:, 1], m[:, 2]) > T), m

    return bab(cert, classify, **kw)


def sound_max(cert: Cert, g_iv, g_np, estimate, scale, region_eps=0.0, rel=1e-4):
    """Smallest T in estimate + scale*rel*{1,4,16,...} that the BaB proves. Returns (T, stats)."""
    st = None
    for i in range(10):
        T = estimate + scale * rel * (4 ** i)
        ok, st = verify_bound(cert, g_iv, g_np, T, region_eps)
        if ok:
            st["estimate"] = float(estimate)
            st["bound"] = float(T)
            return float(T), st
    raise RuntimeError("could not bound objective: " + str(st))


# ------------------------------------------------------------------ candidate search (sampling only)
def sampled_violation(cert: Cert, n_pts=41):
    """Max of grad h_k . f over sampled band points, per face. Uses the exact P_L breakpoints,
    since the Lie derivative is piecewise linear in P_L. Faces with no edge get -inf.
    This only proposes candidates; it proves nothing."""
    d = cert.d
    V = polygon(cert.N, cert.b)
    viol = np.full(len(cert.b), -np.inf)
    if len(V) < 3:
        return viol
    t = np.linspace(0, 1, n_pts)
    for i in range(len(V)):
        a, c = V[i], V[(i + 1) % len(V)]
        mid = 0.5 * (a + c)
        k = int(np.argmin(np.abs(cert.N @ mid - cert.b)))
        seg = a[None] + t[:, None] * (c - a)[None]
        pts = np.concatenate([seg, seg + cert.eps * cert.N[k][None]])
        Pg, e = cert.x(pts)
        s = soc_term(d, e, Pg)
        cand = np.stack([np.full_like(Pg, C.P_MIN), np.full_like(Pg, C.P_MAX), Pg,
                         Pg - s + d.R * d.tau, Pg - s - d.R * d.tau])
        cand = np.clip(cand, C.P_MIN, C.P_MAX)
        L = cert.lie(k, Pg[None], e[None], cand)
        viol[k] = max(viol[k], L.max())
    return viol


def certify_design(d: Design, max_outer=12, step=0.005, b_cap=8.0, max_rounds=3000, n_dirs=C.N_DIRS, b0=0.02):
    """Candidate search by sampling, then sound BaB; failing faces are pushed out and re-searched.
    Returns (Cert or None, info)."""
    N = directions(n_dirs)
    b = np.full(len(N), b0)
    log = []
    gamma = 1e-3 / d.tau  # small strict margin asked of candidates so the BaB can close
    for outer in range(max_outer):
        cert = None
        for r in range(max_rounds):
            cand = Cert(d, N, b)
            v = sampled_violation(cand)
            bad = v > -gamma
            if not bad.any():
                cert = cand
                break
            b = b + step * bad
            if b.max() > b_cap:
                break
        if cert is None:
            return None, dict(proved=False, reason="no candidate (offsets diverged)", log=log)
        ok, faces, summ = verify_invariance(cert)
        log.append(summ)
        if ok:
            return cert, dict(faces=faces, **summ, outer_rounds=outer + 1, log=log)
        for f in faces:
            if f["status"] != "proved":
                b[f["face"]] += step
    return None, dict(proved=False, reason="BaB kept failing", log=log)


# ------------------------------------------------------------------ sound bounds on the certified set
def analyze(cert: Cert):
    """Sound upper/lower bounds (BaB proved) on the quantities the theorems need."""
    V = cert.vertices_x()
    Ve = cert.vertices_x(cert.b + cert.eps)
    E0 = cert.E0
    ident = lambda Pg, e, PL: e
    neg = lambda Pg, e, PL: e * -1.0
    negnp = lambda Pg, e, PL: -e
    pgf = lambda Pg, e, PL: Pg
    pgn = lambda Pg, e, PL: Pg * -1.0
    pgnn = lambda Pg, e, PL: -Pg
    e_hi, s1 = sound_max(cert, ident, ident, V[:, 1].max(), E0)
    m_e_lo, s2 = sound_max(cert, neg, negnp, -V[:, 1].min(), E0)
    pg_hi, s3 = sound_max(cert, pgf, pgf, V[:, 0].max(), C.DP)
    m_pg_lo, s4 = sound_max(cert, pgn, pgnn, -V[:, 0].min(), C.DP)
    pb_est = max(C.P_MAX - Ve[:, 0].min(), Ve[:, 0].max() - C.P_MIN)
    pb, s5 = sound_max(cert, lambda Pg, e, PL: (PL - Pg).abs(), lambda Pg, e, PL: np.abs(PL - Pg),
                       pb_est, C.DP, region_eps=cert.eps)
    # whether the rate clip ever binds on S (estimate from vertices, P_L at the extremes)
    from .controller import soc_term
    Pg, e = V[:, 0], V[:, 1]
    u_minus = np.stack([C.P_MIN + soc_term(cert.d, e, Pg) - Pg, C.P_MAX + soc_term(cert.d, e, Pg) - Pg])
    filt_rate = float(np.abs(u_minus).max() / cert.d.tau)
    return dict(
        e_hi_MWh=e_hi, e_lo_MWh=-m_e_lo, energy_window_MWh=e_hi + m_e_lo,
        Pg_hi_MW=pg_hi, Pg_lo_MW=-m_pg_lo, Pb_max_MW=pb,
        unclipped_rate_max_est_MW_per_s=filt_rate, clip_binds_on_S=bool(filt_rate > cert.d.R),
        bound_proofs=dict(e_hi=s1, e_lo=s2, Pg_hi=s3, Pg_lo=s4, Pb=s5),
    )
