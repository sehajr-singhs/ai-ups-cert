"""Reproduce every number and figure: results/*.json and figures/*.{pdf,png}.

    .venv/Scripts/python run_all.py            # full run
    .venv/Scripts/python run_all.py --reuse    # reuse results/design_search.json (skip the slow grid)
"""
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

from aiups import config as C  # noqa: E402
from aiups import voltage as V  # noqa: E402
from aiups.certify import Cert, analyze, certify_design, directions, verify_invariance, P_MID  # noqa: E402
from aiups.controller import Design, pg_rate, pg_rate_naive_limiter, pg_rate_no_soc  # noqa: E402
from aiups.interval import I  # noqa: E402
from aiups.model import phi  # noqa: E402
from aiups.profiles import checkpoint, family, square, time_grid  # noqa: E402
from aiups.sim import simulate  # noqa: E402
from aiups.figs import AQUA, BLUE, INK2, ORANGE, VIOLET, RED, plt, save  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_quick" if C.QUICK else "results")
FIG = os.path.join(ROOT, "figures_quick" if C.QUICK else "figures")
HSCALE = 0.1 if C.QUICK else 1.0  # horizon scale (smoke mode only)
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def dump(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def strip(info):
    """Drop bulky per-face lists from BaB info for the summary files."""
    return {k: v for k, v in info.items() if k not in ("faces", "log")}


# =========================================================================== 1. design search
def _search_one(args):
    R, alpha, c, kf = args
    d = Design.from_alpha(R, alpha, c, kf, mode="w")
    t0 = time.perf_counter()
    cert, info = certify_design(d)
    row = dict(alpha=alpha, c=c, ke_frac=kf, design=d.to_dict(), proved=cert is not None,
               search_seconds=time.perf_counter() - t0)
    if cert is not None:
        a = analyze(cert)
        row.update(offsets=cert.b.tolist(), bab=strip(info),
                   energy_window_MWh=a["energy_window_MWh"], Pb_max_MW=a["Pb_max_MW"],
                   window_over_E0=a["energy_window_MWh"] / d.e_scale)
    else:
        row.update(reason=info.get("reason"))
    log("search", alpha, c, kf, row.get("energy_window_MWh"), row.get("Pb_max_MW"), row["proved"])
    return row


def pmap(fn, items):
    """Parallel map over CPU cores (each worker is single-threaded numpy)."""
    from multiprocessing import Pool
    n = max(1, min(len(items), os.cpu_count() or 1))
    if n == 1:
        return [fn(x) for x in items]
    with Pool(n) as pool:
        return pool.map(fn, items, chunksize=1)


def design_search(R):
    jobs = [(R, a, c, kf) for a in C.ALPHA_GRID for c in C.C_GRID for kf in C.KE_FRAC_GRID]
    return pmap(_search_one, jobs)


def lower_bound(R_mw_per_min):
    R = R_mw_per_min / 60.0
    return dict(Pb_MW=C.DP, energy_MWh=C.DP ** 2 / (2 * C.ETA * R) / C.SEC_PER_H)


def certify_at(R, row, mode="w"):
    """Re-certify a design at ramp limit R: reuse the normalized offsets (the normalized problem is
    the same for every R), BaB from scratch; fall back to a fresh search if that fails."""
    d = Design.from_alpha(R, row["alpha"], row["c"], row["ke_frac"], mode=mode)
    cert = Cert(d, directions(), np.array(row["offsets"]))
    ok, faces, summ = verify_invariance(cert)
    if not ok:
        cert, info = certify_design(d)
        if cert is None:
            return None, None, dict(proved=False)
        summ, faces = strip(info), info["faces"]
    return cert, faces, summ


def _sizing_one(args):
    R, top = args
    cands = []
    for row in top:
        c_R, _, s_R = certify_at(R, row)
        if c_R is None:
            cands.append(dict(alpha=row["alpha"], c=row["c"], ke_frac=row["ke_frac"], proved=False))
            continue
        a_R = analyze(c_R)
        cands.append(dict(alpha=row["alpha"], c=row["c"], ke_frac=row["ke_frac"], proved=True,
                          tau_s=c_R.d.tau, energy_window_MWh=a_R["energy_window_MWh"],
                          Pb_max_MW=a_R["Pb_max_MW"], bab=s_R))
    ok = [c for c in cands if c["proved"]]
    bestR = min(ok, key=lambda c: (c["energy_window_MWh"], c["Pb_max_MW"]))
    lbR = lower_bound(R)
    log("sizing", R, bestR["energy_window_MWh"], bestR["Pb_max_MW"])
    return dict(R_mw_per_min=R, best=bestR, candidates=cands, lower_bound=lbR,
                energy_ratio_to_lower_bound=bestR["energy_window_MWh"] / lbR["energy_MWh"])


def main():
    reuse = "--reuse" in sys.argv
    T0 = time.perf_counter()
    Rn = C.R_NOMINAL_MW_PER_MIN

    # ------------------------------------------------------------- search
    path = os.path.join(RES, "design_search.json")
    if reuse and os.path.exists(path):
        rows = json.load(open(path))["rows"]
    else:
        rows = design_search(Rn)
        dump("design_search.json", dict(R_mw_per_min=Rn, n_dirs=C.N_DIRS, band_eps=C.BAND_EPS, rows=rows))
    proved = sorted([r for r in rows if r["proved"]], key=lambda r: (r["energy_window_MWh"], r["Pb_max_MW"]))
    best = proved[0]
    log("best", best["alpha"], best["c"], best["ke_frac"], best["energy_window_MWh"])

    # ------------------------------------------------------------- nominal certificate in full
    cert, faces, summ = certify_at(Rn, best)
    an = analyze(cert)
    d = cert.d
    lb = lower_bound(Rn)
    nominal = dict(
        R_mw_per_min=Rn, design=d.to_dict(), certificate=cert.to_dict(), bab_invariance=summ,
        bab_faces=faces, bounds={k: v for k, v in an.items() if k != "bound_proofs"},
        bound_proofs=an["bound_proofs"], lower_bound_any_controller=lb,
        ratio_energy_to_lower_bound=an["energy_window_MWh"] / lb["energy_MWh"],
        ratio_power_to_lower_bound=an["Pb_max_MW"] / lb["Pb_MW"],
        vertices_Pg_e=cert.vertices_x().tolist(),
        P_min=C.P_MIN, P_max=C.P_MAX, eta=C.ETA,
    )
    # plain SOC loop (restore e itself) with the same (alpha, c, ke_frac): certified for comparison
    de = Design.from_alpha(Rn, best["alpha"], best["c"], best["ke_frac"], mode="e")
    cert_e, info_e = certify_design(de)
    if cert_e is not None:
        an_e = analyze(cert_e)
        nominal["plain_soc_loop"] = dict(design=de.to_dict(), proved=True, bab=strip(info_e),
                                         bounds={k: v for k, v in an_e.items() if k != "bound_proofs"},
                                         vertices_Pg_e=cert_e.vertices_x().tolist())
    else:
        nominal["plain_soc_loop"] = dict(design=de.to_dict(), proved=False, reason=info_e.get("reason"))
    dump("certificate.json", nominal)
    log("nominal certified", summ, an["energy_window_MWh"], an["Pb_max_MW"])

    # ------------------------------------------------------------- sizing curve
    top = proved[:3]
    sizing = pmap(_sizing_one, [(R, top) for R in C.R_SWEEP_MW_PER_MIN])
    dump("sizing.json", dict(rows=sizing, note="best over the top-3 designs of the nominal search, each BaB-certified at that R"))

    # ------------------------------------------------------------- simulations (illustration only)
    rng = np.random.default_rng(C.SEED)
    Pb_cap, e_lo, e_hi = an["Pb_max_MW"], an["e_lo_MWh"], an["e_hi_MWh"]
    rate = lambda Pg, e, PL: pg_rate(d, Pg, e, PL)

    def x0_for(PL0):
        # start every simulation at the centre (P_g = P_mid, e = e_ref), regardless of the first load value
        Pg0 = np.full_like(PL0, P_MID)
        e0 = np.zeros_like(PL0)
        assert np.all(cert.h(Pg0, e0) <= 0), "initial states must lie in S"
        return Pg0, e0

    def inside_stats(out):
        H = cert.h(out["P_g"].ravel(), out["e"].ravel())
        return float(H.max())

    t = time_grid(3600.0 * HSCALE)
    PLf, labels = family(t, rng)
    Pg0, e0 = x0_for(PLf[0])
    fam = simulate(rate, PLf, Pg0, e0, Pb_cap, e_lo, e_hi, stride=20)
    sim = dict(family=dict(n_profiles=PLf.shape[1], horizon_s=3600.0 * HSCALE, dt_s=C.DT,
                           max_grid_ramp_MW_per_s=float(fam["ramp"].max()),
                           max_abs_Pb_MW=float(fam["pb_abs"].max()), e_min_MWh=float(fam["e_min"].min()),
                           e_max_MWh=float(fam["e_max"].max()), max_B_on_trajectories=inside_stats(fam)))
    log("family sims done")

    # square-wave worst-case search (any period, duty, depth: all admissible)
    periods = [2, 5, 10, 20, 30, 60, 120, 240, 480, 960]
    duties = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    depths = [0.1, 0.25, 0.4]
    grid = [(p, du, de_) for p in periods for du in duties for de_ in depths]
    PLs = np.stack([square(t, p, du, de_) for p, du, de_ in grid], axis=1)
    Pg0, e0 = x0_for(PLs[0])
    sq = simulate(rate, PLs, Pg0, e0, Pb_cap, e_lo, e_hi, stride=20)
    i_min, i_max, i_pb = int(np.argmin(sq["e_min"])), int(np.argmax(sq["e_max"])), int(np.argmax(sq["pb_abs"]))
    desc = lambda i: dict(period_s=grid[i][0], duty=grid[i][1], depth_frac=grid[i][2],
                          e_min_MWh=float(sq["e_min"][i]), e_max_MWh=float(sq["e_max"][i]),
                          max_abs_Pb_MW=float(sq["pb_abs"][i]))
    sim["square_search"] = dict(n_profiles=len(grid), horizon_s=3600.0 * HSCALE, max_grid_ramp_MW_per_s=float(sq["ramp"].max()),
                                max_abs_Pb_MW=float(sq["pb_abs"].max()), e_min_MWh=float(sq["e_min"].min()),
                                e_max_MWh=float(sq["e_max"].max()), max_B_on_trajectories=inside_stats(sq),
                                worst_discharge=desc(i_min), worst_charge=desc(i_max), worst_power=desc(i_pb))

    # greedy adversary: at each step pick P_L in {P_min, P_max} that pushes the active face outward fastest
    V0 = cert.vertices_x()
    cen = V0.mean(0)
    starts = np.array([cen + 0.5 * (v - cen) for v in V0[np.linspace(0, len(V0) - 1, 8).astype(int)]] + [cen])
    Pg, e = starts[:, 0].copy(), starts[:, 1].copy()
    nT = int(7200 * HSCALE / C.DT)
    greedy_B, gtr = -np.inf, []
    for k in range(nT):
        H = cert.h(Pg, e)
        act = H.argmax(0)
        best_L, best_val = None, None
        for PLc in (C.P_MIN, C.P_MAX):
            PLv = np.full_like(Pg, PLc)
            fP_, fe_ = cert.field(Pg, e, PLv)
            allL = cert.N[:, 0:1] * fP_ / C.DP + cert.N[:, 1:2] * fe_ / cert.E0
            val = allL[act, np.arange(len(Pg))]
            if best_val is None:
                best_L, best_val = PLv, val
            else:
                m = val > best_val
                best_L = np.where(m, PLv, best_L); best_val = np.where(m, val, best_val)
        fP = pg_rate(d, Pg, e, best_L)
        pb = best_L - Pg
        e = e + C.DT * (-phi(pb) / C.SEC_PER_H)
        Pg = Pg + C.DT * fP
        greedy_B = max(greedy_B, float(H.max()))
        if k % 20 == 0:
            gtr.append(np.stack([Pg, e, best_L], 1))
    gtr = np.array(gtr)
    sim["greedy_adversary"] = dict(n_starts=len(starts), horizon_s=7200.0 * HSCALE, max_B=greedy_B,
                                   e_min_MWh=float(gtr[:, :, 1].min()), e_max_MWh=float(gtr[:, :, 1].max()))
    dump("simulation.json", sim)
    log("sims done")

    # ------------------------------------------------------------- ablations
    t2 = time_grid(7200.0 * HSCALE)
    abl_grid = [(p, du, de_) for p in [2, 5, 10, 20, 30] for du in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                for de_ in [0.1, 0.2, 0.3, 0.4]]
    PLa = np.stack([square(t2, p, du, de_) for p, du, de_ in abl_grid], axis=1)
    naive = pg_rate_naive_limiter(d.R)
    na = simulate(naive, PLa, PLa[0], np.zeros(PLa.shape[1]), Pb_cap, e_lo, e_hi, stride=10)
    te = na["t_empty"]
    i_dr = int(np.argmin(np.where(np.isnan(te), np.inf, te))) if np.any(~np.isnan(te)) else int(np.argmin(na["e_min"]))
    nosoc = simulate(lambda Pg, e, PL: pg_rate_no_soc(d, Pg, e, PL), PLa, PLa[0], np.zeros(PLa.shape[1]),
                     Pb_cap, e_lo, e_hi, stride=10)
    ours = simulate(rate, PLa, *x0_for(PLa[0]), Pb_cap, e_lo, e_hi, stride=10)
    n_drained = int(np.sum(~np.isnan(te)))
    ab = dict(
        battery=dict(Pb_max_MW=Pb_cap, e_lo_MWh=e_lo, e_hi_MWh=e_hi),
        no_ups=dict(max_step_MW=C.DP, max_grid_ramp_MW_per_s_at_dt=C.DP / C.DT, ramp_limit_MW_per_s=d.R,
                    note="grid sees P_L directly; a load step is a grid step, so |dP/dt| is unbounded"),
        naive_rate_limiter=dict(tau_fast_s=1.0, n_profiles=len(abl_grid), horizon_s=7200.0 * HSCALE,
                                n_drained=n_drained,
                                fastest_drain_profile=dict(period_s=abl_grid[i_dr][0], duty=abl_grid[i_dr][1],
                                                           depth_frac=abl_grid[i_dr][2],
                                                           time_to_empty_s=float(te[i_dr]),
                                                           max_grid_ramp_MW_per_s=float(na["ramp"][i_dr]))),
        filter_without_soc_loop=dict(n_profiles=len(abl_grid), horizon_s=7200.0 * HSCALE,
                                     n_drained=int(np.sum(~np.isnan(nosoc["t_empty"]))),
                                     e_end_min_MWh=float(nosoc["e"][-1].min()),
                                     e_min_MWh=float(nosoc["e_min"].min())),
        certified_controller_same_profiles=dict(n_drained=int(np.sum(~np.isnan(ours["t_empty"]))),
                                                e_min_MWh=float(ours["e_min"].min()),
                                                max_grid_ramp_MW_per_s=float(ours["ramp"].max())),
    )
    dump("ablation.json", ab)
    log("ablation done", ab["naive_rate_limiter"]["fastest_drain_profile"])

    # ------------------------------------------------------------- voltage
    tp = V.TAN_PHI_LOAD
    v_ups = V.sound_range(V.v_exact_iv, an["Pg_lo_MW"], an["Pg_hi_MW"], 0.0)
    v_no = V.sound_range(V.v_exact_iv, C.P_MIN, C.P_MAX, tp)
    dvdp = V.sound_range(V.dv_dp_iv, an["Pg_lo_MW"], an["Pg_hi_MW"], 0.0)
    L_V = max(abs(dvdp[0]), abs(dvdp[1]))
    Ps = np.linspace(min(an["Pg_lo_MW"], C.P_MIN), max(an["Pg_hi_MW"], C.P_MAX), 41)
    pp0 = V.pandapower_check(Ps, 0.0)
    ppq = V.pandapower_check(Ps, tp)
    step_no_ups = float(V.v_exact(C.P_MIN, C.P_MIN * tp) - V.v_exact(C.P_MAX, C.P_MAX * tp))
    volt = dict(
        feeder=dict(V_kV=C.V_BASE_KV, length_km=C.LINE_KM, R_ohm=C.LINE_R_OHM_PER_KM * C.LINE_KM,
                    X_ohm=C.LINE_X_OHM_PER_KM * C.LINE_KM, r_pu=V.R_PU, x_pu=V.X_PU, pf_no_ups=C.PF_LOAD_NO_UPS),
        with_ups=dict(P_range_MW=[an["Pg_lo_MW"], an["Pg_hi_MW"]], Q_MVAr=0.0, V_range_pu=list(v_ups),
                      max_abs_dev_pu=max(abs(1 - v_ups[0]), abs(v_ups[1] - 1)),
                      dVdP_range_pu_per_MW=list(dvdp), dVdt_bound_pu_per_s=L_V * d.R,
                      dV_bound_per_min_pu=L_V * d.R * 60.0,
                      lin_error_bound_pu=V.lin_error_bound(an["Pg_lo_MW"], an["Pg_hi_MW"], 0.0)),
        without_ups=dict(P_range_MW=[C.P_MIN, C.P_MAX], V_range_pu=list(v_no),
                         max_abs_dev_pu=max(abs(1 - v_no[0]), abs(v_no[1] - 1)),
                         step_dV_pu_for_full_load_step=step_no_ups,
                         lin_error_bound_pu=V.lin_error_bound(C.P_MIN, C.P_MAX, tp)),
        band_pu=C.V_BAND,
        pandapower=dict(n_points=len(Ps), P_range_MW=[float(Ps[0]), float(Ps[-1])],
                        max_abs_diff_vs_exact_unity_pf=float(np.abs(pp0 - V.v_exact(Ps, 0.0)).max()),
                        max_abs_diff_vs_exact_pf=float(np.abs(ppq - V.v_exact(Ps, Ps * tp)).max()),
                        max_abs_diff_vs_linear_unity_pf=float(np.abs(pp0 - V.v_lin(Ps, 0.0)).max()),
                        max_abs_diff_vs_linear_pf=float(np.abs(ppq - V.v_lin(Ps, Ps * tp)).max())),
    )
    dump("voltage.json", volt)
    log("voltage done")

    # ------------------------------------------------------------- figures
    # (a) load vs grid on 3 worst-case profiles
    picks = [("worst discharge", i_min), ("worst charge", i_max), ("largest battery power", i_pb)]
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 4.6), sharex=True)
    tt = sq["t"] / 60
    for ax, (name, i) in zip(axs, picks):
        ax.plot(tt, sq["P_L"][:, i], color=INK2, lw=0.6, label="load $P_L$")
        ax.plot(tt, sq["grid"][:, i], color=BLUE, lw=1.4, label="grid $P_g$")
        p, du, de_ = grid[i]
        ax.set_title(f"{name}: {p} s, duty {du}, depth {int(de_*100)}%", loc="left", fontsize=7)
        ax.set_ylabel("MW")
        ax.set_xlim(0, 30)
    axs[0].legend(loc="lower right", ncol=2)
    axs[-1].set_xlabel("time (min)")
    save(fig, os.path.join(FIG, "fig_a_worst_profiles"))

    # (b) certified invariant set with trajectories
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    Vx = np.array(nominal["vertices_Pg_e"])
    if cert_e is not None:
        Ve = cert_e.vertices_x()
        ax.fill(Ve[:, 0], Ve[:, 1], facecolor="none", edgecolor=ORANGE, lw=1.0, ls="--")
        ax.text(Ve[:, 0].mean(), Ve[:, 1].max(), "plain SOC loop (restores e)", color=ORANGE, va="bottom", ha="center", fontsize=6)
    ax.fill(Vx[:, 0], Vx[:, 1], facecolor=BLUE, alpha=0.12, edgecolor=BLUE, lw=1.4)
    for j in range(0, fam["P_g"].shape[1], 6):
        ax.plot(fam["P_g"][:, j], fam["e"][:, j], color=INK2, lw=0.3, alpha=0.5)
    for j in range(gtr.shape[1]):
        ax.plot(gtr[:, j, 0], gtr[:, j, 1], color=AQUA, lw=0.6)
    ax.set_xlabel("grid power $P_g$ (MW)")
    ax.set_ylabel("stored energy $e - e_{ref}$ (MWh)")
    ax.annotate("certified set S", xy=(Vx[:, 0].mean() + 8, Vx[:, 1].mean() - 0.4), xytext=(Vx[:, 0].mean() + 2, Vx[:, 1].min() - 1.0),
                color=BLUE, fontsize=7, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    ax.plot([], [], color=INK2, lw=0.6, label="simulated profiles")
    ax.plot([], [], color=AQUA, lw=0.8, label="greedy adversary")
    ax.legend(loc="lower left", fontsize=6)
    ax.margins(y=0.12)
    save(fig, os.path.join(FIG, "fig_b_invariant_set"))

    # (c) battery size vs ramp limit
    Rs = np.array([r["R_mw_per_min"] for r in sizing])
    Ew = np.array([r["best"]["energy_window_MWh"] for r in sizing])
    Pw = np.array([r["best"]["Pb_max_MW"] for r in sizing])
    Elb = np.array([r["lower_bound"]["energy_MWh"] for r in sizing])
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.2))
    axs[0].loglog(Rs, Ew, "o-", color=BLUE, ms=4, label="certified")
    axs[0].loglog(Rs, Elb, "s--", color=INK2, ms=3, lw=0.8, label="any controller (bound)")
    axs[0].set_xlabel("ramp limit (MW/min)"); axs[0].set_ylabel("usable energy (MWh)")
    axs[0].set_xticks(Rs); axs[0].set_xticklabels([f"{r:g}" for r in Rs]); axs[0].minorticks_off()
    axs[0].legend(loc="lower left", fontsize=6)
    axs[1].plot(Rs, Pw, "o-", color=BLUE, ms=4, label="certified")
    axs[1].axhline(C.DP, color=INK2, ls="--", lw=0.8, label="any controller (bound)")
    axs[1].set_xscale("log"); axs[1].set_ylim(0, max(Pw) * 1.25)
    axs[1].set_xlabel("ramp limit (MW/min)"); axs[1].set_ylabel("battery power (MW)")
    axs[1].set_xticks(Rs); axs[1].set_xticklabels([f"{r:g}" for r in Rs]); axs[1].minorticks_off()
    axs[1].legend(loc="lower left", fontsize=6)
    save(fig, os.path.join(FIG, "fig_c_battery_vs_ramp"))

    # (d) PCC voltage with and without the UPS on the checkpoint profile
    tv = time_grid(1800.0)
    PLv = checkpoint(tv, every=300, dip_len=40)[:, None]
    ov = simulate(rate, PLv, *x0_for(PLv[0]), Pb_cap, e_lo, e_hi, stride=1)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(tv / 60, V.v_exact(PLv[:, 0], PLv[:, 0] * tp), color=ORANGE, lw=1.0, label="no UPS (pf 0.95)")
    ax.plot(tv / 60, V.v_exact(ov["grid"][:, 0], 0.0), color=BLUE, lw=1.4, label="with UPS")
    ax.axhline(1 - C.V_BAND, color=RED, lw=0.8, ls=":")
    ax.set_xlabel("time (min)"); ax.set_ylabel("PCC voltage (pu)")
    ax.set_ylim(0.94, 0.995)
    ax.legend(loc="lower center", ncol=2, fontsize=6)
    save(fig, os.path.join(FIG, "fig_d_pcc_voltage"))
    volt["figure_profile"] = "checkpoint every 300 s, 40 s dip to idle"
    dump("voltage.json", volt)

    # (e) ablation: naive rate limiter drains the battery
    i = i_dr
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.4))
    ta = na["t"] / 60
    axs[0].plot(ta, na["e"][:, i], color=ORANGE, lw=1.2, label="naive rate limiter")
    axs[0].plot(ta, nosoc["e"][:, i], color=VIOLET, lw=1.0, label="filter, no SOC loop")
    axs[0].plot(ta, ours["e"][:, i], color=BLUE, lw=1.4, label="certified")
    axs[0].axhline(e_lo, color=INK2, lw=0.8, ls="--")
    axs[0].set_ylabel("stored energy (MWh)"); axs[0].set_xlabel("time (min)")
    axs[0].legend(loc="lower left", fontsize=6)
    one = simulate(naive, PLa[:, i:i + 1], PLa[0, i:i + 1], np.zeros(1), Pb_cap, e_lo, e_hi, stride=1)
    ts = one["t"]
    seg = ts > ts.max() - 30.0
    axs[1].plot(ts[seg] - ts.max(), one["P_L"][seg, 0], color=INK2, lw=0.6, label="load")
    axs[1].plot(ts[seg] - ts.max(), one["grid"][seg, 0], color=ORANGE, lw=1.0, ls="--", label="grid, naive (battery empty)")
    axs[1].set_ylabel("MW"); axs[1].set_xlabel("time before end of run (s)")
    axs[1].legend(loc="lower left", fontsize=6)
    p, du, de_ = abl_grid[i]
    axs[0].set_title(f"bursts: period {p} s, duty {du}, depth {int(de_*100)}%", loc="left")
    save(fig, os.path.join(FIG, "fig_e_ablation"))

    dump("run_info.json", dict(total_seconds=time.perf_counter() - T0, python=sys.version, numpy=np.__version__))
    log("done", time.perf_counter() - T0)


if __name__ == "__main__":
    main()
