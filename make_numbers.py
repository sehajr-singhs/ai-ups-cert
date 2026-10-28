"""Turn results/*.json into paper/numbers.tex (LaTeX macros) so the paper never retypes a number.

    .venv/Scripts/python make_numbers.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.environ.get("AIUPS_RESULTS", "results")
R = lambda n: json.load(open(os.path.join(ROOT, RESDIR, n)))


def f(x, nd=2):
    return f"{x:.{nd}f}"


def build():
    cert, siz, sim, abl, volt = R("certificate.json"), R("sizing.json"), R("simulation.json"), R("ablation.json"), R("voltage.json")
    srch = R("design_search.json")
    d, b, inv = cert["design"], cert["bounds"], cert["bab_invariance"]
    m = {}
    m["Rnom"] = f(cert["R_mw_per_min"], 0)
    m["RnomS"] = f(d["R"], 4)
    m["tauNom"] = f(d["tau"], 0)
    m["cNom"] = f(d["c"], 0)
    m["keNom"] = f(d["ke"], 2)
    m["alphaNom"] = f(cert["certificate"]["design"]["tau"] * d["R"] / 40.0, 2)
    m["Ezero"] = f(cert["certificate"]["E0_MWh"], 3)
    m["nDirs"] = str(len(cert["certificate"]["offsets"]))
    m["bandEps"] = f(cert["certificate"]["band_eps"], 3)
    m["babBoxes"] = f"{inv['boxes']:,}".replace(",", "{,}")
    m["babDepth"] = str(inv["max_depth"])
    m["babSec"] = f(inv["seconds"], 1)
    m["eHi"] = f(b["e_hi_MWh"], 3)
    m["eLo"] = f(b["e_lo_MWh"], 3)
    m["Ewin"] = f(b["energy_window_MWh"], 3)
    m["PbMax"] = f(b["Pb_max_MW"], 2)
    m["PgLo"] = f(b["Pg_lo_MW"], 2)
    m["PgHi"] = f(b["Pg_hi_MW"], 2)
    m["clipRate"] = f(b["unclipped_rate_max_est_MW_per_s"], 4)
    m["clipBinds"] = "does" if b["clip_binds_on_S"] else "does not"
    m["lbE"] = f(cert["lower_bound_any_controller"]["energy_MWh"], 3)
    m["lbP"] = f(cert["lower_bound_any_controller"]["Pb_MW"], 0)
    m["ratioE"] = f(cert["ratio_energy_to_lower_bound"], 2)
    m["ratioP"] = f(cert["ratio_power_to_lower_bound"], 3)
    pl = cert["plain_soc_loop"]
    m["plainProved"] = "yes" if pl["proved"] else "no"
    if pl["proved"]:
        m["plainEwin"] = f(pl["bounds"]["energy_window_MWh"], 3)
        m["plainPb"] = f(pl["bounds"]["Pb_max_MW"], 2)
        m["plainRatio"] = f(pl["bounds"]["energy_window_MWh"] / b["energy_window_MWh"], 2)
    rows = srch["rows"]
    m["nSearch"] = str(len(rows))
    m["nSearchProved"] = str(sum(r["proved"] for r in rows))
    # sizing table rows
    lines = []
    for r in siz["rows"]:
        bb = r["best"]
        lines.append(f"{r['R_mw_per_min']:.0f} & {bb['alpha']:.2f} & {bb['c']:.0f} & {bb['tau_s']:.0f} & "
                     f"{bb['Pb_max_MW']:.2f} & {bb['energy_window_MWh']:.3f} & {r['lower_bound']['energy_MWh']:.3f} & "
                     f"{r['energy_ratio_to_lower_bound']:.2f} & {bb['bab']['boxes']:,} \\\\".replace(",", "{,}"))
    m["sizingRows"] = "\n".join(lines)
    # simulation
    fam, sq, gr = sim["family"], sim["square_search"], sim["greedy_adversary"]
    m["famN"] = str(fam["n_profiles"])
    m["famRamp"] = f(fam["max_grid_ramp_MW_per_s"], 4)
    m["famPb"] = f(fam["max_abs_Pb_MW"], 2)
    m["famB"] = f(fam["max_B_on_trajectories"], 3)
    m["sqN"] = str(sq["n_profiles"])
    m["sqEmin"] = f(sq["e_min_MWh"], 3)
    m["sqEmax"] = f(sq["e_max_MWh"], 3)
    m["sqPb"] = f(sq["max_abs_Pb_MW"], 2)
    m["sqB"] = f(sq["max_B_on_trajectories"], 3)
    m["grB"] = f(gr["max_B"], 3)
    m["grEmin"] = f(gr["e_min_MWh"], 3)
    m["grEmax"] = f(gr["e_max_MWh"], 3)
    wd = sq["worst_discharge"]
    m["wdPer"], m["wdDuty"], m["wdDepth"] = str(wd["period_s"]), str(wd["duty"]), f(100 * wd["depth_frac"], 0)
    # ablation
    nr = abl["naive_rate_limiter"]
    fd = nr["fastest_drain_profile"]
    m["abN"] = str(nr["n_profiles"])
    m["abDrained"] = str(nr["n_drained"])
    m["abPer"], m["abDuty"], m["abDepth"] = str(fd["period_s"]), str(fd["duty"]), f(100 * fd["depth_frac"], 0)
    m["abTempty"] = f(fd["time_to_empty_s"] / 60.0, 1)
    m["abRamp"] = f(fd["max_grid_ramp_MW_per_s"], 1)
    m["noSocDrained"] = str(abl["filter_without_soc_loop"]["n_drained"])
    m["noSocEmin"] = f(abl["filter_without_soc_loop"]["e_min_MWh"], 3)
    m["oursDrained"] = str(abl["certified_controller_same_profiles"]["n_drained"])
    m["oursEmin"] = f(abl["certified_controller_same_profiles"]["e_min_MWh"], 3)
    m["noUpsRamp"] = f(abl["no_ups"]["max_grid_ramp_MW_per_s_at_dt"], 0)
    # voltage
    w, n, pp = volt["with_ups"], volt["without_ups"], volt["pandapower"]
    m["rpu"], m["xpu"] = f(volt["feeder"]["r_pu"], 4), f(volt["feeder"]["x_pu"], 4)
    m["vUpsLo"], m["vUpsHi"] = f(w["V_range_pu"][0], 4), f(w["V_range_pu"][1], 4)
    m["vUpsDev"] = f(100 * w["max_abs_dev_pu"], 2)
    m["vNoLo"], m["vNoHi"] = f(n["V_range_pu"][0], 4), f(n["V_range_pu"][1], 4)
    m["vNoDev"] = f(100 * n["max_abs_dev_pu"], 2)
    m["vStep"] = f(100 * n["step_dV_pu_for_full_load_step"], 2)
    m["dVdtMin"] = f(100 * w["dV_bound_per_min_pu"], 3)
    m["linErrUps"] = f"{w['lin_error_bound_pu']:.2e}"
    m["linErrNo"] = f"{n['lin_error_bound_pu']:.2e}"
    m["ppExact"] = f"{max(pp['max_abs_diff_vs_exact_unity_pf'], pp['max_abs_diff_vs_exact_pf']):.1e}"
    m["ppLin"] = f"{max(pp['max_abs_diff_vs_linear_unity_pf'], pp['max_abs_diff_vs_linear_pf']):.2e}"
    m["ppN"] = str(pp["n_points"])
    return m


if __name__ == "__main__":
    m = build()
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as fh:
        fh.write("% generated by make_numbers.py from results/*.json; do not edit\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in m.items():
        print(k, "=", v.replace("\n", " | ")[:200])
