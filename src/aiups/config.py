"""Every model parameter, each with a one-line source.

This is a public, assumed-parameter model. None of these numbers come from
ON.energy hardware or data; "assumed" means chosen by the author as a
plausible round number for a ~100 MW AI training campus.
"""

# ---------------------------------------------------------------- load
P_MIN = 60.0   # MW, campus floor (idle + cooling). Assumed (brief example 60 to 100 MW).
P_MAX = 100.0  # MW, campus peak. Assumed (brief example).
DP = P_MAX - P_MIN  # MW, largest admissible load step (steps allowed, |dP_L/dt| unbounded).

# ---------------------------------------------------------------- battery
ETA = 0.95     # one-way battery+converter efficiency (round trip ~0.90). Assumed.
SEC_PER_H = 3600.0  # s per h, converts MW*s to MWh. Definition.

# ---------------------------------------------------------------- ramp limits
R_NOMINAL_MW_PER_MIN = 10.0  # MW/min, nominal grid ramp limit. Assumed (not a quoted ERCOT number).
R_SWEEP_MW_PER_MIN = [2.0, 5.0, 10.0, 20.0, 40.0]  # MW/min, sweep for the sizing curve. Assumed.

# ---------------------------------------------------------------- controller design search
# Filter time constant tau = ALPHA * DP / R. ALPHA >= 1 + 2c/DP means the rate clip never binds.
ALPHA_GRID = [0.5, 0.6, 0.75, 0.9, 1.0]   # dimensionless. Design search grid (assumed).
C_GRID = [3.0, 6.0, 10.0]               # MW, SOC-loop authority |s| <= c. Design search grid (assumed).
KE_FRAC_GRID = [0.5, 0.15, 0.05]       # s saturates when |w - e_ref| = KE_FRAC * tau*DP/3600. Assumed.

# ---------------------------------------------------------------- certificate
N_DIRS = 32          # template polytope directions in normalized (P_g, e). Choice.
BAND_EPS = 2e-3      # band width (normalized units) in the invariance lemma. Choice.
BAB_MAX_DEPTH = 60   # max bisections per box. Choice.
BAB_MAX_BOXES = 4_000_000  # total box budget per face. Choice (keeps memory modest).
BAB_CHUNK = 200_000  # boxes evaluated per numpy batch. Choice (memory).

# ---------------------------------------------------------------- feeder / voltage
V_BASE_KV = 69.0     # kV, PCC nominal voltage. Assumed.
S_BASE_MVA = 100.0   # MVA, per-unit base. Convention.
LINE_KM = 10.0       # km, radial feeder length slack -> PCC. Assumed.
LINE_R_OHM_PER_KM = 0.05  # ohm/km. Assumed (typical large overhead conductor).
LINE_X_OHM_PER_KM = 0.35  # ohm/km. Assumed (typical overhead line).
V_SOURCE_PU = 1.0    # pu, stiff source behind the feeder. Assumed.
PF_LOAD_NO_UPS = 0.95  # lagging power factor the grid sees with no UPS. Assumed.
Q_GRID_UPS = 0.0     # MVAr, grid-side inverter runs at unity power factor. Assumed (no Q support needed, see RESULTS).
V_BAND = 0.05        # pu, |V - 1| <= 5%, a common planning band. Assumed.

# ---------------------------------------------------------------- simulation (illustration only)
DT = 0.1             # s, forward-Euler step for simulations. Choice.
SEED = 7             # RNG seed. Choice.

# ---------------------------------------------------------------- smoke-test mode (never used for reported numbers)
import os as _os
QUICK = _os.environ.get("AIUPS_QUICK") == "1"
if QUICK:
    ALPHA_GRID, C_GRID, KE_FRAC_GRID = [1.0], [6.0], [0.15]
    R_SWEEP_MW_PER_MIN = [10.0, 20.0]
    N_DIRS = 16
