# RESULTS

Every number in this file is filled in by `make_results_md.py` from `results/*.json`, which `run_all.py` writes.
This is a **public, assumed-parameter model** (see `src/aiups/config.py`). It is not ON.energy hardware, data or
controller.

## Model

* Load: any Lebesgue-measurable `P_L(t)` in `[60, 100]` MW. Steps allowed, no bound on `dP_L/dt`. `DP = 40` MW.
* Battery: `P_b = sat_Pbar(P_L - P_g)` (discharge positive), grid draw `P_grid = P_L - P_b`.
* Energy: `de/dt = -phi(P_b)/3600` MWh/s, `phi(p) = p/eta` for `p >= 0`, `eta p` for `p < 0`, `eta = 0.95` (assumed).
* Controller (certified):
  `dP_g/dt = clip((P_L + s - P_g)/tau, -R, R)`, `s = sat_c(k_e (e_ref - w))`, `w = e + tau (P_g - 80)/3600`.
  `w` is the "settled" state of charge: the energy the battery holds once `P_g` settles at 80 MW (lossless).
  Design parameters: `tau = alpha DP / R`, `k_e = c / (kappa E0)`, `E0 = tau DP / 3600`.
* Voltage: exact two-bus solution at the PCC, 69 kV, 10 km, 0.05 + j0.35 ohm/km, 100 MVA base
  (`r = 0.0105` pu, `x = 0.0735` pu), stiff 1.0 pu source, UPS at unity power factor; without UPS the campus is at
  power factor 0.95 lagging (all assumed). No reactive-power support term was needed (the band holds without it).

## Nominal design (R = 10 MW/min = 0.1667 MW/s)

`alpha = 0.75`, `c = 10` MW, `tau = 180` s, `k_e = 100.00` MW/MWh, `E0 = 2.000` MWh.
Certified set: 32-face polygon `S = {B <= 0}`, `B(x) = max_k (n_k . z(x) - b_k)`,
`z = ((P_g - 80)/40, e/E0)`; normals and offsets in `results/certificate.json`. Band width `eps = 0.002`.

### Lemma 1 (closed form, unsaturated filter)
For `dP_g/dt = (P_L - P_g)/tau` (no clip, no SOC loop) and any admissible `P_L`: if `P_g(0)` is in `[60, 100]` it
stays there, so `|dP_g/dt| <= DP/tau` a.e. If `eta = 1`, `w` is constant. With losses `dw/dt <= 0`, so without a
SOC loop the battery drifts empty on any oscillating load.

### Lemma 2 (band lemma)
If for every face `k`, `grad h_k . f(x, p) <= 0` for all `p` in `[60, 100]` and all `x` with
`0 <= h_k(x) <= eps` and `h_j(x) <= eps` (`j != k`), then `S` is forward invariant for every measurable `P_L`.
(Proof: last-exit-time argument on the absolutely continuous `h_k(x(t))`; written out in `paper/main.tex`.)

### Proposition 1 (battery invariance), PROVED
Interval branch-and-bound (outward rounding, one ulp per operation; monotone clip/sat/phi evaluated at endpoints)
verified the band condition for all 32 faces over the full `(P_g, e, P_L)` box:
**27,230 boxes, max depth 40, 1.4 s** (per-face stats in `results/certificate.json`).
Hence for every admissible `P_L` and every `x(0)` in `S`, for all `t >= 0`:

* `x(t)` in `S`
* `|P_L - P_g| <= 40.28` MW (proved on the eps-enlarged set, so the battery limit never binds with `Pbar >= 40.28` MW)
* `e(t)` in `[-1.006, 1.006]` MWh, i.e. a usable window of **2.013 MWh**
* `P_g(t)` in `[59.81, 100.19]` MW

### Theorem 1 (ramp), PROVED
Under Proposition 1 with `Pbar >= 40.28` MW and window `[-1.006, 1.006]` MWh: `P_grid(t) = P_g(t)` and
`|P_grid(t2) - P_grid(t1)| <= R |t2 - t1|` for every admissible `P_L` and every `x(0)` in `S`.
(The clip gives `|dP_g/dt| <= R`; Proposition 1 makes the battery limits inactive so the grid draws exactly `P_g`.)
Estimated max of `|u - P_g|/tau` on the vertices of S: 0.2288 MW/s (vs R = 0.1667 MW/s), so the clip does bind
inside S for this design (estimate, not part of any proof).

### Corollary 1 (voltage), PROVED
With the UPS: `V_PCC(t)` in `[0.9865, 0.9927]` pu, `|V - 1| <= 1.35%` (band is 5%), and `V` changes by at most
**0.165% per minute** (no steps). Linearization error vs the exact formula over the certified `P_g` range:
<= 2.94e-03 pu (sound interval bound). pandapower AC power flow vs the exact formula: max difference 6.7e-12 pu at
41 points; pandapower vs the linear model: 4.07e-03 pu.
Without the UPS: `V` in `[0.9613, 0.9778]` pu (`|V - 1| <= 3.87%`, also inside 5% on this stiff feeder), but a full
40 MW load step moves it by **1.65% instantly**. Linearization error over that range: <= 4.06e-03 pu.
Honest reading: on this assumed feeder the UPS's voltage benefit is removing steps, not the static band.

### Proposition 2 (lower bound for ANY causal controller), PROVED (closed form)
Any controller whose grid power is R-Lipschitz and that keeps the battery inside its limits for every admissible
profile needs `Pbar >= DP = 40` MW and a window `>= DP^2 / (2 eta R 3600)` = **1.404 MWh** at R = 10 MW/min.
The certified design uses 1.007x the power and **1.43x the energy**.

## Battery size vs ramp limit (every row BaB-certified at its own R)

| R (MW/min) | alpha | c (MW) | tau (s) | Pbar (MW) | window (MWh) | lower bound (MWh) | ratio | BaB boxes |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.75 | 10 | 900 | 40.28 | 10.064 | 7.018 | 1.43 | 27,230 |
| 5 | 0.75 | 10 | 360 | 40.28 | 4.026 | 2.807 | 1.43 | 27,230 |
| 10 | 0.75 | 10 | 180 | 40.28 | 2.013 | 1.404 | 1.43 | 27,230 |
| 20 | 0.75 | 10 | 90 | 40.28 | 1.006 | 0.702 | 1.43 | 27,230 |
| 40 | 0.75 | 10 | 45 | 40.28 | 0.503 | 0.351 | 1.43 | 27,230 |

The closed loop is scale-invariant in R (time `t R/DP`, coordinates `z`), so the window scales as `1/R` and `Pbar`
is constant, exactly like the lower bound. Design search at the nominal R: 45 designs
(alpha in 0.5..1.0, c in 3, 6, 10 MW, kappa in 0.5, 0.15, 0.05), 29 certified; the rest had no invariant
32-face candidate (search failure, not a proof that none exists). The curve is "smallest over this grid".

## Simulations (illustration only, forward Euler, dt = 0.1 s)

* 210 profiles from the brief's families (square bursts 2 to 30 s, duty 0.2 to 0.8, depth 10 to 40%; checkpoint
  dips to idle; random walks), 1 h each: max grid ramp 0.1667 MW/s, max |P_b| 39.96 MW, max B on trajectories -0.002
  (<= 0 means inside S).
* 330 square waves, periods 2 to 960 s, duty 0.05 to 0.95, depth 10 to 40%: stored energy in [-1.000, 0.995] MWh,
  max |P_b| 39.96 MW, max B -0.002. Deepest discharge: period 960 s, duty 0.95, depth 10%.
* Greedy adversary (picks the load extreme that pushes the active face outward fastest), 2 h: max B -0.002, stored
  energy in [-1.000, 1.000] MWh vs certified [-1.006, 1.006]. The gap shows the conservatism of the polygon.

## Ablations

* **No UPS:** the grid sees `P_L` directly; a 40 MW step lands in one 0.1 s sample (400 MW/s at that
  resolution, unbounded in the limit) against R = 0.1667 MW/s. Ramp limit violated.
* **Naive ramp-rate limiter, no SOC loop** (`dP_g/dt = clip((P_L - P_g)/1 s, -R, R)`), same certified battery: drained
  it on 60 of 140 burst trains from the brief's family within 2 h. Fastest: period 5 s, duty 0.3,
  depth 20%, empty after 17.2 min; afterwards the grid sees 199.9 MW/s (Fig. e).
* **Certified filter with the SOC loop removed** (`k_e = 0`): drained the battery on 7 of the same profiles
  (min stored energy -1.007 MWh); losses make `w` drift down.
* **Certified controller** on the same profiles: drained on 0, min stored energy -0.931 MWh.
* **Plain SOC loop** (restore `e` instead of `w`, same alpha, c, kappa): certified = yes, window 5.700 MWh,
  Pbar 50.28 MW, i.e. **2.83x** the energy of the settled-SOC loop.

## Figures

`figures/fig_a_worst_profiles`, `fig_b_invariant_set`, `fig_c_battery_vs_ramp`, `fig_d_pcc_voltage`, `fig_e_ablation`
(each as .pdf and .png).

## What this does not prove

* It is a public, assumed-parameter model. No parameter comes from ON.energy; the controller is ours, not theirs.
* Averaged power model only: no inverter current/PLL dynamics, switching, faults, fault ride-through, protection,
  communication delay or measurement noise on `P_L`, battery aging/temperature/state-dependent efficiency.
* Loads outside `[60, 100]` MW (e.g. a full campus trip to zero) are not covered.
* Stiff source behind one line; real PCC voltage depends on the network and the campus's reactive power.
* Sizes are upper bounds for this controller family and this 32-face template on a small design grid; part of the
  gap to Proposition 2 may be the template, not the controller.
* The BaB is sound by construction but its implementation (`src/aiups/certify.py`, `interval.py`) is tested, not
  formally verified. Simulations are illustrations, not evidence for the theorems.
