# ai-ups-cert

A sound certificate for the grid-side behavior of an "AI UPS": a battery plus inverter that sits between an AI
training campus and the grid. The campus load can jump between its floor and its peak at any instant (synchronized
GPU bursts, checkpoint dips). The question is whether a simple grid-side controller can **guarantee**, for every
such load profile, that

1. the power the grid sees never ramps faster than a limit R,
2. the battery never exceeds its power rating or leaves its energy window,
3. the PCC voltage stays in a band and never steps.

This is a proof on a public, assumed-parameter model. It is not ON.energy's hardware, data, or controller.

## What is in here

| path | what |
|---|---|
| `src/aiups/config.py` | every parameter, each with a one-line source ("assumed" where it is a choice) |
| `src/aiups/model.py` | battery and state-of-charge model with one-way efficiency |
| `src/aiups/controller.py` | the certified controller (rate-clipped low-pass + saturated settled-SOC loop) and ablations |
| `src/aiups/interval.py` | interval arithmetic with outward rounding (one ulp per operation) |
| `src/aiups/certify.py` | polytope invariant set, sampling-based candidate search, sound interval branch-and-bound |
| `src/aiups/voltage.py` | exact two-bus PCC voltage, its linearization, interval bounds, pandapower AC check |
| `src/aiups/profiles.py`, `sim.py` | load families and forward-Euler simulation (illustration only) |
| `run_all.py` | reproduces `results/*.json` and `figures/*` |
| `RESULTS.md` | theorems, exact hypotheses, every number (copied from `results/`), and what this does not prove |
| `paper/` | the write-up (LaTeX) |

## Run

```
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy matplotlib pytest pandapower
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python run_all.py          # full run (design search included)
.venv/Scripts/python run_all.py --reuse  # reuse results/design_search.json
```

## How the proof works, in one paragraph

The certified set S is a 32-sided polygon in the (grid power, stored energy) plane, written as
B(x) = max_k (n_k . z(x) - b_k) <= 0. A short lemma shows S is forward invariant for every measurable load profile if,
on a thin band just outside each face, the face's outward derivative is nonpositive for every load value. Sampling
proposes the offsets b_k. Interval branch-and-bound over the full (P_g, e, P_L) box then proves the band condition
for every face: a box is discarded only when outward-rounded interval arithmetic shows it misses the band or shows the
derivative is <= 0 on all of it. The same branch-and-bound produces sound bounds on stored energy, battery power and
grid power over S, which give the battery rating, the energy window and the voltage band.

## Where the numbers were computed

The reported `results/*.json` and `figures/*` come from a Kaggle CPU run of `run_all.py`
(`kaggle/build_kernel.py` packs the code into one script; its log is `results/kaggle_run.log`).
`AIUPS_QUICK=1 python run_all.py` is a small smoke test that writes to `results_quick/` instead and is never reported.
`make_results_md.py` writes `RESULTS.md` and `make_numbers.py` writes `paper/numbers.tex`, both straight from
`results/*.json`, so no number is retyped by hand.
