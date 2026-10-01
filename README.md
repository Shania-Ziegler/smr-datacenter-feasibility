# Runtime Power Feasibility for SMR-Powered Datacenters

Simulation software for studying how a datacenter workload scheduler can
avoid asking a small modular reactor (SMR) for power trajectories that the
reactor cannot follow because of iodine-xenon dynamics.

**All reported results are simulation results. No reactor or datacenter
hardware is controlled by this software.**

## Overview

A datacenter powered by an SMR can save energy, follow cheap grid prices or
absorb workload dips by turning the reactor down. After a power reduction,
however, xenon-135 continues to build up from the decay of iodine-135 for
several hours. The negative reactivity it adds can exceed the reactor's
remaining reactivity margin.

A power request can therefore look feasible when it is made and become
infeasible hours later. The reactor is then forced to shut down, and it
cannot restart until the xenon decays, typically after more than a day.

This project exposes reactor feasibility to the scheduler **before** a
maneuver is committed. Each policy is evaluated against the same reactor
model and the same workloads. They differ only in what reactor information
they use: none, the current state, or a projection of the future state.

## Core idea

The reactor's xenon headroom is

```
H(t) = R_cap − ρ_Xe(t) − σ_m        [pcm]
```

where:
- `R_cap` is the excess reactivity available for xenon override. It
  shrinks with fuel age; the evaluation uses fixed snapshots of 8000, 3500
  and 2700 pcm, from beginning-of-cycle-like to end-of-cycle-like.
- `ρ_Xe(t)` is the reactivity worth of xenon-135.
- `σ_m` = 100 pcm is a model and measurement margin.

`H < 0` is a ceiling crossing. In the simulated plant, a crossing trips the
reactor, and it stays down until `H ≥ 50` pcm.

Policies that look only at the current state can approve a power reduction
whose xenon peak arrives hours later. **Predictive policies instead project
the trajectory.** Before accepting a power path, they integrate the
iodine-xenon equations forward over a 10 h horizon. They accept the path
only if the projected peak stays below an enforcement ceiling
`R_cap − σ_m − ε`, where ε = 20 pcm is an enforcement buffer that absorbs
discretization error. If it does not, they raise the reactor power floor
(by bisection) until it does. The projection is re-evaluated causally every
simulated minute.

```
   workload scheduler
          |
          v
   requested power trajectory
          |
          v
   reactor feasibility layer   (projects xenon over the horizon;
          |                     compares with R_cap − σ_m − ε)
          v
   feasible / limited action
```

### Experimental extension: partial grid import

Cheap-price grid imports displace reactor output one-for-one, so they
are also reactor turndowns. The canonical **Consolidated** controller
either accepts a requested cheap-price import whole, or cancels it whole
if its projected path is unsafe.

The opt-in extension **`consolidated_partial_grid`** asks the feasibility
layer a different question: what is the *largest* partial import whose own
projected path stays under the same ceiling? It finds that import by
bisection, and accepts only imports whose projections it has checked
directly. Everything else is identical to Consolidated. This is an
experimental extension with narrower validation (see
[Limitations](#limitations)), and it never replaces Consolidated in any
default run.

## What is included

- **Iodine-xenon reactor model** (`src/sim.py`): the I-135/Xe-135 balance
  equations, conversion of xenon to reactivity (pcm), a ramp limit of 5 %
  of rating per minute, the shutdown (SCRAM) and restart rule, and
  facility power with cooling.
  - Physics reference checks: `python src/sim.py --physics-audit`.
- **Feasibility projection:**
  - constant-hold projection: `sim.project_peak_rho`;
  - time-varying path projection: `sim.project_peak_rho_path`;
  - a standalone, scheduler-independent guard API in `src/xenon_guard.py`
    (`is_dip_safe`, `feasible_floor_checked`, `deadtime_if_tripped`).
- **Reactor policies** (`sim.run`): Limit-Agnostic, Limit-Reactive,
  fixed-delay baselines and Limit-Predictive.
- **Workload-aware controllers** (`src/consolidated_scheduler.py`, run by
  `sim.run_controller`): Computing-Only, Consolidated and the experimental
  `consolidated_partial_grid`.
- **Stress scenarios:** demand spike, synthetic deadline shock, surprise
  blackout, scheduled outage, and two compound events. All are
  deterministic and preregistered.
- **Sensitivity experiments:** look-ahead horizon, enforcement buffer,
  reactor-model mismatch, grid-interconnection capacity and the partial-grid
  extension.
- **Reproducibility utilities:**
  - write-once run manifests with code and input SHA-256 hashes;
  - energy-conservation residuals in every result row;
  - row-level regression against a reference run
    (`experiments/reproduce_reference_rows.py`);
  - an independent cross-check of shutdown dead time
    (`src/validate_deadtime.py`);
  - a test suite.

### Policies

| Policy | `rx` / mode name | Reactor information used | Behaviour |
|---|---|---|---|
| Limit-Agnostic | `free` | none | reactor follows the facility's net demand within its ramp limit |
| Limit-Reactive | `reactive` | current headroom | scales the permitted power *increase* by the current headroom relative to the equilibrium margin; no look-ahead |
| Fixed delay (2, 4, 6 h) | `hysteresis` | none (elapsed time only) | after any power reduction, blocks power increases for the delay |
| Limit-Predictive | `predictive` | current I-135/Xe-135 + constant-hold projection | before a reduction, projects 10 h at the target power; if unsafe, raises the power floor by bisection |
| Computing-Only | `computing_only` | none | workload-aware: defers batch and mid-tier work least-slack-first within deadlines, detects demand spikes, bridges ramp shortfalls with grid; never receives reactor state |
| Consolidated | `consolidated` | current I-135/Xe-135 + projection of the requested power path | Computing-Only's workload logic, plus feasibility: cancels the cheap-price import, then raises the power floor, if the projected path is unsafe |
| *Experimental:* partial grid | `consolidated_partial_grid` | as Consolidated | as Consolidated, but accepts the largest projected-safe partial cheap-price import instead of cancelling it |

In code comments, "legacy" refers to the reactor-policy path in `sim.run()`
(the first four rows). The controller extension leaves that path unchanged.

## Evaluation scope

- **Workloads:** Google Borg 2019 cells A–H, one 31-day trace per cell, at
  five-minute resolution. They are simulated at **one-minute** steps on a
  4,000-node facility (peak utilisation 85 %, PUE 1.2). Batch and mid-tier
  work is deferrable within 24 h and 1 h deadlines; production and
  monitoring work is rigid.
- **Reactor margin snapshots:** R_cap = 8000, 3500 and 2700 pcm.
- **Grid:** PJM day-ahead hourly LMP (pnode 34885183, DOM zone; January
  2025).
  - Price imports are allowed only at or below the window's 25th-percentile
    price.
  - The import cap is G = 0.4 × reactor rating unless stated otherwise.
  - Emergency import is allowed during a shutdown.
- **Comparisons:** every policy on every cell, R_cap snapshot and stress
  scenario, including the fixed-delay baselines.
- **Sensitivity:**
  - look-ahead horizon 0–15 h;
  - enforcement buffer ε = 0–100 pcm;
  - reactor-model mismatch: perturbed I-135/Xe-135 decay constants and
    flux, with the plant and the controller's projection using different
    parameters;
  - grid import capacity G = 0–1.0;
  - the partial-grid extension at G = 0–1.0.
- **Water accounting:**
  - reactor water from generation;
  - grid-embedded water from the hourly PJM generation mix with per-fuel
    water intensities (`sim.FUEL_WATER_GAL_MWH`);
  - datacenter WUE.

  These are estimates, not plant-specific measurements.

## External data

Neither external data source is distributed with this repository.
`data/README.md` gives the exact file names, schemas and windows, and
explains how to supply your own legitimately obtained copies.

- **Google Borg 2019 traces:** obtain them from Google's `cluster-data`
  repository, extract the per-window aggregates described in
  `data/README.md`, and convert them with `scripts/prepare_borg_cell.py`.
- **PJM Data Miner inputs** (day-ahead LMP and generation by fuel type):
  download them yourself under PJM's terms of use. **PJM data are not
  distributed in this repository.**

## Quick start

Requires Python 3.13 (other recent versions are untested).

```bash
# 1. install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run the tests (tests that need external data are skipped, with the
#    missing file named, until the data are present)
python -m pytest -q tests

# 3. a small example that needs no external data
python examples/synthetic_example.py

# 4. place your external inputs in data/ (see data/README.md), then check them
python scripts/check_external_inputs.py

# 5. main experiment: all policies x cells A-H x R_cap snapshots x scenarios
cd data && mkdir -p controller_runs
python ../src/sim.py --computing-only --consolidated --stress-suite \
    --decision-log --decision-log-level episodes --timing-mode detailed \
    --workers 8 --price-file pjm_dom_34885183.csv \
    --genmix-file load_pjm_genmix.csv --out controller_runs/main_suite.csv
```

The main suite is 2,112 one-month simulations. In the reported run it took
about 2.7 h with 9 worker processes. Use the name
`controller_runs/main_suite.csv`, and keep `--timing-mode detailed`: the
scripts in `experiments/` validate against this reference run and its
manifest.

## Reproducing the evaluation

Run `src/sim.py` from `data/`. Run the scripts in `experiments/` from the
repository root; they switch to `data/` themselves. Every run writes new
files only and refuses to overwrite existing outputs. All of these need the
external inputs.

| Question | Command | Main outputs (`data/controller_runs/`) |
|---|---|---|
| Main A–H comparison: all policies, fixed delays, R_cap snapshots, stress scenarios, water | step 5 of the quick start | `main_suite*.csv`, `main_suite_manifest.json` |
| Does the code reproduce the reference rows exactly? | `python experiments/reproduce_reference_rows.py --cells abcdefgh --rcaps 2700 --scenarios none --workers 8` | console report (`--report FILE` to save) |
| How much look-ahead is needed? | `cd data && python ../src/sim.py --consolidated --stress-suite --scenarios none --rcaps 2700,3500 --predict-horizon-h 0,2,4,6,10,15 --arms predictive,consolidated --timing-mode detailed --workers 8 --out controller_runs/horizon_sensitivity.csv` | `horizon_sensitivity*` |
| How large must the enforcement buffer be? | `cd data && python ../src/sim.py --consolidated --stress-suite --scenarios none --rcaps 2700,3500 --eps-buffer-pcm 0,10,20,50,100 --arms predictive,consolidated --timing-mode detailed --workers 8 --out controller_runs/eps_sensitivity.csv` | `eps_sensitivity*` |
| What if the controller's xenon model is wrong? | `python experiments/model_mismatch_experiment.py --workers 8 --out controller_runs/model_mismatch.csv`, then `python experiments/model_mismatch_analysis.py` | `model_mismatch*`, `data/analysis/model_mismatch*` |
| How does grid interconnection capacity change the comparison? | `python experiments/grid_capacity_experiment.py --workers 8 --out controller_runs/grid_capacity.csv`, then `python experiments/grid_capacity_summary.py` | `grid_capacity*` |
| *Extension:* how much cheap-price import can feasibility safely expose? (needs `grid_capacity.csv`) | `python experiments/partial_grid_feasibility_experiment.py --stage g04 --workers 8 --out controller_runs/partial_grid_feasibility_g04.csv`, then `... --stage sweep --workers 8 --out controller_runs/partial_grid_feasibility.csv --g04 controller_runs/partial_grid_feasibility_g04.csv`, then `python experiments/partial_grid_feasibility_summary.py controller_runs/partial_grid_feasibility.csv --computing-only controller_runs/grid_capacity.csv` | `partial_grid_feasibility*` |
| Which reactor information does a scheduler need? (read-only analysis) | `python experiments/info_ablation_summary.py` | `data/analysis/info_ablation*` |
| Is the simulated shutdown dead time pure xenon physics? | `cd data && python -c "import sys; sys.path.insert(0, '../src'); import validate_deadtime as V; V.check_logged_trips('controller_runs/main_suite_trip_log.csv')"` | console report |

### Reported numbers and figures

After the runs above, these read-only steps regenerate the reported
numbers and figures. Run them from the repository root.

```bash
python experiments/info_ablation_summary.py      # data/analysis/info_ablation_*
python experiments/reported_evidence.py          # data/analysis/reported_{numbers,claims}.csv, reported_evidence.md
python experiments/minute_traces.py --workers 8  # data/controller_runs/minute_traces/ (24 re-simulations)
python experiments/model_mismatch_analysis.py    # data/analysis/model_mismatch_*
python figures/plot_multicell.py                 # then any figure script below
```

Figures are written to `data/figures/` as PNG and SVG:

| Figure | Script | Needs |
|---|---|---|
| Mean unserved work per policy, cells A–H, ample vs low margin; per-policy means and decision-time overhead (CSV) | `figures/plot_multicell.py` | main suite |
| Fixed-delay baselines vs state-dependent feasibility (table figure) | `figures/plot_design_table.py` | main suite |
| Xenon and reactor power for one representative cell; A–H mean unserved-power timeline | `figures/plot_behavior_and_loss.py` | minute traces, reported numbers |
| Summary table and water accounting | `figures/plot_summary_table_and_water.py` | main suite, reported numbers |
| Summary findings / summary strip | `figures/plot_summary_findings.py`, `figures/plot_summary_strip.py` | main suite, reported numbers |
| Shutdowns and unserved work vs look-ahead | `figures/plot_lookahead.py`, `figures/plot_lookahead_strip.py` | look-ahead sweep, main suite |
| Shutdowns vs look-ahead and buffer, with model mismatch | `figures/plot_robustness_sensitivity.py`, `figures/plot_validation.py` | look-ahead and buffer sweeps, model mismatch, main suite |
| Unserved work and shutdowns vs grid import capacity (Computing-Only) | `figures/plot_grid_capacity.py` | grid-capacity sweep, main suite |
| Requested vs feasibility-limited cheap-price import (extension) | `figures/plot_safe_grid_flexibility.py` | partial-grid stages and summary |

Every figure script recomputes its plotted values from the per-cell rows
before drawing, and checks them in one of three ways:
- **Run manifests:** the experiment-figure scripts verify the manifests'
  output hashes and their anchors against the main suite.
- **Reported numbers:** the summary scripts cross-check against
  `reported_numbers.csv`.
- **Reported means:** `plot_multicell.py` and `plot_design_table.py` assert
  the reported headline means, so they fail deliberately if the inputs are
  not those of the reported evaluation.

`plot_multicell.py` and `plot_design_table.py` rewrite their own outputs.
Every other script refuses to replace an existing figure.

The output columns, reason codes and modelling assumptions are documented
in [`docs/controller_output_schema.md`](docs/controller_output_schema.md).
`results/README.md` describes the reported results and why they are not
included.

## Repository organization

| Path | Contents |
|---|---|
| `src/` | the simulator and controllers: `sim.py` (reactor model, reactor policies, plant for the controllers, experiment CLI), `consolidated_scheduler.py` (workload-aware controllers; no reactor physics), `xenon_guard.py` (standalone feasibility guard), `validate_deadtime.py` (independent dead-time check) |
| `experiments/` | one-factor experiments, the minute-trace re-run and read-only analyses (including `reported_evidence.py`, which produces every reported number), plus the row-level reference regression |
| `figures/` | plotting scripts that turn completed runs into the reported figures (no simulation) |
| `scripts/` | Borg preprocessing (`prepare_borg_cell.py`, `build_year.py`) and the external-input checker |
| `examples/` | a synthetic example that needs no external data |
| `tests/` | the pytest suite; tests that need external inputs skip until they are in `data/` |
| `data/` | working directory for external inputs and run outputs (only its README is tracked) |
| `results/` | notes on the reported results and how to regenerate them |
| `docs/` | output schema, controller definition and assumptions |

## Limitations

- **This is a simulation model, not reactor-control software.** It has not
  been validated against a plant and must not be used to operate one.
- **The reactor model is parameterized, not plant-specific.**
  - The xenon model is a two-equation I-135/Xe-135 balance with fixed
    constants. It has no spatial effects, and reactor power follows its
    setpoint within the ramp limit (no kinetics model). It is not a plant-specific safety analysis.
  - R_cap is applied as fixed snapshots rather than a simulated fuel-cycle
    trajectory (see `docs/controller_output_schema.md`).
- **Borg cells A–H are eight cells of one provider over about the same
  month.** They do not establish generalization to other HPC or AI
  workloads. Deadline shocks are synthetic, because the trace has no
  job-level deadlines.
- **Projections assume persistence.** Current grid use and a causal demand
  forecast persist over the horizon, so a single projection is not a
  guarantee. Safety in the evaluation comes from re-checking every minute.
  This is reported per run as crossings, shutdowns and false-safe
  decisions.
- **External PJM inputs are required** for the grid, price and water
  results, and are not distributed. The query that produced the Borg
  per-window aggregates is also not included; `data/README.md` documents
  the schema it must produce.
- **The cheap-price import rule is a fixed heuristic, not an economic
  optimization.**
- **Limit-Predictive skips its projection by a speed heuristic.** It
  skips the projection when the ceiling is more than 1,200 pcm above the
  current xenon worth (`XE_OVERSHOOT_BOUND` in `sim.run`). That is not a
  physical bound: a dip from full power can overshoot by about 3,900 pcm.
  The skip is harmless for the evaluated snapshots, since the check always
  runs at 2700 and 3500 pcm, and at 8000 pcm the ceiling exceeds any
  reachable peak. For R_cap between roughly 3,700 and 6,400 pcm, however,
  it could admit unsafe reductions. Consolidated always projects.
- **The flux constant `phi_full` (5×10¹³ n/cm²/s) has no documented
  source.** `python src/sim.py --physics-audit` flags it as such. Xenon
  reactivity is independent of `Sigma_f`, and the audit checks this.
- **`consolidated_partial_grid` has narrower validation than Consolidated.**
  It was evaluated only on cells A–H at R_cap = 2700 pcm without stress
  events, with the controller's xenon model equal to the plant's. It
  operates at the ε buffer far more often than Consolidated, so stress
  scenarios and model mismatch, which were not evaluated for it, matter
  more for it.

## Citation

Citation metadata for this software is in [`CITATION.cff`](CITATION.cff).
A publication reference will be added when it is available.

## Data licenses and attribution

The license of this software is separate from the terms of the external
datasets.

- **Google Borg cluster traces (2019):** published by Google in
  <https://github.com/google/cluster-data>. That repository states the
  license and the paper to cite when using the traces. This project
  redistributes no trace data.
- **PJM Data Miner:** PJM market and generation data are subject to PJM's
  terms of use. This repository contains no PJM data and does not download
  it. Users obtain the data themselves and are responsible for complying
  with those terms.
- **Reactor-margin background:** the treatment of R_cap as a fuel-age
  dependent margin follows the discussion in Choudhury et al.,
  arXiv:2507.18150 (see `docs/controller_output_schema.md`).

## License

The software is released under the [MIT License](LICENSE). The external
datasets are not covered by it; see *Data licenses and attribution* above.
