# Workload-aware controller: outputs, schema and assumptions

> **Deadline shocks are synthetic stress events; the Borg trace does not
> provide job-level deadline arrivals.** Every table or figure that contains
> deadline-shock rows should carry this sentence.

Code: `src/consolidated_scheduler.py` holds the controller logic only, and
`sim.run_controller` in `src/sim.py` is the plant. The reactor-policy path
in `sim.run` (called "legacy" in code comments) is unchanged by the
controller extension. The workload-aware arms are reached only through the
dispatch at the top of `run()` (`controller_mode=...`) or through the
opt-in CLI flags below.

## Commands

All commands are run from the `data/` directory, which holds the external
inputs (see `data/README.md`); outputs go to `data/controller_runs/`.

```bash
cd data
mkdir -p controller_runs

# main suite: 8 Borg cells x R_cap {8000, 3500, 2700} x 7 scenarios x
# 8 arms, plus the spike-threshold sensitivity grid
python ../src/sim.py --computing-only --consolidated --stress-suite \
    --decision-log --decision-log-level episodes --timing-mode detailed \
    --workers 8 --price-file pjm_dom_34885183.csv \
    --genmix-file load_pjm_genmix.csv --out controller_runs/main_suite.csv

# natural (no-event) controller comparison on one trace
python ../src/sim.py --computing-only --consolidated \
    --borg demand_curve_cell_a.csv --out controller_runs/cell_a.csv

# explicit stress parameters (all optional; defaults shown)
python ../src/sim.py --computing-only --consolidated --stress-suite \
    --cells abcdefgh --scenarios spike,surprise_blackout --stress-G 0.4 \
    --stress-start-h 254 --spike-mult 1.25 --spike-duration-h 2 \
    --deadline-shock-fraction 0.5 --deadline-shock-minutes 60 \
    --blackout-duration-h 6 --compound-offset-h 2 --out controller_runs/x.csv
```

Controller flags fail hard when combined with `--year`, `--r-cap-file`,
`--frontier`, `--hysteresis`, `--quick`, `--self-test` or
`--physics-audit`. These are also rejected:
- stress parameters without `--stress-suite`;
- `--decision-log*` without a controller arm;
- `--borg` together with `--stress-suite`.

Controller runs refuse to overwrite an existing file unless `--overwrite`
is given. They always refuse the protected output names listed in
`sim.PROTECTED_OUTPUTS` and `sim.CANONICAL_RUN_PREFIXES`.

## Files (`<stem>` = `--out` without `.csv`)

| file | content |
|---|---|
| `<stem>.csv` | one row per run (primary and sensitivity suites) |
| `<stem>_trip_log.csv` | every SCRAM with the exact (I0, X0) at the trip; columns include `cell, scenario, suite`; checkable with `validate_deadtime.check_logged_trips` |
| `<stem>_comparisons.csv` | per (cell, R_cap, scenario): computing_only − free, consolidated − computing_only, consolidated − predictive |
| `<stem>_spike_sensitivity.csv` | per cell × (k_sigma, floor) × R_cap × arm, on natural and injected-spike traces |
| `<stem>_detection.csv` | detector-only passes: detections and the binding term (natural traces); detection delay, misses and attributable false positives (injected spikes) |
| `<stem>_cell_characteristics.csv` | per-cell utilisation, burstiness and flexibility |
| `<stem>_manifest.json` | write-once, read-only run manifest. Records: code sha256 (at start and end), input sha256, argv, R_cap snapshots, G, controller horizon and configuration, eps buffer per arm, stress-event metadata, cooling convention, evidence classes and output sha256 |
| `<stem>_decision_episodes.csv` | with `--decision-log`: consecutive identical reason sets merged into episodes |
| `<stem>_decisions_minute.csv` | with `--decision-log-level minute`: every modified minute (large) |

## Result-row columns

**Evidence class (every row).** `evidence_class` takes one of these values:
- `primary_cross_arm`: the only rows used in cross-arm comparisons.
- `controller_diagnostic_endogenous_deadline_shock`: deadline-shock
  scenarios. The shock promotes a fraction of each policy's *own* queue, so
  its size differs between policies.
- `sensitivity`, or the label of a one-factor sweep.

**Identity.**
- `experiment` is unique together with `sched` and `rx`. Rows with
  synthetic deadlines end in `[synthetic_deadline]`.
- `sched` is always `tiered`; `rx` is the arm.
- `suite`, `cell`, `R_cap`, `G`, `eps`, `hyst_h`.
- `controller_mode`: `computing_only`, `consolidated`,
  `consolidated_partial_grid` or `legacy`.
- `controller_k_sigma`, `controller_rated_floor_fraction`.

**Stress metadata (every row).**
- `scenario`, and `stress_seed` (always `none (deterministic)`).
- Event: `event_start_h`, `event_duration_h`, `event_magnitude`.
- Spike: `spike_multiplier`, `spike_start_h`, `spike_duration_h`.
- Blackout: `blackout_type` (`none` | `surprise` | `scheduled`),
  `blackout_start_h`, `blackout_duration_h`, `blackout_known_in_advance`.
- `cycle_fraction`: always NaN, because no cited R_cap(t) exists.
- State at the event: `R_cap_at_event_pcm`, `pre_event_headroom_pcm`,
  `pre_event_reactor_power`, `grid_Wh_during_blackout`.

**Synthetic-deadline provenance (every row).**
- `synthetic_deadline`.
- `deadline_source`: `existing_queue_horizon` or
  `synthetic_deadline_shock`.
- `deadline_shock_fraction`, `deadline_shock_minutes`,
  `deadline_shock_start_h`.
- `deadline_shock_applied`: False for legacy arms, whose `[amount, age]`
  queues have no deadline field.
- `deadline_shock_vacuous`: True when no queued batch work existed to
  promote.

**Reactor-policy metrics.** These use the same definitions as `run()`, and
the first 6 h settling period is excluded wherever `run()` excludes it.
- Unserved work: `true_unserved`, `unmet`, `dropped`, `dropped_b`,
  `dropped_m`, `leftover`, `shed`, `drained`, `delay_b`, `delay_m`.
- Ceiling crossings and headroom: `crossings`, `crossings_op`,
  `crossings_outage`, `cross_episodes`, `cross_episodes_op`,
  `longest_viol_min`, `max_exceed`, `min_headroom`, `peak_rho`.
- Shutdowns: `n_trips`, `deadtime_h`.
- Energy, cost and water: `grid_Wh`, `grid_cost`, `surplus_Wh`, `gen_Wh`,
  `served_it_Wh`, `water_*`, `residual`.
- Reactor dispatch: `refused_steps`, `floor_raise_Wh`, `p_lo`, `p_hi`,
  `p_mean`, `ramp_distance`, `unmet_cap/force/trip/ramp/xen`.

For controller arms, `unmet` is exactly
`rigid_unserved_Wh + facility_unserved_Wh`, and the `unmet_*` columns
attribute it to whichever limit set the reachable maximum. `refused_steps`
counts post-settle steps whose decision reasons include
`projected_ceiling`.

**Controller metrics (controller arms; NaN for legacy arms).**
- States: `minutes_NORMAL/STRESSED/ISLANDED`, `state_transitions`.
- Spikes: `spike_detections`, `spike_episodes`, `spike_bind_sigma`,
  `spike_bind_rated_floor`, `spike_frac_sigma`, `spike_frac_rated_floor`,
  and `spike_energy_Wh` (sum of demand above the previous EWMA mean at
  detected minutes).
- Deadlines: `deadline_promoted_Wh`, `deadline_miss_b_Wh`,
  `deadline_miss_m_Wh`, `deadline_miss_steps`, `dropped_natural_Wh`,
  `dropped_synthetic_Wh`.
- Queue: `deferred_b_Wh`, `deferred_m_Wh`, `max_queue_age_min`,
  `mean_queue_age_min`, `min_queue_slack_min`,
  `mean_min_queue_slack_min`.
- Rigid shortfall: `rigid_unserved_Wh`, `facility_unserved_Wh`,
  `rigid_shortfall_steps`, and `rigid_shortfall_reason`
  (`insufficient_feasible_supply` | `none`).
- Grid:
  - `grid_price_Wh`, `grid_bridge_Wh`.
  - `grid_emergency_Wh`: also filled for legacy arms, as grid use while
    tripped.
  - `grid_unavailable_min` (all arms).
- Decisions: `projections`, `no_feasible_action_steps`,
  `price_import_canceled_steps`, `reason_<code>_steps`,
  `decision_episodes_n`.

**Price-import diagnostics.** These appear only when a run requests them
(`price_diagnostics=True`, as set by
`experiments/partial_grid_feasibility_experiment.py`):
- Energy: `price_requested_Wh` (after the G cap), `price_approved_Wh`
  (after reactor feasibility) and the delivered `grid_price_Wh` (after
  surplus displacement).
- Steps: `price_opportunity_steps`, `price_full_accept_steps`,
  `price_partial_steps`, `price_full_cancel_steps`.
- Import size and slack: `price_approved_mean_frac`,
  `price_delivered_mean_frac` and `price_opp_mean_proj_slack_pcm`.
- Monotonicity audit: `partial_audit_missed_steps` and
  `partial_audit_max_gap_frac`, for `consolidated_partial_grid` with
  `partial_grid_audit > 0`.

**Energy conservation.**
- `energy_queue_residual_Wh`: deferred − drained − dropped − leftover.
- `energy_work_residual_Wh`: offered work − served − deferred − rigid
  shortfall.
- `energy_supply_residual_Wh`: reactor + grid − served demand − surplus.
- `energy_residual_Wh` = max |·| of the three. Suites flag any row above
  1 Wh.

**Forecast validation (computed only after the run).**
- Forecast error: `fc_mae_1h_W`, `fc_bias_1h_W`, `fc_mae_horizon_W`.
- Projection vs realized: `proj_vs_realized_mean_pcm`,
  `proj_vs_realized_min_pcm`, `proj_underpredict_frac`.
- `false_safe_decisions`: the accepted path was projected under the
  ceiling, but the realized peak over the horizon went above
  `R_cap − sigma_m`.
- `false_safe_eps_decisions`: the same, against the eps-buffered ceiling.
- `unnecessarily_conservative_decisions`, out of
  `modified_decisions_evaluated`: the rejected candidate, rebuilt from the
  realized requested demand with the same grid persistence, would have
  projected under the ceiling. This is a post-hoc counterfactual with no
  re-simulated queue response.
- `deadline_pred_mae_Wh`, `deadline_pred_bias_Wh`: hourly samples of
  predicted misses among queued items due within the horizon, against
  realized drops over the next horizon.

**Timing** (`--timing-mode coarse|detailed`; wall-clock, not scientific):
`policy_*_us`, `projection_*_us`.

## Decision / refusal log

- **Reason codes:** `projected_ceiling`, `ramp_limit`, `grid_unavailable`,
  `grid_capacity`, `deadline_pressure`, `demand_spike`,
  `rigid_supply_shortfall`. A minute is logged when any code applies.
- **Episode rows** carry `reason_code, start_h, end_h, steps`, the first
  minute's fields (prefixed `first_`), `max_projected_peak_pcm` and
  `max_requested_grid_W`.
- **Minute fields:** `t_h, policy, state, pressure_tier,
  req_p0/min/mean/max, acc_p0/min/mean/max, reactor_power, accepted_power,
  projected_peak_pcm, accepted_peak_pcm, applicable_ceiling_pcm,
  headroom_pcm, ramp_limit, requested_grid_W, available_grid_W, grid_W,
  min_queue_slack_min, no_feasible_action, reason_code`, plus scenario and
  synthetic-deadline metadata.
- **Plant diagnostics:** `applicable_ceiling_pcm` and `headroom_pcm` are
  written after the decision; `computing_only` never receives them.

## Controller definition (summary)

- **Workload.** `P_batch = fb·P_work`, `P_mid = fm·P_work`,
  `P_rigid = (1 − fb − fm)·P_work`. The fractions are validated at every
  input sample and every step. `P_FIXED` and cooling are inflexible
  facility demand. Borg has no per-job rigid class, so rigid compute is the
  remainder.
- **Deadlines.** Batch work has 24 h and mid-tier work `mid_horizon_s` (the
  existing queue horizons), with
  `remaining_slack = deadline_steps − age_steps`. Work is served least slack
  first; ties go to mid-tier before batch, then earlier arrival, then
  creation order.
- **Spike detector.** EWMA mean and variance of requested power
  (`P_FIXED + PUE·P_work`), with τ = 60 min, `alpha = 1 − exp(−dt/τ)` and a
  60 min warm-up.
  - A spike is `x − mean_prev > max(k_sigma·std_prev, floor·P_rated)`.
  - Primary values are k_sigma = 3 and floor = 0.05; a tie counts as
    `sigma`.
- **States**, in priority order:
  - ISLANDED (grid availability = 0) uses no grid.
  - STRESSED (a spike within the last 60 min, or queued slack ≤ 15 min)
    cancels price imports and bridges the reactor's shortfall with
    available grid.
  - NORMAL allows price imports.
- **Causal forecast.** The EWMA of observed requested power, plus queued
  work due within the horizon, spread evenly up to its deadline.
  - Current grid use is assumed to persist, except inside explicitly
    scheduled outages.
  - A surprise blackout is visible only through the current availability
    sample.
- **Consolidated feasibility.** The requested 10 h path, in 300 s segments,
  is projected with `sim.project_peak_rho_path`. If the projected peak
  exceeds `R_cap − sigma_m − eps_buffer`:
  1. The controller first cancels the price-driven import.
  2. It then bisects a power floor (12 iterations).
  3. If even the maximum reachable power is projected unsafe, the minute is
     recorded as `no_feasible_action`. The lower-projected-peak setpoint is
     applied as a best-effort UNSAFE action (`p_floor` = NaN). It is never
     labelled an accepted safe floor, and its outcome (crossing, trip,
     rigid shortfall) is whatever the plant records.
- **consolidated_partial_grid (opt-in, experimental).** This arm differs
  from Consolidated in one way. Where the requested cheap-price import is
  projected unsafe but a zero import is projected safe, it bisects the
  import size (12 iterations) for the largest price import whose own
  projected path stays under the same ceiling, instead of cancelling the
  import.
  - An import is accepted only if its own projection has been checked.
  - The import displaces reactor output one-for-one.
  - It is not part of `CONTROLLER_MODES`, and no default run uses it.
- **Surplus.** Surplus from a raised floor displaces grid import. It never
  fabricates demand: whenever surplus exists, all offered work has already
  been served.

## Assumptions and limitations

1. **No R_cap(t) trajectory.**
   - Choudhury et al. (arXiv:2507.18150v2) give
     `Δρ_marg,n = (k_eff,n − 1)/k_eff,n × 10⁵` pcm (Eq. 7) and
     `k_eff,n+1 = k_eff,n − m·α_n`, with m = 3.8×10⁻⁴ day⁻¹ at full power
     (Eq. 8).
   - They do not give a numeric `k_eff,BOL` for this reactor, and α_n
     depends on the simulated dispatch, so it forms a closed loop.
   - The paper also does not say whether Δρ_marg is compared with the
     absolute xenon worth (the `sim.py` convention, rho_eq = 2365 pcm) or
     with the defect relative to equilibrium.
   - Runs therefore use fixed snapshots, and `cycle_fraction` is NaN.
2. **Fixed R_cap snapshots.** The values are controlled BOC-, intermediate-
   and EOC-like snapshots, not points on a simulated fuel cycle.
3. **Stress events** are deterministic and preregistered:
   - They start at 254 h.
   - Spike: ×1.25 on IT work (utilisation capped at 1.0) for 2 h.
   - Deadline shock: 50 % of queued batch work moved to a 60 min deadline.
     If no batch work is queued at 254 h it promotes nothing, which
     `deadline_shock_vacuous` records. The event is not moved to where it
     would bite.
   - Blackout: 6 h. Compound events start 2 h into the blackout.
4. **Legacy arms and deadline shocks.** Legacy arms cannot represent
   deadline shocks, because their queues are `[amount, age]`. Their
   deadline-shock rows equal their no-event rows and are marked
   `deadline_shock_applied = False`.
5. **Legacy arms and rigid shortfall.** Legacy arms do not separate rigid
   shortfall from other unmet demand, so their `rigid_unserved_Wh` is NaN.
6. **Fair comparison.** Every primary cross-arm row uses the same cell,
   trace interval, R_cap snapshot, initial reactor state, price and grid
   inputs, event definition and facility/cooling convention.
   - In every arm, cooling follows non-deferred IT work
     (`P_work − deferred + drained`), as in `run()`. Cooling owed to
     unserved rigid work is booked as unserved facility demand, and a test
     checks this.
   - Emergency grid while tripped does not require a price series in the
     controller arms. A price series is always present in the suite, so the
     two conditions coincide.
7. **Generation-mix input.** If it is absent, grid-embedded water is
   reported as n/a. If the price file is shorter than the simulated window,
   `load_pjm` tiles it and discloses this.
8. **No spike ground truth in natural traces.** Natural detections are
   counts, never true or false positives.
9. **The Borg cells are not eight workload types.** They differ
   substantially in flexibility and moderately in utilisation and
   burstiness; see `<stem>_cell_characteristics.csv`.
10. **Claim boundary.** The projection is a runtime decision aid,
    re-evaluated causally every minute and scored afterwards against
    realized trajectories. A single 10 h projection is not a guarantee.
    - Realized dispatch differs from the persistence forecast, so a
      decision projected under the eps-buffered ceiling can be followed by
      a realized peak above it. `false_safe_eps_decisions` counts these.
    - Physical safety against `R_cap − sigma_m` comes from re-evaluating
      every minute. It is reported per run as `crossings`, `n_trips` and
      `false_safe_decisions`.
