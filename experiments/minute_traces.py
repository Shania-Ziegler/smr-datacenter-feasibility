#!/usr/bin/env python3
"""
minute_traces.py -- re-run 24 configurations of the main suite only to
save minute-level trajectories (the main suite keeps summary rows only).

  cells A-H x {free, reactive, predictive}, scenario none, R_cap 2700,
  G 0.4, eps 20 (predictive), horizon 10 h, tiered, trips on, emergency
  grid, January-2025 PJM inputs, timing detailed -- i.e. the identical
  task dicts from sim._controller_tasks, executed through the canonical
  sim._controller_task. sim.run is wrapped ONLY to request
  return_series='full' (record-only arrays; no physics change).

Acceptance: every regenerated summary row must equal its canonical row in
controller_runs/main_suite.csv in every non-timing
column (exact, after the same CSV round trip). Traces are written ONLY if
24/24 reproduce; otherwise the differing fields are printed and nothing is
written.

Outputs (refuses to overwrite):
  controller_runs/minute_traces/trace_<cell>_<policy>.npz
  controller_runs/minute_traces/reproduction.csv
  controller_runs/minute_traces_manifest.json
Run from the repository root:  python3 experiments/minute_traces.py --workers 8
"""
import argparse
import dataclasses
import datetime
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
HERE = os.path.join(ROOT, "data")      # working directory: inputs, outputs
sys.path.insert(0, SRC_DIR)
import consolidated_scheduler as CS  # noqa: E402
import sim as S  # noqa: E402

CANON = "controller_runs/main_suite"
OUTDIR = "controller_runs/minute_traces"
MANIFEST = OUTDIR + "_manifest.json"
PRICE, GENMIX = "pjm_dom_34885183.csv", "load_pjm_genmix.csv"
ARMS = ("free", "reactive", "predictive")
CELLS = "abcdefgh"
TIMING = ("policy_total_us", "policy_mean_us", "policy_p95_us",
          "policy_p99_us", "policy_max_us", "projection_total_us",
          "projection_mean_us", "projection_p95_us", "projection_p99_us",
          "projection_max_us")
REQUIRED = ("true_unserved", "unmet", "n_trips", "deadtime_h", "crossings",
            "crossings_op", "grid_Wh", "grid_cost", "gen_Wh", "served_it_Wh",
            "demand_Wh", "water_L", "water_grid_L", "water_dc_L",
            "water_sys_L", "min_headroom", "peak_rho", "refused_steps",
            "residual")
FIELDS = ("time_s", "P_rx", "Pd", "rho", "iodine", "xenon", "R_cap",
          "ceiling", "headroom", "grid", "tripped", "unmet_W", "dropped_Wh",
          "shed_W", "drained_W", "served_it_W", "surplus_W", "unmet_xen_W",
          "unmet_trip_W", "unmet_ramp_W")


def traced_task(t):
    cap = {}
    real = S.run

    def run_full(*a, **k):
        k["return_series"] = "full"
        r = real(*a, **k)
        cap["series"] = {f: np.array(r["series"][f]) for f in FIELDS}
        return r
    S.run = run_full
    try:
        _i, row, _tr, _ep, _mn = S._controller_task(t)
    finally:
        S.run = real
    C = S._CTX["cells"][t["cell"]]
    tg = C["t_grid"]
    ser = cap["series"]
    ser["demand_W"] = np.array([S.P_FIXED + S.PUE * S.work_power(C["u_at"](
        tt)) for tt in tg])
    ser["P_rated_W"] = np.array(C["P_rated"])
    step_Wh = ser["unmet_W"] * 60.0 / 3600.0 + ser["dropped_Wh"]
    ser["cumulative_unserved_Wh"] = np.cumsum(step_Wh)
    return t, row, ser


def same(a, b):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    os.chdir(HERE)
    if os.path.exists(OUTDIR) or os.path.exists(MANIFEST):
        sys.exit(f"{OUTDIR} or its manifest exists; refusing to overwrite.")
    man0 = json.load(open(CANON + "_manifest.json"))
    scfg = CS.StressConfig()
    assert dataclasses.asdict(scfg) == man0["stress_config"]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    code_files = {"sim.py": os.path.join(SRC_DIR, "sim.py"),
                  "consolidated_scheduler.py":
                  os.path.join(SRC_DIR, "consolidated_scheduler.py"),
                  os.path.basename(__file__): os.path.abspath(__file__)}
    code0 = {f: S._sha256(p) for f, p in code_files.items()}
    inputs = {p: S._sha256(p) for p in
              [f"demand_curve_cell_{c}.csv" for c in CELLS] + [PRICE, GENMIX]}
    for p, h in inputs.items():
        assert man0["input_sha256"][p] == h, p

    scenarios = {s.name: s for s in CS.build_scenarios(scfg,
                                                       ["none", "spike"])}
    ctx = {c: S._load_cell_ctx(c, f"demand_curve_cell_{c}.csv", PRICE,
                               GENMIX, 60.0) for c in CELLS}
    S._CTX.update(cells=ctx, scenarios=scenarios, stress_cfg=scfg,
                  timing=man0["timing_mode"], decision_log="episodes")
    tasks = [t for t in S._controller_tasks(
        list(CELLS), ["none"], ["computing_only", "consolidated"], scenarios,
        0.4, "ST", True)
        if t["suite"] == "primary" and t["R_cap"] == 2700.0
        and t["arm"] in ARMS]
    assert len(tasks) == 24
    print(f"[trace] 24 canonical configurations on {a.workers} workers")
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(traced_task, tasks, chunksize=1)
    print(f"[trace] simulated in {time.time() - t0:.0f} s")

    buf = io.StringIO()
    pd.DataFrame([r for _, r, _ in res]).to_csv(buf, index=False)
    buf.seek(0)
    new = pd.read_csv(buf)
    canon = pd.read_csv(CANON + ".csv")
    rep, n_ok = [], 0
    for (t, _, ser), (_, nr) in zip(res, new.iterrows()):
        m = canon[(canon.experiment == nr.experiment) & (canon.rx == nr.rx)
                  & (canon.suite == "primary")]
        assert len(m) == 1, nr.experiment
        cr = m.iloc[0]
        cols = [c for c in canon.columns if c not in TIMING]
        diff = [c for c in cols if (c in new.columns and not same(cr[c], nr[c]))
                or (c not in new.columns and not pd.isna(cr[c]))]
        req_ok = all(same(cr[c], nr[c]) for c in REQUIRED)
        cum_end = float(ser["cumulative_unserved_Wh"][-1]) + float(nr.leftover)
        endpoint_ok = bool(np.isclose(cum_end, float(cr.true_unserved),
                                      rtol=1e-9, atol=1e-6))
        ok = not diff and req_ok and endpoint_ok
        n_ok += ok
        rep.append(dict(cell=t["cell"], policy=t["arm"],
                        canonical_experiment=cr.experiment,
                        canonical_row_index=int(m.index[0]),
                        compared_columns=len(cols), identical=ok,
                        differing=";".join(diff),
                        trace_cumulative_plus_leftover_Wh=cum_end,
                        canonical_true_unserved_Wh=float(cr.true_unserved),
                        residual=float(nr.residual)))
        print(f"  {'OK  ' if ok else 'FAIL'} {t['cell']} {t['arm']:>10} | "
              f"{len(cols)} cols | true_unserved {nr.true_unserved:,.3f} Wh "
              f"| trace endpoint {cum_end:,.3f} Wh"
              + (f" | DIFF {diff[:8]}" if diff else ""))
    print(f"\n[trace] {n_ok}/24 canonical rows reproduced exactly")
    if n_ok != 24:
        sys.exit("[trace] STOP: reproduction failed; no traces written.")

    os.makedirs(OUTDIR)
    paths = {}
    for t, _, ser in res:
        p = f"{OUTDIR}/trace_{t['cell']}_{t['arm']}.npz"
        np.savez_compressed(p, **ser)
        paths[f"{t['cell']}_{t['arm']}"] = dict(path=p, sha256=S._sha256(p))
    pd.DataFrame(rep).to_csv(f"{OUTDIR}/reproduction.csv", index=False,
                             mode="x")
    code1 = {f: S._sha256(p) for f, p in code_files.items()}
    man = dict(
        purpose="minute-level trajectories for the behavior and loss-"
                "timeline figures; "
                "re-run of 24 existing canonical configurations, NOT a new "
                "experiment",
        started_utc=started,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=sys.argv, python=sys.version.split()[0], numpy=np.__version__,
        pandas=pd.__version__, code_sha256_at_start=code0,
        code_sha256_at_end=code1, code_unchanged_during_run=code0 == code1,
        canonical_run=CANON, canonical_code_sha256=man0["code_sha256_at_start"],
        input_sha256=inputs,
        configuration=dict(cells=list(CELLS), policies=list(ARMS),
                           scenario="none", R_cap_pcm=2700.0, G=0.4,
                           eps_buffer_pcm=dict(free=0.0, reactive=0.0,
                                               predictive=20.0),
                           predict_horizon_h=10.0, sigma_m_pcm=S.sigma_m,
                           ramp_pct_per_min=S.ramp_lim, sched="tiered",
                           trip=True, emergency_grid=True, dt_s=60.0,
                           settle_h=6.0, timing_mode=man0["timing_mode"]),
        reproduction=dict(rows_reproduced=n_ok, of=24,
                          criterion="all non-timing columns identical after "
                                    "CSV round trip; trace cumulative + "
                                    "leftover == true_unserved (rtol 1e-9)",
                          max_abs_residual_Wh=float(max(abs(r["residual"])
                                                        for r in rep))),
        trace_fields={f: "see sim.run return_series='full'" for f in FIELDS},
        derived_fields=dict(
            demand_W="P_FIXED + PUE * work_power(u(t)) (facility request)",
            P_rated_W="reactor rating for the cell (W)",
            cumulative_unserved_Wh="cumsum(unmet_W*dt/3600 + dropped_Wh); "
                                   "add leftover (canonical row) at month "
                                   "end to obtain true_unserved"),
        units=dict(P_rx="fraction of P_rated", Pd="fraction of P_rated "
                   "(reactor-side net demand)", rho="pcm", iodine="atoms/cm3",
                   xenon="atoms/cm3", ceiling="pcm (R_cap - sigma_m)",
                   headroom="pcm", grid="W", tripped="bool",
                   unmet_W="W (zero in first 6 h settle window)",
                   dropped_Wh="Wh per step"),
        traces=paths,
        canonical_rows={f"{r['cell']}_{r['policy']}":
                        dict(experiment=r["canonical_experiment"],
                             row_index=r["canonical_row_index"])
                        for r in rep},
        conservation="all |residual| <= 1 Wh" if all(
            abs(r["residual"]) <= 1.0 for r in rep) else "FAILED")
    man.update(S._git_state(HERE))
    S._write_manifest(MANIFEST, man)
    print(f"[trace] wrote 24 traces to {OUTDIR}/ and {MANIFEST}")


if __name__ == "__main__":
    main()
