#!/usr/bin/env python3
"""
model_mismatch_experiment.py -- focused plant/controller xenon-model
mismatch experiment (Option A: the controller observes the TRUE current I/X
and projects with its own parameters).

  7 plant parameterizations x 2 arms (predictive, consolidated)
  x 8 Borg cells (A-H) x 2 model conditions = 224 runs
  R_cap = 2700 pcm, nominal scenario, G = 0.4, H = 10 h, eps = 20 pcm,
  every other setting canonical (same _controller_task as the suite).

  mismatched: plant = theta_true,  controller projection = NOMINAL
  matched   : plant = theta_true,  controller projection = theta_true

Parameterizations (one factor at a time; only kinetic parameters):
  DOE_halflives   SOURCE-SUPPORTED ALTERNATIVE: DOE-HDBK-1019/2-93
                  half-lives I-135 6.57 h, Xe-135 9.10 h (canonical Table V
                  values correspond to 6.71 h / 9.21 h)
  lambda_I x0.95/x1.05, lambda_X x0.95/x1.05, phi x0.9/x1.1
                  EXPLICIT SENSITIVITY RANGES -- not calibrated measurement
                  uncertainties (no project source gives such bounds)

Run from the repository root (the script works in data/; it refuses
to overwrite anything):
  python3 experiments/model_mismatch_experiment.py --workers 11 \
      --out controller_runs/model_mismatch.csv
"""
import argparse
import dataclasses
import datetime
import json
import math
import multiprocessing as mp
import os
import platform
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

CANON_MANIFEST = "controller_runs/main_suite_manifest.json"
PRICE, GENMIX = "pjm_dom_34885183.csv", "load_pjm_genmix.csv"
R_CAP = 2700.0
ARMS = ("predictive", "consolidated")
N = S.NOMINAL_XE
LN2 = math.log(2.0)
SRC = "source-supported alternative (DOE-HDBK-1019/2-93 half-lives)"
SENS = "explicit sensitivity range (not a calibrated uncertainty)"
CASES = [
    ("DOE_halflives", dataclasses.replace(
        N, lambda_I=LN2 / (6.57 * 3600.0), lambda_X=LN2 / (9.10 * 3600.0)),
     SRC),
    ("lambda_I_x0.95", dataclasses.replace(N, lambda_I=N.lambda_I * 0.95),
     SENS),
    ("lambda_I_x1.05", dataclasses.replace(N, lambda_I=N.lambda_I * 1.05),
     SENS),
    ("lambda_X_x0.95", dataclasses.replace(N, lambda_X=N.lambda_X * 0.95),
     SENS),
    ("lambda_X_x1.05", dataclasses.replace(N, lambda_X=N.lambda_X * 1.05),
     SENS),
    ("phi_x0.9", dataclasses.replace(N, phi_full=N.phi_full * 0.9), SENS),
    ("phi_x1.1", dataclasses.replace(N, phi_full=N.phi_full * 1.1), SENS),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--timing-mode", default="detailed",
                    choices=("off", "coarse", "detailed"))
    a = ap.parse_args()
    os.chdir(HERE)
    stem = a.out[:-4] if a.out.endswith(".csv") else a.out
    outputs = dict(results=a.out, trips=f"{stem}_trip_log.csv",
                   manifest=f"{stem}_manifest.json")
    for p in outputs.values():
        if S._is_canonical_output(p) or os.path.exists(p):
            sys.exit(f"refusing to write {p} (exists or canonical)")

    man0 = json.load(open(CANON_MANIFEST))
    scfg = CS.StressConfig()
    assert dataclasses.asdict(scfg) == man0["stress_config"]
    code_files = {"sim.py": os.path.join(SRC_DIR, "sim.py"),
                  "consolidated_scheduler.py":
                  os.path.join(SRC_DIR, "consolidated_scheduler.py"),
                  os.path.basename(__file__): os.path.abspath(__file__)}
    code0 = {f: S._sha256(p) for f, p in code_files.items()}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cells = list("abcdefgh")
    scenarios = {s.name: s for s in CS.build_scenarios(scfg,
                                                       ["none", "spike"])}
    ctx, inputs = {}, {}
    for p in [f"demand_curve_cell_{c}.csv" for c in cells] + [PRICE, GENMIX]:
        inputs[p] = S._sha256(p)
        assert inputs[p] == man0["input_sha256"][p], p
    for c in cells:
        ctx[c] = S._load_cell_ctx(c, f"demand_curve_cell_{c}.csv", PRICE,
                                  GENMIX, 60.0)
    S._CTX.update(cells=ctx, scenarios=scenarios, stress_cfg=scfg,
                  timing=a.timing_mode, decision_log=None)

    base = [t for t in S._controller_tasks(cells, ["none"], ["consolidated"],
                                           scenarios, 0.4, "ST", False)
            if t["arm"] in ARMS and t["R_cap"] == R_CAP]
    assert len(base) == 16
    suite = S.SWEEP_SUITES["model"][0]
    tasks = []
    for name, theta, label in CASES:
        for cond in ("mismatched", "matched"):
            for t in base:
                tasks.append(dict(
                    t, idx=len(tasks), suite=suite, plant_xe=theta,
                    model_xe=theta if cond == "matched" else None,
                    mismatch_case=name, model_condition=cond,
                    range_label=label))
    assert len(tasks) == 224
    print(f"[mismatch] {len(tasks)} runs on {a.workers} workers; R_cap "
          f"{R_CAP:.0f}, nominal, G=0.4, H=10 h, eps=20 pcm; timing "
          f"{a.timing_mode}")
    for name, theta, label in CASES:
        d = {f.name: getattr(theta, f.name) for f in dataclasses.fields(N)
             if getattr(theta, f.name) != getattr(N, f.name)}
        print(f"  {name:>15}: {d}  [{label}]")

    t0 = time.time()
    rows, trips, bad = [None] * len(tasks), [], 0
    pool = mp.get_context("fork").Pool(a.workers)
    try:
        it = pool.imap_unordered(S._controller_task, tasks, chunksize=1)
        for done, (i, row, tr, _ep, _mn) in enumerate(it, 1):
            rows[i] = row
            trips.extend(tr)
            flag = ""
            if abs(row["residual"]) > 1.0 or \
                    abs(row.get("energy_residual_Wh", 0.0) or 0.0) > 1.0:
                flag = "  <-- CONSERVATION BROKEN"
                bad += 1
            print(f"  [{done:3d}/{len(tasks)}] {row['experiment']:>62} | "
                  f"{row['rx']:>12} | uns {row['true_unserved']/1e3:9.1f} kWh "
                  f"| trips {row['n_trips']:2d} | {time.time()-t0:6.0f} s"
                  f"{flag}", flush=True)
    finally:
        pool.close()
        pool.join()
    pd.DataFrame(rows).to_csv(outputs["results"], index=False, mode="x")
    pd.DataFrame(trips).to_csv(outputs["trips"], index=False, mode="x")

    code1 = {f: S._sha256(p) for f, p in code_files.items()}
    man = dict(
        manifest_version=1, experiment="plant/controller xenon-model "
        "mismatch (Option A: true current I/X observed)",
        started_utc=started,
        finished_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        argv=sys.argv, cwd=os.getcwd(), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__,
        code_sha256_at_start=code0, code_sha256_at_end=code1,
        code_unchanged_during_run=code0 == code1,
        input_sha256=inputs, runs=len(tasks), conservation_failures=bad,
        cells=cells, arms=list(ARMS), R_cap_pcm=R_CAP, scenario="none",
        G_max_frac=0.4, predict_horizon_h=S.CANONICAL_HORIZON_H,
        eps_buffer_pcm=S.CANONICAL_EPS_PCM, timing_mode=a.timing_mode,
        workers=a.workers, suite=suite,
        evidence_class=S.SWEEP_SUITES["model"][1],
        nominal_xe=dataclasses.asdict(N),
        cases=[dict(name=n_, range_label=l_,
                    plant_xe=dataclasses.asdict(q_),
                    changed={f.name: getattr(q_, f.name)
                             for f in dataclasses.fields(N)
                             if getattr(q_, f.name) != getattr(N, f.name)})
               for n_, q_, l_ in CASES],
        conditions=dict(
            mismatched="plant = theta_true; controller projection = NOMINAL",
            matched="plant = theta_true; controller projection = theta_true"),
        option_A="controller observes the true current I and X; only its "
                 "projection parameters differ; trip rule and metrics use "
                 "the true plant",
        excluded_parameters="Sigma_f (cancels from rho_Xe), sigma_aX and nu "
                            "(atom->pcm conversion; state-reading artifact "
                            "under Option A), gamma_I/gamma_X (small effect)",
        not_included="R_cap/headroom mismatch; R_cap = 3500 control",
        outputs={k: dict(path=v, sha256=S._sha256(v))
                 for k, v in outputs.items() if k != "manifest"})
    man.update(S._git_state(HERE))
    S._write_manifest(outputs["manifest"], man)
    print(f"\n[mismatch] wrote {', '.join(outputs.values())} in "
          f"{(time.time()-t0)/60:.1f} min; conservation failures: {bad}")


if __name__ == "__main__":
    main()
