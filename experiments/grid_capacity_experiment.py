#!/usr/bin/env python3
"""
grid_capacity_experiment.py -- one-factor grid-interconnection-capacity
sensitivity (G_max_frac) for the two controller arms.

  7 G values x 2 arms (computing_only, consolidated) x 8 Borg cells (A-H)
  = 112 runs. R_cap = 2700 pcm, scenario none, H = 10 h, eps = 20 pcm
  (consolidated; computing_only consumes no eps), every other setting the
  canonical controller-suite value (same _controller_task as the suite,
  only the task's G differs).

G definition (sim.run_controller): G_max_frac is the import limit as a
fraction of the cell's reactor rating P_rated (= mean facility demand /
0.95), i.e. grid_W <= G * P_rated at every step (times grid availability,
which is 1 throughout under scenario none). Imports only; surplus reactor
power is never exported. G = 0 means no interconnection at all
(grid_connected False: no price imports, no bridging, no emergency supply
while tripped).

These are grid-capacity SENSITIVITY cases, not claims that each G is an
economically realistic interconnection design.

The G = 0.4 rows must reproduce the canonical primary rows bit-for-bit
(timing columns and the two label columns suite / evidence_class excluded);
the script exits non-zero otherwise. Nothing canonical is written.

Run from the repository root (the script works in data/; it refuses
to overwrite anything):
  python3 experiments/grid_capacity_experiment.py --workers 11 \
      --out controller_runs/grid_capacity.csv
"""
import argparse
import dataclasses
import datetime
import io
import json
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

CANON = "controller_runs/main_suite.csv"
CANON_MANIFEST = "controller_runs/main_suite_manifest.json"
PRICE, GENMIX = "pjm_dom_34885183.csv", "load_pjm_genmix.csv"
R_CAP = 2700.0
ARMS = ("computing_only", "consolidated")
G_VALUES = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
G_CANON = 0.4
SUITE = "grid_capacity_sensitivity"
EVIDENCE = "sensitivity_grid_capacity"
TIMING = ("policy_total_us", "policy_mean_us", "policy_p95_us",
          "policy_p99_us", "policy_max_us", "projection_total_us",
          "projection_mean_us", "projection_p95_us", "projection_p99_us",
          "projection_max_us")
LABELS = ("suite", "evidence_class")


def _same(a, b):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return a == b


def _check_canonical(rows):
    """G = 0.4 rows vs the canonical primary rows, after the same CSV
    round trip. Returns (n_identical, n_compared, report rows)."""
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    new = pd.read_csv(buf)
    new = new[new.G == G_CANON]
    canon = pd.read_csv(CANON)
    skip = set(TIMING) | set(LABELS)
    rep, ok_n = [], 0
    for _, nr in new.iterrows():
        m = canon[(canon.cell == nr.cell) & (canon.R_cap == nr.R_cap) &
                  (canon.scenario == nr.scenario) & (canon.rx == nr.rx) &
                  (canon.suite == "primary")]
        assert len(m) == 1, (nr.experiment, len(m))
        cr = m.iloc[0]
        cols = [c for c in canon.columns if c not in skip]
        missing = [c for c in cols if c not in new.columns
                   and not pd.isna(cr[c])]
        diffs = [c for c in cols if c in new.columns
                 and not _same(cr[c], nr[c])]
        extra = [c for c in new.columns if c not in canon.columns]
        ok = not diffs and not missing and not extra
        ok_n += ok
        rep.append(dict(cell=nr.cell, rx=nr.rx, identical=ok,
                        compared_columns=len(cols) - len(missing),
                        differing=";".join(diffs), missing=";".join(missing),
                        extra=";".join(extra)))
    return ok_n, len(rep), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    os.chdir(HERE)
    stem = a.out[:-4] if a.out.endswith(".csv") else a.out
    outputs = dict(results=a.out, trips=f"{stem}_trip_log.csv",
                   validation=f"{stem}_g04_vs_canonical.csv",
                   manifest=f"{stem}_manifest.json")
    for p in outputs.values():
        if S._is_canonical_output(p) or os.path.exists(p):
            sys.exit(f"refusing to write {p} (exists or canonical)")

    man0 = json.load(open(CANON_MANIFEST))
    scfg = CS.StressConfig()
    assert dataclasses.asdict(scfg) == man0["stress_config"]
    assert man0["G_max_frac"] == G_CANON and \
        man0["timing_mode"] == "detailed"
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
    # same context as the canonical suite / reproduce_canonical_rows.py
    S._CTX.update(cells=ctx, scenarios=scenarios, stress_cfg=scfg,
                  timing="detailed", decision_log="episodes")

    base = [t for t in S._controller_tasks(cells, ["none"], list(ARMS),
                                           scenarios, G_CANON, "ST", False)
            if t["arm"] in ARMS and t["R_cap"] == R_CAP
            and t["suite"] == "primary"]
    assert len(base) == 16
    tasks = []
    for G in G_VALUES:
        for t in base:
            # G = 0.4 tasks keep suite 'primary' so they are the canonical
            # task verbatim; labels are rewritten on every row below
            tasks.append(dict(t, idx=len(tasks), G=float(G)))
    assert len(tasks) == 112
    print(f"[grid] {len(tasks)} runs on {a.workers} workers; R_cap "
          f"{R_CAP:.0f}, scenario none, G in {list(G_VALUES)}, H=10 h, "
          f"eps=20 pcm (consolidated); timing detailed")

    t0 = time.time()
    rows, trips, bad = [None] * len(tasks), [], 0
    pool = mp.get_context("fork").Pool(a.workers)
    try:
        it = pool.imap_unordered(S._controller_task, tasks, chunksize=1)
        for done, (i, row, tr, _ep, _mn) in enumerate(it, 1):
            rows[i] = row
            for r_ in tr:
                r_.update(suite=SUITE)
            trips.extend(tr)
            flag = ""
            if abs(row["residual"]) > 1.0 or \
                    abs(row.get("energy_residual_Wh", 0.0) or 0.0) > 1.0:
                flag = "  <-- CONSERVATION BROKEN"
                bad += 1
            print(f"  [{done:3d}/{len(tasks)}] {row['experiment']:>34} | "
                  f"{row['rx']:>14} | uns {row['true_unserved']/1e6:8.2f} "
                  f"MWh | trips {row['n_trips']:2d} | "
                  f"{time.time()-t0:6.0f} s{flag}", flush=True)
    finally:
        pool.close()
        pool.join()

    n_ok, n_cmp, rep = _check_canonical(rows)
    for r in rows:
        r.update(suite=SUITE, evidence_class=EVIDENCE,
                 sweep_parameter="G_max_frac")
    pd.DataFrame(rows).to_csv(outputs["results"], index=False, mode="x")
    pd.DataFrame(trips).to_csv(outputs["trips"], index=False, mode="x")
    pd.DataFrame(rep).to_csv(outputs["validation"], index=False, mode="x")
    print(f"\n[grid] G={G_CANON} vs canonical: {n_ok}/{n_cmp} rows "
          f"bit-identical (timing + suite/evidence_class labels excluded)")
    for r in rep:
        if not r["identical"]:
            print(f"  DIFFERS {r['cell']} {r['rx']}: {r['differing']} "
                  f"missing {r['missing']} extra {r['extra']}")

    code1 = {f: S._sha256(p) for f, p in code_files.items()}
    man = dict(
        manifest_version=1,
        experiment="grid-interconnection capacity one-factor sensitivity",
        started_utc=started,
        finished_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        argv=sys.argv, cwd=os.getcwd(), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__,
        code_sha256_at_start=code0, code_sha256_at_end=code1,
        code_unchanged_during_run=code0 == code1,
        input_sha256=inputs, runs=len(tasks), conservation_failures=bad,
        cells=cells, arms=list(ARMS), R_cap_pcm=R_CAP, scenario="none",
        G_values=list(G_VALUES), G_canonical=G_CANON,
        G_definition="import limit grid_W <= G * P_rated (P_rated = cell "
                     "mean facility demand / 0.95); imports only, no "
                     "export; G = 0 -> no interconnection",
        predict_horizon_h=S.CANONICAL_HORIZON_H,
        eps_buffer_pcm=S.CANONICAL_EPS_PCM, timing_mode="detailed",
        decision_log="episodes", workers=a.workers, suite=SUITE,
        evidence_class=EVIDENCE,
        canonical_reproduction=dict(rows_identical=n_ok, rows_compared=n_cmp,
                                    excluded=list(TIMING) + list(LABELS)),
        note="sensitivity cases, not claims that each G is an economically "
             "realistic interconnection design",
        outputs={k: dict(path=v, sha256=S._sha256(v))
                 for k, v in outputs.items() if k != "manifest"})
    man.update(S._git_state(HERE))
    S._write_manifest(outputs["manifest"], man)
    print(f"[grid] wrote {', '.join(outputs.values())} in "
          f"{(time.time()-t0)/60:.1f} min; conservation failures: {bad}")
    sys.exit(0 if (n_ok == n_cmp == 16 and bad == 0) else 1)


if __name__ == "__main__":
    main()
