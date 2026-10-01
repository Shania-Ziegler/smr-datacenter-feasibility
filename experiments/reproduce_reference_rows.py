#!/usr/bin/env python3
"""
reproduce_reference_rows.py -- regression check: re-simulate selected rows
of a completed main controller suite (the reference run) with the CURRENT code and
require exact equality with the stored rows.

Reference: controller_runs/main_suite.csv and its manifest (read only).
Compared: every column of the stored row except wall-clock timing columns
(policy_* / projection_*_us, which are host-load dependent). Floats are
compared after the same CSV round trip the canonical file went through;
NaN == NaN. Nothing is written except an optional --report CSV.

--as-sweep horizon|eps additionally re-runs the predictive/consolidated
rows as ONE-FACTOR SWEEP tasks at the canonical value (H = 10 h, or
eps = 20 pcm); those must reproduce the canonical physics/decision columns
too (the sweep only adds labels: experiment, suite, evidence_class,
sweep_parameter, predict_horizon_h).

Examples (run from the repository root; the script works in data/):
  python3 experiments/reproduce_reference_rows.py --cells ah --rcaps 2700 \
      --scenarios none,compound_blackout_spike --workers 11
  python3 experiments/reproduce_reference_rows.py --cells ah --rcaps 2700 --as-sweep eps
"""
import argparse
import dataclasses
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
HERE = os.path.join(ROOT, "data")      # working directory: inputs, outputs
sys.path.insert(0, SRC_DIR)
import consolidated_scheduler as CS  # noqa: E402
import sim as S  # noqa: E402

CANON = "controller_runs/main_suite.csv"
CANON_MANIFEST = "controller_runs/main_suite_manifest.json"
TIMING = ("policy_total_us", "policy_mean_us", "policy_p95_us",
          "policy_p99_us", "policy_max_us", "projection_total_us",
          "projection_mean_us", "projection_p95_us", "projection_p99_us",
          "projection_max_us")
SWEEP_LABELS = ("experiment", "suite", "evidence_class", "sweep_parameter",
                "predict_horizon_h")


def _roundtrip(rows):
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and \
            math.isnan(a) and math.isnan(b):
        return True
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="ah")
    ap.add_argument("--rcaps", default="2700")
    ap.add_argument("--scenarios", default="none")
    ap.add_argument("--as-sweep", choices=("horizon", "eps"), default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    os.chdir(HERE)

    man = json.load(open(CANON_MANIFEST))
    code_now = {f: S._sha256(os.path.join(SRC_DIR, f))
                for f in ("sim.py", "consolidated_scheduler.py")}
    print("canonical code sha256:", man["code_sha256_at_start"])
    print("current   code sha256:", code_now)
    scfg = CS.StressConfig()
    assert dataclasses.asdict(scfg) == man["stress_config"], \
        "default StressConfig differs from the canonical run"
    assert man["G_max_frac"] == 0.4 and man["timing_mode"] == "detailed"

    canon = pd.read_csv(CANON)
    cells = list(a.cells)
    rcaps = [float(x) for x in a.rcaps.split(",")]
    scen_names = a.scenarios.split(",")
    price, genmix = "pjm_dom_34885183.csv", "load_pjm_genmix.csv"
    assert man["input_sha256"][price] == S._sha256(price)
    assert man["input_sha256"][genmix] == S._sha256(genmix)

    scenarios = {s.name: s for s in CS.build_scenarios(
        scfg, scen_names + [x for x in ("none", "spike")
                            if x not in scen_names])}
    ctx = {}
    for c in cells:
        p = f"demand_curve_cell_{c}.csv"
        assert man["input_sha256"][p] == S._sha256(p), p
        ctx[c] = S._load_cell_ctx(c, p, price, genmix, 60.0)
    S._CTX.update(cells=ctx, scenarios=scenarios, stress_cfg=scfg,
                  timing="detailed", decision_log="episodes")

    all_tasks = S._controller_tasks(cells, scen_names,
                                    ["computing_only", "consolidated"],
                                    scenarios, 0.4, "ST", True)
    tasks = [t for t in all_tasks
             if t["suite"] == "primary" and t["R_cap"] in rcaps]
    if a.as_sweep:
        key, val = (("horizon_h", S.CANONICAL_HORIZON_H)
                    if a.as_sweep == "horizon"
                    else ("eps", S.CANONICAL_EPS_PCM))
        suite = S.SWEEP_SUITES[a.as_sweep][0]
        tasks = [dict(t, suite=suite, **{key: val}) for t in tasks
                 if t["arm"] in S.SWEEP_ARMS]
    for i, t in enumerate(tasks):
        t["idx"] = i
    print(f"re-simulating {len(tasks)} canonical task(s) "
          f"(cells {a.cells}, R_cap {rcaps}, scenarios {scen_names}"
          f"{', as ' + a.as_sweep + ' sweep' if a.as_sweep else ''})")

    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(S._controller_task, tasks, chunksize=1)
    else:
        res = list(map(S._controller_task, tasks))
    rows = [r[1] for r in sorted(res, key=lambda r: r[0])]
    new = _roundtrip(rows)
    print(f"  done in {time.time() - t0:.0f} s")

    skip = set(TIMING) | (set(SWEEP_LABELS) if a.as_sweep else set())
    report, n_bad = [], 0
    for _, nr in new.iterrows():
        m = canon[(canon.cell == nr.cell) & (canon.R_cap == nr.R_cap) &
                  (canon.scenario == nr.scenario) & (canon.rx == nr.rx) &
                  (canon.suite == "primary") &
                  ((canon.hyst_h == nr.hyst_h) if pd.notna(nr.hyst_h)
                   else canon.hyst_h.isna())]
        assert len(m) == 1, (nr.experiment, len(m))
        cr = m.iloc[0]
        cols = [c for c in canon.columns if c not in skip]
        # a column only other arms/scenarios populate (e.g. reactive's
        # margin_guard_steps) is absent from a smaller batch: it must then
        # be NaN in the canonical row too
        missing = [c for c in cols if c not in new.columns
                   and not pd.isna(cr[c])]
        cols = [c for c in cols if c in new.columns or c in missing]
        diffs = [c for c in cols if c in new.columns
                 and not _same(cr[c], nr[c])]
        ok = not diffs and not missing
        n_bad += not ok
        report.append(dict(experiment=cr.experiment, rx=nr.rx,
                           compared_columns=len(cols) - len(missing),
                           identical=ok, differing=";".join(diffs),
                           missing=";".join(missing)))
        print(f"  {'IDENTICAL' if ok else 'DIFFERS  '} {cr.experiment:>48} "
              f"| {nr.rx:>14} | {len(cols) - len(missing)} columns"
              + (f" | diff {diffs[:6]} missing {missing[:6]}" if not ok
                 else ""))
    if a.report:
        pd.DataFrame(report).to_csv(a.report, index=False)
    print(f"\n{len(report) - n_bad}/{len(report)} rows bit-identical to "
          f"the canonical CSV (timing columns excluded"
          f"{'; sweep label columns excluded' if a.as_sweep else ''}).")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
