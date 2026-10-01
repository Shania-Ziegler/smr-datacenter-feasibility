#!/usr/bin/env python3
"""
partial_grid_feasibility_experiment.py -- can the reactor-feasibility layer
expose the LARGEST SAFE cheap-price import instead of vetoing it?

Arms: computing_only, consolidated (canonical all-or-nothing gate) and the
opt-in consolidated_partial_grid (largest projected-feasible partial
import; identical safety definition: 10 h projection, R_cap - sigma_m -
eps = 2700 - 100 - 20 pcm). Cells A-H, R_cap 2700, scenario none; every
other setting canonical (same _controller_task as the suite). All runs add
the price-import diagnostic columns; the partial arm also runs the
8-point monotonicity audit (diagnostic only, never changes a decision).

  --stage g04   : G = 0.4, all three arms (24 runs). computing_only and
                  consolidated must reproduce the canonical rows on every
                  canonical column (timing excluded).
  --stage sweep : G in {0, 0.2, 0.4, 0.6, 0.8, 1.0}, consolidated and
                  partial (96 runs). consolidated must reproduce the
                  completed grid-capacity sweep rows; G = 0.4 rows must
                  repeat the g04 stage exactly (determinism).
                  computing_only rows are NOT rerun: they are reused from
                  controller_runs/grid_capacity.csv (hash recorded).

Run from the repository root (the script works in data/; it refuses
to overwrite anything):
  python3 experiments/partial_grid_feasibility_experiment.py --stage g04 --workers 11 \
      --out controller_runs/partial_grid_feasibility_g04.csv
  python3 experiments/partial_grid_feasibility_experiment.py --stage sweep --workers 11 \
      --out controller_runs/partial_grid_feasibility.csv \
      --g04 controller_runs/partial_grid_feasibility_g04.csv
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
GRID_SWEEP = "controller_runs/grid_capacity.csv"
PRICE, GENMIX = "pjm_dom_34885183.csv", "load_pjm_genmix.csv"
R_CAP = 2700.0
PARTIAL = CS.CONSOLIDATED_PARTIAL_GRID
AUDIT_POINTS = 8
SUITE = "partial_grid_feasibility"
EVIDENCE = "extension_partial_grid_import"
STAGES = {"g04": dict(G=[0.4], arms=["computing_only", "consolidated",
                                     PARTIAL]),
          "sweep": dict(G=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        arms=["consolidated", PARTIAL])}
TIMING = ("policy_total_us", "policy_mean_us", "policy_p95_us",
          "policy_p99_us", "policy_max_us", "projection_total_us",
          "projection_mean_us", "projection_p95_us", "projection_p99_us",
          "projection_max_us")
LABELS = ("suite", "evidence_class", "sweep_parameter")


def _roundtrip(rows):
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def _same(a, b):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return a == b


def compare(new, ref, ref_name, skip, key=("cell", "rx", "G")):
    """Every column of the reference rows must be identical in the new
    rows (columns only the new run adds are ignored)."""
    rep = []
    for _, nr in new.iterrows():
        m = ref
        for k in key:
            m = m[m[k] == nr[k]]
        if "suite" in ref.columns and ref_name == "canonical":
            m = m[(m.suite == "primary") & (m.scenario == "none") &
                  (m.R_cap == R_CAP)]
        assert len(m) == 1, (ref_name, tuple(nr[k] for k in key), len(m))
        rr = m.iloc[0]
        cols = [c for c in ref.columns if c not in skip]
        missing = [c for c in cols if c not in new.columns
                   and not pd.isna(rr[c])]
        diffs = [c for c in cols if c in new.columns
                 and not _same(rr[c], nr[c])]
        rep.append(dict(reference=ref_name, cell=nr.cell, rx=nr.rx, G=nr.G,
                        identical=not diffs and not missing,
                        compared_columns=len(cols) - len(missing),
                        differing=";".join(diffs),
                        missing=";".join(missing)))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--g04", default=None,
                    help="sweep stage: the completed g04 stage CSV")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    os.chdir(HERE)
    stem = a.out[:-4] if a.out.endswith(".csv") else a.out
    outputs = dict(results=a.out, trips=f"{stem}_trip_log.csv",
                   validation=f"{stem}_validation.csv",
                   manifest=f"{stem}_manifest.json")
    for p in outputs.values():
        if S._is_canonical_output(p) or os.path.exists(p) or \
                os.path.basename(p).startswith(("grid_capacity",
                                                "horizon", "eps",
                                                "model_mismatch", "stress")):
            sys.exit(f"refusing to write {p} (exists or protected)")
    if a.stage == "sweep" and not a.g04:
        sys.exit("--stage sweep needs --g04 (determinism check)")

    man0 = json.load(open(CANON_MANIFEST))
    scfg = CS.StressConfig()
    assert dataclasses.asdict(scfg) == man0["stress_config"]
    code_files = {"sim.py": os.path.join(SRC_DIR, "sim.py"),
                  "consolidated_scheduler.py":
                  os.path.join(SRC_DIR, "consolidated_scheduler.py"),
                  os.path.basename(__file__): os.path.abspath(__file__)}
    code0 = {f: S._sha256(p) for f, p in code_files.items()}
    ref_files = [CANON, GRID_SWEEP] + ([a.g04] if a.g04 else [])
    ref_hash0 = {p: S._sha256(p) for p in ref_files}
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
                  timing="detailed", decision_log="episodes")

    base = [t for t in S._controller_tasks(cells, ["none"],
                                           ["computing_only",
                                            "consolidated"],
                                           scenarios, 0.4, "ST", False)
            if t["arm"] in ("computing_only", "consolidated")
            and t["R_cap"] == R_CAP and t["suite"] == "primary"]
    assert len(base) == 16
    st = STAGES[a.stage]
    tasks = []
    for G in st["G"]:
        for arm in st["arms"]:
            src = "consolidated" if arm == PARTIAL else arm
            for t in base:
                if t["arm"] != src:
                    continue
                t2 = dict(t, idx=len(tasks), G=float(G), arm=arm,
                          price_diagnostics=True)
                if arm == PARTIAL:
                    t2["partial_grid_audit"] = AUDIT_POINTS
                tasks.append(t2)
    print(f"[partial-grid {a.stage}] {len(tasks)} runs on {a.workers} "
          f"workers; R_cap {R_CAP:.0f}, scenario none, G {st['G']}, "
          f"arms {st['arms']}")

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
            print(f"  [{done:3d}/{len(tasks)}] {row['experiment']:>30} | "
                  f"{row['rx']:>25} | uns {row['true_unserved']/1e6:7.3f} "
                  f"MWh | trips {row['n_trips']:2d} | price acc "
                  f"{row['grid_price_Wh']/1e6:7.2f} MWh | "
                  f"{time.time()-t0:6.0f} s{flag}", flush=True)
    finally:
        pool.close()
        pool.join()

    new = _roundtrip(rows)
    skip = set(TIMING) | set(LABELS)
    rep = []
    if a.stage == "g04":
        ref = pd.read_csv(CANON)
        rep += compare(new[new.rx != PARTIAL], ref, "canonical", skip)
    else:
        ref = pd.read_csv(GRID_SWEEP)
        rep += compare(new[new.rx == "consolidated"], ref,
                       "grid_capacity", skip)
        g4 = pd.read_csv(a.g04)
        g4n = new[(new.G == 0.4)]
        rep += compare(g4n, g4[g4.rx != "computing_only"], "g04_stage",
                       skip | {"experiment"})
    n_ok = sum(r["identical"] for r in rep)
    for r in rows:
        r.update(suite=SUITE, evidence_class=EVIDENCE,
                 sweep_parameter="G_max_frac" if a.stage == "sweep"
                 else "none")
    pd.DataFrame(rows).to_csv(outputs["results"], index=False, mode="x")
    pd.DataFrame(trips).to_csv(outputs["trips"], index=False, mode="x")
    pd.DataFrame(rep).to_csv(outputs["validation"], index=False, mode="x")
    print(f"\n[partial-grid] reference reproduction: {n_ok}/{len(rep)} rows "
          f"identical (timing + label columns excluded)")
    for r in rep:
        if not r["identical"]:
            print(f"  DIFFERS vs {r['reference']}: {r['cell']} {r['rx']} "
                  f"G={r['G']}: {r['differing']} missing {r['missing']}")

    code1 = {f: S._sha256(p) for f, p in code_files.items()}
    man = dict(
        manifest_version=1,
        experiment="partial cheap-price import under the reactor-"
                   "feasibility layer",
        stage=a.stage, started_utc=started,
        finished_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        argv=sys.argv, cwd=os.getcwd(), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__,
        code_sha256_at_start=code0, code_sha256_at_end=code1,
        code_unchanged_during_run=code0 == code1,
        reference_sha256_at_start=ref_hash0,
        reference_unchanged=ref_hash0 == {p: S._sha256(p)
                                          for p in ref_files},
        input_sha256=inputs, runs=len(tasks), conservation_failures=bad,
        cells=cells, arms=st["arms"], G_values=st["G"], R_cap_pcm=R_CAP,
        scenario="none", eps_buffer_pcm=S.CANONICAL_EPS_PCM,
        sigma_m_pcm=S.sigma_m, predict_horizon_h=S.CANONICAL_HORIZON_H,
        bisect_iters=CS.ControllerConfig().bisect_iters,
        partial_audit_points=AUDIT_POINTS, timing_mode="detailed",
        workers=a.workers, suite=SUITE, evidence_class=EVIDENCE,
        computing_only_source=(None if a.stage == "g04" else dict(
            path=GRID_SWEEP, sha256=ref_hash0[GRID_SWEEP],
            note="computing_only rows reused, not rerun")),
        reference_reproduction=dict(rows_identical=n_ok,
                                    rows_compared=len(rep)),
        outputs={k: dict(path=v, sha256=S._sha256(v))
                 for k, v in outputs.items() if k != "manifest"})
    man.update(S._git_state(HERE))
    S._write_manifest(outputs["manifest"], man)
    print(f"[partial-grid] wrote {', '.join(outputs.values())} in "
          f"{(time.time()-t0)/60:.1f} min; conservation failures: {bad}")
    sys.exit(0 if (n_ok == len(rep) and bad == 0) else 1)


if __name__ == "__main__":
    main()
