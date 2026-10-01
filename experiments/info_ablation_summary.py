#!/usr/bin/env python3
"""
info_ablation_summary.py -- read-only analysis of the COMPLETED controller
suite, framed as an information ablation: which reactor-state information
does a scheduler need to avoid requesting infeasible power trajectories?

No simulation is run and no controller is added. The existing arms already
differ in the information they are given:

  free            requested power / ordinary signals only
  hysteresis 2-6h elapsed time since the last down-ramp; no reactor state
  computing_only  workload queue + causal demand forecast; no reactor state
  reactive        current reactor state (xenon headroom) only
  predictive      current reactor state + projected iodine-xenon trajectory
                  (constant-hold projection)
  consolidated    workload queue/forecast + projected reactor feasibility
                  (projection of the requested power path)

These are different information sets, NOT a monotonic sophistication
ladder; predictive and consolidated also differ in mechanism (constant-hold
floor vs. path projection), not only in information.

Input  (never modified): controller_runs/main_suite.csv
Rows   : evidence_class == "primary_cross_arm" only (no detector-sensitivity
         rows, no synthetic deadline-shock diagnostics). Every primary
         scenario is kept and reported separately; rows never mix scenarios.
Outputs (new files, refuses to overwrite):
  analysis/info_ablation_cells.csv    one row per cell x R_cap x scenario x arm
  analysis/info_ablation_summary.csv  mean / std / min / max over cells A-H
  analysis/info_ablation_manifest.json

Run from the repository root:  python3 experiments/info_ablation_summary.py
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

WORK = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")   # inputs and outputs
SRC = "controller_runs/main_suite.csv"
OUT_DIR = "analysis"
OUT_CELLS = os.path.join(OUT_DIR, "info_ablation_cells.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "info_ablation_summary.csv")
OUT_MANIFEST = os.path.join(OUT_DIR, "info_ablation_manifest.json")

# arm -> (information set, reactor state, xenon projection,
#         workload queue/forecast, elapsed-time rule)
INFO = {
    "free": ("requested power only", False, False, False, False),
    "hysteresis": ("elapsed time since down-ramp", False, False, False, True),
    "computing_only": ("workload queue + demand forecast", False, False, True,
                       False),
    "reactive": ("current reactor headroom", True, False, False, False),
    "predictive": ("current reactor state + constant-hold Xe projection",
                   True, True, False, False),
    "consolidated": ("workload + projected reactor feasibility (path)", True,
                     True, True, False),
}
ARM_ORDER = ["free", "hysteresis 2h", "hysteresis 4h", "hysteresis 6h",
             "computing_only", "reactive", "predictive", "consolidated"]

# metric -> output column (energy converted Wh -> kWh)
METRICS = {
    "true_unserved": "unserved_kWh",
    "dropped": "dropped_kWh",
    "rigid_unserved_Wh": "rigid_unserved_kWh",
    "n_trips": "trips",
    "deadtime_h": "deadtime_h",
    "crossings": "crossing_min",
    "cross_episodes": "crossing_episodes",
    "max_exceed": "max_exceed_pcm",
    "grid_Wh": "grid_kWh",
    "grid_cost": "grid_cost_usd",
    "water_sys_L": "water_sys_L",
    "refused_steps": "refused_or_modified_steps",
    "policy_mean_us": "policy_mean_us",
    "policy_p95_us": "policy_p95_us",
    "projection_mean_us": "projection_mean_us",
    "projection_calls": "projection_calls",
}
KWH = {"true_unserved", "dropped", "rigid_unserved_Wh", "grid_Wh"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.chdir(WORK)
    for p in (OUT_CELLS, OUT_SUMMARY, OUT_MANIFEST):
        if os.path.exists(p):
            sys.exit(f"{p} exists; refusing to overwrite (move it first).")
    src_hash = sha256(SRC)
    d = pd.read_csv(SRC)
    assert len(d) == 2112, len(d)
    p = d[d.evidence_class == "primary_cross_arm"].copy()
    assert len(p) == 960, len(p)
    assert set(p.cell) == set("abcdefgh")

    p["arm"] = np.where(p.rx == "hysteresis",
                        "hysteresis " + p.hyst_h.fillna(0).astype(int)
                        .astype(str) + "h", p.rx)
    base = p.rx.map(lambda a: INFO[a])
    p["information_set"] = [b[0] for b in base]
    p["uses_reactor_state"] = [b[1] for b in base]
    p["uses_xenon_projection"] = [b[2] for b in base]
    p["uses_workload_queue_forecast"] = [b[3] for b in base]
    p["uses_elapsed_time_rule"] = [b[4] for b in base]

    keep = ["R_cap", "scenario", "arm", "cell", "information_set",
            "uses_reactor_state", "uses_xenon_projection",
            "uses_workload_queue_forecast", "uses_elapsed_time_rule",
            "eps", "timing_mode"]
    cells = p[keep].copy()
    for src, dst in METRICS.items():
        v = p[src].astype(float)
        cells[dst] = v / 1e3 if src in KWH else v

    # every (R_cap, scenario, arm) group must hold exactly cells A-H
    grp = cells.groupby(["R_cap", "scenario", "arm"])
    sizes = grp.cell.agg(lambda s: "".join(sorted(s)))
    assert (sizes == "abcdefgh").all(), sizes[sizes != "abcdefgh"]

    order = {a: i for i, a in enumerate(ARM_ORDER)}
    cells["_o"] = cells.arm.map(order)
    cells = cells.sort_values(["R_cap", "scenario", "_o", "cell"],
                              ascending=[False, True, True, True]) \
                 .drop(columns="_o")

    info_cols = ["information_set", "uses_reactor_state",
                 "uses_xenon_projection", "uses_workload_queue_forecast",
                 "uses_elapsed_time_rule"]
    mcols = list(METRICS.values())
    g = cells.groupby(["R_cap", "scenario", "arm"], sort=False)
    summ = g[info_cols].first()
    summ["n_cells"] = g.size()
    stats = g[mcols].agg(["mean", "std", "min", "max"])
    stats.columns = [f"{m}_{s}" for m, s in stats.columns]
    summ = summ.join(stats).reset_index()

    os.makedirs(OUT_DIR, exist_ok=True)
    cells.to_csv(OUT_CELLS, index=False)
    summ.to_csv(OUT_SUMMARY, index=False)
    man = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script=os.path.basename(__file__),
        script_sha256=sha256(os.path.abspath(__file__)),
        source=SRC, source_sha256=src_hash,
        source_rows=len(d), selected_rows=len(p),
        selection="evidence_class == 'primary_cross_arm' (all primary "
                  "scenarios, reported separately)",
        excluded=["sensitivity (detector-threshold grid)",
                  "controller_diagnostic_endogenous_deadline_shock"],
        cells="Borg cells A-H: they differ mainly in workload flexibility "
              "(batch/mid fraction) with some utilization/burstiness "
              "variation; not eight distinct workload types",
        R_cap_note="fixed-margin snapshots 8000/3500/2700 pcm; no fuel-cycle "
                   "R_cap(t)",
        G_max_frac=sorted(p.G.unique().tolist()),
        framing="information ablation over existing arms; different "
                "information sets, not a monotonic sophistication ladder",
        timing_note="policy/projection timings are wall-clock measurements "
                    "from the parallel canonical run (timing_mode=detailed); "
                    "they depend on host load and are reported descriptively",
        outputs={o: sha256(o) for o in (OUT_CELLS, OUT_SUMMARY)})
    with open(OUT_MANIFEST, "x") as f:
        json.dump(man, f, indent=2, default=str)
        f.write("\n")

    print(f"source {SRC}: {len(d)} rows; primary_cross_arm {len(p)}")
    print(f"wrote {OUT_CELLS} ({len(cells)} rows), {OUT_SUMMARY} "
          f"({len(summ)} rows), {OUT_MANIFEST}")
    show = summ[summ.scenario == "none"]
    print("\nNominal scenario, mean over cells A-H:")
    print(show[["R_cap", "arm", "unserved_kWh_mean", "trips_mean",
                "deadtime_h_mean", "crossing_min_mean", "grid_cost_usd_mean",
                "policy_mean_us_mean"]].to_string(index=False,
                                                  float_format="%.1f"))


if __name__ == "__main__":
    main()
