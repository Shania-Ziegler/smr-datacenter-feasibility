#!/usr/bin/env python3
"""
partial_grid_feasibility_summary.py -- A-H means for the partial-grid
feasibility experiment (read only; writes one new summary CSV).

  python3 partial_grid_feasibility_summary.py \
      controller_runs/partial_grid_feasibility_g04.csv
  python3 partial_grid_feasibility_summary.py \
      controller_runs/partial_grid_feasibility.csv \
      --computing-only controller_runs/grid_capacity.csv

Energies in MWh per 31-day cell; import magnitudes as a fraction of the
cell's reactor rating; grid cost in $. Run from the repository root (works in data/).
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")   # inputs and outputs
MWH = ("true_unserved", "grid_Wh", "grid_price_Wh", "grid_bridge_Wh",
       "grid_emergency_Wh", "price_requested_Wh", "price_approved_Wh",
       "gen_Wh", "served_it_Wh")


def summarize(d):
    g = d.groupby(["G", "rx"], sort=True)
    assert (g.size() == 8).all()
    m = g.mean(numeric_only=True)
    s = pd.DataFrame(index=m.index)
    for c in MWH:
        if c in m:
            s[c.replace("_Wh", "") + "_MWh"] = m[c] / 1e6
    s["cells_with_trip"] = g.n_trips.apply(lambda x: int((x > 0).sum()))
    for c in ("n_trips", "deadtime_h", "crossings", "min_headroom",
              "refused_steps", "grid_cost", "price_opportunity_steps",
              "price_full_accept_steps", "price_partial_steps",
              "price_full_cancel_steps", "price_import_canceled_steps",
              "price_approved_mean_frac", "price_delivered_mean_frac",
              "price_opp_mean_proj_slack_pcm", "partial_audit_missed_steps",
              "partial_audit_max_gap_frac", "no_feasible_action_steps",
              "energy_residual_Wh", "false_safe_decisions"):
        if c in m:
            s[c] = m[c]
    s["min_headroom_worst"] = g.min_headroom.min()
    if "price_requested_Wh" in m:
        s["price_accepted_frac"] = m.grid_price_Wh / m.price_requested_Wh
        s["price_approved_frac"] = m.price_approved_Wh / \
            m.price_requested_Wh
    return s.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--computing-only", default=None)
    a = ap.parse_args()
    os.chdir(HERE)
    d = pd.read_csv(a.src)
    assert not d.duplicated(["G", "rx", "cell"]).any()
    if a.computing_only:
        co = pd.read_csv(a.computing_only)
        co = co[(co.rx == "computing_only") & co.G.isin(d.G.unique())]
        d = pd.concat([d, co], ignore_index=True)
    s = summarize(d)
    out = a.src[:-4] + "_summary.csv"
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")
    s.to_csv(out, index=False, mode="x")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(s.set_index(["G", "rx"]).T.to_string(
        float_format=lambda v: f"{v:,.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
