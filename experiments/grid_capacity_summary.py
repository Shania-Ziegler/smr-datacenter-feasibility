#!/usr/bin/env python3
"""
grid_capacity_summary.py -- A-H summary of grid_capacity_experiment.py.

Reads controller_runs/grid_capacity.csv (read only) and writes
controller_runs/grid_capacity_summary.csv (refuses to overwrite).
Energies in MWh, grid cost in $ (same PJM price series at every G, so
directly comparable). Run from the repository root (works in data/).
"""
import os
import sys

import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")   # inputs and outputs
SRC = "controller_runs/grid_capacity.csv"
OUT = "controller_runs/grid_capacity_summary.csv"


def main():
    os.chdir(HERE)
    d = pd.read_csv(SRC)
    key = ["G", "rx", "cell"]
    assert len(d) == 112 and not d.duplicated(key).any()
    assert (d.groupby(["G", "rx"]).size() == 8).all()
    g = d.groupby(["G", "rx"])
    s = pd.DataFrame(dict(
        cells=g.size(),
        true_unserved_MWh=g.true_unserved.mean() / 1e6,
        true_unserved_MWh_min=g.true_unserved.min() / 1e6,
        true_unserved_MWh_max=g.true_unserved.max() / 1e6,
        unserved_pct=g.unserved_pct.mean(),
        n_trips=g.n_trips.mean(),
        cells_with_trip=g.n_trips.apply(lambda x: int((x > 0).sum())),
        deadtime_h=g.deadtime_h.mean(),
        crossings=g.crossings.mean(),
        refused_steps=g.refused_steps.mean(),
        grid_MWh=g.grid_Wh.mean() / 1e6,
        grid_emergency_MWh=g.grid_emergency_Wh.mean() / 1e6,
        grid_cost_usd=g.grid_cost.mean(),
        max_abs_residual_Wh=g.energy_residual_Wh.apply(
            lambda x: x.abs().max()),
    )).reset_index()
    if os.path.exists(OUT):
        sys.exit(f"refusing to overwrite {OUT}")
    s.to_csv(OUT, index=False, mode="x")
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
