#!/usr/bin/env python3
"""
synthetic_example.py -- a self-contained demonstration that needs NO
external data (no PJM, no Borg).

Part 1 uses the standalone feasibility interface (src/xenon_guard.py):
starting from full-power xenon equilibrium at an end-of-cycle margin
(R_cap = 2700 pcm), it asks whether a requested power reduction is
projected safe over the next 10 h, the shallowest safe setpoint, and how
long the reactor would be unavailable if it were forced to shut down.

Part 2 runs the reactor policies and workload-aware controllers on a
synthetic 4-day workload (a smooth daily cycle between 30 % and 85 %
utilisation, with fixed batch / mid-tier fractions) without any grid
connection. The workload is invented for illustration only; its numbers
say nothing about the Borg evaluation.

Usage (from the repository root):
  python examples/synthetic_example.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import sim as S  # noqa: E402
import xenon_guard as XG  # noqa: E402

R_CAP = 2700.0          # pcm, end-of-cycle-like reactivity margin
DT = 60.0               # s, simulator step


def part1_feasibility_interface():
    print("== Part 1: feasibility interface (xenon_guard)")
    state = XG.equilibrium_state(1.0)
    for p in (0.9, 0.7, 0.5):
        safe, peak = XG.is_dip_safe(state, p, R_CAP)
        print(f"  hold {p:.0%} power for 10 h: projected xenon peak "
              f"{peak:7.1f} pcm -> {'safe' if safe else 'NOT safe'} "
              f"(ceiling R_cap - sigma_m - eps = "
              f"{R_CAP - XG.SIGMA_M - 20.0:.0f} pcm)")
    r = XG.feasible_floor_checked(state, 0.5, R_CAP)
    print(f"  shallowest projected-safe setpoint for a request of 50 %: "
          f"{r.floor:.3f} (peak {r.peak_pcm:.1f} pcm)" if r.feasible else
          "  no projected-safe setpoint exists")
    print(f"  dead time if forced to shut down now: "
          f"{XG.deadtime_if_tripped(state, R_CAP):.1f} h")


def part2_policies():
    print("\n== Part 2: policies on a synthetic 4-day workload, no grid")
    t_grid = np.arange(0.0, 4 * 86400.0, DT)
    u_at = lambda t: 0.575 + 0.275 * np.sin(2 * np.pi * t / 86400.0)
    fb_at = lambda t: 0.30          # batch fraction of IT work
    fm_at = lambda t: 0.10          # mid-tier fraction of IT work
    P_rated = np.mean([S.P_FIXED + S.work_power(u_at(t)) * S.PUE
                       for t in t_grid]) / 0.95
    common = dict(trip=True, G_max_frac=0.0)
    arms = [("Limit-Agnostic", dict(rx_mode="free")),
            ("Limit-Reactive", dict(rx_mode="reactive")),
            ("Limit-Predictive", dict(rx_mode="predictive",
                                      eps_buffer=20.0)),
            ("Fixed delay 4 h", dict(rx_mode="hysteresis",
                                     hysteresis_h=4.0)),
            ("Computing-Only", dict(controller_mode="computing_only")),
            ("Consolidated", dict(controller_mode="consolidated",
                                  eps_buffer=20.0))]
    print(f"  {'policy':>17} | {'shutdowns':>9} | {'dead time h':>11} | "
          f"{'crossings':>9} | {'unserved MWh':>12} | {'min headroom pcm':>16}")
    for name, kw in arms:
        r = S.run(fb_at, fm_at, "tiered", t_grid, u_at, P_rated, DT, R_CAP,
                  **common, **kw)
        print(f"  {name:>17} | {r['n_trips']:9d} | {r['deadtime_h']:11.1f} | "
              f"{r['crossings']:9d} | {r['true_unserved'] / 1e6:12.3f} | "
              f"{r['min_headroom']:16.1f}")


if __name__ == "__main__":
    part1_feasibility_interface()
    part2_policies()
