#!/usr/bin/env python3
"""
plot_summary_findings.py -- three short findings with large numbers, computed from the canonical A-H evaluation.

Source (read only): controller_runs/main_suite.csv,
primary_cross_arm, scenario none, R_cap 2700 (low margin) and 8000
(ample margin), cells A-H; means cross-checked against
analysis/reported_numbers.csv.
Output (refuses to overwrite): figures/summary_findings.*
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# inputs are read from, and figures written to, the data/ working directory
os.chdir(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data"))
os.makedirs("figures", exist_ok=True)

CSV = "controller_runs/main_suite.csv"
NUMBERS = "analysis/reported_numbers.csv"
OUT, STEM = "figures", "summary_findings"
FIGSIZE = (13.60, 3.20)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 600,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
for ext in ("png", "svg"):
    if os.path.exists(f"{OUT}/{STEM}.{ext}"):
        sys.exit(f"{OUT}/{STEM}.{ext} exists; refusing to overwrite.")

d = pd.read_csv(CSV)
n = d[(d.evidence_class == "primary_cross_arm") & (d.scenario == "none")]
n = n.assign(arm=np.where(n.rx == "hysteresis", "hyst" + n.hyst_h.fillna(0)
                          .astype(int).astype(str), n.rx))
nm = pd.read_csv(NUMBERS)


def m(R, arm, col):
    g = n[(n.R_cap == R) & (n.arm == arm)]
    assert len(g) == 8 and "".join(sorted(g.cell)) == "abcdefgh"
    v = g[col].mean()
    key = "unserved_kWh" if col == "true_unserved" else "trips"
    ev = nm[(nm.section == "2 main") & (nm.metric == key) & (nm.arm == arm)
            & (nm.R_cap.astype(str) == str(float(R)))].value
    ref = float(ev.iloc[0]) * (1e3 if col == "true_unserved" else 1)
    assert len(ev) == 1 and np.isclose(v, ref, rtol=1e-12), (R, arm, col)
    return v


MWh = lambda R, a: m(R, a, "true_unserved") / 1e6
fail = ["free", "reactive", "computing_only"]
hyst = ["hyst2", "hyst4", "hyst6"]
f_lo, f_hi = min(MWh(2700, a) for a in fail), max(MWh(2700, a) for a in fail)
f_tr = {round(m(2700, a, "n_trips"), 1) for a in fail}
assert f_tr == {7.4}
p_k, c_k = MWh(2700, "predictive") * 1e3, MWh(2700, "consolidated") * 1e3
assert m(2700, "predictive", "n_trips") == 0 == m(2700, "consolidated",
                                                  "n_trips")
h_lo, h_hi = min(MWh(2700, a) for a in hyst), max(MWh(2700, a) for a in hyst)
a_lo, a_hi = min(MWh(8000, a) for a in hyst), max(MWh(8000, a) for a in hyst)
print(f"source: {CSV}; cross-check vs {NUMBERS}: PASS")
print(f"  no-projection arms @2700: {f_lo:.3f}-{f_hi:.3f} MWh, trips {f_tr}")
print(f"  predictive {p_k:.1f} kWh, consolidated {c_k:.1f} kWh, 0 trips")
print(f"  fixed delays @2700: {h_lo:.3f}-{h_hi:.3f} MWh; @8000: "
      f"{a_lo:.3f}-{a_hi:.3f} MWh")

F = [("Reactor state matters\nat low margin.",
      f"{f_lo:.0f}–{f_hi:.0f} MWh",
      "unserved · 7.4 shutdowns",
      "Agnostic · Reactive · Computing-Only"),
     ("Forward feasibility avoids\nthe observed shutdowns.",
      "0 shutdowns",
      f"Predictive {p_k:.0f} kWh · Consolidated {c_k:.0f} kWh", ""),
     ("Simple waiting\nis not enough.",
      f"{h_lo:.0f}–{h_hi:.0f} MWh",
      "unserved at low margin · 2/4/6 h delays",
      f"{a_lo:.0f}–{a_hi:.0f} MWh stranded at ample margin")]

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
w = 1.0 / 3
for i, (head, big, l1, l2) in enumerate(F):
    cx = (i + 0.5) * w
    ax.text(cx, 0.83, head, ha="center", va="center", fontsize=18,
            fontweight="bold", linespacing=1.1)
    ax.text(cx, 0.51, big, ha="center", va="center", fontsize=40,
            fontweight="bold")
    ax.text(cx, 0.28, l1, ha="center", va="center", fontsize=16)
    if l2:
        ax.text(cx, 0.13, l2, ha="center", va="center", fontsize=16)
    if i:
        ax.plot([i * w, i * w], [0.08, 0.92], color="black", lw=2.0)
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{ext}", facecolor="white")
print(f"wrote {OUT}/{STEM}.png + .svg ({FIGSIZE[0]:.2f}x{FIGSIZE[1]:.2f} in)")
