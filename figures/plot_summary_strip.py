#!/usr/bin/env python3
"""
plot_summary_strip.py -- full-width SUMMARY results strip:
five adjacent policy cards (mean unserved work, mean shutdowns).

Source (read only): controller_runs/main_suite.csv,
primary_cross_arm, scenario none, R_cap 2700, cells A-H (8 per policy);
cross-checked against analysis/reported_numbers.csv.
Style: policy fills, 2-pt black borders, sans-serif, white background.
Output (refuses to overwrite): figures/summary_strip.*
Run from the repository root (works in data/).
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
OUT = "figures"
STEM = "summary_strip"
FIGSIZE = (13.60, 3.20)
BAR_LW = 2.0
FILL = {"free": "#F8CECC", "reactive": "#7DC4B4",
        "computing_only": "#A9C4EB", "predictive": "#B85450",
        "consolidated": "#E1D5E7"}
LABEL = {"free": "Limit-Agnostic", "reactive": "Limit-Reactive",
         "computing_only": "Computing-Only",
         "predictive": "Limit-Predictive", "consolidated": "Consolidated"}
ARMS = ["free", "reactive", "computing_only", "predictive", "consolidated"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "pdf.fonttype": 42, "savefig.dpi": 600,
    "figure.facecolor": "white", "savefig.facecolor": "white"})

for ext in ("png", "svg"):
    if os.path.exists(f"{OUT}/{STEM}.{ext}"):
        sys.exit(f"{OUT}/{STEM}.{ext} exists; refusing to overwrite.")

d = pd.read_csv(CSV)
sub = d[(d.evidence_class == "primary_cross_arm") & (d.scenario == "none")
        & (d.R_cap == 2700.0)]
nm = pd.read_csv(NUMBERS)
V = {}
for a in ARMS:
    g = sub[sub.rx == a]
    assert len(g) == 8 and "".join(sorted(g.cell)) == "abcdefgh", a
    V[a] = (g.true_unserved.mean() / 1e3, g.n_trips.mean())
    for m, v in (("unserved_kWh", V[a][0]), ("trips", V[a][1])):
        ev = nm[(nm.section == "2 main") & (nm.metric == m) & (nm.arm == a)
                & (nm.R_cap.astype(str) == "2700.0")].value
        assert len(ev) == 1 and np.isclose(float(ev.iloc[0]), v,
                                           rtol=1e-12), (a, m)
print(f"source: {CSV} (primary_cross_arm, none, R_cap 2700, A-H); "
      f"cross-check vs {NUMBERS}: PASS")


def uns(kwh):
    return f"{kwh/1e3:,.1f} MWh" if kwh >= 1000 else f"{kwh:,.0f} kWh"


def shut(n):
    return "0 shutdowns" if n == 0 else f"{n:.1f} shutdowns"


# table style: coloured bold policy headers above contiguous
# policy-filled cells, thick black borders, centred numbers, no gaps
LINE = {"free": "#D98984", "reactive": "#4F9F91",
        "computing_only": "#6C8EBF", "predictive": "#A84440",
        "consolidated": "#9673A6"}
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
L, R, B, T = 0.006, 0.994, 0.155, 0.775
w = (R - L) / len(ARMS)
for i, a in enumerate(ARMS):
    x0 = L + i * w
    cx = x0 + w / 2
    ax.text(cx, T + 0.105, LABEL[a], ha="center", va="center",
            fontsize=21, fontweight="bold", color=LINE[a])
    ax.add_patch(plt.Rectangle((x0, B), w, T - B, facecolor=FILL[a],
                               edgecolor="black", lw=BAR_LW * 1.5))
    ax.text(cx, B + 0.64 * (T - B), uns(V[a][0]), ha="center",
            va="center", fontsize=32, fontweight="bold")
    ax.text(cx, B + 0.25 * (T - B), shut(V[a][1]), ha="center",
            va="center", fontsize=21)
    print(f"  {LABEL[a]:>16}: {uns(V[a][0]):>10} | {shut(V[a][1])}  "
          f"(exact {V[a][0]:,.4f} kWh, {V[a][1]:.3f} trips)")
ax.text(0.5, B - 0.085, "Mean across Borg A–H at low reactor margin, "
        "$R_\\mathrm{cap}$ = 2700 pcm.", ha="center", va="center",
        fontsize=16, fontweight="bold")
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{ext}", facecolor="white")
print(f"wrote {OUT}/{STEM}.png + .svg ({FIGSIZE[0]:.2f}x{FIGSIZE[1]:.2f} in)")
