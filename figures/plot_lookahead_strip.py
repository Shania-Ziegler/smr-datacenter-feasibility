#!/usr/bin/env python3
"""
plot_lookahead_strip.py -- lookahead result strip from the
COMPLETED prediction-horizon sweep (no simulation; sim.py not imported).

Source (read only): controller_runs/horizon_sensitivity.csv
  predictive, R_cap 2700, scenario none, cells A-H, eps 20 pcm.
Checks: sweep outputs match their manifest; means match the sweep summary;
10 h equals the canonical predictive A-H result.
Output (never overwritten): figures/lookahead_strip.*
"""
import hashlib
import json
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

HZ = "controller_runs/horizon_sensitivity"
CANON = "controller_runs/main_suite.csv"
OUT, STEM = "figures", "lookahead_strip"
H = [0.0, 2.0, 4.0, 6.0, 10.0, 15.0]
FAIL = "#B85450"        # Limit-Predictive fill
STABLE = "#F3DEDD"      # light tint of the same identity
LINE = "#A84440"        # Limit-Predictive line colour
FIGSIZE = (13.60, 2.70)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 600,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
for e in ("png", "svg"):
    if os.path.exists(f"{OUT}/{STEM}.{e}"):
        sys.exit(f"{OUT}/{STEM}.{e} exists; refusing to overwrite.")

man = json.load(open(HZ + "_manifest.json"))
for k, o in man["outputs"].items():
    assert hashlib.sha256(open(o["path"], "rb").read()).hexdigest() == \
        o["sha256"], k
d = pd.read_csv(HZ + ".csv")
sel = d[(d.rx == "predictive") & (d.R_cap == 2700.0) & (d.scenario == "none")]
assert (sel.eps == 20.0).all()
summ = pd.read_csv(HZ + "_sweep_summary.csv")
M = {}
for h in H:
    g = sel[sel.predict_horizon_h == h]
    assert len(g) == 8 and "".join(sorted(g.cell)) == "abcdefgh", h
    M[h] = (g.true_unserved.mean() / 1e3, g.n_trips.mean())
    s = summ[(summ.rx == "predictive") & (summ.R_cap == 2700.0) &
             (summ.predict_horizon_h == h)].iloc[0]
    assert np.isclose(s.true_unserved_mean / 1e3, M[h][0], rtol=1e-12)
    assert np.isclose(s.n_trips_mean, M[h][1], rtol=1e-12)
c = pd.read_csv(CANON)
c = c[(c.evidence_class == "primary_cross_arm") & (c.scenario == "none") &
      (c.R_cap == 2700.0) & (c.rx == "predictive")]
assert np.isclose(c.true_unserved.mean() / 1e3, M[10.0][0], rtol=1e-12)
assert c.n_trips.mean() == M[10.0][1]
print("A-H means, Limit-Predictive, R_cap 2700, nominal:")
for h in H:
    print(f"  {h:4.0f} h: unserved {M[h][0]:14,.4f} kWh | shutdowns "
          f"{M[h][1]:.3f}")
print("summary cross-check PASS; 10 h == canonical predictive "
      f"({c.true_unserved.mean()/1e3:,.4f} kWh, {c.n_trips.mean():.0f}) PASS")


def uns(kwh):
    return f"{kwh/1e3:,.0f} MWh" if kwh >= 1000 else f"{kwh:,.0f} kWh"


fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
L0, L1, R = 0.004, 0.155, 0.996          # label column | data columns
B, T = 0.035, 0.80
rows = [T, T - 0.22, T - 0.22 - (T - 0.22 - B) / 2, B]  # header/uns/shut
w = (R - L1) / len(H)
xs = [L1 + i * w for i in range(len(H) + 1)]
# fills: failure column vs one stable region
ax.add_patch(plt.Rectangle((xs[0], B), w, rows[1] - B, facecolor=FAIL,
                           edgecolor="none"))
ax.add_patch(plt.Rectangle((xs[1], B), R - xs[1], rows[1] - B,
                           facecolor=STABLE, edgecolor="none"))
# grid: thin internal lines, heavy frame and heavy 0 h | 2 h divider
for x in xs[2:-1]:
    ax.plot([x, x], [B, T], color="black", lw=1.2)
for y in rows[1:3]:
    ax.plot([L0, R], [y, y], color="black", lw=1.6)
ax.plot([L1, L1], [B, T], color="black", lw=3.0)
ax.plot([xs[1], xs[1]], [B, T], color="black", lw=5.5)
ax.add_patch(plt.Rectangle((L0, B), R - L0, T - B, facecolor="none",
                           edgecolor="black", lw=3.0))
# row labels
for y0, y1, t in ((rows[1], rows[0], "LOOKAHEAD"),
                  (rows[2], rows[1], "UNSERVED"),
                  (rows[3], rows[2], "SHUTDOWNS")):
    ax.text((L0 + L1) / 2, (y0 + y1) / 2, t, ha="center", va="center",
            fontsize=16, fontweight="bold")
# cells
for i, h in enumerate(H):
    cx = xs[i] + w / 2
    kwh, n = M[h]
    fail, first_ok = i == 0, i == 1
    ax.text(cx, (rows[0] + rows[1]) / 2, f"{h:g} h", ha="center",
            va="center", fontsize=21, fontweight="bold")
    ax.text(cx, (rows[1] + rows[2]) / 2, uns(kwh), ha="center",
            va="center", fontsize=26 if fail else 24 if first_ok else 20,
            fontweight="bold" if (fail or first_ok) else "normal")
    ax.text(cx, (rows[2] + rows[3]) / 2, f"{n:g}", ha="center",
            va="center", fontsize=30 if fail else 22,
            fontweight="bold" if (fail or first_ok) else "normal")
# one short annotation over the stable region
ax.annotate("", xy=(xs[1] + 0.012, 0.905), xytext=(xs[0] + w / 2, 0.905),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=LINE,
                            mutation_scale=18))
ax.text(xs[1] + 0.018, 0.905, "Observed shutdowns eliminated by 2 h",
        ha="left", va="center", fontsize=16, fontweight="bold", color=LINE)
for e in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{e}", facecolor="white")
print(f"wrote {OUT}/{STEM}.png + .svg ({FIGSIZE[0]:.2f}x{FIGSIZE[1]:.2f} in)")
