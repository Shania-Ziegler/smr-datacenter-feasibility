#!/usr/bin/env python3
"""
plot_lookahead.py -- "How much lookahead is needed?" figure
from the COMPLETED prediction-horizon sweep (no simulation; sim.py is not
imported).

Source (read only): controller_runs/horizon_sensitivity.csv
  rx == predictive, R_cap == 2700, scenario none, cells A-H,
  predict_horizon_h in {0, 2, 4, 6, 10, 15}; eps fixed at 20 pcm.
Cross-checks: the sweep summary file, and the 10 h anchor against the
canonical predictive row set in controller_runs/main_suite.csv.
Output (versioned, never overwritten): figures/lookahead.*
"""
import hashlib
import json
import os

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
OUT, STEM = "figures", "lookahead"
H = [0.0, 2.0, 4.0, 6.0, 10.0, 15.0]
PRED = "#A84440"                    # Limit-Predictive line colour
FIGSIZE = (10.00, 3.40)
FS, AXIS_LW = 18, 1.5
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "axes.linewidth": AXIS_LW,
    "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 600,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white"})

stem = STEM
v = 1
while any(os.path.exists(f"{OUT}/{stem}.{e}") for e in ("png", "svg")):
    v += 1
    stem = f"{STEM}_v{v}"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- provenance: the sweep outputs are exactly the manifest's ----
man = json.load(open(HZ + "_manifest.json"))
for k, o in man["outputs"].items():
    assert sha(o["path"]) == o["sha256"], k
d = pd.read_csv(HZ + ".csv")
sel = d[(d.rx == "predictive") & (d.R_cap == 2700.0) & (d.scenario == "none")]
assert (sel.eps == 20.0).all()
print(f"source rows: {HZ}.csv, rx=predictive, R_cap=2700, scenario=none")
print(sel.sort_values(["predict_horizon_h", "cell"])[
    ["predict_horizon_h", "cell", "true_unserved", "n_trips",
     "deadtime_h", "crossings"]].to_string(index=False))

M = {}
for h in H:
    g = sel[sel.predict_horizon_h == h]
    assert len(g) == 8 and "".join(sorted(g.cell)) == "abcdefgh", h
    M[h] = (g.true_unserved.mean() / 1e3, g.n_trips.mean(),
            int((g.n_trips > 0).sum()))
summ = pd.read_csv(HZ + "_sweep_summary.csv")
for h in H:
    s = summ[(summ.rx == "predictive") & (summ.R_cap == 2700.0) &
             (summ.predict_horizon_h == h)].iloc[0]
    assert np.isclose(s.true_unserved_mean / 1e3, M[h][0], rtol=1e-12)
    assert np.isclose(s.n_trips_mean, M[h][1], rtol=1e-12)
c = pd.read_csv(CANON)
c = c[(c.evidence_class == "primary_cross_arm") & (c.scenario == "none") &
      (c.R_cap == 2700.0) & (c.rx == "predictive")]
anchor_ok = (np.isclose(c.true_unserved.mean() / 1e3, M[10.0][0],
                        rtol=1e-12) and c.n_trips.mean() == M[10.0][1])
assert anchor_ok
print("\nA-H means (predictive, R_cap 2700):")
for h in H:
    print(f"  H = {h:4.0f} h: unserved {M[h][0]:14,.4f} kWh | shutdowns "
          f"{M[h][1]:.3f} ({M[h][2]}/8 cells)")
print(f"summary-file cross-check: PASS; 10 h anchor vs canonical "
      f"({c.true_unserved.mean()/1e3:,.4f} kWh, {c.n_trips.mean():.3f}): "
      f"PASS")
r35 = d[(d.rx == "predictive") & (d.R_cap == 3500.0)]
print("R_cap 3500 (not plotted): " + ", ".join(
    f"{h:g} h {r35[r35.predict_horizon_h == h].true_unserved.mean()/1e3:.2f}"
    f" kWh/{r35[r35.predict_horizon_h == h].n_trips.mean():.0f}"
    for h in H))


def lab(kwh):
    return f"{kwh/1e3:,.0f} MWh" if kwh >= 1000 else f"{kwh:,.0f} kWh"


# ---- figure: log unserved vs lookahead + slim shutdown strip ----
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0.15, 0.36, 0.83, 0.60])
st = fig.add_axes([0.15, 0.03, 0.83, 0.10], sharex=ax)
y = [M[h][0] for h in H]
ax.plot(H, y, color=PRED, lw=3.6, marker="o", ms=11, mfc=PRED,
        mec="black", mew=1.8, zorder=3)
for h, v_ in zip(H, y):
    first = h == H[0]           # the 0 h label sits right of its point
    ax.annotate(lab(v_), xy=(h, v_), xytext=(14, 0) if first else (0, 11),
                textcoords="offset points", ha="left" if first else "center",
                va="center" if first else "bottom", fontsize=16)
ax.set_yscale("log")
ax.set_ylim(60, 8e5)
ax.set_yticks([1e2, 1e3, 1e4, 1e5])
ax.set_ylabel("Mean Unserved\nWork (kWh)")
ax.set_xticks(H)
ax.set_xticklabels([f"{h:g}" for h in H])
ax.set_xlim(-0.8, 15.8)
ax.set_xlabel("Lookahead (hours)", labelpad=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out")
st.set_axis_off()
st.set_ylim(0, 1)
st.text(-0.95, 0.5, "Shutdowns", ha="right", va="center", fontsize=16,
        fontweight="bold", transform=st.transData)
for h in H:
    n = M[h][1]
    st.text(h, 0.5, f"{n:g}", ha="center", va="center", fontsize=17,
            fontweight="bold", color=PRED if n else "black")
for e in ("png", "svg"):
    fig.savefig(f"{OUT}/{stem}.{e}", facecolor="white")
print(f"wrote {OUT}/{stem}.png + .svg ({FIGSIZE[0]:.2f}x{FIGSIZE[1]:.2f} in)")
