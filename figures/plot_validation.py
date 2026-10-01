#!/usr/bin/env python3
"""
plot_validation.py -- one wide three-column validation graphic
(lookahead | enforcement buffer | model mismatch) from COMPLETED results
only (no simulation; sim.py not imported).

Sources (read only):
  controller_runs/horizon_sensitivity.csv   predictive, R_cap 2700
  controller_runs/eps_sensitivity.csv       predictive, R_cap 2700
  controller_runs/model_mismatch.csv        224 runs, R_cap 2700
Anchors: horizon 10 h and eps 20 pcm == canonical predictive A-H means.
Output (refuses to overwrite): figures/validation.*
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
EP = "controller_runs/eps_sensitivity"
MM = "controller_runs/model_mismatch"
CANON = "controller_runs/main_suite.csv"
OUT, STEM = "figures", "validation"
RED = "#A84440"                    # Limit-Predictive accent
FIGSIZE = (13.60, 3.30)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 600,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
for e in ("png", "svg"):
    if os.path.exists(f"{OUT}/{STEM}.{e}"):
        sys.exit(f"{OUT}/{STEM}.{e} exists; refusing to overwrite.")


def verified(stem):
    man = json.load(open(stem + "_manifest.json"))
    for k, o in man["outputs"].items():
        assert hashlib.sha256(open(o["path"], "rb").read()).hexdigest() \
            == o["sha256"], (stem, k)
    return pd.read_csv(stem + ".csv"), man


def ah(df, col, val):
    g = df[(df.rx == "predictive") & (df.R_cap == 2700.0) &
           (df.scenario == "none") & (df[col] == val)]
    assert len(g) == 8 and "".join(sorted(g.cell)) == "abcdefgh", (col, val)
    return g.true_unserved.mean() / 1e3, g.n_trips.mean(), \
        int((g.n_trips > 0).sum())


c = pd.read_csv(CANON)
c = c[(c.evidence_class == "primary_cross_arm") & (c.scenario == "none") &
      (c.R_cap == 2700.0) & (c.rx == "predictive")]
canon = (c.true_unserved.mean() / 1e3, c.n_trips.mean())

hz, _ = verified(HZ)
Hs = [0.0, 2.0, 4.0, 6.0, 10.0, 15.0]
HV = {h: ah(hz, "predict_horizon_h", h) for h in Hs}
ep, _ = verified(EP)
Es = [0.0, 10.0, 20.0, 50.0, 100.0]
EV = {e: ah(ep, "eps", e) for e in Es}
for name, v in (("horizon 10 h", HV[10.0]), ("eps 20 pcm", EV[20.0])):
    assert np.isclose(v[0], canon[0], rtol=1e-12) and v[1] == canon[1], name
mm, mman = verified(MM)
assert len(mm) == 224 == mman["runs"] and mman["conservation_failures"] == 0
assert not mm[["mismatch_case", "model_condition", "rx", "cell"]] \
    .duplicated().any()
mm_trips, mm_cross = int(mm.n_trips.max()), int(mm.crossings.max())
mmm = mm[mm.model_condition == "mismatched"]
worst = mmm.loc[mmm.min_headroom.idxmin()]
consumed = 20.0 - worst.min_headroom
assert mm_trips == 0 and mm_cross == 0

print("LOOKAHEAD (predictive, R_cap 2700, A-H mean):")
for h in Hs:
    print(f"  {h:4.0f} h: {HV[h][0]:14,.4f} kWh | shutdowns {HV[h][1]:.3f} "
          f"({HV[h][2]}/8 cells)")
print("ENFORCEMENT BUFFER (predictive, R_cap 2700, A-H mean):")
for e in Es:
    print(f"  {e:5.0f} pcm: {EV[e][0]:14,.4f} kWh | shutdowns "
          f"{EV[e][1]:.3f} ({EV[e][2]}/8 cells)")
print(f"anchors: horizon 10 h and eps 20 pcm == canonical predictive "
      f"({canon[0]:,.4f} kWh, {canon[1]:.0f}): PASS")
print(f"MODEL MISMATCH: {len(mm)} runs (7 cases x 2 arms x 8 cells x "
      f"matched/mismatched); max shutdowns {mm_trips}, max crossing min "
      f"{mm_cross}; worst mismatched buffer use {consumed:.1f} of 20 pcm "
      f"({worst.mismatch_case}, {worst.rx}, cell {worst.cell}); cases "
      f"{sorted(mm.mismatch_case.unique())}")
assert all(HV[h][1] == 0 for h in Hs[1:]) and HV[0.0][1] > 0
assert all(EV[e][1] == 0 for e in Es[1:]) and EV[0.0][1] > 0

# ---------------- graphic ----------------
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
L, R, B, T = 0.004, 0.996, 0.02, 0.98
w = (R - L) / 3
ax.add_patch(plt.Rectangle((L, B), R - L, T - B, facecolor="white",
                           edgecolor="black", lw=3.0))
for i in (1, 2):
    ax.plot([L + i * w] * 2, [B, T], color="black", lw=3.0)
ax.plot([L, R], [0.80, 0.80], color="black", lw=1.4)

Y = dict(head=0.89, c1=0.715, v1=0.60, arrow=0.47, c2=0.365, v2=0.25,
         foot=0.095)


def col(i, head, c1, v1, c2, v2, foot, v1_red=True):
    cx = L + (i + 0.5) * w
    ax.text(cx, Y["head"], head, ha="center", va="center", fontsize=17,
            fontweight="bold")
    ax.text(cx, Y["c1"], c1, ha="center", va="center", fontsize=16)
    ax.text(cx, Y["v1"], v1, ha="center", va="center", fontsize=27,
            fontweight="bold", color=RED if v1_red else "black")
    if c2 is not None:
        ax.text(cx, Y["arrow"], "↓", ha="center", va="center", fontsize=22)
        ax.text(cx, Y["c2"], c2, ha="center", va="center", fontsize=16)
        ax.text(cx, Y["v2"], v2, ha="center", va="center", fontsize=27,
                fontweight="bold")
    ax.text(cx, Y["foot"], foot, ha="center", va="center", fontsize=13.5,
            style="italic")


col(0, "LOOKAHEAD", "0 h", f"{HV[0.0][1]:g} shutdowns",
    "2 h  (2–15 h tested)", "0 shutdowns",
    "10 h used in main evaluation")
col(1, "ENFORCEMENT BUFFER", "0 pcm", f"{EV[0.0][1]:.2f} shutdowns",
    "≥ 10 pcm tested", "0 shutdowns",
    "20 pcm used in main evaluation")
# model mismatch: no failure condition -> conditions, then the result
cx = L + 2.5 * w
ax.text(cx, Y["head"], "MODEL MISMATCH", ha="center", va="center",
        fontsize=17, fontweight="bold")
ax.text(cx, 0.69, "±5% decay constants  ·  ±10% flux", ha="center",
        va="center", fontsize=16)
ax.text(cx, 0.595, "DOE alternative half-lives", ha="center", va="center",
        fontsize=16)
ax.text(cx, 0.40, "0 shutdowns", ha="center", va="center", fontsize=30,
        fontweight="bold")
ax.text(cx, 0.265, "across Borg A–H", ha="center", va="center",
        fontsize=16)
ax.text(cx, Y["foot"], f"worst tested case used {consumed:.1f} of the "
        f"20-pcm buffer", ha="center", va="center", fontsize=13.5,
        style="italic")
for e in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{e}", facecolor="white")
print(f"wrote {OUT}/{STEM}.png + .svg ({FIGSIZE[0]:.2f}x{FIGSIZE[1]:.2f} in)")
