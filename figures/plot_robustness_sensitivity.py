#!/usr/bin/env python3
"""
plot_robustness_sensitivity.py -- two-panel sensitivity figure
(reactor shutdowns vs prediction lookahead | vs enforcement buffer) plus one
model-mismatch line, from COMPLETED experiments only (sim.py not imported).

Sources (read only, manifests verified):
  controller_runs/horizon_sensitivity.csv  predictive, R_cap 2700
  controller_runs/eps_sensitivity.csv      predictive, R_cap 2700
  controller_runs/model_mismatch.csv       all 224 runs
Anchors: 10 h and 20 pcm == canonical predictive A-H means.
Output (refuses to overwrite): figures/robustness_sensitivity.*
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
OUT, STEM = "figures", "robustness_sensitivity"
FS = 24                                   # the ONE font size
LW, AX_LW, MS, MEW = 5.0, 3.0, 17, 2.8    # ~1.5-2x the other figures
RED = "#A84440"                           # Limit-Predictive line colour
FIGSIZE = (15.2, 4.3)
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "axes.linewidth": AX_LW,
    "xtick.major.width": AX_LW, "ytick.major.width": AX_LW,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white"})
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
    return g.n_trips.mean(), g.true_unserved.mean() / 1e3


c = pd.read_csv(CANON)
c = c[(c.evidence_class == "primary_cross_arm") & (c.scenario == "none") &
      (c.R_cap == 2700.0) & (c.rx == "predictive")]
canon = (c.n_trips.mean(), c.true_unserved.mean() / 1e3)
hz, _ = verified(HZ)
ep, _ = verified(EP)
mm, mman = verified(MM)
H = [0, 2, 4, 6, 10, 15]
E = [0, 10, 20, 50, 100]
HV = {h: ah(hz, "predict_horizon_h", h) for h in H}
EV = {e: ah(ep, "eps", e) for e in E}
print("lookahead shutdown means:", {h: HV[h][0] for h in H})
print("buffer shutdown means:   ", {e: EV[e][0] for e in E})
for name, v in (("10 h", HV[10]), ("20 pcm", EV[20])):
    assert v[0] == canon[0] and np.isclose(v[1], canon[1], rtol=1e-12), name
print(f"anchors 10 h / 20 pcm == canonical predictive ({canon[0]:.0f} "
      f"shutdowns, {canon[1]:.4f} kWh): PASS")
assert len(mm) == 224 == mman["runs"]
per_case = mm.groupby("mismatch_case").n_trips.max()
print("model mismatch max shutdowns per case:", per_case.to_dict())
assert (per_case == 0).all() and mm.crossings.max() == 0

# ---------------- figure ----------------
fig = plt.figure(figsize=FIGSIZE)
L, R, GAP, B, T = 0.078, 0.99, 0.06, 0.475, 0.965
wpan = (R - L - GAP) / 2
axa = fig.add_axes([L, B, wpan, T - B])
axb = fig.add_axes([L + wpan + GAP, B, wpan, T - B], sharey=axa)
panels = ((axa, H, HV, 10, "Prediction Lookahead (h)", "(a)"),
          (axb, E, EV, 20, "Enforcement Buffer (pcm)", "(b)"))
for ax, xs, V, used, xlabel, letter in panels:
    # tested values at equal spacing, labelled with their actual values
    pos = list(range(len(xs)))
    ys = [V[x][0] for x in xs]
    ax.plot(pos, ys, color=RED, lw=LW, zorder=3, clip_on=False)
    ax.plot(pos, ys, ls="none", marker="o", ms=MS, mfc=RED, mec="black",
            mew=MEW, zorder=4, clip_on=False)
    # canonical setting: hollow black ring + "used"
    pu = xs.index(used)
    ax.plot([pu], [V[used][0]], ls="none", marker="o", ms=MS + 13,
            mfc="none", mec="black", mew=MEW, zorder=5, clip_on=False)
    ax.annotate("used", xy=(pu, V[used][0]), xytext=(0, 26),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=FS)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.tick_params(axis="x", pad=18)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_xlim(-0.3, len(xs) - 0.7)
    ax.text(0.985, 0.95, letter, transform=ax.transAxes, ha="right",
            va="top", fontsize=FS, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
axa.set_ylim(0, 6.0)
axa.set_yticks([0, 2, 4, 6])
axa.set_ylabel("Reactor\nShutdowns")
plt.setp(axb.get_yticklabels(), visible=False)
fig.text(0.5 * (L + R), 0.025,
         "MODEL MISMATCH: 0 shutdowns across Borg A–H under all tested\n"
         "±5% decay-rate and ±10% flux sensitivities.",
         ha="center", va="bottom", fontsize=FS, linespacing=1.15)
for e in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{e}", facecolor="white")
print(f"wrote {OUT}/{STEM}.png + .svg ({FIGSIZE[0]:.2f} x {FIGSIZE[1]:.2f} "
      f"in, {round(FIGSIZE[0]*300)} x {round(FIGSIZE[1]*300)} px)")
