#!/usr/bin/env python3
"""
plot_behavior_and_loss.py -- reactor behaviour and where the unserved work
occurs, from the minute-level traces written by
experiments/minute_traces.py (whose 24 summary rows must reproduce the
main-suite rows exactly).

behavior_cell_a: representative Cell A, days 10-15 (the day-11.5
    price-driven event, shared by all eight cells): xenon reactivity
    against the ceiling and reactor power, per reactor policy.
loss_timeline_AH: A-H MEAN instantaneous unserved power and its
    cumulative integral; endpoints validated against the main-suite A-H
    mean true_unserved.
Outputs (refuse to overwrite): figures/behavior_cell_a.*,
figures/loss_timeline_AH.*
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# inputs are read from, and figures written to, the data/ working directory
os.chdir(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data"))
os.makedirs("figures", exist_ok=True)

TR = "controller_runs/minute_traces"
CANON = "controller_runs/main_suite.csv"
NUMBERS = "analysis/reported_numbers.csv"
OUT = "figures"
CELL, D0, D1 = "a", 10.0, 15.0
ARMS = ("free", "reactive", "predictive")
CELLS = "abcdefgh"
DT = 60.0

FIGSIZE_TALL = (6.80, 4.60)
FS, LW, LW_REF, AXIS_LW = 18, 3.6, 2.8, 1.5
LINE = {"free": "#D98984", "reactive": "#4F9F91", "predictive": "#A84440"}
LABEL = {"free": "Limit-Agnostic", "reactive": "Limit-Reactive",
         "predictive": "Limit-Predictive"}
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 600, "font.size": FS,
    "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS, "axes.linewidth": AXIS_LW,
    "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
    "xtick.major.size": 5, "ytick.major.size": 5, "xtick.major.pad": 3,
    "ytick.major.pad": 3, "axes.labelpad": 4, "legend.frameon": True,
    "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "legend.borderpad": 0.25,
    "legend.labelspacing": 0.30, "legend.handletextpad": 0.40,
    "legend.columnspacing": 0.80, "legend.handlelength": 1.15,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42,
    "lines.solid_capstyle": "round", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white"})


DRAW = ("reactive", "predictive", "free")          # z-order: free last
LWD = {"free": LW * 0.45, "reactive": LW, "predictive": LW}
# mechanism trace only: Agnostic thin dashed light pink on top, so it stays visible
# where it coincides with Reactive (styling only; data untouched)
G1_LS = {"free": (0, (2.0, 1.2)), "reactive": "-", "predictive": "-"}
G1_LW = {"free": LW * 0.5, "reactive": LW, "predictive": LW}


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{stem}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}.png + .svg "
          f"(exact {fig.get_size_inches()[0]:.2f}x"
          f"{fig.get_size_inches()[1]:.2f} in)")


def clean(*axes):
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", width=AXIS_LW, length=5)


def pad(fig, h=0.06, w=0.06):
    fig.get_layout_engine().set(h_pad=h, w_pad=w)


def panel_letter(ax, letter, x=0.015, y=0.975):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS,
            fontweight="bold", ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="square,pad=0.15"), zorder=5)


for s in ("behavior_cell_a", "loss_timeline_AH"):
    for ext in ("png", "svg"):
        if os.path.exists(f"{OUT}/{s}.{ext}"):
            sys.exit(f"{OUT}/{s}.{ext} exists; refusing to overwrite.")

man = json.load(open(TR + "_manifest.json"))
assert man["reproduction"]["rows_reproduced"] == 24
T = {(c, a): np.load(f"{TR}/trace_{c}_{a}.npz") for c in CELLS for a in ARMS}
canon = pd.read_csv(CANON)
cn = canon[(canon.evidence_class == "primary_cross_arm") &
           (canon.scenario == "none") & (canon.R_cap == 2700.0)]
day = T[(CELL, "free")]["time_s"] / 86400.0
ceil = float(T[(CELL, "free")]["ceiling"][0])
assert np.all(T[(CELL, "free")]["ceiling"] == ceil)

# ================= representative mechanism trace (Cell A) ==============
print(f"G1 source: {TR}/trace_{CELL}_{{free,reactive,predictive}}.npz, "
      f"days {D0}-{D1}")
m = (day >= D0) & (day <= D1)
for a in ARMS:
    r = cn[(cn.cell == CELL) & (cn.rx == a)].iloc[0]
    rho = T[(CELL, a)]["rho"][m]
    trips = np.nonzero(np.diff(T[(CELL, a)]["tripped"].astype(int)) == 1)[0]
    in_w = [round(float(day[j + 1]), 2) for j in trips
            if D0 <= day[j + 1] <= D1]
    print(f"  {a:>10}: window max rho {rho.max():.1f} pcm, trips in window "
          f"{in_w}; canonical month: unserved {r.true_unserved/1e3:,.1f} kWh"
          f", trips {r.n_trips}, deadtime {r.deadtime_h:.1f} h")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True,
                               layout="constrained",
                               gridspec_kw={"height_ratios": [3, 2]})
ax1.axhline(ceil, color="black", ls="--", lw=LW_REF)
# Agnostic coincides with Reactive in the data: draw it last and thinner
# so the overlap is visible instead of hidden.
for a in DRAW:
    ax1.plot(day[m], T[(CELL, a)]["rho"][m], color=LINE[a], lw=G1_LW[a],
             ls=G1_LS[a])
    ax2.plot(day[m], T[(CELL, a)]["P_rx"][m] * 100.0, color=LINE[a],
             lw=G1_LW[a], ls=G1_LS[a])
# exactness: every plotted line is the unmodified trace slice
for ax_, key, sc in ((ax1, "rho", 1.0), (ax2, "P_rx", 100.0)):
    for ln, a in zip(ax_.get_lines()[-3:], DRAW):
        assert np.array_equal(ln.get_ydata(), T[(CELL, a)][key][m] * sc)
        assert np.array_equal(ln.get_xdata(), day[m])
print("  G1 plotted arrays identical to trace data: PASS")
ax1.set_ylabel("Xenon\nReactivity\n(pcm)")
ax2.set_ylabel("Reactor\nPower (%)")
ax2.set_xlabel("Day")
ax1.set_xlim(D0, D1)
_lo, _hi = ax1.get_ylim()
ax1.set_yticks([v for v in sorted({round(ceil), 4000.0, 6000.0})
                if _lo <= v <= _hi])
ax2.set_yticks([0, 50, 100])
_span = _hi - _lo
ax1.set_ylim(_lo - 0.04 * _span, _hi + 0.07 * _span)
handles = [Line2D([], [], color=LINE[a], lw=G1_LW[a], ls=G1_LS[a],
                  label=LABEL[a]) for a in ARMS]
handles.append(Line2D([], [], color="black", lw=LW_REF, ls="--",
                      label="Xenon Ceiling"))
fig.legend(handles=handles, loc="outside upper center", ncol=2)
fig.align_ylabels([ax1, ax2])
clean(ax1, ax2)
pad(fig, h=0.10)
save(fig, "behavior_cell_a")

# ================= A-H mean loss timeline ===============================
print(f"\nG2 source: {TR}/trace_<a..h>_<policy>.npz (all 24 traces)")
fig, (axa, axb) = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True,
                               layout="constrained")
nm = pd.read_csv(NUMBERS)
for a in DRAW:
    inst_W = np.mean([T[(c, a)]["unmet_W"] + T[(c, a)]["dropped_Wh"]
                      * 3600.0 / DT for c in CELLS], axis=0)
    step_Wh = inst_W * DT / 3600.0
    cum_Wh = np.cumsum(step_Wh)
    left = np.mean([cn[(cn.cell == c) & (cn.rx == a)].leftover.iloc[0]
                    for c in CELLS])
    cum_Wh[-1] += left              # work still queued at month end
    canon_mean = cn[cn.rx == a].true_unserved.mean()
    ev = nm[(nm.section == "2 main") & (nm.metric == "unserved_kWh") &
            (nm.arm == a) & (nm.R_cap.astype(str) == "2700.0")].value.iloc[0]
    ok = np.isclose(cum_Wh[-1], canon_mean, rtol=1e-9) and \
        np.isclose(cum_Wh[-1] / 1e3, ev, rtol=1e-9)
    print(f"  {a:>10}: A-H cumulative endpoint {cum_Wh[-1]/1e3:,.4f} kWh | "
          f"canonical A-H mean {canon_mean/1e3:,.4f} kWh | evidence "
          f"{ev:,.4f} kWh | {'MATCH' if ok else 'MISMATCH'}")
    assert ok, a
    axa.plot(day, inst_W / 1e6, color=LINE[a], lw=LWD[a])
    axb.plot(day, cum_Wh / 1e3, color=LINE[a], lw=LWD[a])
axa.set_ylabel("Unserved\nPower (MW)")
axb.set_yscale("symlog", linthresh=10)
axb.set_ylabel("Cumulative\nUnserved\n(kWh)")
axb.set_xlabel("Day")
axa.set_xlim(0, float(day[-1]))
panel_letter(axa, "(a)")
panel_letter(axb, "(b)", x=0.10, y=0.66)
fig.legend(handles=[Line2D([], [], color=LINE[a], lw=LWD[a],
                           label=LABEL[a]) for a in ARMS],
           loc="outside upper center", ncol=2)
fig.align_ylabels([axa, axb])
clean(axa, axb)
pad(fig, h=0.08)
save(fig, "loss_timeline_AH")
