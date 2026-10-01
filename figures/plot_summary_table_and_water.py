#!/usr/bin/env python3
"""
plot_summary_table_and_water.py -- summary table and water figure from the
main A-H evaluation (no simulation).

Source (read only): controller_runs/main_suite.csv
  evidence_class == "primary_cross_arm", scenario == "none", R_cap == 2700
  (all 8 Borg cells A-H per arm). Values are cross-checked against
  analysis/reported_numbers.csv before anything is drawn.
Outputs (refuses to overwrite):
  figures/summary_table.{png,svg}
  figures/water.{png,svg}
Run from the repository root (works in data/).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# inputs are read from, and figures written to, the data/ working directory
os.chdir(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data"))
os.makedirs("figures", exist_ok=True)

CSV = "controller_runs/main_suite.csv"
NUMBERS = "analysis/reported_numbers.csv"
OUT = "figures"
STEMS = ("summary_table", "water")

# ---------------- style ----------------
FIGSIZE_TALL = (6.80, 4.60)
FIGSIZE_TABLE = (6.80, 3.60)
FS, FS_TABLE, FS_HDR = 18, 14, 13
BAR_LW, AXIS_LW = 2.0, 1.5
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
    "patch.linewidth": BAR_LW, "hatch.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
# policy fills / header colours; computing_only and consolidated use
# the fills already chosen for multicell_unserved
FILL = {"free": "#F8CECC", "reactive": "#7DC4B4", "predictive": "#B85450",
        "computing_only": "#A9C4EB", "consolidated": "#E1D5E7"}
LINE = {"free": "#D98984", "reactive": "#4F9F91", "predictive": "#A84440",
        "computing_only": "#6C8EBF", "consolidated": "#9673A6"}
LABEL = {"free": "Limit-Agnostic", "reactive": "Limit-Reactive",
         "computing_only": "Computing-Only",
         "predictive": "Limit-Predictive", "consolidated": "Consolidated"}
ARMS = ["free", "reactive", "computing_only", "predictive", "consolidated"]


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{stem}.{ext}", facecolor="white")
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  wrote {OUT}/{stem}.png + .svg (exact {w:.2f}x{h:.2f} in)")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=AXIS_LW, length=5)


for s in STEMS:
    for ext in ("png", "svg"):
        if os.path.exists(f"{OUT}/{s}.{ext}"):
            sys.exit(f"{OUT}/{s}.{ext} exists; refusing to overwrite.")

# ---------------- data ----------------
d = pd.read_csv(CSV)
sub = d[(d.evidence_class == "primary_cross_arm") & (d.scenario == "none")
        & (d.R_cap == 2700.0) & d.rx.isin(ARMS)]
print(f"source: {CSV}\n  subset primary_cross_arm & scenario=none & "
      f"R_cap=2700 & rx in {ARMS}: {len(sub)} rows")
V = {}
for a in ARMS:
    g = sub[sub.rx == a]
    assert len(g) == 8 and "".join(sorted(g.cell)) == "abcdefgh", a
    V[a] = dict(
        uns_kWh=g.true_unserved.mean() / 1e3, trips=g.n_trips.mean(),
        trip_cells=int((g.n_trips > 0).sum()),
        onsite=g.water_L.sum() / (g.served_it_Wh.sum() / 1e3),
        grid=g.water_grid_L.sum() / (g.served_it_Wh.sum() / 1e3),
        dc=g.water_dc_L.sum() / (g.served_it_Wh.sum() / 1e3),
        total=g.water_sys_L.sum() / (g.served_it_Wh.sum() / 1e3))
    assert np.isclose(V[a]["onsite"] + V[a]["grid"] + V[a]["dc"],
                      V[a]["total"], rtol=1e-12)

# cross-check against the frozen evidence numbers
nm = pd.read_csv(NUMBERS)
def ev(section, metric, arm):
    x = nm[(nm.section == section) & (nm.metric == metric) & (nm.arm == arm)
           & (nm.R_cap.astype(str) == "2700.0")]    # column mixes labels
    assert len(x) == 1, (section, metric, arm)
    return float(x.value.iloc[0])
for a in ARMS:
    assert np.isclose(V[a]["uns_kWh"], ev("2 main", "unserved_kWh", a),
                      rtol=1e-12), a
    assert np.isclose(V[a]["trips"], ev("2 main", "trips", a), rtol=1e-12)
    assert np.isclose(V[a]["total"], ev("6 grid+water", "L_per_kWh_pooled",
                                        a), rtol=1e-12), a
print("  cross-check vs analysis/reported_numbers.csv: PASS")
print(f"\n  {'arm':>15} | unserved kWh | trips (cells) | onsite | grid | "
      f"DC | system L/kWh served")
for a in ARMS:
    v = V[a]
    print(f"  {a:>15} | {v['uns_kWh']:12,.1f} | {v['trips']:.3f} "
          f"({v['trip_cells']}/8) | {v['onsite']:.3f} | {v['grid']:.3f} | "
          f"{v['dc']:.3f} | {v['total']:.3f}")


def fmt_uns(kwh):
    return f"{kwh/1e3:,.1f} MWh" if kwh >= 1000 else f"{kwh:,.0f} kWh"


# ================= summary table ========================================
fig, ax = plt.subplots(figsize=FIGSIZE_TABLE)
ax.set_axis_off()
left, right, bottom, top = 0.345, 0.985, 0.025, 0.855
col_w = (right - left) / 2.0
row_h = (top - bottom) / len(ARMS)
for j, h in enumerate(("Unserved Work", "Reactor Trips")):
    ax.text(left + (j + 0.5) * col_w, top + 0.075, h, ha="center",
            va="center", fontsize=FS_HDR + 1, fontweight="bold")
ax.text(0.01, top + 0.075, "Mean, Borg A–H\n$R_\\mathrm{cap}$ = 2700 pcm",
        ha="left", va="center", fontsize=FS_HDR - 2, linespacing=1.1)
for i, a in enumerate(ARMS):
    y = top - (i + 1) * row_h
    ax.text(left - 0.015, y + row_h / 2.0, LABEL[a], ha="right",
            va="center", fontsize=FS_HDR, fontweight="bold", color=LINE[a])
    for j, txt in enumerate((fmt_uns(V[a]["uns_kWh"]),
                             f"{V[a]['trips']:.1f}" if V[a]["trips"] else "0")):
        ax.add_patch(plt.Rectangle((left + j * col_w, y), col_w, row_h,
                                   facecolor=FILL[a], edgecolor="black",
                                   lw=BAR_LW))
        ax.text(left + (j + 0.5) * col_w, y + row_h / 2.0, txt,
                ha="center", va="center", fontsize=FS_TABLE + 2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
print()
save(fig, "summary_table")

# ================= water: service-normalised system water ================
COMP = [("onsite", "On-Site SMR Cooling", FILL["free"], ".."),
        ("grid", "Grid Generation", FILL["reactive"], "//"),
        ("dc", "Data Center", "white", "--")]
fig, ax = plt.subplots(figsize=FIGSIZE_TALL, layout="constrained")
ys = np.arange(len(ARMS))[::-1]
for y, a in zip(ys, ARMS):
    x0 = 0.0
    for key, _, fc, hat in COMP:
        v = V[a][key]
        ax.barh(y, v, 0.62, left=x0, facecolor=fc, edgecolor="black",
                lw=BAR_LW, hatch=hat)
        x0 += v
    ax.annotate(f"{V[a]['total']:.1f}", xy=(x0, y), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=FS)
ax.set_yticks(ys)
ax.set_yticklabels([LABEL[a] for a in ARMS])
ax.set_xlabel("Water Intensity (L/kWh served)")
ax.set_xlim(0, max(V[a]["total"] for a in ARMS) * 1.16)
ax.set_ylim(-0.6, len(ARMS) - 0.4)
fig.legend(handles=[Patch(facecolor=fc, edgecolor="black", hatch=hat,
                          label=lab) for _, lab, fc, hat in COMP],
           loc="outside upper center", ncol=3, fontsize=FS - 3,
           handlelength=1.4)
clean(ax)
eng = fig.get_layout_engine()
eng.set(h_pad=0.08, w_pad=0.06)
save(fig, "water")
print("\nNOTE: data-center WUE = 1.1 L/kWh (sim.WUE_L_PER_KWH) is an "
      "assumed value; cite its source when reporting water results.")
