#!/usr/bin/env python3
"""
plot_safe_grid_flexibility.py -- "How much of a requested grid
maneuver can the reactor-feasibility layer safely expose?" figure from the
COMPLETED partial-grid feasibility sweep (no simulation; sim.py is not
imported).

Source (read only): controller_runs/partial_grid_feasibility.csv
  rx == consolidated_partial_grid, R_cap 2700, scenario none, cells A-H,
  G in {0, 0.2, 0.4, 0.6, 0.8, 1.0}.
Plotted (A-H means, MWh per 31-day trace):
  requested = price_requested_Wh (cheap-price import after the G cap,
              before the feasibility layer)
  accepted  = price_approved_Wh  (largest projected-safe import found)
Cross-checks: sweep manifest hashes, summary CSV, stage-1 G = 0.4 rows.
Output (refuses to overwrite):
  figures/safe_grid_flexibility.{svg,png}
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
SW = "controller_runs/partial_grid_feasibility"
G04 = "controller_runs/partial_grid_feasibility_g04"
OUT, STEM = "figures", "safe_grid_flexibility"
G = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ARM = "consolidated_partial_grid"
C_REQ = "#7F7F7F"                  # neutral: the scheduler's request
C_ACC = "#A84440"                  # Limit-Predictive line colour
FIGSIZE = (10.0, 3.9)
FS, AXIS_LW, LINE_LW, MS, MEW = 20, 1.8, 3.4, 10, 1.8
TIMING = ("policy_", "projection_")
LABELS = ("suite", "evidence_class", "sweep_parameter", "experiment")
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": AXIS_LW, "axes.edgecolor": "black",
    "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.pad": 3, "ytick.major.pad": 3, "axes.labelpad": 5,
    "legend.frameon": True, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.35, "legend.labelspacing": 0.35,
    "legend.handletextpad": 0.5, "legend.handlelength": 2.4,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 600,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white"})

for e in ("png", "svg"):
    if os.path.exists(f"{OUT}/{STEM}.{e}"):
        sys.exit(f"refusing to overwrite {OUT}/{STEM}.{e}")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- provenance ----
src = [SW + ".csv", SW + "_summary.csv", SW + "_manifest.json",
       G04 + ".csv", G04 + "_manifest.json"]
src_hash = {p: sha(p) for p in src}
for m in (SW, G04):
    man = json.load(open(m + "_manifest.json"))
    for k, o in man["outputs"].items():
        assert sha(o["path"]) == o["sha256"], (m, k)
    assert man["conservation_failures"] == 0
    assert man["reference_reproduction"]["rows_identical"] == \
        man["reference_reproduction"]["rows_compared"]

d = pd.read_csv(SW + ".csv")
p = d[d.rx == ARM]
assert len(p) == 48 and not p.duplicated(["G", "cell"]).any()
assert (p.R_cap == 2700.0).all() and (p.scenario == "none").all()
assert sorted(p.G.unique()) == G
M = {}
for g in G:
    s = p[p.G == g]
    assert len(s) == 8 and "".join(sorted(s.cell)) == "abcdefgh", g
    assert (s.n_trips == 0).all() and (s.crossings == 0).all(), g
    assert (s.energy_residual_Wh.abs() <= 1.0).all() and \
        (s.residual.abs() <= 1.0).all(), g
    M[g] = dict(req=s.price_requested_Wh.mean() / 1e6,
                acc=s.price_approved_Wh.mean() / 1e6,
                dlv=s.grid_price_Wh.mean() / 1e6,
                trips=s.n_trips.mean(), uns=s.true_unserved.mean() / 1e6,
                hd=s.min_headroom.mean(), hd_min=s.min_headroom.min())
print(f"aggregation: {len(p)}/48 partial-grid rows (6 G x 8 cells); "
      f"0 shutdowns and 0 crossings in every row; |residual| <= 1 Wh")

summ = pd.read_csv(SW + "_summary.csv")
for g in G:
    r = summ[(summ.rx == ARM) & (summ.G == g)].iloc[0]
    assert np.isclose(r.price_requested_MWh, M[g]["req"], rtol=1e-12)
    assert np.isclose(r.price_approved_MWh, M[g]["acc"], rtol=1e-12)
print("summary-file cross-check: PASS")

g4 = pd.read_csv(G04 + ".csv")
g4 = g4[g4.rx == ARM].set_index("cell").sort_index()
s4 = p[p.G == 0.4].set_index("cell").sort_index()
cols = [c for c in g4.columns if not c.startswith(TIMING) and c not in LABELS]
for c in cols:
    a_, b_ = g4[c], s4[c]
    assert ((a_ == b_) | (a_.isna() & b_.isna())).all(), c
print(f"G=0.4 vs stage-1 result: {len(cols)} columns x 8 cells identical "
      f"-- PASS")

print("\npartial-grid controller, A-H means (MWh per 31-day trace):")
print(f"{'G':>4} | {'requested':>9} | {'approved':>9} | {'frac':>6} | "
      f"{'delivered':>9} | {'shutdowns':>9} | {'unserved':>9} | "
      f"{'min headroom mean / worst (pcm)':>31}")
for g in G:
    m = M[g]
    fr = m["acc"] / m["req"] if m["req"] > 0 else float("nan")
    print(f"{g:4.1f} | {m['req']:9.3f} | {m['acc']:9.3f} | {fr:6.3f} | "
          f"{m['dlv']:9.3f} | {m['trips']:9.3f} | {m['uns']:9.4f} | "
          f"{m['hd']:14.3f} / {m['hd_min']:8.3f}")
assert "sim" not in sys.modules

# ---- figure ----
fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
ax.plot(G, [M[g]["req"] for g in G], color=C_REQ, lw=LINE_LW,
        ls=(0, (3.2, 1.6)), marker="s", ms=MS - 1, mfc="white", mec="black",
        mew=MEW, zorder=3, clip_on=False, label="Requested grid opportunity")
ax.plot(G, [M[g]["acc"] for g in G], color=C_ACC, lw=LINE_LW, ls="-",
        marker="o", ms=MS, mfc=C_ACC, mec="black", mew=MEW, zorder=4,
        clip_on=False, label="Feasibility-limited import")
ax.vlines(0.4, 0, 160, color="black", ls=(0, (3, 3)), lw=1.3,
          zorder=1)          # stops below the legend
ax.set_xlim(-0.03, 1.03)
ax.set_xticks(G)
ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
ax.set_ylim(0, 260)
ax.set_yticks([0, 50, 100, 150, 200, 250])
ax.set_xlabel("Grid Import Capacity G")
ax.set_ylabel("MWh per 31-day trace")
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.tick_params(direction="out", colors="black")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))

for e in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{e}", facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
w, h = plt.imread(f"{OUT}/{STEM}.png").shape[1::-1]
for q, h0 in src_hash.items():
    assert sha(q) == h0, f"{q} changed"
assert "sim" not in sys.modules
print(f"\nwrote {OUT}/{STEM}.png + .svg; PNG {w}x{h} px at 600 dpi = "
      f"{w/600:.2f} x {h/600:.2f} in (aspect {w/h:.2f}:1); font {FS} pt "
      f"throughout; source files unchanged; sim.py not imported")
