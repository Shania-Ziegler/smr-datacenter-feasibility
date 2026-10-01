#!/usr/bin/env python3
"""
plot_grid_capacity.py -- Computing-Only only grid-capacity
figure (Consolidated summarized in one note beneath the panels):
(a) service outcome (mean true unserved work) and (b) reactor operation
(mean reactor shutdowns per 31-day Borg trace) vs grid import capacity G,
from the COMPLETED sweep (no simulation; sim.py is not imported).

Source (read only): controller_runs/grid_capacity.csv
  rx in {computing_only, consolidated}, R_cap 2700, scenario none,
  cells A-H, G in {0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0}.
Cross-checks: sweep manifest hashes, the summary CSV, and the G = 0.4
means against the canonical primary rows.
Output (refuses to overwrite):
  figures/grid_capacity.{svg,png}
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
SW = "controller_runs/grid_capacity"
CANON = "controller_runs/main_suite.csv"
OUT, STEM = "figures", "grid_capacity"
G = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
ARMS = ("computing_only", "consolidated")
LABEL = {"computing_only": "Computing-Only", "consolidated": "Consolidated"}
LINE = {"computing_only": "#6C8EBF", "consolidated": "#9673A6"}  # policy colours
LS = {"computing_only": "-", "consolidated": (0, (3.2, 1.6))}
FIGSIZE = (12.6, 3.9)
FS, AXIS_LW, LINE_LW, MS, MEW = 21, 1.8, 3.4, 10, 1.8
plt.rcParams.update({
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": AXIS_LW, "axes.edgecolor": "black",
    "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.major.pad": 3, "ytick.major.pad": 3, "axes.labelpad": 4,
    "legend.frameon": True, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.30, "legend.labelspacing": 0.30,
    "legend.handletextpad": 0.45, "legend.handlelength": 2.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path", "savefig.dpi": 600,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white"})

for e in ("png", "svg"):
    if os.path.exists(f"{OUT}/{STEM}.{e}"):
        sys.exit(f"refusing to overwrite {OUT}/{STEM}.{e}")
assert "sim" not in sys.modules


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- provenance ----
man = json.load(open(SW + "_manifest.json"))
for k, o in man["outputs"].items():
    assert sha(o["path"]) == o["sha256"], k
src = [SW + ".csv", SW + "_summary.csv", SW + "_manifest.json", CANON]
src_hash = {p: sha(p) for p in src}

d = pd.read_csv(SW + ".csv")
assert len(d) == 112 and not d.duplicated(["G", "rx", "cell"]).any()
assert (d.R_cap == 2700.0).all() and (d.scenario == "none").all()
assert set(d.rx) == set(ARMS) and sorted(d.G.unique()) == G
M, used = {}, 0
for a in ARMS:
    for g in G:
        s = d[(d.rx == a) & (d.G == g)]
        assert len(s) == 8 and "".join(sorted(s.cell)) == "abcdefgh", (a, g)
        used += len(s)
        M[a, g] = (s.true_unserved.mean() / 1e6, s.n_trips.mean())
assert used == 112

summ = pd.read_csv(SW + "_summary.csv")
for (a, g), (u, n) in M.items():
    r = summ[(summ.rx == a) & (summ.G == g)].iloc[0]
    assert np.isclose(r.true_unserved_MWh, u, rtol=1e-12) and \
        np.isclose(r.n_trips, n, rtol=1e-12), (a, g)

c = pd.read_csv(CANON)
c = c[(c.suite == "primary") & (c.scenario == "none") & (c.R_cap == 2700.0)]
print(f"aggregation covers {used}/112 sweep rows; summary cross-check PASS")
for a in ARMS:
    ca = c[c.rx == a]
    assert len(ca) == 8
    cu, cn = ca.true_unserved.mean() / 1e6, ca.n_trips.mean()
    assert cu == M[a, 0.4][0] and cn == M[a, 0.4][1], a
    print(f"G=0.4 vs canonical {LABEL[a]:>15}: {cu:.6f} MWh, "
          f"{cn:.3f} shutdowns -- PASS (exact)")

print("\nplotted A-H means:")
print(f"{'G':>4} | {'CO unserved MWh':>15} | {'Cons unserved MWh':>17} | "
      f"{'CO shutdowns':>12} | {'Cons shutdowns':>14}")
for g in G:
    print(f"{g:4.1f} | {M['computing_only', g][0]:15.4f} | "
          f"{M['consolidated', g][0]:17.4f} | "
          f"{M['computing_only', g][1]:12.3f} | "
          f"{M['consolidated', g][1]:14.3f}")

# ---- Consolidated note: every claim checked against the data ----
cons = d[d.rx == "consolidated"]
assert (cons.n_trips == 0).all()
assert cons.true_unserved.groupby(cons.G).mean().max() / 1e6 <= 0.2
NOTE = ("Consolidated: 0 shutdowns at every tested G; "
        "\u22640.2 MWh unserved.")
print(f"Consolidated note verified: max A-H mean unserved "
      f"{cons.true_unserved.groupby(cons.G).mean().max()/1e6:.4f} MWh; "
      f"shutdowns {int(cons.n_trips.sum())} in all 56 runs")

# ---- figure: Computing-Only only ----
CO = "computing_only"
fig, (axa, axb) = plt.subplots(1, 2, figsize=FIGSIZE, layout="constrained")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.01, wspace=0.04)


def draw(ax, idx):
    ax.plot(G, [M[CO, g][idx] for g in G], color=LINE[CO], lw=LINE_LW,
            marker="o", ms=MS, mfc=LINE[CO], mec="black", mew=MEW,
            zorder=3, clip_on=False)
    ax.axvline(0.4, color="black", ls=(0, (3, 3)), lw=1.3, zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", colors="black")


draw(axa, 0)
axa.set_ylim(0, 200)
axa.set_yticks([0, 50, 100, 150, 200])
axa.set_ylabel("Mean Unserved\nWork (MWh)")
axa.annotate("0 MWh", xy=(1.0, 0.0), xytext=(0, 14),
             textcoords="offset points", ha="center", va="bottom",
             fontsize=FS)
axa.text(0.415, 6, "main G", ha="left", va="bottom", fontsize=FS)

draw(axb, 1)
axb.set_ylim(0, 10)
axb.set_yticks([0, 2, 4, 6, 8, 10])
axb.set_ylabel("Reactor Shutdowns\n(mean per 31-day trace)")
axb.annotate("8.25", xy=(1.0, M[CO, 1.0][1]), xytext=(0, -14),
             textcoords="offset points", ha="right", va="top", fontsize=FS)

for ax, lab in ((axa, "(a)"), (axb, "(b)")):
    ax.text(0.02, 1.0, lab, transform=ax.transAxes, fontsize=FS,
            fontweight="bold", ha="left", va="top", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.text(0.02, 1.0, f"{lab} Computing-Only", transform=ax.transAxes,
            fontsize=FS, ha="left", va="top", alpha=0)      # width probe
xl = fig.supxlabel("Grid Import Capacity G (fraction of reactor rating)",
                   fontsize=FS)

# series name next to each panel tag (single series: no legend box)
fig.canvas.draw()
r = fig.canvas.get_renderer()
for ax in (axa, axb):
    tag = [t for t in ax.texts if t.get_text() in ("(a)", "(b)")][0]
    bb = tag.get_window_extent(r).transformed(ax.transAxes.inverted())
    ax.text(bb.x1 + 0.012, 1.0, "Computing-Only", transform=ax.transAxes,
            fontsize=FS, ha="left", va="top", color="black", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1.0))
    for t in [t for t in ax.texts if t.get_alpha() == 0]:
        t.remove()
# the one note, directly beneath the shared x-axis label
xb = xl.get_window_extent(r).transformed(fig.transFigure.inverted())
fig.text(0.5, xb.y0 - 0.012, NOTE, ha="center", va="top", fontsize=FS)

for e in ("png", "svg"):
    fig.savefig(f"{OUT}/{STEM}.{e}", facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
w, h = plt.imread(f"{OUT}/{STEM}.png").shape[1::-1]
for p, h0 in src_hash.items():
    assert sha(p) == h0, f"{p} changed"
assert "sim" not in sys.modules
print(f"\nwrote {OUT}/{STEM}.png + .svg; PNG {w}x{h} px at 600 dpi = "
      f"{w/600:.2f} x {h/600:.2f} in (aspect {w/h:.2f}:1); font {FS} pt "
      f"throughout; source files unchanged; sim.py not imported")
