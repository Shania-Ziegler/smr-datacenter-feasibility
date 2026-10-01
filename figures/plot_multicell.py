#!/usr/bin/env python3
"""
plot_multicell.py -- mean unserved work per policy across Borg cells A-H
(log-scale grouped bars at ample and low reactor margin), the fixed-delay
comparison, and the per-policy means and decision-time overhead as CSV.
The aggregate values are checked against the reported means before
plotting, so the script fails if the inputs differ from the reported
evaluation.

Source of truth: controller_runs/main_suite.csv
  evidence_class == "primary_cross_arm" and scenario == "none" only.
  No sensitivity rows, no synthetic deadline-shock diagnostics.

Outputs (nothing old is overwritten):
  figures/multicell_unserved.{png,svg}
  figures/fixed_delay.{png,svg}
  figures/unserved_means.csv
  figures/policy_overhead.csv

Run from the repository root (works in data/).
"""
import os
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
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------- style ----------------
FIGSIZE = (6.80, 3.98)
FS = 18
BAR_LW = 2.0
AXIS_LW = 1.5

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.linewidth": AXIS_LW,
    "xtick.major.width": AXIS_LW,
    "ytick.major.width": AXIS_LW,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "axes.labelpad": 4,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.25,
    "legend.labelspacing": 0.30,
    "legend.handletextpad": 0.40,
    "legend.columnspacing": 0.80,
    "legend.handlelength": 1.15,
    "patch.linewidth": BAR_LW,
    "hatch.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.antialiased": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Policy fills/hatches; the workload-aware arms take
# muted fills from the same draw.io-style pastel family.
FILL = {
    "free": "#F8CECC",
    "computing_only": "#A9C4EB",
    "predictive": "#B85450",
    "consolidated": "#E1D5E7",
    "hyst2": "#D5E8D4",
    "hyst4": "#B3D3A5",
    "hyst6": "#8DBA7A",
}
HATCH = {
    "free": "//",
    "computing_only": "\\\\",
    "predictive": "xx",
    "consolidated": "..",
    "hyst2": "--",
    "hyst4": "||",
    "hyst6": "++",
}
LABEL = {
    "free": "Limit-Agnostic",
    "computing_only": "Computing-Only",
    "predictive": "Limit-Predictive",
    "consolidated": "Consolidated",
    "hyst2": "Fixed Delay 2 h",
    "hyst4": "Fixed Delay 4 h",
    "hyst6": "Fixed Delay 6 h",
}
GROUPS = [(8000.0, "Ample Margin\n$R_\\mathrm{cap}$ = 8000 pcm"),
          (2700.0, "Low Margin\n$R_\\mathrm{cap}$ = 2700 pcm")]


def save(fig, stem):
    """EXACT-size save (no tight bbox)."""
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{stem}.{ext}", facecolor="white")
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  wrote {OUT}/{stem}.png + .svg  (exact {w:.2f}x{h:.2f} in)")


def pad(fig, h=0.06, w=0.06):
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(h_pad=h, w_pad=w)


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=AXIS_LW, length=5)


# ---------------- data: primary nominal rows only ----------------
d = pd.read_csv(CSV)
assert len(d) == 2112, len(d)
p = d[d.evidence_class == "primary_cross_arm"]
assert len(p) == 960, len(p)
base = p[p.scenario == "none"].copy()
assert len(base) == 192, len(base)
assert set(base.cell) == set("abcdefgh")
assert np.allclose(base.G, 0.4)

# hyst_h and hysteresis_h must agree wherever both are set.
both = base.hyst_h.notna() & base.hysteresis_h.notna()
assert np.allclose(base.hyst_h[both], base.hysteresis_h[both])
hh = base.hyst_h.fillna(base.hysteresis_h)


def arm_key(row, h):
    cm = str(row.controller_mode)
    if cm in ("computing_only", "consolidated"):
        return cm
    if row.rx == "hysteresis":
        return f"hyst{int(h)}"
    return row.rx


base["arm"] = [arm_key(r, h) for r, h in zip(base.itertuples(), hh)]
base["unserved_kWh"] = base.true_unserved / 1e3

print(f"\nSelected rows: all={len(d)}  primary_cross_arm={len(p)}  "
      f"primary & scenario=none={len(base)}")
print(base.groupby(["R_cap", "arm"]).size().unstack(0).to_string())

ARMS_ALL = ["free", "computing_only", "hyst2", "hyst4", "hyst6",
            "predictive", "consolidated"]
means = {}
for R, _ in GROUPS:
    for a in ARMS_ALL:
        q = base[(base.R_cap == R) & (base.arm == a)]
        assert len(q) == 8 and set(q.cell) == set("abcdefgh"), (R, a)
        means[(R, a)] = (q.unserved_kWh.mean(), q.n_trips.mean())

print("\nMean across cells A-H (unserved kWh, trips):")
for (R, a), (u, t) in means.items():
    print(f"  R_cap={R:6.0f}  {a:15s} {u:12.1f} kWh   {t:5.2f} trips")

pd.DataFrame(
    [(R, a, LABEL[a], u, t) for (R, a), (u, t) in means.items()],
    columns=["R_cap", "arm", "label", "mean_unserved_kWh", "mean_trips"],
).to_csv(f"{OUT}/unserved_means.csv", index=False)

# ---------------- aggregate assertions ----------------
EXPECT = {
    (2700.0, "free"): 182898.9,
    (2700.0, "computing_only"): 171992.8,
    (2700.0, "predictive"): 264.4,
    (2700.0, "consolidated"): 168.7,
    (8000.0, "hyst2"): 17968.2,
    (8000.0, "hyst4"): 31272.0,
    (8000.0, "hyst6"): 40432.3,
    (8000.0, "predictive"): 475.6,
    (8000.0, "consolidated"): 0.0,
    (2700.0, "hyst2"): 181856.2,
    (2700.0, "hyst4"): 180949.3,
    (2700.0, "hyst6"): 180614.5,
}
EXPECT_TRIPS = {
    (2700.0, "hyst2"): 7.4, (2700.0, "hyst4"): 7.0,
    (2700.0, "hyst6"): 7.0, (2700.0, "predictive"): 0.0,
    (2700.0, "consolidated"): 0.0,
}
print("\nAggregate assertions (tol 1 kWh / 0.1 trips):")
for k, exp in EXPECT.items():
    got = means[k][0]
    assert abs(got - exp) <= 1.0, (k, got, exp)
    print(f"  PASS  R_cap={k[0]:.0f} {k[1]:15s} {got:10.1f} ~ {exp}")
for k, exp in EXPECT_TRIPS.items():
    got = means[k][1]
    assert abs(got - exp) <= 0.1, (k, got, exp)
    print(f"  PASS  R_cap={k[0]:.0f} {k[1]:15s} trips {got:.2f} ~ {exp}")


# ---------------- grouped log bar chart ----------------
def fmt(v):
    # Compact thousands so adjacent six-digit labels do not collide;
    # exact means are in unserved_means.csv.  A true zero is
    # labelled literally, stacked to fit the bar width.
    if v == 0:
        return "0\nkWh"
    if v >= 10000:
        return f"{v / 1e3:,.0f}k"
    return f"{v:,.0f}"


def grouped_bars(arms, stem, ncol, w, label_fs):
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    x = np.arange(len(GROUPS))
    off = lambda i: (i - (len(arms) - 1) / 2.0) * w
    vals = {a: [means[(R, a)][0] for R, _ in GROUPS] for a in arms}
    pos = [v for vs in vals.values() for v in vs if v > 0]
    lo, hi = min(pos) / 2.2, max(pos) * 4.0   # axis-limit rule
    for i, a in enumerate(arms):
        for xi, v in zip(x + off(i), vals[a]):
            if v > 0:
                ax.bar(xi, v, w, facecolor=FILL[a], edgecolor="black",
                       lw=BAR_LW, hatch=HATCH[a])
                ax.annotate(fmt(v), xy=(xi, v), xytext=(0, 5),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=label_fs)
            else:
                # True zero cannot sit on a log axis: mark the floor
                # with a thick outline stub and state the real value.
                ax.plot([xi - w / 2, xi + w / 2], [lo, lo], color="black",
                        lw=BAR_LW * 2.5, solid_capstyle="butt",
                        clip_on=False, zorder=5)
                ax.annotate(fmt(v), xy=(xi, lo), xytext=(0, 5),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=label_fs,
                            linespacing=0.95)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels([g for _, g in GROUPS])
    ax.set_ylabel("Mean Unserved\nEnergy (kWh)")
    ax.set_xlim(-0.5, 1.5)
    handles = [Patch(facecolor=FILL[a], edgecolor="black", hatch=HATCH[a],
                     label=LABEL[a]) for a in arms]
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol)
    clean(ax)
    pad(fig, h=0.08)
    save(fig, stem)


print()
grouped_bars(["free", "computing_only", "predictive", "consolidated"],
             "multicell_unserved", ncol=2, w=0.22, label_fs=FS)
grouped_bars(["hyst2", "hyst4", "hyst6", "predictive", "consolidated"],
             "fixed_delay", ncol=3, w=0.18, label_fs=FS - 2)

# ---------------- decision-time overhead (text/CSV only) ----------
tim = base[base.arm.isin(["computing_only", "consolidated"])]
over = tim.groupby("arm").agg(
    n_rows=("cell", "size"),
    mean_decision_us=("policy_mean_us", "mean"),
    p95_decision_us=("policy_p95_us", "mean"),
    p99_decision_us=("policy_p99_us", "mean"),
    max_decision_us=("policy_max_us", "max"),
    mean_projection_us=("projection_mean_us", "mean"),
)
over.to_csv(f"{OUT}/policy_overhead.csv")
print("\nPolicy overhead (primary nominal rows, all R_cap):")
print(over.to_string())
co, cs = over.loc["computing_only"], over.loc["consolidated"]
print(f"  Computing-only mean decision {co.mean_decision_us/1e3:.2f} ms; "
      f"Consolidated mean {cs.mean_decision_us/1e3:.2f} ms "
      f"(projection {cs.mean_projection_us/1e3:.2f} ms), "
      f"p95 {cs.p95_decision_us/1e3:.2f} ms")
print("\nDONE")
