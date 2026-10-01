#!/usr/bin/env python3
"""
plot_design_table.py -- fixed-delay baselines versus state-dependent
feasibility (predictive, consolidated) at ample and low reactor margin, as
a table figure, plus the per-policy decision-time overhead as CSV.

Source (read only): controller_runs/main_suite.csv, primary_cross_arm,
scenario none. Headline means are checked against the reported values
before plotting. Outputs: figures/design_table.{png,svg},
figures/policy_overhead_summary.csv.
Run from the repository root (works in data/).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
# inputs are read from, and figures written to, the data/ working directory
os.chdir(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data"))
os.makedirs("figures", exist_ok=True)

CSV = "controller_runs/main_suite.csv"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------
# Policy colours
# ------------------------------------------------------------
C_FREE = "#A13C3A"
C_REAC = "#3A9FA1"
C_PRED = "#A13A9F"
C_COMP = "#C9953D"
C_CONS = "#4E6A9F"
C_GRAY = "#666666"

for name in ("TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"):
    if any(name.lower() in f.name.lower()
           for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "svg.fonttype": "none",
})

def save(fig, stem):
    for ext in ("png", "svg"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{stem}.png/.svg")

# ------------------------------------------------------------
# Load ONLY primary evidence.
# Never mix sensitivity or synthetic deadline diagnostics.
# ------------------------------------------------------------
d = pd.read_csv(CSV)

assert len(d) == 2112
assert set(d["cell"]) == set("abcdefgh")
assert set(d["R_cap"]) == {8000.0, 3500.0, 2700.0}

p = d[d["evidence_class"] == "primary_cross_arm"].copy()
assert len(p) == 960

# Headline figures use nominal/no-injected-stress scenario.
base = p[p["scenario"] == "none"].copy()

# Determine arm name robustly.
# rx is used by legacy arms; controller_mode identifies controller arms.
def arm_of(row):
    cm = str(row.get("controller_mode", "")).strip().lower()
    rx = str(row.get("rx", "")).strip().lower()

    if cm in ("computing_only", "consolidated"):
        return cm

    if rx in ("free", "reactive", "predictive", "hysteresis"):
        return rx

    # fallbacks
    if cm not in ("", "nan", "none"):
        return cm
    return rx

base["arm_plot"] = base.apply(arm_of, axis=1)

# Hysteresis hours can appear under hyst_h or hysteresis_h.
if "hyst_h" in base:
    base["h_plot"] = base["hyst_h"]
else:
    base["h_plot"] = base["hysteresis_h"]

# true_unserved is Wh -> kWh
base["unserved_kWh"] = base["true_unserved"] / 1000.0

print("\nPrimary/no-stress rows:", len(base))
print(base.groupby(["R_cap", "arm_plot"]).size())

# ------------------------------------------------------------
# Verify against completed-run printed aggregates.
# This prevents accidentally plotting wrong evidence classes.
# ------------------------------------------------------------
def mean_metric(R, arm, col):
    x = base[(base.R_cap == R) & (base.arm_plot == arm)]
    assert len(x) == 8, (R, arm, len(x))
    return x[col].mean()

checks = {
    ("free", 2700): 182898.9,
    ("computing_only", 2700): 171992.8,
    ("predictive", 2700): 264.4,
    ("consolidated", 2700): 168.7,
}

print("\nHeadline validation:")
for (arm, R), expected in checks.items():
    got = mean_metric(R, arm, "unserved_kWh")
    print(f"  {arm:16s} R={R}: {got:.1f} kWh")
    assert abs(got - expected) < 1.0, (arm, R, got, expected)

print("HEADLINE AGGREGATE CHECK: PASS")


# ============================================================
# DESIGN TABLE: fixed-delay baseline vs state-dependent feasibility.
# ============================================================

def hyst_rows(R, h):
    q = base[
        (base.R_cap == R) &
        (base.arm_plot == "hysteresis") &
        (np.isclose(base.h_plot.astype(float), h))
    ]
    assert len(q) == 8, (R, h, len(q))
    return q

def arm_rows(R, arm):
    q = base[(base.R_cap == R) & (base.arm_plot == arm)]
    assert len(q) == 8, (R, arm, len(q))
    return q

cols = [
    ("Fixed\n2 h", lambda R: hyst_rows(R, 2)),
    ("Fixed\n4 h", lambda R: hyst_rows(R, 4)),
    ("Fixed\n6 h", lambda R: hyst_rows(R, 6)),
    ("Predictive", lambda R: arm_rows(R, "predictive")),
    ("Consolidated", lambda R: arm_rows(R, "consolidated")),
]

Rs = [
    (8000, "Ample margin\n8000 pcm"),
    (2700, "Low margin\n2700 pcm"),
]

fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.axis("off")

left = 0.20
right = 0.985
bottom = 0.10
top = 0.87

ncols = len(cols)
nrows = len(Rs)

cw = (right - left) / ncols
rh = (top - bottom) / nrows

# column headers
for j, (lab, _) in enumerate(cols):
    x0 = left + j*cw
    ax.add_patch(Rectangle(
        (x0, top), cw, 0.105,
        transform=ax.transAxes,
        facecolor="#EFEFEF", edgecolor="black", lw=1.2
    ))
    ax.text(
        x0 + cw/2, top + 0.052, lab,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, fontweight="bold"
    )

# row labels + cells
for i, (R, rlab) in enumerate(Rs):
    y0 = top - (i+1)*rh

    ax.add_patch(Rectangle(
        (0.01, y0), left-0.01, rh,
        transform=ax.transAxes,
        facecolor="#EFEFEF", edgecolor="black", lw=1.2
    ))
    ax.text(
        0.105, y0 + rh/2, rlab,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, fontweight="bold"
    )

    for j, (_, getter) in enumerate(cols):
        q = getter(R)
        uns = q.unserved_kWh.mean()
        trips = q.n_trips.mean()

        x0 = left + j*cw

        # emphasize physics-aware columns subtly
        fc = "#F7F7F7" if j < 3 else "#F1EDF5"

        ax.add_patch(Rectangle(
            (x0, y0), cw, rh,
            transform=ax.transAxes,
            facecolor=fc, edgecolor="black", lw=1.0
        ))

        if uns >= 10000:
            uns_txt = f"{uns/1000:.1f} MWh"
        else:
            uns_txt = f"{uns:.0f} kWh"

        ax.text(
            x0 + cw/2, y0 + rh*0.61,
            uns_txt,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="bold"
        )
        ax.text(
            x0 + cw/2, y0 + rh*0.35,
            f"{trips:.1f} trips",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9, color="#444444"
        )

ax.text(
    0.5, 0.025,
    "Mean across Borg cells A-H, nominal scenario. "
    "Fixed-delay policies wait 2/4/6 h; predictive and consolidated "
    "use reactor-state feasibility.",
    transform=ax.transAxes,
    ha="center", va="bottom",
    fontsize=8.5, color="#444444"
)

save(fig, "design_table")


# ============================================================
# Decision-time overhead summary.
# Not a giant figure: write a CSV/text callout.
# ============================================================

tim = base[
    base.arm_plot.isin(["computing_only", "consolidated"])
].copy()

over = (
    tim.groupby("arm_plot")
       .agg(
           mean_decision_us=("policy_mean_us", "mean"),
           p95_decision_us=("policy_p95_us", "mean"),
           p99_decision_us=("policy_p99_us", "mean"),
           max_decision_us=("policy_max_us", "max"),
           mean_projection_us=("projection_mean_us", "mean"),
       )
)

over.to_csv(os.path.join(OUT, "policy_overhead_summary.csv"))

print("\nPOLICY OVERHEAD")
print(over.to_string())

print("\nDONE")
