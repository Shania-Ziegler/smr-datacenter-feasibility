# ============================================================
# prepare_borg_cell.py — Borg 2019 cell (a-h): demand + deferrability
# Steps: (1) load + verify, (2) demand & split + plots,
#        (3) diurnal profile + deferrable fraction over time
# Run from data/ (reads borg_cell_<x>.csv, writes demand_curve_cell_<x>.csv
# and four diagnostic PNGs there):
#   cd data && python ../scripts/prepare_borg_cell.py --cell a
# ============================================================

import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1 — LOAD + VERIFY
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cell", required=True, choices=list("abcdefgh"))
cell = parser.parse_args().cell
label = f"Cell {cell.upper()}"

CSV = f"borg_cell_{cell}.csv"    # in the working directory (data/)
df = pd.read_csv(CSV)

print("Columns:", list(df.columns))
print("Total rows:", len(df))
print(df.head(), "\n")

# is it a clean month?
n_windows = df["window_5min"].nunique()
days = n_windows * 5 / 60 / 24                     # windows -> minutes -> days
print(f"Unique 5-min windows: {n_windows}")
print(f"Days covered: {days:.2f}")
print("Max rows per window (should be <=20):",
      df.groupby("window_5min").size().max())

# first-window total (all tiers summed)
w1 = df.loc[df["window_5min"] == 1, "total_cpu"].sum()
print(f"\nWindow 1 total_cpu (all tiers summed): {w1:.2f}")

print("\nPriority tiers:", sorted(df["priority_tier"].unique()))
print("Scheduling classes:", sorted(df["scheduling_class"].unique()))

# ------------------------------------------------------------
# STEP 2 — TOTAL DEMAND, DEFERRABLE VS RIGID, FIRST PLOTS
# ------------------------------------------------------------
demand = df.groupby("window_5min")["total_cpu"].sum().reset_index()
demand["hours"] = (demand["window_5min"] - 1) * 5 / 60   # window index -> hours

def tier_sum(tiers):
    return (df[df["priority_tier"].isin(tiers)]
            .groupby("window_5min")["total_cpu"].sum()
            .reindex(demand["window_5min"], fill_value=0).values)

defer_low  = tier_sum(["1_free", "2_best_effort_batch"])              # lower bound
defer_high = tier_sum(["1_free", "2_best_effort_batch", "3_mid"])     # upper bound
rigid      = tier_sum(["4_production", "5_monitoring"])

total = demand["total_cpu"].values
print(f"\nMonth-average deferrable fraction: "
      f"{defer_low.sum()/total.sum():.1%} (batch only) "
      f"to {defer_high.sum()/total.sum():.1%} (incl. mid)")
print(f"Month-average rigid fraction:       {rigid.sum()/total.sum():.1%}")
print(f"Peak-to-mean demand ratio:          {total.max()/total.mean():.2f}")

assert demand["window_5min"].diff().dropna().max() == 1, "gap in window sequence!"
assert not df[["total_cpu","total_mem","peak_cpu"]].isna().any().any(), "NaNs in usage!"

# Plot 1: total demand over the month
plt.figure(figsize=(14, 4))
plt.plot(demand["hours"] / 24, total, linewidth=0.6)
plt.xlabel("Days"); plt.ylabel("Total CPU demand (normalized)")
plt.title(f"{label} — total datacenter demand over the month")
plt.tight_layout(); plt.savefig(f"demand_total_{cell}.png", dpi=130)

# Plot 2: deferrable (batch) vs rigid, stacked
plt.figure(figsize=(14, 4))
plt.stackplot(demand["hours"] / 24, rigid, defer_low,
              labels=["Rigid (production+monitoring)", "Deferrable (batch)"],
              colors=["#c44", "#4a8"])
plt.xlabel("Days"); plt.ylabel("CPU demand (normalized)")
plt.title(f"{label} — rigid vs. deferrable demand")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(f"demand_split_{cell}.png", dpi=130)

print(f"Saved demand_total_{cell}.png and demand_split_{cell}.png")

# ------------------------------------------------------------
# STEP 3 — DIURNAL PROFILE + DEFERRABLE FRACTION OVER TIME
# ------------------------------------------------------------
demand["hour"] = (demand["hours"] % 24).astype(int)   # hour of day, 0..23
demand["day"]  = (demand["hours"] // 24).astype(int)  # day index, 0..30
frac_defer = defer_low / total                        # deferrable share per window

# Plot 3: average day (diurnal profile), total vs rigid
prof_total = demand.groupby("hour")["total_cpu"].mean()
rigid_s = pd.Series(rigid, index=demand.index)
prof_rigid = rigid_s.groupby(demand["hour"]).mean()

plt.figure(figsize=(8, 4.5))
plt.plot(prof_total.index, prof_total.values, label="Total", lw=2)
plt.plot(prof_rigid.index, prof_rigid.values, label="Rigid only", lw=2)
plt.xlabel("Hour of day (trace time)"); plt.ylabel("Mean CPU demand")
plt.title(f"{label} — average daily profile")
plt.legend(); plt.tight_layout(); plt.savefig(f"diurnal_profile_{cell}.png", dpi=130)

# Plot 4: deferrable fraction over the month
plt.figure(figsize=(14, 3.5))
plt.plot(demand["hours"] / 24, frac_defer, lw=0.5)
plt.xlabel("Days"); plt.ylabel("Deferrable fraction")
plt.title(f"{label} — share of demand that is deferrable (batch), over time")
plt.tight_layout(); plt.savefig(f"defer_fraction_{cell}.png", dpi=130)

print(f"\nDiurnal swing (total, avg day): "
      f"{(prof_total.max()-prof_total.min())/prof_total.mean():.1%}")
print(f"Deferrable fraction range over month: "
      f"{frac_defer.min():.1%} to {frac_defer.max():.1%}")
print(f"Saved diurnal_profile_{cell}.png and defer_fraction_{cell}.png")

# ------------------------------------------------------------
# STEP 4 — EXPORT: per-window demand + flexibility for the sim
# ------------------------------------------------------------
out = pd.DataFrame({
    "window_5min": demand["window_5min"],
    "hours":       demand["hours"],
    "total_cpu":   total,
    "rigid_cpu":   rigid,
    "defer_low":   defer_low,      # batch only  (conservative flexibility)
    "defer_high":  defer_high,     # batch + mid (optimistic flexibility)
    "frac_defer":  frac_defer,
})
out.to_csv(f"demand_curve_cell_{cell}.csv", index=False)
print(f"\nWrote demand_curve_cell_{cell}.csv  ({len(out)} rows)")