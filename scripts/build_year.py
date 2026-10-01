#!/usr/bin/env python3
"""
build_year.py -- construct SYNTHETIC 365-day workload chronologies from the
processed Google Borg Cell A-H demand curves (demand_curve_cell_<x>.csv).

Scientific framing (disclose wherever these outputs are used):
  - Cells A-H are DIFFERENT Borg cells (distinct workload regimes) observed
    over roughly the SAME ~31-day period. They are not eight consecutive
    months, and they are not eight workload types.
  - The output is a synthetic 365-day workload assembled by appending whole
    31-day segments of real cell traces in a fixed order and repeating that order cyclically
    until 105120 five-minute windows exist. Real windows are therefore
    REUSED to fill the year; no workload values are synthesized or
    interpolated. It is not a recorded year.
  - Each source file holds 8929 windows, hour 0.0 through 744.0 INCLUSIVE.
    The 744.0 h sample is the start of day 32, so only the first
    SEG_WIN = 31*24*12 = 8928 windows form a segment; appending all 8929
    would repeat 00:00 at every splice (a 5-min phase slip per splice).
    Source files are not modified; the endpoint is omitted only here.
  - Cell boundaries are splice points: workload level and time-of-day phase
    can jump there. Provenance columns identify every splice.

Cross-cell normalization:
  peak      [PRIMARY, default] scale_i = U_PK / max_i(total_cpu), U_PK = 0.85.
            This is sim.py's per-trace workload scaling
            (u = 0.85 * tot / tot.max(), used for Cell A) applied to each
            cell independently BEFORE concatenation. Every cell peaks at
            0.85 utilization; the finished year is NOT renormalized.
Diagnostic / sensitivity modes only (not the primary output):
  mean      scale_i = ref / mean_i(total_cpu), ref = mean of the eight cell
            means. Every cell ends with mean total_cpu == ref (CPU units).
  capacity  scale_i = 1 / total_cpu_capacity_i from the cell metadata, so
            workload becomes utilization of that cell's CPU capacity.
            (Metadata capacity is never used to scale the peak-mode output.)
The same per-cell factor multiplies total_cpu, rigid_cpu, defer_low and
defer_high; frac_defer is left as-is (a ratio, unchanged by scaling). Each
cell's within-trace temporal shape is preserved.

Usage (run from data/, which holds the demand curves and cell metadata):
    cd data && python ../scripts/build_year.py --normalization peak [--dry-run]    (default)
    cd data && python ../scripts/build_year.py --normalization mean [--dry-run]
    cd data && python ../scripts/build_year.py --normalization capacity [--dry-run]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

CELLS = "ABCDEFGH"
WORKLOAD = ["total_cpu", "rigid_cpu", "defer_low", "defer_high"]
REQUIRED = ["window_5min", "hours"] + WORKLOAD + ["frac_defer"]
N_WIN = 365 * 24 * 12          # 105120 five-minute windows = 365 days
SEG_WIN = 31 * 24 * 12         # 8928 usable windows per 31-day cell segment
ORDERINGS = {
    "order1": "ABCDEFGH",
    "order2": "HGFEDCBA",
    "order3": "CFAHDBGE",
}
FRAC_ATOL = 1e-9
U_PK = 0.85                    # == sim.load_borg default u_pk
PEAK_ATOL = 1e-9
BANNER = ("Synthetic 365-day workload assembled from repeated real Borg "
          "Cell A-H traces (different cells over the same ~31-day period; "
          "not a recorded year).")


def fail(msg):
    sys.exit(f"[build_year] ERROR: {msg}")


def load_cell(c):
    path = f"demand_curve_cell_{c.lower()}.csv"
    if not os.path.exists(path):
        fail(f"{path} not found.")
    d = pd.read_csv(path)
    missing = [k for k in REQUIRED if k not in d.columns]
    if missing:
        fail(f"{path}: missing columns {missing}; has {list(d.columns)}.")
    if d[REQUIRED].isna().any().any():
        fail(f"{path}: NaNs in {d[REQUIRED].columns[d[REQUIRED].isna().any()].tolist()}.")
    v = d[REQUIRED].to_numpy(dtype=float)
    if not np.isfinite(v).all():
        fail(f"{path}: non-finite values.")
    w = d["window_5min"].to_numpy()
    if len(d) < 2 or not (np.diff(w) == 1).all():
        fail(f"{path}: window_5min is not strictly continuous (gap/reorder).")
    if len(d) < SEG_WIN:
        fail(f"{path}: {len(d)} rows < {SEG_WIN} needed for a 31-day segment.")
    checks = {
        "total_cpu > 0": d.total_cpu > 0,
        "rigid_cpu >= 0": d.rigid_cpu >= 0,
        "defer_low >= 0": d.defer_low >= 0,
        "defer_high >= defer_low": d.defer_high >= d.defer_low,
        "rigid_cpu <= total_cpu": d.rigid_cpu <= d.total_cpu,
        "defer_high <= total_cpu": d.defer_high <= d.total_cpu,
        "0 <= frac_defer <= 1": d.frac_defer.between(0.0, 1.0),
        "frac_defer == defer_low/total_cpu":
            np.isclose(d.frac_defer, d.defer_low / d.total_cpu,
                       rtol=0, atol=FRAC_ATOL),
    }
    for name, ok in checks.items():
        if not ok.all():
            fail(f"{path}: {int((~ok).sum())} row(s) violate {name}.")
    return d


def cell_stats(d):
    tc = d.total_cpu
    return dict(rows=len(d), days=len(d) * 5 / 1440.0, mean=tc.mean(),
                peak=tc.max(), pm=tc.max() / tc.mean(),
                fmean=d.frac_defer.mean(), fmin=d.frac_defer.min(),
                fmax=d.frac_defer.max())


def load_capacity():
    """Per-cell CPU capacity. Uses cell_metadata.csv (one row per cell, with
    a 'cell' column) if present, else cell<x>_metadata.csv per cell. The
    capacity column must be the UNIQUE column naming both 'cpu' and
    'capacity'; anything else is treated as ambiguous."""
    def pick(cols, where):
        cand = [c for c in cols if "cpu" in c.lower()
                and "capacity" in c.lower()]
        if len(cand) != 1:
            fail(f"cannot identify a unique CPU-capacity column in {where}; "
                 f"columns: {list(cols)} (candidates: {cand}).")
        return cand[0]

    cap, cols_seen = {}, {}
    if os.path.exists("cell_metadata.csv"):
        m = pd.read_csv("cell_metadata.csv")
        cols_seen["cell_metadata.csv"] = list(m.columns)
        if "cell" not in m.columns:
            fail(f"cell_metadata.csv has no 'cell' column: {list(m.columns)}.")
        col = pick(m.columns, "cell_metadata.csv")
        keyed = m.set_index(m["cell"].astype(str).str.strip().str.upper())
        for c in CELLS:
            if c not in keyed.index:
                fail(f"cell_metadata.csv has no row for cell {c}.")
            cap[c] = float(keyed.loc[c, col])
    else:
        col = None
        for c in CELLS:
            path = f"cell{c.lower()}_metadata.csv"
            if not os.path.exists(path):
                fail(f"no cell_metadata.csv and {path} not found.")
            m = pd.read_csv(path)
            cols_seen[path] = list(m.columns)
            if len(m) != 1:
                fail(f"{path}: expected exactly 1 row, got {len(m)}.")
            if "cell" in m.columns and \
               str(m["cell"].iloc[0]).strip().upper() != c:
                fail(f"{path}: 'cell' column says {m['cell'].iloc[0]!r}.")
            this = pick(m.columns, path)
            if col is not None and this != col:
                fail(f"capacity column differs across files ({col} vs {this}).")
            col = this
            cap[c] = float(m[col].iloc[0])
    for c, v in cap.items():
        if not np.isfinite(v) or v <= 0:
            fail(f"cell {c}: capacity {v} is not a positive finite number.")
    return cap, col, cols_seen


def normalize(cells, mode):
    if mode == "peak":
        scale = {c: U_PK / cells[c].total_cpu.max() for c in CELLS}
        print(f"[peak] scale_i = {U_PK} / max_i(total_cpu), per cell, before "
              f"concatenation (sim.py's u = {U_PK} * tot / tot.max() applied "
              f"cell by cell); finished year is not renormalized")
    elif mode == "mean":
        means = {c: cells[c].total_cpu.mean() for c in CELLS}
        ref = float(np.mean(list(means.values())))
        scale = {c: ref / means[c] for c in CELLS}
        print(f"[mean] reference = mean of the 8 cell mean total_cpu = "
              f"{ref:.4f} (CPU units); scale_i = ref / mean_i")
    else:
        cap, col, cols_seen = load_capacity()
        print("[capacity] metadata columns discovered:")
        for f, cols in cols_seen.items():
            print(f"    {f}: {cols}")
        print(f"[capacity] capacity field used: '{col}'; "
              f"scale_i = 1 / {col}_i (output = utilization of capacity)")
        scale = {c: 1.0 / cap[c] for c in CELLS}
        for c in CELLS:
            print(f"    {c}: {col} = {cap[c]:.4f}")

    out = {}
    print(f"\n[{mode}] per-cell normalization:")
    print(f"  {'cell':>4} | {'scale':>12} | {'mean':>10} | {'peak':>10} | "
          f"{'peak/mean':>9} | {'max |dfrac|':>11}")
    for c in CELLS:
        d = cells[c].copy()
        for k in WORKLOAD:
            d[k] = d[k] * scale[c]
        dfrac = float(np.max(np.abs(d.defer_low / d.total_cpu
                                    - cells[c].frac_defer)))
        out[c] = d
        print(f"  {c:>4} | {scale[c]:12.6g} | {d.total_cpu.mean():10.4f} | "
              f"{d.total_cpu.max():10.4f} | "
              f"{d.total_cpu.max() / d.total_cpu.mean():9.4f} | {dfrac:11.2e}")
    allv = pd.concat([out[c].total_cpu for c in CELLS])
    maxd = max(float(np.max(np.abs(out[c].defer_low / out[c].total_cpu
                                   - cells[c].frac_defer))) for c in CELLS)
    print(f"  overall total_cpu min/mean/max = {allv.min():.4f} / "
          f"{allv.mean():.4f} / {allv.max():.4f}")
    print(f"  flexibility fractions changed? "
          f"{'NO' if maxd <= FRAC_ATOL else 'YES'} "
          f"(max |defer_low/total_cpu - frac_defer| = {maxd:.2e}; "
          f"frac_defer column copied unmodified)")
    if mode == "peak":
        validate_peak(cells, out, scale)
    return out, scale


def validate_peak(cells, out, scale):
    print(f"\n[peak] per-cell normalized workload regimes:")
    print(f"  {'cell':>4} | {'scale':>12} | {'mean util':>9} | "
          f"{'peak util':>9} | {'peak/mean':>9} | {'mean frac_defer':>15}")
    mu, pmr, fd = {}, {}, {}
    for c in CELLS:
        d, src = out[c], cells[c]
        mu[c] = d.total_cpu.mean()
        pm_src = src.total_cpu.max() / src.total_cpu.mean()
        pm = d.total_cpu.max() / mu[c]
        print(f"  {c:>4} | {scale[c]:12.6g} | {mu[c]:9.4f} | "
              f"{d.total_cpu.max():9.6f} | {pm:9.4f} | "
              f"{d.frac_defer.mean():15.4f}")
        if abs(d.total_cpu.max() - U_PK) > PEAK_ATOL:
            fail(f"peak: cell {c} peak {d.total_cpu.max()!r} != {U_PK}.")
        if (d[WORKLOAD] > U_PK + PEAK_ATOL).any().any():
            fail(f"peak: cell {c} has normalized workload > {U_PK}.")
        # The scale uses the full source trace; the year uses only the first
        # SEG_WIN windows. They agree only if the omitted endpoint is not the
        # cell's peak -- otherwise a segment would never reach U_PK.
        if abs(d.total_cpu.iloc[:SEG_WIN].max() - U_PK) > PEAK_ATOL:
            fail(f"peak: cell {c} peak lies in the omitted endpoint window.")
        if not np.array_equal(d.frac_defer.to_numpy(),
                              src.frac_defer.to_numpy()):
            fail(f"peak: cell {c} frac_defer changed.")
        # Regression guard only: constant scaling preserves peak/mean by
        # construction, so passing this is not evidence of anything.
        if not np.isclose(pm, pm_src, rtol=1e-12, atol=0):
            fail(f"peak: cell {c} peak/mean {pm} != source {pm_src}.")
        pmr[c] = pm
        fd[c] = d.frac_defer.mean()
    v = np.array(list(mu.values()))
    lo_c, hi_c = min(mu, key=mu.get), max(mu, key=mu.get)
    print(f"  mean-utilization spread across A-H: min {v.min():.4f} ({lo_c}) "
          f"/ max {v.max():.4f} ({hi_c}) / range {v.max() - v.min():.4f} / "
          f"max:min {v.max() / v.min():.3f} / std {v.std():.4f}")
    r = np.array(list(pmr.values()))
    lo_c, hi_c = min(pmr, key=pmr.get), max(pmr, key=pmr.get)
    print(f"  peak/mean (burstiness) spread across A-H: min {r.min():.4f} "
          f"({lo_c}) / max {r.max():.4f} ({hi_c}) / range "
          f"{r.max() - r.min():.4f} / std {r.std():.4f}")
    f = np.array(list(fd.values()))
    lo_c, hi_c = min(fd, key=fd.get), max(fd, key=fd.get)
    print(f"  mean frac_defer spread across A-H: min {f.min():.4f} ({lo_c}) / "
          f"max {f.max():.4f} ({hi_c}) / range {f.max() - f.min():.4f} / "
          f"std {f.std():.4f}")
    print(f"  peak validation: PASS (every cell peak == {U_PK} within "
          f"{PEAK_ATOL:g}; no value > {U_PK}; frac_defer unchanged; "
          f"each {SEG_WIN}-window segment still reaches {U_PK})")
    print("  regression guard: peak/mean equals source within rtol 1e-12 "
          "(holds by construction under constant scaling; not evidence)")


def idempotence_check(y, name):
    """sim.load_borg's u = 0.85 * tot / tot.max() must leave a peak-mode
    year unchanged (its global max is already 0.85)."""
    tc = y.total_cpu.to_numpy()
    dev = float(np.max(np.abs(U_PK * tc / tc.max() - tc)))
    if dev > PEAK_ATOL:
        fail(f"{name}: idempotence check failed (max |dev| = {dev:.2e}).")
    print(f"  idempotence check: PASS -- {U_PK} * total_cpu / "
          f"total_cpu.max() == total_cpu (year max {float(tc.max())!r}, "
          f"max |dev| = {dev:.2e})")


def build(norm, order):
    """Append the first SEG_WIN windows of each cell in `order`, cycling,
    then truncate the final segment so the year is exactly N_WIN."""
    parts, segs, n, seg, cyc = [], [], 0, 0, 0
    while n < N_WIN:
        for c in order:
            if n >= N_WIN:
                break
            d = norm[c].iloc[:SEG_WIN]
            take = min(SEG_WIN, N_WIN - n)
            p = d.iloc[:take][WORKLOAD + ["frac_defer"]].copy()
            p["source_cell"] = c
            p["source_window_5min"] = d["window_5min"].iloc[:take].to_numpy()
            p["segment_index"] = seg
            p["cycle_index"] = cyc
            parts.append(p)
            segs.append((seg, cyc, c, n, take, take == SEG_WIN))
            n += take; seg += 1
        cyc += 1
    y = pd.concat(parts, ignore_index=True)
    y.insert(0, "window_5min", np.arange(1, N_WIN + 1))
    y.insert(1, "hours", (y["window_5min"] - 1) * 5 / 60.0)
    return y, segs


def splice_report(norm, segs):
    """DIAGNOSTIC ONLY -- nothing is shifted, rotated or altered.
    Hour-of-day is TRACE-RELATIVE (source `hours` mod 24, trace start =
    00:00), not wall-clock. Across a splice the synthetic clock advances one
    5-min step; the phase jump is how far the source clock departs from
    that: jump = hod(next first) - (hod(prev last) + 5 min), wrapped to
    (-12, 12] h. drift = synthetic hod - source hod after the splice."""
    step = 5 / 60.0
    wrap = lambda x: ((x + 12.0) % 24.0 - 12.0) if ((x + 12.0) % 24.0) else 12.0
    tod = lambda h: f"{int(round(h * 60)) // 60 % 24:02d}:{int(round(h * 60)) % 60:02d}"
    print(f"  splice phase (trace-relative hour-of-day; diagnostic only):")
    print(f"    {'#':>2} | {'prev':>4} -> {'next':<4} | {'prev last w':>11} | "
          f"{'next first w':>12} | {'prev TOD':>8} | {'next TOD':>8} | "
          f"{'jump h':>7} | {'drift h':>7}")
    jumps, drift = [], 0.0
    for i in range(len(segs) - 1):
        _, _, ca, na, ta, _ = segs[i]
        _, _, cb, nb, _, _ = segs[i + 1]
        wa = int(norm[ca].window_5min.iloc[ta - 1])
        wb = int(norm[cb].window_5min.iloc[0])
        ha = float(norm[ca].hours.iloc[ta - 1]) % 24.0
        hb = float(norm[cb].hours.iloc[0]) % 24.0
        j = wrap(hb - (ha + step))
        drift = wrap((nb * 5 / 60.0) % 24.0 - hb)
        jumps.append(j)
        print(f"    {i + 1:>2} | {ca:>4} -> {cb:<4} | {wa:>11d} | {wb:>12d} | "
              f"{tod(ha):>8} | {tod(hb):>8} | {j:+7.4f} | {drift:+7.4f}")
    a = np.abs(jumps)
    print(f"    splice boundaries: {len(jumps)} | max |jump| {a.max():.2e} h "
          f"| mean |jump| {a.mean():.2e} h | accumulated drift after last "
          f"splice {drift:+.2e} h")


def validate_year(y, name):
    def req(ok, msg):
        if not ok:
            fail(f"{name}: {msg}")
    req(len(y) == N_WIN, f"{len(y)} rows != {N_WIN}")
    req(len(y) * 5 / 1440.0 == 365.0, "does not span exactly 365 days")
    req((np.diff(y.window_5min.to_numpy()) == 1).all()
        and y.window_5min.iloc[0] == 1, "window_5min not continuous from 1")
    req(np.allclose(np.diff(y.hours.to_numpy()), 5 / 60.0, rtol=0,
                    atol=1e-9) and y.hours.iloc[0] == 0.0,
        "hours not continuous from 0 at 5-min steps")
    num = y[["window_5min", "hours"] + WORKLOAD + ["frac_defer"]]
    req(not y.isna().any().any(), "NaNs present")
    req(np.isfinite(num.to_numpy(dtype=float)).all(), "infinities present")
    req((y[WORKLOAD] >= 0).all().all(), "negative workload")
    req((y.total_cpu > 0).all(), "total_cpu <= 0")
    req((y.defer_low <= y.defer_high).all()
        and (y.defer_high <= y.total_cpu).all(),
        "defer_low <= defer_high <= total_cpu violated")
    req((y.rigid_cpu <= y.total_cpu).all(), "rigid_cpu > total_cpu")
    req(set(y.source_cell) <= set(CELLS), "source_cell outside A-H")
    req(y.source_cell.nunique() > 1, "only one source cell used")
    dev = float(np.max(np.abs(y.frac_defer - y.defer_low / y.total_cpu)))
    req(dev <= FRAC_ATOL, f"frac_defer disagrees with defer_low/total_cpu "
                          f"(max dev {dev:.2e})")
    return dev


def main():
    ap = argparse.ArgumentParser(description=BANNER)
    ap.add_argument("--normalization", default="peak",
                    choices=("peak", "mean", "capacity"),
                    help="peak = primary (default); mean/capacity = "
                         "diagnostic/sensitivity only")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print statistics; write no CSVs")
    args = ap.parse_args()

    print(BANNER)
    print("A-H are different Borg cells (workload regimes), not workload "
          "types.")
    print("Reusing real A-H windows cyclically to fill 365 days is "
          "intentional (disclosed); no workload values are synthesized.\n")

    cells = {c: load_cell(c) for c in CELLS}
    print("Input cells (all validated):")
    print(f"  {'cell':>4} | {'rows':>5} | {'days':>7} | {'mean cpu':>10} | "
          f"{'peak cpu':>10} | {'pk/mean':>7} | {'batch frac mean':>15} | "
          f"{'frac min/max':>13}")
    for c in CELLS:
        s = cell_stats(cells[c])
        print(f"  {c:>4} | {s['rows']:5d} | {s['days']:7.3f} | "
              f"{s['mean']:10.2f} | {s['peak']:10.2f} | {s['pm']:7.3f} | "
              f"{s['fmean']:15.4f} | {s['fmin']:.4f}/{s['fmax']:.4f}")
    print()

    norm, _ = normalize(cells, args.normalization)

    for oname, order in ORDERINGS.items():
        y, segs = build(norm, order)
        dev = validate_year(y, oname)
        fname = f"demand_curve_year_{oname}_{args.normalization}.csv"
        full = sum(1 for s in segs if s[5])
        n_cyc = segs[-1][1] + 1
        cyc_full = sum(1 for k in range(n_cyc)
                       if sum(1 for s in segs if s[1] == k and s[5]) == 8)
        tc = y.total_cpu
        print(f"\n== {fname}{'  [dry-run: not written]' if args.dry_run else ''}")
        print(f"  normalization: {args.normalization} | rows: {len(y)} | "
              f"days: {len(y) * 5 / 1440.0:.6f} | order: {' '.join(order)}")
        print(f"  usable windows per complete cell segment: {SEG_WIN} "
              f"(source endpoint window omitted)")
        print(f"  cycles: {cyc_full} complete + {n_cyc - cyc_full} partial; "
              f"segments: {len(segs)} ({full} whole cell traces, "
              f"{len(segs) - full} truncated)")
        print("  chronology (segment: cell, cycle, day span):")
        print("    " + " | ".join(
            f"{s[0]}:{s[2]} c{s[1]} d{s[3] * 5 / 1440:.1f}-"
            f"{(s[3] + s[4]) * 5 / 1440:.1f}{'' if s[5] else '*'}"
            for s in segs) + "   (* = truncated)")
        cnt = y.source_cell.value_counts()
        splice_report(norm, segs)
        print("  windows per source cell: " +
              ", ".join(f"{c}={int(cnt.get(c, 0))}" for c in CELLS))
        print(f"  total_cpu mean/peak/peak:mean = {tc.mean():.4f} / "
              f"{tc.max():.4f} / {tc.max() / tc.mean():.4f}")
        print(f"  batch-deferrable frac mean/min/max = "
              f"{y.frac_defer.mean():.4f} / {y.frac_defer.min():.4f} / "
              f"{y.frac_defer.max():.4f}")
        print(f"  validation: PASS (max |frac_defer - defer_low/total_cpu| "
              f"= {dev:.2e})")
        if args.normalization == "peak":
            idempotence_check(y, oname)
        if not args.dry_run:
            y.to_csv(fname, index=False)
            print(f"  wrote {fname}")

    print("\n" + BANNER)
    print("Done" + (" (dry run; no files written)." if args.dry_run else "."))


if __name__ == "__main__":
    main()
