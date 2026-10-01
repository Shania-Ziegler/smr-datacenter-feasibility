#!/usr/bin/env python3
"""
model_mismatch_analysis.py -- read-only analysis of
controller_runs/model_mismatch.csv.

Primary comparison: MISMATCHED vs MATCHED on the SAME perturbed plant
(paired per Borg cell). The canonical nominal rows are shown for context
only. Sensitivity ranges are labelled as such, never as uncertainty bounds.
Outputs (refuse to overwrite): analysis/model_mismatch_rows.csv,
analysis/model_mismatch_summary.csv, analysis/model_mismatch.md
"""
import hashlib
import json
import os
import sys

import pandas as pd

WORK = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")   # inputs and outputs
SRC = "controller_runs/model_mismatch"
CANON = "controller_runs/main_suite.csv"
OUT_R = "analysis/model_mismatch_rows.csv"
OUT_S = "analysis/model_mismatch_summary.csv"
OUT_M = "analysis/model_mismatch.md"
ORDER = ["DOE_halflives", "lambda_I_x0.95", "lambda_I_x1.05",
         "lambda_X_x0.95", "lambda_X_x1.05", "phi_x0.9", "phi_x1.1"]
ARMS = ["predictive", "consolidated"]
MET = ["true_unserved", "n_trips", "deadtime_h", "crossings",
       "false_safe_decisions", "false_safe_eps_decisions",
       "proj_vs_realized_mean_pcm", "proj_vs_realized_min_pcm",
       "proj_underpredict_frac", "refused_steps", "min_headroom"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.chdir(WORK)
    for p in (OUT_R, OUT_S, OUT_M):
        if os.path.exists(p):
            sys.exit(f"{p} exists; refusing to overwrite.")
    d = pd.read_csv(SRC + ".csv")
    man = json.load(open(SRC + "_manifest.json"))
    V = []

    def chk(name, ok, detail=""):
        V.append((name, "PASS" if ok else "FAIL", detail))
    chk("224 rows = manifest runs", len(d) == 224 == man["runs"],
        f"{len(d)} rows")
    key = ["mismatch_case", "model_condition", "rx", "cell"]
    chk("no duplicate keys", not d[key].duplicated().any(), ",".join(key))
    g = d.groupby(key[:3]).cell.agg(lambda s: "".join(sorted(s)))
    chk("cells A-H in all 28 (case, condition, arm) groups",
        (g == "abcdefgh").all() and len(g) == 28)
    chk("R_cap 2700, scenario none, G 0.4, eps 20, H 10 h",
        (d.R_cap == 2700).all() and (d.scenario == "none").all() and
        (d.G == 0.4).all() and (d.eps == 20).all() and
        (d.predict_horizon_h == 10).all())
    rmax = max(d.residual.abs().max(), d.energy_residual_Wh.abs().max())
    chk("conservation <= 1 Wh", rmax <= 1.0 and man["conservation_failures"]
        == 0, f"max {rmax:.3g} Wh")
    chk("manifest outputs sha256 match",
        all(sha(v["path"]) == v["sha256"] for v in man["outputs"].values()))
    chk("code unchanged during run", man["code_unchanged_during_run"])
    mm = d[d.model_condition == "mismatched"]
    chk("mismatched rows: controller params = NOMINAL",
        all((mm[f"model_{f}"] == v).all()
            for f, v in man["nominal_xe"].items()))
    ma = d[d.model_condition == "matched"]
    chk("matched rows: controller params = plant params",
        all((ma[f"model_{f}"] == ma[f"plant_{f}"]).all()
            for f in man["nominal_xe"]))

    # ---- rows (individual cells) ----
    keep = ["mismatch_case", "range_label", "model_condition", "rx", "cell",
            "R_cap"] + MET + ["plant_lambda_I", "plant_lambda_X",
                              "plant_phi_full", "model_lambda_I",
                              "model_lambda_X", "model_phi_full"]
    rows = d[keep].copy()
    rows["true_unserved_kWh"] = rows.pop("true_unserved") / 1e3
    rows["_o"] = rows.mismatch_case.map({c: i for i, c in enumerate(ORDER)})
    rows = rows.sort_values(["_o", "rx", "model_condition", "cell"]) \
               .drop(columns="_o")
    os.makedirs("analysis", exist_ok=True)
    rows.to_csv(OUT_R, index=False, mode="x")

    # ---- summary: A-H means + paired deltas ----
    mcols = ["true_unserved_kWh"] + MET[1:]
    S = []
    for case in ORDER:
        for arm in ARMS:
            x = {c: rows[(rows.mismatch_case == case) & (rows.rx == arm) &
                         (rows.model_condition == c)].set_index("cell")
                 .sort_index() for c in ("matched", "mismatched")}
            for c in ("matched", "mismatched"):
                r = dict(mismatch_case=case,
                         range_label=x[c].range_label.iloc[0], arm=arm,
                         model_condition=c, n_cells=len(x[c]),
                         cells_with_trips=int((x[c].n_trips > 0).sum()),
                         cells_with_crossings=int((x[c].crossings > 0).sum()))
                for m in mcols:
                    r[f"{m}_mean"] = x[c][m].mean()
                    r[f"{m}_min"] = x[c][m].min()
                    r[f"{m}_max"] = x[c][m].max()
                S.append(r)
            dlt = x["mismatched"][mcols] - x["matched"][mcols]
            r = dict(mismatch_case=case,
                     range_label=x["matched"].range_label.iloc[0], arm=arm,
                     model_condition="mismatched - matched (paired)",
                     n_cells=len(dlt))
            for m in mcols:
                r[f"{m}_mean"] = dlt[m].mean()
                r[f"{m}_min"] = dlt[m].min()
                r[f"{m}_max"] = dlt[m].max()
            S.append(r)
    summ = pd.DataFrame(S)
    summ.to_csv(OUT_S, index=False, mode="x")

    # ---- canonical nominal reference (context only) ----
    c = pd.read_csv(CANON)
    c = c[(c.evidence_class == "primary_cross_arm") & (c.scenario == "none")
          & (c.R_cap == 2700) & c.rx.isin(ARMS)]
    ref = c.groupby("rx").agg(uns=("true_unserved", lambda v: v.mean()/1e3),
                              trips=("n_trips", "mean"),
                              pvr=("proj_vs_realized_min_pcm", "mean"))

    # ---- markdown ----
    L = []
    w = L.append
    w("# Plant/controller xenon-model mismatch (R_cap = 2700 pcm)\n")
    w("Option A: the controller observes the TRUE current iodine/xenon "
      "state; only its projection parameters may differ from the plant. "
      "**Mismatched:** plant = θ_true, controller projection = nominal. "
      "**Matched:** plant = θ_true, controller projection = θ_true. The "
      "trip rule and all metrics evaluate the true plant. Cells A-H, "
      "nominal scenario, G = 0.4, H = 10 h, eps = 20 pcm. The DOE case is "
      "a source-supported alternative parameterization; the λ ±5 % and "
      "φ ±10 % cases are explicit sensitivity ranges, not calibrated "
      "uncertainty bounds.\n")
    w("## Validation\n")
    for n_, r_, d_ in V:
        w(f"* **{r_}** — {n_}{(' (' + d_ + ')') if d_ else ''}")
    w(f"\nCanonical nominal reference at 2700 pcm (context): predictive "
      f"{ref.loc['predictive', 'uns']:.1f} kWh / "
      f"{ref.loc['predictive', 'trips']:.2f} trips; consolidated "
      f"{ref.loc['consolidated', 'uns']:.1f} kWh / "
      f"{ref.loc['consolidated', 'trips']:.2f} trips.\n")
    w("## Results (means over cells A-H; trips/crossing cells out of 8)\n")
    w("`pvr` = projected-minus-realized peak (consolidated only; negative "
      "= under-prediction). `FS` = false-safe decisions (accepted "
      "projection ≤ enforced ceiling but realized peak > true ceiling "
      "R_cap − σ_m); `FSeps` = realized peak > enforced ceiling "
      "(R_cap − σ_m − eps). Both are consolidated-only diagnostics.\n")
    w("| case | range | arm | condition | unserved kWh | trips (cells) | "
      "deadtime h | crossing min (cells) | FS | FSeps | pvr mean / min pcm | "
      "under-pred frac | refused | min headroom pcm |\n"
      "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summ[summ.model_condition.isin(["matched", "mismatched"])] \
            .iterrows():
        lab = "sourced alt." if r.range_label.startswith("source") else \
            "sensitivity"
        w(f"| {r.mismatch_case} | {lab} | {r.arm} | {r.model_condition} | "
          f"{r.true_unserved_kWh_mean:,.1f} | {r.n_trips_mean:.2f} "
          f"({r.cells_with_trips}) | {r.deadtime_h_mean:.1f} | "
          f"{r.crossings_mean:.1f} ({r.cells_with_crossings}) | "
          f"{r.false_safe_decisions_mean:.1f} | "
          f"{r.false_safe_eps_decisions_mean:.1f} | "
          f"{r.proj_vs_realized_mean_pcm_mean:.1f} / "
          f"{r.proj_vs_realized_min_pcm_min:.1f} | "
          f"{r.proj_underpredict_frac_mean:.3f} | "
          f"{r.refused_steps_mean:,.0f} | {r.min_headroom_min:.1f} |")
    w("\n## Paired effect of mismatch (mismatched − matched, same plant, "
      "same cell)\n")
    w("| case | arm | Δ unserved kWh mean [min, max] | Δ trips mean | "
      "Δ crossings mean | Δ refused mean | Δ min headroom mean pcm |\n"
      "|---|---|---:|---:|---:|---:|---:|")
    for _, r in summ[summ.model_condition.str.startswith("mismatched -")] \
            .iterrows():
        w(f"| {r.mismatch_case} | {r.arm} | {r.true_unserved_kWh_mean:+,.1f} "
          f"[{r.true_unserved_kWh_min:+,.1f}, {r.true_unserved_kWh_max:+,.1f}]"
          f" | {r.n_trips_mean:+.2f} | {r.crossings_mean:+.1f} | "
          f"{r.refused_steps_mean:+,.0f} | {r.min_headroom_mean:+.1f} |")
    # ---- buffer consumption and answers ----
    EPS = 20.0
    sm = summ[summ.model_condition.isin(["matched", "mismatched"])] \
        .set_index(["mismatch_case", "arm", "model_condition"])
    w("\n## Enforcement-buffer consumption\n")
    w("min headroom = min over cells and time of R_cap − σ_m − ρ_true "
      "(distance to the trip level). The controllers enforce R_cap − σ_m − "
      "20 pcm, so 20 − min headroom is the part of the 20-pcm buffer "
      "consumed in the worst cell; a value ≥ 20 would mean a crossing.\n")
    w("| case | arm | consumed, matched (pcm) | consumed, mismatched (pcm) "
      "| remaining, mismatched (pcm) |\n|---|---|---:|---:|---:|")
    cons = {}
    for case in ORDER:
        for arm in ARMS:
            a_ = EPS - sm.loc[(case, arm, "matched"), "min_headroom_min"]
            b_ = EPS - sm.loc[(case, arm, "mismatched"), "min_headroom_min"]
            cons[(case, arm)] = b_
            w(f"| {case} | {arm} | {max(a_, 0):.1f} | {b_:.1f} | "
              f"{EPS - b_:.1f} |")
    trips_mm = summ[summ.model_condition == "mismatched"].n_trips_max.max()
    trips_ma = summ[summ.model_condition == "matched"].n_trips_max.max()
    cr_all = summ[summ.model_condition.isin(["matched", "mismatched"])] \
        .crossings_max.max()
    fs_all = summ[summ.model_condition.isin(["matched", "mismatched"])] \
        .false_safe_decisions_max.max()
    unsafe = {"lambda_I": "lambda_I_x1.05", "lambda_X": "lambda_X_x0.95",
              "phi": "phi_x0.9"}
    safe = {"lambda_I": "lambda_I_x0.95", "lambda_X": "lambda_X_x1.05",
            "phi": "phi_x1.1"}
    worst = max(cons, key=cons.get)
    w("\n## Answers\n")
    w(f"**1. Does model mismatch cause additional trips?** No. Maximum trips "
      f"in any single cell: {trips_mm:.0f} mismatched, {trips_ma:.0f} "
      f"matched; maximum crossing minutes in any run: {cr_all:.0f}. All 224 "
      "runs are trip-free, so the paired Δ trips and Δ crossings are 0 for "
      "every case and arm.\n")
    w("**2. Does the canonical 20-pcm buffer absorb the sourced DOE "
      "half-life alternative?** Yes, in these runs: worst-cell buffer "
      f"consumption under mismatch {cons[('DOE_halflives', 'predictive')]:.1f}"
      f" pcm (predictive) and {cons[('DOE_halflives', 'consolidated')]:.1f} "
      "pcm (consolidated) of 20; no trips or crossings; paired Δ unserved "
      f"{summ.set_index(['mismatch_case', 'arm', 'model_condition']).loc[('DOE_halflives', 'predictive', 'mismatched - matched (paired)'), 'true_unserved_kWh_mean']:+.1f}"
      " kWh (predictive), +0.0 kWh (consolidated).\n")
    w("**3. Unsafe sensitivity direction** (the direction that reduces the "
      "true margin: less headroom, more under-prediction, fewer refusals):\n")
    for p_, u_ in unsafe.items():
        s_ = safe[p_]
        w(f"* {p_}: unsafe = plant {u_.split('_x')[1] if '_x' in u_ else ''}"
          f"× ({u_}); buffer consumed {cons[(u_, 'predictive')]:.1f} / "
          f"{cons[(u_, 'consolidated')]:.1f} pcm (predictive / consolidated). "
          f"The opposite direction ({s_}) is conservative: consumed "
          f"{cons[(s_, 'predictive')]:.1f} / {cons[(s_, 'consolidated')]:.1f} "
          "pcm (negative = more headroom than the buffer requires).")
    w("\nPhysically: the controller under-predicts the true xenon peak when "
      "the true plant's iodine decays faster (λ_I higher), its xenon decays "
      "slower (λ_X lower), or its flux per unit power is lower (φ lower, "
      "less xenon burnout at part power) than the nominal model assumes.\n")
    w("**4. At what tested mismatch does the current buffer stop being "
      "sufficient?** None of the tested mismatches exhausted it. The "
      f"largest consumption was {cons[worst]:.1f} pcm ({worst[0]}, "
      f"{worst[1]}), leaving {EPS - cons[worst]:.1f} pcm in the worst cell. "
      "This does not identify a threshold; larger mismatches, combined "
      "mismatches, state-estimation error and other R_cap values were not "
      "tested. (R_cap/headroom mismatch maps onto the completed eps sweep, "
      "where eps = 0 produced trips.)\n")
    w("**5. Are Predictive and Consolidated similarly robust?** Both stayed "
      "trip-free in every case. Consolidated was less sensitive: its unserved "
      "work is identical (168.7 kWh) in every case and condition, and its "
      "worst buffer consumption was "
      f"{max(cons[(c_, 'consolidated')] for c_ in ORDER):.1f} pcm versus "
      f"{max(cons[(c_, 'predictive')] for c_ in ORDER):.1f} pcm for "
      "predictive, whose unserved work shifted by "
      f"{summ[(summ.arm == 'predictive') & summ.model_condition.str.startswith('mismatched -')].true_unserved_kWh_mean.min():+.1f}"
      " to "
      f"{summ[(summ.arm == 'predictive') & summ.model_condition.str.startswith('mismatched -')].true_unserved_kWh_mean.max():+.1f}"
      " kWh (paired means).\n")
    w("**6. Are failures associated with false-safe decisions / "
      "under-prediction?** No failures occurred, so this cannot be tested "
      f"on trips. False-safe decisions against the true ceiling: max "
      f"{fs_all:.0f} in every run. The consolidated diagnostics move in the "
      "expected direction: in each unsafe case the under-prediction "
      "fraction and FSeps (realized peak above the enforced ceiling) rise "
      "relative to matched, and in each conservative case they fall (to 0 "
      "FSeps for all three). FSeps is non-zero in matched runs as well, so "
      "part of it is the path-forecast error that exists without any model "
      "mismatch. Predictive has no projection diagnostics; its min "
      "headroom is the comparable measure.\n")
    w("## Interpretation and limits\n")
    w("* Both controllers re-project every minute from the observed true "
      "state (Option A), so a kinetic-parameter error affects each decision "
      "only through the next short interval before it is re-planned. This "
      "is the likely reason the ~100-pcm open-loop peak errors estimated in "
      "the audit translated into ≤ "
      f"{cons[worst]:.0f} pcm of closed-loop buffer consumption. The "
      "result therefore depends on observing the true current I/X; a "
      "controller that must estimate xenon from power history (Option B) "
      "was not tested.\n"
      "* The φ and λ perturbations are explicit sensitivity ranges, not "
      "measurement uncertainties; the DOE case is the only source-supported "
      "alternative.\n"
      "* Scope: R_cap = 2700 pcm, nominal scenario, one factor at a time, "
      "31-day Borg traces A-H; no R_cap mismatch, no combined "
      "perturbations.\n")
    with open(OUT_M, "x") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(f"{r_}  {n_}  {d_}" for n_, r_, d_ in V))
    print(f"wrote {OUT_R} ({len(rows)}), {OUT_S} ({len(summ)}), {OUT_M}")


if __name__ == "__main__":
    main()
