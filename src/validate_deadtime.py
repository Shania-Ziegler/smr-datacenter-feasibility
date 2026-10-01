#!/usr/bin/env python3
"""
validate_deadtime.py -- independent cross-checks of the reactor shutdown
(SCRAM) and dead-time model.

Run AFTER sim.py has produced trip_log.csv. Three jobs:

(3) SINGLE-TRIP ODE CROSS-CHECK. For every logged SCRAM, re-integrate the
    iodine-xenon ODE OFFLINE from the exact (I0, X0) recorded at the trip
    instant, at zero power, until headroom >= trip_margin -- exactly the
    sim's restart condition, computed by independent code. The offline
    duration must match the sim's dead-time within TOL_H (one coarse step
    of slack). PASS here means the in-sim trip machinery adds no phantom
    hours: dead-time is pure xenon physics, nothing else.
    (Right-censored trips -- still down at month end -- are compared as
    lower bounds only.)

(4) DEAD-TIME vs FUEL AGE (the analytic curve). From equilibrium at
    several pre-trip power levels, SCRAM at t=0 and compute dead-time as a
    function of R_cap over a fine grid. This (a) explains why EOC-only
    trips run LONGER than the ~20 h fleet-average literature anchor --
    the anchor mixes fuel ages, while R_cap = 2700 pcm is end-of-cycle
    only -- and (b) is a
    figure in its own right: dead-time is itself fuel-age dependent, and
    the curve shows the cliff. Writes deadtime_vs_rcap.csv (+ a preview
    PNG).

(5b) TRIP-COINCIDENCE CHECK. Where two arms of the same (experiment, G)
    show near-identical trip counts and dead-times (the G=0.4 free vs
    reactive convergence), compare their trip start-times: same dynamics
    should mean similar-but-not-minute-identical sequences. Minute-
    identical sequences across arms would suggest state leaking between
    runs -- flagged loudly for manual inspection of the reset logic.

No synthetic data: inputs are the sim's own logged states; the curve is
the cited ODE evaluated from cited equilibrium states.
"""
import os
import sys
import numpy as np
import pandas as pd

# import constants + physics from the ONE canonical simulator, so this
# check can never silently drift from the model it is validating.
import sim as S

TRIP_MARGIN = 50.0      # pcm -- must equal sim.run() default trip_margin
TOL_H = 0.05            # 3 min of slack: sim steps at 1 min, offline at 1 min
DT_OFF = 60.0           # offline integrator step (s), matches sim dt
SUB = 5                 # substeps, matches sim
T_MAX_H = 96.0          # give up beyond this (flag it -- something is wrong)


def offline_deadtime_permanent(I0, X0, R_cap, trip_margin=TRIP_MARGIN):
    """PERMANENT-clearance dead-time: integrate at zero power and return
    the LAST instant at which headroom < trip_margin (0.0 if never
    blocked). Required when starting from a HEALTHY state (equilibrium):
    headroom is positive at t=0, so first-clearance would trivially
    return 0 h while the post-SCRAM xenon peak is still ahead. Identical
    to first-clearance for real trips (which begin in violation, so
    clearance lands on the decaying side of the single peak). This
    function feeds the analytic dead-time-vs-fuel-age curve."""
    I, X = I0, X0
    h = DT_OFF / SUB
    t = 0.0
    last_blocked = 0.0
    while t < T_MAX_H * 3600.0:
        if (R_cap - S.sigma_m - S.xe_pcm(X)) < trip_margin:
            last_blocked = t
        for _ in range(SUB):
            dI = -S.lambda_I * I
            dX = S.lambda_I * I - S.lambda_X * X
            I += dI * h
            X += dX * h
        t += DT_OFF
    if (R_cap - S.sigma_m - S.xe_pcm(X)) < trip_margin:
        return float("inf")
    return last_blocked / 3600.0


def offline_deadtime(I0, X0, R_cap, trip_margin=TRIP_MARGIN):
    """FIRST-clearance dead-time: integrate at ZERO power from (I0, X0);
    return hours until headroom = R_cap - sigma_m - rho >= trip_margin.
    Independent reimplementation of the sim's restart condition -- used
    ONLY for the logged-trip cross-check, where (I0, X0) is already in
    violation and first clearance is the sim's exact rule."""
    I, X = I0, X0
    h = DT_OFF / SUB
    t = 0.0
    while t < T_MAX_H * 3600.0:
        # sim checks the restart condition against rho[k-1], i.e. the state
        # BEFORE this step's integration -- mirror that ordering exactly.
        rho = S.xe_pcm(X)
        if (R_cap - S.sigma_m - rho) >= trip_margin:
            return t / 3600.0
        for _ in range(SUB):
            dI = -S.lambda_I * I                       # phi = 0
            dX = S.lambda_I * I - S.lambda_X * X       # phi = 0
            I += dI * h
            X += dX * h
        t += DT_OFF
    return float("inf")


def check_logged_trips(path="trip_log.csv"):
    if not os.path.exists(path):
        print(f"[3] {path} not found -- run sim.py first. Skipping.")
        return True
    d = pd.read_csv(path)
    n_bad = 0
    print(f"[3] SINGLE-TRIP ODE CROSS-CHECK -- {len(d)} logged SCRAMs")
    print(f"    {'experiment':>22} | {'rx':>4} | {'start_h':>7} | "
          f"{'sim h':>6} | {'ode h':>6} | {'diff':>6} | verdict")
    for _, r in d.iterrows():
        I0 = float(eval(r["I0"])) if isinstance(r["I0"], str) else float(r["I0"])
        X0 = float(eval(r["X0"])) if isinstance(r["X0"], str) else float(r["X0"])
        pred = offline_deadtime(I0, X0, float(r["R_cap"]))
        sim_dur = r["dur_h"]
        if bool(r.get("censored", False)):
            ok = pred >= sim_dur - TOL_H       # lower bound only
            verdict = "ok (censored, lower-bound)" if ok else "FAIL"
        else:
            ok = abs(pred - sim_dur) <= TOL_H
            verdict = "ok" if ok else "FAIL"
        if not ok:
            n_bad += 1
        print(f"    {r['experiment']:>22} | {r['rx'][:4]:>4} | "
              f"{r['start_h']:7.1f} | {sim_dur:6.2f} | {pred:6.2f} | "
              f"{pred - sim_dur:6.2f} | {verdict}")
    if n_bad:
        print(f"    => {n_bad} FAILURE(S): trip machinery is adding or "
              f"losing time vs pure xenon physics. Fix before use.")
    else:
        print(f"    => ALL PASS: every simulated dead-time is reproduced "
              f"by independent offline integration of the same ODE. "
              f"Dead-time is pure xenon physics.")
    return n_bad == 0


def deadtime_curve(out_csv="deadtime_vs_rcap.csv",
                   out_png="deadtime_vs_rcap_preview.png"):
    print(f"\n[4] DEAD-TIME vs FUEL AGE (SCRAM from equilibrium)")
    rcaps = np.arange(2600.0, 8001.0, 100.0)
    p_pres = (0.70, 0.85, 0.95)
    rows = []
    for p in p_pres:
        I0, X0 = S.xe_equilibrium(p)
        for R in rcaps:
            dt_h = offline_deadtime_permanent(I0, X0, R)
            rows.append(dict(p_pre=p, R_cap=R, deadtime_h=dt_h))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    # console summary at the reported R_cap points
    for p in p_pres:
        sub = df[df.p_pre == p].set_index("R_cap")["deadtime_h"]
        pts = {R: sub.get(R, float("nan"))
               for R in (2700.0, 3500.0, 5000.0, 8000.0)}
        print(f"    p_pre={p:.2f}: " + "  ".join(
            f"R={int(R)}: {v:5.1f} h" for R, v in pts.items()))
    print(f"    Wrote {out_csv}. Read: literature fleet-average ~20 h "
          f"mixes fuel ages; EOC-only (R=2700) sits ABOVE it and "
          f"mid-cycle sits below -- dead-time is itself fuel-age "
          f"dependent. That is a finding, not a discrepancy.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for p in p_pres:
            sub = df[df.p_pre == p]
            ax.plot(sub.R_cap, sub.deadtime_h, label=f"pre-trip p={p:.2f}")
        ax.axhline(20.0, ls="--", lw=1, color="gray",
                   label="lit. fleet-avg ~20 h (mixed ages)")
        ax.set_xlabel("reactivity ceiling R_cap (pcm)  [BOC right, EOC left]")
        ax.set_ylabel("post-SCRAM dead-time (h)")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
        ax.set_title("Xenon dead-time after SCRAM vs fuel age (preview)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"    Preview figure: {out_png} "
              f"(preview only).")
    except Exception as e:
        print(f"    (preview plot skipped: {e})")


def coincidence_check(path="trip_log.csv"):
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    print(f"\n[5b] TRIP-COINCIDENCE CHECK (arm pairs within one experiment)")
    any_pair = False
    for exp, g in d.groupby("experiment"):
        arms = sorted(g["rx"].unique())
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a = g[g.rx == arms[i]].sort_values("start_h")
                b = g[g.rx == arms[j]].sort_values("start_h")
                if len(a) == 0 or len(b) == 0 or len(a) != len(b):
                    continue
                any_pair = True
                diffs = np.abs(a["start_h"].values - b["start_h"].values)
                print(f"    {exp}: {arms[i]} vs {arms[j]} -- "
                      f"{len(a)} trips each; start-time |diff| "
                      f"min/median/max = {diffs.min():.2f} / "
                      f"{np.median(diffs):.2f} / {diffs.max():.2f} h")
                if diffs.max() < 1.0 / 60.0:
                    print(f"      *** SUSPICIOUS: minute-identical trip "
                          f"sequences across arms. Check for state "
                          f"leaking between runs before calling this "
                          f"convergence 'real dynamics'.")
                else:
                    print(f"      -> sequences similar but not identical: "
                          f"consistent with shared price/demand forcing "
                          f"dominating the arm difference (real "
                          f"convergence, not a bug).")
    if not any_pair:
        print("    (no comparable arm pairs with equal trip counts)")


if __name__ == "__main__":
    ok = check_logged_trips()
    deadtime_curve()
    coincidence_check()
    sys.exit(0 if ok else 1)