#!/usr/bin/env python3

import sys
from time import perf_counter_ns
import numpy as np
import pandas as pd

# Iodine-135 / xenon-135 model constants (see physics_audit()).
lambda_I = 2.87e-5; lambda_X = 2.09e-5   # decay constants, 1/s
gamma_I  = 0.0639;  gamma_X  = 0.00237   # fission yields (atoms/fission)
sigma_aX = 2.65e-18                      # Xe-135 absorption cross section, cm^2
nu       = 2.42                          # neutrons per fission
Sigma_f  = 0.39497                       # macroscopic fission cross section, 1/cm
phi_full = 5.0e13                        # flux at rated power, n/cm^2/s
sigma_m  = 100.0                         # model/measurement margin, pcm
ramp_lim = 5.0                           # ramp limit, % of rating per minute

VIOL_TOL_PCM = 0.0   # pcm; see crossing definition in run()


SUB_STEPS = 5   # xenon ODE sub-steps per dt; 5 => 12 s


def xe_pcm(X): return sigma_aX * X / (nu * Sigma_f) * 1e5

def xe_equilibrium(p):
    I = gamma_I*Sigma_f*phi_full*p / lambda_I
    X = (gamma_X*Sigma_f*phi_full*p + lambda_I*I) / (lambda_X + sigma_aX*phi_full*p)
    return I, X

_, X_full = xe_equilibrium(1.0)
rho_eq = xe_pcm(X_full)

def project_peak_rho(I0, X0, p_hold, horizon_s, dt_proj=300.0):

    I, X = I0, X0
    f = phi_full * p_hold
    peak = xe_pcm(X)
    steps = int(horizon_s / dt_proj)
    sub = 2; h = dt_proj / sub
    for _ in range(steps):
        for _ in range(sub):
            dI = gamma_I*Sigma_f*f - lambda_I*I
            dX = gamma_X*Sigma_f*f + lambda_I*I - lambda_X*X - sigma_aX*f*X
            I += dI*h; X += dX*h
        r = xe_pcm(X)
        if r > peak: peak = r
    return peak


def project_peak_rho_path(I0, X0, p_path, path_dt_s):
    """Time-varying counterpart of project_peak_rho (which is unchanged):
    segment j holds power fraction p_path[j] for path_dt_s seconds. Same
    constants, units and forward-Euler convention (2 sub-steps per
    segment); the peak includes the initial state. A constant path of
    int(horizon_s / dt_proj) segments with path_dt_s = dt_proj reproduces
    project_peak_rho exactly (tested)."""
    I, X = I0, X0
    peak = xe_pcm(X)
    sub = 2; h = path_dt_s / sub
    for p in p_path:
        f = phi_full * p
        for _ in range(sub):
            dI = gamma_I*Sigma_f*f - lambda_I*I
            dX = gamma_X*Sigma_f*f + lambda_I*I - lambda_X*X - sigma_aX*f*X
            I += dI*h; X += dX*h
        r = xe_pcm(X)
        if r > peak: peak = r
    return peak


def xe_advance(I, X, p, dt):
    """One plant step at power fraction p with the same arithmetic as the
    inline integration in run() (left inline there so the frozen legacy
    loop is untouched). Used only by run_controller; parity with run() is
    tested."""
    sub = SUB_STEPS; h = dt/sub; f = phi_full * p
    for _ in range(sub):
        dI = gamma_I*Sigma_f*f - lambda_I*I
        dX = gamma_X*Sigma_f*f + lambda_I*I - lambda_X*X - sigma_aX*f*X
        I += dI*h; X += dX*h
    return I, X


# ---- opt-in plant/controller model mismatch (dependency injection) --------
# The module constants above stay the canonical model. run()/run_controller
# accept plant_xe (the simulated reactor's true parameters) and model_xe
# (the parameters used by the controller's xenon projection). None = the
# canonical functions above, executed unchanged. The *_p functions below
# repeat the canonical arithmetic expression-for-expression, so passing
# NOMINAL_XE explicitly reproduces the canonical floats exactly (tested).
# Option A: the controller always observes the TRUE current I and X; only
# its projection dynamics may differ from the plant.
from dataclasses import dataclass as _dataclass, fields as _fields


@_dataclass(frozen=True)
class XeParams:
    lambda_I: float
    lambda_X: float
    gamma_I: float
    gamma_X: float
    sigma_aX: float
    phi_full: float
    nu: float
    Sigma_f: float


NOMINAL_XE = XeParams(lambda_I, lambda_X, gamma_I, gamma_X, sigma_aX,
                      phi_full, nu, Sigma_f)
# Parameters that only enter the atom -> pcm conversion (Sigma_f also the
# production term, where it cancels exactly; see physics_audit). Under
# Option A the controller receives atom counts, so a plant/controller
# difference in these would be a state-reading artifact, not a kinetic
# prediction error: such pairs are rejected.
XE_CONVERSION_FIELDS = ('sigma_aX', 'nu', 'Sigma_f')


def _check_xe_pair(plant_xe, model_xe):
    """Validate an opt-in (plant, model) parameter pair."""
    for q in (plant_xe, model_xe):
        if q is not None and not isinstance(q, XeParams):
            raise TypeError("plant_xe / model_xe must be XeParams or None")
        if q is not None and not all(np.isfinite(getattr(q, f.name)) and
                                     getattr(q, f.name) > 0
                                     for f in _fields(q)):
            raise ValueError("XeParams values must be finite and > 0")
    p = plant_xe or NOMINAL_XE
    m = model_xe or NOMINAL_XE
    bad = [f for f in XE_CONVERSION_FIELDS if getattr(p, f) != getattr(m, f)]
    if bad:
        raise ValueError(
            f"plant/controller mismatch in {bad} is not a kinetic mismatch: "
            f"Sigma_f cancels from xenon reactivity and sigma_aX / nu enter "
            f"the atom->pcm conversion, so with the true I/X handed to the "
            f"controller (Option A) the difference is a state-reading "
            f"artifact. Rejected.")


def _xe_pcm_p(X, q):
    return q.sigma_aX * X / (q.nu * q.Sigma_f) * 1e5


def _xe_equilibrium_p(p, q):
    I = q.gamma_I*q.Sigma_f*q.phi_full*p / q.lambda_I
    X = (q.gamma_X*q.Sigma_f*q.phi_full*p + q.lambda_I*I) / \
        (q.lambda_X + q.sigma_aX*q.phi_full*p)
    return I, X


def _xe_advance_p(I, X, p, dt, q):
    sub = SUB_STEPS; h = dt/sub; f = q.phi_full * p
    for _ in range(sub):
        dI = q.gamma_I*q.Sigma_f*f - q.lambda_I*I
        dX = q.gamma_X*q.Sigma_f*f + q.lambda_I*I - q.lambda_X*X - \
            q.sigma_aX*f*X
        I += dI*h; X += dX*h
    return I, X


def _project_peak_rho_p(I0, X0, p_hold, horizon_s, q, dt_proj=300.0):
    I, X = I0, X0
    f = q.phi_full * p_hold
    peak = _xe_pcm_p(X, q)
    steps = int(horizon_s / dt_proj)
    sub = 2; h = dt_proj / sub
    for _ in range(steps):
        for _ in range(sub):
            dI = q.gamma_I*q.Sigma_f*f - q.lambda_I*I
            dX = q.gamma_X*q.Sigma_f*f + q.lambda_I*I - q.lambda_X*X - \
                q.sigma_aX*f*X
            I += dI*h; X += dX*h
        r = _xe_pcm_p(X, q)
        if r > peak: peak = r
    return peak


def _project_peak_rho_path_p(I0, X0, p_path, path_dt_s, q):
    I, X = I0, X0
    peak = _xe_pcm_p(X, q)
    sub = 2; h = path_dt_s / sub
    for p in p_path:
        f = q.phi_full * p
        for _ in range(sub):
            dI = q.gamma_I*q.Sigma_f*f - q.lambda_I*I
            dX = q.gamma_X*q.Sigma_f*f + q.lambda_I*I - q.lambda_X*X - \
                q.sigma_aX*f*X
            I += dI*h; X += dX*h
        r = _xe_pcm_p(X, q)
        if r > peak: peak = r
    return peak


def physics_audit():
    """--physics-audit: print reference quantities and check numerically
    that xenon reactivity is invariant to Sigma_f. Uses LOCAL copies of the
    equations with Sigma_f as an argument; module constants are untouched.
    The local copies are first checked against the module's own functions."""
    def pcm(X, sf): return sigma_aX * X / (nu * sf) * 1e5
    def eq(p, sf):
        I = gamma_I*sf*phi_full*p / lambda_I
        X = (gamma_X*sf*phi_full*p + lambda_I*I) / (lambda_X + sigma_aX*phi_full*p)
        return I, X
    def proj(I, X, p_hold, horizon_s, sf, dt_proj=300.0):
        f = phi_full * p_hold; peak = pcm(X, sf); h = dt_proj / 2
        for _ in range(int(horizon_s / dt_proj)):
            for _ in range(2):
                dI = gamma_I*sf*f - lambda_I*I
                dX = gamma_X*sf*f + lambda_I*I - lambda_X*X - sigma_aX*f*X
                I += dI*h; X += dX*h
            peak = max(peak, pcm(X, sf))
        return peak

    p_pre, p_hold, hz = 1.0, 0.3, 10*3600.0     # representative dip
    print("PHYSICS AUDIT (constants unchanged; nothing else runs)")
    print(f"  rho_eq (xenon at 100% equilibrium) = {rho_eq:.6f} pcm")
    print(f"  sigma_m                            = {sigma_m:.1f} pcm")
    print(f"  EOC R_cap                          = 2700 pcm")
    print(f"  EOC usable ceiling = 2700 - sigma_m          = {2700.0 - sigma_m:.1f} pcm")
    print(f"  EOC equilibrium headroom = 2700 - sigma_m - rho_eq = "
          f"{2700.0 - sigma_m - rho_eq:.6f} pcm")
    print(f"  phi_full = {phi_full:.6g} n/cm^2/s  [source to be verified/"
          f"documented separately]")
    print(f"  Sigma_f  = {Sigma_f!r} /cm")

    ok = True
    r_mod = rho_eq
    r_loc = pcm(eq(1.0, Sigma_f)[1], Sigma_f)
    I0, X0 = xe_equilibrium(p_pre)
    pk_mod = project_peak_rho(I0, X0, p_hold, hz)
    pk_loc = proj(*eq(p_pre, Sigma_f), p_hold, hz, Sigma_f)
    print(f"  local copies vs module: rho_eq diff {r_loc - r_mod:+.3e}, "
          f"projection diff {pk_loc - pk_mod:+.3e} pcm")
    ok &= (r_loc == r_mod) and (pk_loc == pk_mod)
    for fac in (2.0, 1.37):
        sf = fac * Sigma_f
        r2 = pcm(eq(1.0, sf)[1], sf)
        pk2 = proj(*eq(p_pre, sf), p_hold, hz, sf)
        d_eq, d_pk = r2 - r_loc, pk2 - pk_loc
        good = abs(d_eq) <= 1e-9*abs(r_loc) and abs(d_pk) <= 1e-9*abs(pk_loc)
        ok &= good
        print(f"  Sigma_f x{fac:<4}: rho_eq {r2:.9f} (diff {d_eq:+.3e}); "
              f"projected peak p={p_pre}->{p_hold}, {hz/3600:.0f} h: "
              f"{pk2:.9f} (diff {d_pk:+.3e})  {'OK' if good else 'FAIL'}")
    if not ok:
        sys.exit("PHYSICS AUDIT FAILED: xenon reactivity depends on Sigma_f "
                 "beyond floating-point tolerance (rel 1e-9).")
    print("  Sigma_f invariance: PASS (rel tol 1e-9). Production gamma*Sigma_f"
          "*phi scales X by Sigma_f; rho divides by nu*Sigma_f.")


# Facility model: per-node power (W) at idle and peak, node count, PUE.
P_idle = 208.3; P_peak = 523.7
dyn = P_peak - P_idle
N_nodes = 4000; PUE = 1.2
P_FIXED = PUE * N_nodes * P_idle

def work_power(u): return N_nodes * u * dyn


R_CAP_GRID = (8000.0, 6500.0, 5000.0, 3500.0, 2700.0)   # BOC ... EOC


GAL_TO_L = 3.785411784
WATER_GAL_PER_MWH = 672.0
WATER_GAL_PER_MWH_MIN = 581.0
WATER_GAL_PER_MWH_MAX = 845.0
WATER_L_PER_MWH = WATER_GAL_PER_MWH * GAL_TO_L        # ~2543 L/MWh


FUEL_WATER_GAL_MWH = {
    "gas": 198.0, "coal": 687.0, "nuclear": 672.0, "oil": 826.0,
    "wind": 1.0, "solar": 26.0, "hydro": 0.0, "storage": 0.0,
    "multiple fuels": 198.0, "other renewables": 0.0, "other": 0.0,
}

WUE_L_PER_KWH = 1.1


def load_borg(path, u_pk=0.85):
    d  = pd.read_csv(path)
    t  = d["hours"].values * 3600.0
    tot = d["total_cpu"].values
    u  = u_pk * tot / tot.max()
    fb = (d["defer_low"].values / tot)                             # batch
    fm = ((d["defer_high"].values - d["defer_low"].values) / tot)  # mid
    u_at  = lambda tt: float(np.interp(tt, t, u))
    fb_at = lambda tt: float(np.interp(tt, t, fb))
    fm_at = lambda tt: float(np.interp(tt, t, fm))
    return t, u_at, fb_at, fm_at

def _utc_series(d, path, tag):
    """[strict] parse the UTC timestamp column; cross-check EPT offsets."""
    ucol = next((c for c in d.columns if "datetime" in c.lower()
                 and "utc" in c.lower()), None)
    if ucol is None:
        sys.exit(f"{tag} {path}: no UTC datetime column (EPT alone is "
                 f"ambiguous across DST); columns {list(d.columns)}.")
    utc = pd.to_datetime(d[ucol], format="mixed", errors="coerce")
    if utc.isna().any():
        sys.exit(f"{tag} {path}: {int(utc.isna().sum())} unparseable UTC "
                 f"timestamps, e.g. {d.loc[utc.isna(), ucol].head(5).tolist()}.")
    ecol = next((c for c in d.columns if "datetime" in c.lower()
                 and "ept" in c.lower()), None)
    if ecol is not None:
        off = (pd.to_datetime(d[ecol], format="mixed", errors="coerce")
               - utc).dt.total_seconds() / 3600.0
        bad = ~off.isin((-4.0, -5.0))
        if bad.any():
            sys.exit(f"{tag} {path}: {int(bad.sum())} row(s) whose EPT-UTC "
                     f"offset is not -4/-5 h (ambiguous), e.g. "
                     f"{d.loc[bad, [ucol, ecol]].head(5).values.tolist()}.")
    return utc, ucol


def _coverage(stamps, n_rows, n_needed, path, tag, start=None):
    """[strict] stamps: one UTC stamp per logical hourly sample. Exit, listing
    every problem, unless n_needed consecutive hours from `start` (default:
    first stamp) are present exactly once. Nothing is filled or repaired."""
    st = pd.Series(pd.DatetimeIndex(stamps)).sort_values().reset_index(drop=True)
    t0 = st.iloc[0] if start is None else start
    want = pd.date_range(t0, periods=n_needed, freq="h")
    dups = sorted(set(st[st.duplicated()]))
    offh = sorted(set(st[(st.dt.minute != 0) | (st.dt.second != 0)]))
    missing = list(want.difference(pd.DatetimeIndex(st)))
    if not (dups or offh or missing):
        return t0
    msg = [f"{tag} {path}: INCOMPLETE chronological coverage -- refusing to "
           f"tile, pad, hold, interpolate across or delete anything.",
           f"  total input rows: {n_rows}; distinct UTC timestamps: "
           f"{st.nunique()}; first {st.iloc[0]} / last {st.iloc[-1]} UTC",
           f"  required: {n_needed} consecutive hours from {t0} to "
           f"{want[-1]} UTC"]
    msg.append(f"  missing hours ({len(missing)}):")
    msg += [f"    {m} UTC" for m in missing]
    msg.append(f"  duplicate/ambiguous timestamps ({len(dups)}):")
    msg += [f"    {x} UTC" for x in dups]
    msg.append(f"  non-hourly timestamps ({len(offh)}):")
    msg += [f"    {x} UTC" for x in offh]
    sys.exit("\n".join(msg))


def load_pjm(path, t_grid, strict=False, start_utc=None):
    # strict=True (--year): real chronological UTC hours only, gap-free over
    # the whole grid; never tiled. Returns the same (price, thresh, tiled).
    try:
        d = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"[EXP E] Could not read {path}: {e}")

    if strict:
        utc, _ = _utc_series(d, path, "[YEAR price]")
        d = d.assign(_utc=utc).sort_values("_utc", kind="stable") \
             .reset_index(drop=True)

    cols = {c.lower().strip(): c for c in d.columns}
    # find a price column
    price_col = None
    for key in ("total_lmp_da", "total_lmp_rt", "lmp", "price",
                "system_energy_price_da", "total_lmp"):
        for lc, orig in cols.items():
            if key in lc:
                price_col = orig; break
        if price_col: break
    if price_col is None:
        num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if len(num) == 1:
            price_col = num[0]
    if price_col is None:
        sys.exit(f"[EXP E] No price column found in {path}. Columns: "
                 f"{list(d.columns)}.\nDownload 'Day-Ahead Hourly LMPs' "
                 f"from PJM Data Miner (one pricing node, >= 31 days); "
                 f"expected a column like total_lmp_da.")

    if strict:
        bad = pd.to_numeric(d[price_col], errors="coerce").isna()
        if bad.any():
            sys.exit(f"[YEAR price] {path}: {int(bad.sum())} missing/non-"
                     f"numeric {price_col} values at UTC "
                     f"{[str(x) for x in d.loc[bad, '_utc']]}; not dropped.")
    p = pd.to_numeric(d[price_col], errors="coerce").dropna().values
    problems = []
    if len(p) < 24:
        problems.append(f"only {len(p)} rows (< 1 day of hourly data)")
    if np.nanmax(np.abs(p)) <= 2.0:
        problems.append("values within [-2, 2] -- looks like a normalized "
                        "regulation signal (RegD), not a price trace")
    if np.nanmedian(p) < 1.0 or np.nanmedian(p) > 500.0:
        problems.append(f"median {np.nanmedian(p):.2f} outside plausible "
                        f"$/MWh range")
    if problems:
        sys.exit("[EXP E] error:\n  - "
                 + "\n  - ".join(problems))

    tcol = None
    for key in ("datetime_beginning_ept", "datetime_beginning_utc",
                "datetime", "timestamp", "hour"):
        for lc, orig in cols.items():
            if key in lc:
                tcol = orig; break
        if tcol: break
    hour_of_day = None
    if tcol is not None:

        try:
            ts = pd.to_datetime(d.loc[d[price_col].notna(), tcol],
                                format="mixed")
            hour_of_day = ts.dt.hour.values
        except Exception:
            hour_of_day = None


    n_hours_needed = int(np.ceil((t_grid[-1] - t_grid[0]) / 3600.0)) + 1
    if strict:
        t0 = _coverage(d["_utc"], len(d), n_hours_needed, path,
                       "[YEAR price]", start_utc)
        if d["_utc"].iloc[0] != t0:
            sys.exit(f"[YEAR price] {path}: data before {t0} UTC.")
        print(f"[YEAR price] {path}: {n_hours_needed} real consecutive UTC "
              f"hours from {t0} (file rows {len(d)}); P25 over that window.")
    tiled = False
    if len(p) < n_hours_needed:
        reps = int(np.ceil(n_hours_needed / len(p)))
        p_full = np.tile(p, reps)[:n_hours_needed]
        tiled = True
    else:
        p_full = p[:n_hours_needed]
    th = t_grid[0] + 3600.0 * np.arange(n_hours_needed)
    price_at_grid = np.interp(t_grid, th, p_full)

    thresh = float(np.percentile(p_full, 25))

    print(f"[EXP E] price file: {path}  column: {price_col}  "
          f"rows used: {len(p)}{' (TILED to cover month -- DISCLOSED; '
          'prefer full-month download)' if tiled else ''}")
    print(f"[EXP E] $/MWh min/median/mean/max = {p.min():.1f} / "
          f"{np.median(p):.1f} / {p.mean():.1f} / {p.max():.1f}; "
          f"cheap threshold (P25) = {thresh:.1f}")
    if hour_of_day is not None:
        prof = pd.Series(p).groupby(hour_of_day).mean()
        cheap_hours = list(prof.sort_values().index[:6])
        print(f"[EXP E] cheapest hours of day (mean): {cheap_hours} ")
    return price_at_grid, thresh, tiled

def load_pjm_genmix(path, t_grid, strict=False, start_utc=None):
    # strict=True (--year): every fallback below becomes a hard exit.
    import os
    if strict:
        return _load_genmix_strict(path, t_grid, start_utc)
    if not os.path.exists(path):
        print(f"[genmix] {path} not found -- grid-embedded water will be "
              f"reported as n/a. Download 'Generation by Fuel Type' "
              f"(hourly, same date window) from PJM Data Miner to enable.")
        return None
    try:
        d = pd.read_csv(path)
    except Exception as e:
        print(f"[genmix] could not read {path}: {e} -- skipping.")
        return None
    cols = {c.lower().strip(): c for c in d.columns}
    fcol = next((cols[c] for c in cols if "fuel" in c and "pct" not in c
                 and "percentage" not in c), None)
    mcol = next((cols[c] for c in cols if c in ("mw", "mwh")
                 or c.endswith("_mw")), None)
    tcol = next((cols[c] for c in cols if "datetime" in c or c == "hour"), None)
    if not (fcol and mcol and tcol):
        print(f"[genmix] REFUSING {path}: need fuel/mw/datetime columns, "
              f"got {list(d.columns)}. No fallback will be invented.")
        return None
    d = d[[tcol, fcol, mcol]].dropna()
    d[tcol] = pd.to_datetime(d[tcol], format="mixed")
    d["fuel_key"] = d[fcol].astype(str).str.lower().str.strip()
    unknown = sorted(set(d["fuel_key"]) - set(FUEL_WATER_GAL_MWH))
    if unknown:
        print(f"[genmix] WARNING: unmapped fuel types {unknown} -> treated "
              f"as 0 gal/MWh [DISCLOSED]. Add them to FUEL_WATER_GAL_MWH.")
    d["wf"] = d["fuel_key"].map(FUEL_WATER_GAL_MWH).fillna(0.0)
    g = d.groupby(tcol).apply(
        lambda x: (x[mcol] * x["wf"]).sum() / max(x[mcol].sum(), 1e-9),
        include_groups=False)
    g = g.sort_index()
    hours = ((g.index - g.index[0]).total_seconds().values)
    ewif_gal = g.values                                  # gal/MWh hourly
    n_hours_needed = int(np.ceil((t_grid[-1] - t_grid[0]) / 3600.0)) + 1
    if len(ewif_gal) < n_hours_needed:
        pad = n_hours_needed - len(ewif_gal)
        ewif_gal = np.concatenate([ewif_gal,
                                   np.full(pad, ewif_gal[-1])])
        print(f"[genmix] mix trace shorter than sim by {pad} h -> last real "
              f"value held flat [DISCLOSED]. Prefer full-window download.")
    th = t_grid[0] + 3600.0 * np.arange(n_hours_needed)
    ewif_L = np.interp(t_grid, th, ewif_gal[:n_hours_needed]) * GAL_TO_L
    print(f"[genmix] {path}: hourly EWIF computed. gal/MWh "
          f"min/mean/max = {ewif_gal.min():.0f}/{ewif_gal.mean():.0f}/"
          f"{ewif_gal.max():.0f}  (L/MWh mean {ewif_L.mean():.0f}). "
          f"Fuel->water map printed above; verify vs Macknick tables.")
    return ewif_L


def _load_genmix_strict(path, t_grid, start_utc):
    import os
    tag = "[YEAR genmix]"
    if not os.path.exists(path):
        sys.exit(f"{tag} {path} not found.")
    try:
        d = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"{tag} could not read {path}: {e}")
    cols = {c.lower().strip(): c for c in d.columns}
    fcol = next((cols[c] for c in cols if "fuel" in c and "pct" not in c
                 and "percentage" not in c), None)
    mcol = next((cols[c] for c in cols if c in ("mw", "mwh")
                 or c.endswith("_mw")), None)
    if not (fcol and mcol):
        sys.exit(f"{tag} {path}: need fuel and MW columns; got "
                 f"{list(d.columns)}.")
    utc, _ = _utc_series(d, path, tag)
    d = d.assign(_utc=utc)
    blank = d[[fcol, mcol]].isna().any(axis=1) | \
        pd.to_numeric(d[mcol], errors="coerce").isna()
    if blank.any():
        sys.exit(f"{tag} {path}: {int(blank.sum())} row(s) with blank fuel/"
                 f"MW at UTC {sorted(set(str(x) for x in d.loc[blank, '_utc']))};"
                 f" not dropped.")
    d["fuel_key"] = d[fcol].astype(str).str.lower().str.strip()
    unk = ~d["fuel_key"].isin(FUEL_WATER_GAL_MWH.keys())
    if unk.any():
        lines = [f"{tag} {path}: fuel categories not in FUEL_WATER_GAL_MWH "
                 f"(no zero/merge/invented factor; needs a documented "
                 f"decision):"]
        for raw, g in d[unk].groupby(fcol):
            lines.append(f"  {raw!r}: {len(g)} rows, first {g['_utc'].min()}"
                         f" / last {g['_utc'].max()} UTC")
        sys.exit("\n".join(lines))
    dup = d.duplicated(["_utc", "fuel_key"], keep=False)
    fuels = set(d["fuel_key"])
    per = d.groupby("_utc")["fuel_key"].agg(set)
    partial = per[per.apply(lambda s_: s_ != fuels)]
    n_hours_needed = int(np.ceil((t_grid[-1] - t_grid[0]) / 3600.0)) + 1
    if dup.any() or len(partial):
        msg = [f"{tag} {path}: malformed hours -- nothing repaired."]
        if dup.any():
            msg.append(f"  duplicate (UTC, fuel) rows: "
                       f"{d.loc[dup, ['_utc', fcol]].values.tolist()}")
        for ts, have in partial.items():
            msg.append(f"  {ts} UTC missing fuels {sorted(fuels - have)}")
        sys.exit("\n".join(msg))
    t0 = _coverage(per.index, len(d), n_hours_needed, path, tag, start_utc)
    if per.index[0] != t0:
        sys.exit(f"{tag} {path}: data before {t0} UTC.")
    # existing Macknick-weighted EWIF, unchanged
    d["wf"] = d["fuel_key"].map(FUEL_WATER_GAL_MWH)
    mw = pd.to_numeric(d[mcol])
    g = (mw * d["wf"]).groupby(d["_utc"]).sum() / \
        mw.groupby(d["_utc"]).sum().clip(lower=1e-9)
    ewif_gal = g.sort_index().values[:n_hours_needed]
    th = t_grid[0] + 3600.0 * np.arange(n_hours_needed)
    ewif_L = np.interp(t_grid, th, ewif_gal) * GAL_TO_L
    print(f"{tag} {path}: {n_hours_needed} real consecutive UTC hours from "
          f"{t0}; gal/MWh min/mean/max = {ewif_gal.min():.0f}/"
          f"{ewif_gal.mean():.0f}/{ewif_gal.max():.0f}")
    return ewif_L


YEAR_WINDOWS = 365 * 24 * 12          # five-minute windows in 365 days
YEAR_STEPS = 365 * 24 * 60            # one-minute simulation steps


def validate_year_workload(path, u_pk=0.85):
    """[--year] synthetic 365-day workload from build_year.py (peak mode).
    Exit on any violation; the file is never altered. Provenance columns
    are reported only -- they never reach the scheduler."""
    tag = "[YEAR workload]"
    try:
        d = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"{tag} could not read {path}: {e}")
    req = ["window_5min", "hours", "total_cpu", "rigid_cpu", "defer_low",
           "defer_high", "frac_defer"]
    miss = [c for c in req if c not in d.columns]
    problems = []
    if miss:
        sys.exit(f"{tag} {path}: missing columns {miss}.")
    v = d[req].to_numpy(dtype=float)
    if len(d) != YEAR_WINDOWS:
        problems.append(f"{len(d)} rows != {YEAR_WINDOWS}")
    if not np.isfinite(v).all():
        problems.append("non-finite workload values")
    if not (np.diff(d.window_5min.to_numpy()) == 1).all():
        problems.append("window_5min not continuous")
    if not np.allclose(np.diff(d.hours.to_numpy()), 5/60.0, rtol=0, atol=1e-9):
        problems.append("hours not uniform 5-min window starts")
    if not (d.total_cpu > 0).all():
        problems.append("total_cpu <= 0")
    if abs(d.total_cpu.max() - u_pk) > 1e-9:
        problems.append(f"total_cpu.max() = {d.total_cpu.max()!r} != {u_pk} "
                        f"(not a peak-mode year)")
    if not ((d.defer_low >= 0) & (d.defer_low <= d.defer_high)
            & (d.defer_high <= d.total_cpu)).all():
        problems.append("need 0 <= defer_low <= defer_high <= total_cpu")
    if not np.allclose(d.frac_defer, d.defer_low / d.total_cpu, rtol=0,
                       atol=1e-9):
        problems.append("frac_defer != defer_low / total_cpu")
    if problems:
        sys.exit(f"{tag} {path} rejected:\n  - " + "\n  - ".join(problems))
    prov = [c for c in ("source_cell", "source_window_5min", "segment_index",
                        "cycle_index") if c in d.columns]
    order = ""
    if "segment_index" in d.columns and "source_cell" in d.columns:
        order = " ".join(d.groupby("segment_index")["source_cell"].first())
    print(f"{tag} {path}: PASS -- {len(d)} five-minute windows, max "
          f"utilization {d.total_cpu.max():.12g}, frac_defer consistent. "
          f"Provenance columns (descriptive only): {prov}"
          + (f"; segment order: {order}" if order else ""))


def load_r_cap_file(path, n):
    """--r-cap-file format: plain text, exactly one finite positive R_cap
    (pcm) per simulation minute, one value per line, no header. Lines
    starting with '#' are comments (e.g. the generator's citation/
    provenance). Wrong length / NaN / inf / <= 0 -> exit. Never resized,
    tiled, padded, extrapolated or interpolated."""
    try:
        v = np.loadtxt(path, dtype=float, comments="#", ndmin=1)
    except Exception as e:
        sys.exit(f"[R_cap] could not parse {path} as one number per line "
                 f"(no header allowed): {e}")
    if v.ndim != 1 or len(v) != n:
        sys.exit(f"[R_cap] {path}: {v.size} values, need exactly {n} "
                 f"(one per simulation minute).")
    bad = ~np.isfinite(v) | (v <= 0)
    if bad.any():
        sys.exit(f"[R_cap] {path}: {int(bad.sum())} non-finite or "
                 f"non-positive values (first at index "
                 f"{int(np.argmax(bad))}).")
    print(f"[R_cap] {path}: {n} values OK; min/mean/max = {v.min():.1f} / "
          f"{v.mean():.1f} / {v.max():.1f} pcm")
    return v


def _per_step(x, n, name, lo=None, hi=None, positive=False):
    """Scalar -> constant array; 1-D array -> validated copy. No resizing."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        a = np.full(n, float(a))
    elif a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name}: need a scalar or 1-D array of length "
                         f"{n}, got shape {a.shape}")
    else:
        a = a.copy()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name}: values must be > 0")
    if lo is not None and np.any(a < lo) or hi is not None and np.any(a > hi):
        raise ValueError(f"{name}: values must lie in [{lo}, {hi}]")
    return a


def make_dip_cap(t_grid, day=10.0, hour=4.0, dip_to=0.6,
                 dip_len_h=2.0, spacing_h=5.0, n_dips=2):
    cap = np.ones(len(t_grid))
    t0 = (day*24.0 + hour) * 3600.0 + t_grid[0]
    for j in range(n_dips):
        a = t0 + j * spacing_h * 3600.0
        b = a + dip_len_h * 3600.0
        cap[(t_grid >= a) & (t_grid < b)] = dip_to
    return cap

def run(fb_at, fm_at, mode, t_grid, u_at, P_rated, dt, R_cap,
        mid_horizon_s=3600.0, tau_cool=300.0, settle_h=6.0,
        rx_gate=None, rx_mode=None, force_cap=None,
        price=None, cheap_thresh=None, G_max_frac=0.0,
        grid_override=None,
        predict_horizon_s=10*3600.0, ewif=None, return_series=False,
        eps_buffer=0.0, trip=False, trip_margin=50.0, emergency_grid=True,
        hysteresis_h=4.0, hysteresis_tol=1e-6, grid_availability=None,
        timing='off', controller_mode=None, plant_xe=None, model_xe=None,
        **controller_kw):
    if controller_mode in ("computing_only", "consolidated",
                           "consolidated_partial_grid"):
        if rx_gate is not None or rx_mode is not None or \
           grid_override is not None or hysteresis_h != 4.0 or \
           hysteresis_tol != 1e-6:
            raise ValueError("controller_mode cannot be combined with "
                             "rx_gate / rx_mode / grid_override / "
                             "hysteresis settings")
        return run_controller(
            fb_at, fm_at, mode, t_grid, u_at, P_rated, dt, R_cap,
            controller_mode, mid_horizon_s=mid_horizon_s, tau_cool=tau_cool,
            settle_h=settle_h, force_cap=force_cap, price=price,
            cheap_thresh=cheap_thresh, G_max_frac=G_max_frac,
            predict_horizon_s=predict_horizon_s, ewif=ewif,
            return_series=return_series, eps_buffer=eps_buffer, trip=trip,
            trip_margin=trip_margin, emergency_grid=emergency_grid,
            grid_availability=grid_availability, timing=timing,
            plant_xe=plant_xe, model_xe=model_xe, **controller_kw)
    if controller_mode is not None or controller_kw:
        raise TypeError(f"run(): unknown controller_mode {controller_mode!r}"
                        f" / arguments {sorted(controller_kw)}")

    # R_cap: scalar (fixed fuel age, as in every monthly experiment) or a
    #   1-D array with one value per t_grid step (supplied by the caller;
    #   no fuel-age trajectory is generated here). Policy decisions at step k
    #   use R_cap_ts[k]; headroom of the state rho[k] uses R_cap_ts[k].
    # rx_mode='hysteresis': fixed-delay comparison baseline.
    #   After a genuine down-ramp (P_rx drop > hysteresis_tol) up-ramps are
    #   blocked for hysteresis_h hours; down-ramps stay allowed and restart
    #   the timer. Decides from P_rx history and time only -- never rho,
    #   I, X, projections or R_cap headroom. Its blocked up-ramp shortfall is
    #   booked in the same policy-gate bucket (unmet_xen) as reactive's.
    # grid_availability: None (unchanged behavior) or per-step fraction in
    #   [0, 1] of the G_max_frac interconnection that is physically usable;
    #   caps cheap-price, emergency and override imports.
    # timing: 'off' (no timers) | 'coarse' (one timer pair around each
    #   step's reactor-policy decision) | 'detailed' (coarse + a timer
    #   around every project_peak_rho call). Never alters any decision.

    if rx_mode is None:
        rx_mode = 'reactive' if (rx_gate is None or rx_gate) else 'free'
    assert rx_mode in ('free', 'reactive', 'predictive', 'hysteresis')
    assert timing in ('off', 'coarse', 'detailed')
    n = len(t_grid)
    R_cap_ts = _per_step(R_cap, n, "R_cap", positive=True)
    if rx_mode == 'hysteresis' and not (hysteresis_h > 0):
        raise ValueError("hysteresis_h must be > 0")
    if grid_availability is not None:
        grid_availability = _per_step(grid_availability, n,
                                      "grid_availability", lo=0.0, hi=1.0)
        if grid_override is not None and not G_max_frac > 0.0:
            raise ValueError("grid_availability with grid_override needs "
                             "G_max_frac > 0 (the interconnection size)")
    hyst_s = hysteresis_h * 3600.0
    t_last_down = None
    margin_guard_steps = 0
    pol_ns = np.zeros(n, dtype=np.int64) if timing != 'off' else None
    proj_ns = []
    # opt-in model mismatch: plant_xe drives the plant only; model_xe only
    # the predictive projection (the one legacy arm that projects)
    _check_xe_pair(plant_xe, model_xe)
    if model_xe is not None and rx_mode != 'predictive':
        raise ValueError(f"model_xe only applies to rx_mode='predictive' "
                         f"(the {rx_mode!r} arm has no xenon projection)")
    if model_xe is None:
        _base_proj = None
    else:
        def _base_proj(I_, X_, p_, hz_):
            return _project_peak_rho_p(I_, X_, p_, hz_, model_xe)
    if timing == 'detailed':
        def proj(I_, X_, p_, hz_):
            a_ = perf_counter_ns()
            v_ = (project_peak_rho(I_, X_, p_, hz_) if _base_proj is None
                  else _base_proj(I_, X_, p_, hz_))
            proj_ns.append(perf_counter_ns() - a_)
            return v_
    else:
        proj = project_peak_rho if _base_proj is None else _base_proj
    P_rx = np.zeros(n); Pd = np.zeros(n); rho = np.zeros(n)
    unmet = np.zeros(n)
    sc_cap = np.zeros(n); sc_force = np.zeros(n)
    sc_ramp = np.zeros(n); sc_xen = np.zeros(n)
    shed_ts = np.zeros(n); drained_ts = np.zeros(n)
    grid_ts = np.zeros(n)
    drop_ts = np.zeros(n)      # per-step expired work (W-steps); record only

    qb, qm = [], []
    dropped_b = dropped_m = 0.0
    wsum_b = esum_b = wsum_m = esum_m = 0.0
    cap_binds = 0; gap_windows = 0; cheap_steps = 0
    shed_acc = 0.0; drained_acc = 0.0
    refused_steps = 0
    floor_raise_sum = 0.0

    tripped = False; n_trips = 0; trip_steps = 0
    sc_trip = np.zeros(n)

    tripped_ts = np.zeros(n, dtype=bool)
    trip_log = []
    surplus_ts = np.zeros(n)
    gridcost_ts = np.zeros(n)
    gridwater_ts = np.zeros(n)
    served_ts = np.zeros(n)
    I_ts = np.zeros(n); X_ts = np.zeros(n)

    mw_b = int(24*3600 / dt)
    mw_m = int(mid_horizon_s / dt)
    ramp_step = ramp_lim/100.0 * (dt/60.0)
    a_cool = 1.0 - np.exp(-dt / tau_cool) if tau_cool > 0 else 1.0

    u0 = u_at(t_grid[0]); Pw0 = work_power(u0)
    P_cool = (PUE - 1.0) * Pw0
    p0 = min(1.0, (P_FIXED + Pw0 + P_cool) / P_rated)
    if plant_xe is None:
        I, X = xe_equilibrium(p0)
        P_rx[0] = p0; rho[0] = xe_pcm(X)
    else:                               # true plant parameters
        I, X = _xe_equilibrium_p(p0, plant_xe)
        P_rx[0] = p0; rho[0] = _xe_pcm_p(X, plant_xe)
    I_ts[0] = I; X_ts[0] = X

    for k in range(1, n):
        u = u_at(t_grid[k]); P_work = work_power(u)


        if trip:
            hd_prev = R_cap_ts[k-1] - sigma_m - rho[k-1]
            if (not tripped) and hd_prev < 0.0:
                tripped = True; n_trips += 1
                trip_log.append(dict(
                    start_h=(t_grid[k] - t_grid[0]) / 3600.0,
                    end_h=None, I0=I, X0=X,
                    rho0=rho[k-1], censored=False))
            elif tripped and hd_prev >= trip_margin:
                tripped = False
                if trip_log and trip_log[-1]["end_h"] is None:
                    trip_log[-1]["end_h"] = (t_grid[k] - t_grid[0]) / 3600.0
        if tripped:
            trip_steps += 1
            tripped_ts[k] = True

        demand_total = P_FIXED + P_work + P_cool
        G_eff = G_max_frac if grid_availability is None \
            else G_max_frac * grid_availability[k]
        grid_use = 0.0
        if grid_override is not None:
            # ORACLE PATH: explicit import schedule replaces the
            # automatic price rule. Physics/accounting unchanged.
            if grid_override[k] > 0.0:
                cheap_steps += 1
                grid_use = min(grid_override[k] * P_rated,
                               demand_total)
                if grid_availability is not None:
                    grid_use = min(grid_use, G_eff * P_rated)
        elif price is not None and G_max_frac > 0.0 and \
           price[k] <= cheap_thresh:
            cheap_steps += 1
            grid_use = min(G_eff * P_rated, demand_total)

        if tripped and emergency_grid and price is not None \
           and G_max_frac > 0.0:
            grid_use = min(G_eff * P_rated, demand_total)
        grid_ts[k] = grid_use
        demand_rx = demand_total - grid_use

        lim_cap   = 1.0
        lim_force = force_cap[k] if force_cap is not None else 1.0
        lim_trip  = 0.0 if tripped else 1.0
        lim_ramp  = P_rx[k-1] + ramp_step

        # ---- reactor-policy decision (timed when timing != 'off') ----
        if pol_ns is not None:
            _t_pol = perf_counter_ns()
        if rx_mode == 'reactive':
            margin_eq = R_cap_ts[k] - rho_eq - sigma_m
            if margin_eq > 0.0:
                rfac = np.clip((R_cap_ts[k] - rho[k-1] - sigma_m) / margin_eq,
                               0.0, 1.0)
            else:                      # no equilibrium margin: gate closed
                rfac = 0.0; margin_guard_steps += 1
        elif rx_mode == 'hysteresis':
            if k >= 2 and P_rx[k-1] < P_rx[k-2] - hysteresis_tol:
                t_last_down = t_grid[k-1]
            rfac = 0.0 if (t_last_down is not None and
                           t_grid[k] - t_last_down < hyst_s) else 1.0
        else:
            rfac = 1.0
        lim_xen  = P_rx[k-1] + ramp_step * rfac
        reach_hi = min(lim_cap, lim_force, lim_trip, lim_xen)

        reach_lo = 0.0 if tripped else (P_rx[k-1] - ramp_step)
        if rx_mode == 'predictive':
            # requested setpoint: facility demand net of grid, within reach
            p_target = max(reach_lo, min(reach_hi,
                          (P_FIXED + P_work + P_cool - grid_use) / P_rated))
            p_target = max(p_target, 0.0)
            # Speed heuristic, not a physical bound: the projection is
            # skipped when the ceiling is more than XE_OVERSHOOT_BOUND above
            # the current xenon worth. A full-power-to-zero dip can overshoot
            # by ~3900 pcm in 10 h, so the skip is safe only when the
            # ceiling also exceeds any reachable peak (true for R_cap 8000;
            # at 2700 and 3500 the check always runs).
            XE_OVERSHOOT_BOUND = 1200.0
            ceil_pred = R_cap_ts[k] - sigma_m - eps_buffer
            need_check = (p_target < P_rx[k-1] - 1e-4) and \
                         (ceil_pred < rho[k-1] + XE_OVERSHOOT_BOUND)
            peak_proj = (proj(I, X, max(p_target, 1e-3),
                              predict_horizon_s)
                         if need_check else 0.0)
            if peak_proj > ceil_pred:
                # raise the floor: shallowest projected-safe setpoint
                lo, hi = p_target, min(reach_hi, 1.0)
                for _ in range(12):
                    mid = 0.5*(lo+hi)
                    if proj(I, X, max(mid,1e-3),
                            predict_horizon_s) > ceil_pred:
                        lo = mid         # still unsafe
                    else:
                        hi = mid         # safe
                reach_lo = max(reach_lo, hi)
                if hi > p_target + 1e-9:
                    refused_steps += 1
                    floor_raise_sum += (hi - p_target) * P_rated
        if pol_ns is not None:
            pol_ns[k] = perf_counter_ns() - _t_pol
        # ---- end reactor-policy decision ----

        gap = max(0.0, demand_rx - reach_hi * P_rated)

        cap_b = fb_at(t_grid[k]) * P_work
        cap_m = fm_at(t_grid[k]) * P_work
        if gap > 1e-9:
            gap_windows += 1
            if gap > cap_b + cap_m + 1e-9:
                cap_binds += 1

        if mode == "blind":
            tot_cap = cap_b + cap_m
            if tot_cap > 1e-9:
                take_total = min(gap, tot_cap)
                shed_b = take_total * (cap_b / tot_cap)
                shed_m = take_total * (cap_m / tot_cap)
            else:
                shed_b = shed_m = 0.0
        else:
            shed_b = min(gap, cap_b)
            shed_m = min(max(0.0, gap - shed_b), cap_m)
        shed_ts[k] = shed_b + shed_m
        shed_acc  += shed_b + shed_m
        P_work_eff = P_work - shed_b - shed_m

        if mode == "single":
            if shed_b + shed_m > 1e-9: qb.append([shed_b + shed_m, 0])
        else:
            if shed_b > 1e-9: qb.append([shed_b, 0])
            if shed_m > 1e-9: qm.append([shed_m, 0])

        spare = reach_hi * P_rated - (P_FIXED + P_work_eff + P_cool - grid_use)
        if spare > 1e-9:
            if mode == "blind":
                order = [(it, 'b') for it in qb] + [(it, 'm') for it in qm]
                order.sort(key=lambda p_: -p_[0][1])
            else:
                order = [(it, 'm') for it in qm] + [(it, 'b') for it in qb]
            for item, tier in order:
                if spare <= 1e-9: break
                take = min(item[0], spare)
                item[0] -= take; spare -= take
                P_work_eff += take
                drained_ts[k] += take
                drained_acc   += take
                if tier == 'b': wsum_b += take*item[1]; esum_b += take
                else:           wsum_m += take*item[1]; esum_m += take
            qm = [it for it in qm if it[0] > 1e-9]
            qb = [it for it in qb if it[0] > 1e-9]

        P_cool += a_cool * ((PUE - 1.0) * P_work_eff - P_cool)
        Pd[k] = (P_FIXED + P_work_eff + P_cool - grid_use) / P_rated
        Pd[k] = max(Pd[k], 0.0)
        P_rx[k] = min(max(Pd[k], reach_lo), reach_hi)

        surplus_ts[k] = max(0.0, P_rx[k] - Pd[k]) * P_rated
        if grid_use > 0.0 and price is not None:
            gridcost_ts[k] = price[k] * (grid_use * dt / 3600.0) / 1e6
        if grid_use > 0.0 and ewif is not None:
            gridwater_ts[k] = ewif[k] * (grid_use * dt / 3600.0) / 1e6
        served_ts[k] = P_work_eff

        s  = lambda L: max(0.0, Pd[k] - L)
        s0 = s(lim_cap)
        s1 = s(min(lim_cap, lim_force))
        s1t = s(min(lim_cap, lim_force, lim_trip))
        s2 = s(min(lim_cap, lim_force, lim_trip, lim_ramp))
        s3 = s(reach_hi)
        sc_cap[k]   = s0        * P_rated
        sc_force[k] = (s1 - s0) * P_rated
        sc_trip[k]  = (s1t - s1) * P_rated
        sc_ramp[k]  = (s2 - s1t) * P_rated
        sc_xen[k]   = (s3 - s2) * P_rated
        unmet[k]    = s3        * P_rated

        # xenon ODE: forward Euler, SUB_STEPS sub-steps per dt
        if plant_xe is None:
            sub = SUB_STEPS; h = dt/sub; f = phi_full * P_rx[k]
            for _ in range(sub):
                dI = gamma_I*Sigma_f*f - lambda_I*I
                dX = gamma_X*Sigma_f*f + lambda_I*I - lambda_X*X - sigma_aX*f*X
                I += dI*h; X += dX*h
            rho[k] = xe_pcm(X)
        else:                           # true plant parameters
            I, X = _xe_advance_p(I, X, P_rx[k], dt, plant_xe)
            rho[k] = _xe_pcm_p(X, plant_xe)
        I_ts[k] = I; X_ts[k] = X

        for it in qb: it[1] += 1
        for it in qm: it[1] += 1
        _db = sum(it[0] for it in qb if it[1] > mw_b)
        _dm = sum(it[0] for it in qm if it[1] > mw_m)
        dropped_b += _db
        dropped_m += _dm
        drop_ts[k] = _db + _dm
        qb = [it for it in qb if it[1] <= mw_b]
        qm = [it for it in qm if it[1] <= mw_m]

    if trip and trip_log and trip_log[-1]["end_h"] is None:
        trip_log[-1]["end_h"] = (t_grid[-1] - t_grid[0]) / 3600.0
        trip_log[-1]["censored"] = True

    Ws = dt / 3600.0
    leftover = (sum(it[0] for it in qb) + sum(it[0] for it in qm)) * Ws
    dropped  = (dropped_b + dropped_m) * Ws
    residual = shed_acc*Ws - (drained_acc*Ws + dropped + leftover)

    sidx = int(settle_h*3600/dt)
    shed_in_settle = shed_ts[:sidx].sum() * Ws
    for a in (unmet, sc_cap, sc_force, sc_trip, sc_ramp, sc_xen,
              shed_ts, drained_ts, grid_ts, surplus_ts, gridcost_ts,
              gridwater_ts, served_ts):
        a[:sidx] = 0
    Wh = lambda v: v.sum() * dt / 3600.0

    gen_Wh     = float((P_rx[sidx:] * P_rated).sum() * dt / 3600.0)
    surplus_Wh = Wh(surplus_ts)
    served_it_Wh = Wh(served_ts)
    water_L         = gen_Wh     / 1e6 * WATER_L_PER_MWH   # SMR cooling
    water_surplus_L = surplus_Wh / 1e6 * WATER_L_PER_MWH
    water_grid_L    = float(gridwater_ts.sum()) if ewif is not None else None
    water_dc_L      = served_it_Wh / 1e3 * WUE_L_PER_KWH   # facility WUE
    grid_cost = float(gridcost_ts.sum())

    _wgrid = water_grid_L if water_grid_L is not None else 0.0
    water_sys_L = water_L + _wgrid + water_dc_L
    water_L_per_kWh_served = water_sys_L / max(served_it_Wh / 1e3, 1e-9)
    grid_cost_per_MWh_served = grid_cost / max(served_it_Wh / 1e6, 1e-9)

    headroom = R_cap_ts[sidx:] - sigma_m - rho[sidx:]

    # Crossing counted only below -VIOL_TOL_PCM (1 pcm):
    # ~4e-4 of the R_cap ceiling and below the truncation
    # error of the 12 s forward-Euler xenon integration.
    # min_headroom / max_exceed report RAW values.
    viol = headroom < -VIOL_TOL_PCM
    cross_episodes = int((viol[1:] & ~viol[:-1]).sum() + (1 if viol[0] else 0))
    _longest = 0; _cur = 0
    for _v in viol:
        _cur = _cur + 1 if _v else 0
        if _cur > _longest: _longest = _cur
    longest_viol_min = _longest * dt / 60.0
    outage = tripped_ts[sidx:]
    viol_op  = viol & ~outage
    viol_out = viol & outage
    crossings_op     = int(viol_op.sum())
    crossings_outage = int(viol_out.sum())
    cross_episodes_op = int((viol_op[1:] & ~viol_op[:-1]).sum()
                            + (1 if viol_op[0] else 0))
    _pw = np.array([work_power(u_at(tt)) for tt in t_grid[sidx:]])
    demand_Wh = float((P_FIXED + _pw * PUE).sum() * dt / 3600.0)
    out = dict(unmet=Wh(unmet), shed=Wh(shed_ts), drained=Wh(drained_ts),
                dropped=dropped, leftover=leftover,
                true_unserved=Wh(unmet) + dropped + leftover,
                dropped_b=dropped_b*Ws, dropped_m=dropped_m*Ws,
                delay_b=(wsum_b/esum_b*dt/3600.0 if esum_b > 0 else 0.0),
                delay_m=(wsum_m/esum_m*dt/3600.0 if esum_m > 0 else 0.0),
                unmet_cap=Wh(sc_cap), unmet_force=Wh(sc_force),
                unmet_trip=Wh(sc_trip),
                unmet_ramp=Wh(sc_ramp), unmet_xen=Wh(sc_xen),
                n_trips=n_trips, deadtime_h=trip_steps*dt/3600.0,
                grid_Wh=Wh(grid_ts), cheap_steps=cheap_steps,
                peak_rho=float(rho[sidx:].max()),
                min_headroom=float(headroom.min()),
                crossings=int(viol.sum()),
                crossings_op=crossings_op,
                crossings_outage=crossings_outage,
                cross_episodes=cross_episodes,
                cross_episodes_op=cross_episodes_op,
                longest_viol_min=longest_viol_min,
                demand_Wh=demand_Wh,
                unserved_pct=100.0*(Wh(unmet)+dropped+leftover)/demand_Wh,
                max_exceed=float(max(0.0, -headroom.min())),
                xen_binds=int((sc_xen[sidx:]  > 1e-6).sum()),
                ramp_binds=int((sc_ramp[sidx:] > 1e-6).sum()),
                frc_binds=int((sc_force[sidx:] > 1e-6).sum()),
                cap_binds=cap_binds, gap_windows=gap_windows,
                p_lo=float(P_rx[sidx:].min()), p_hi=float(P_rx[sidx:].max()),
                p_mean=float(P_rx[sidx:].mean()),
                refused_steps=refused_steps,
                floor_raise_Wh=floor_raise_sum*Ws,
                ramp_distance=float(np.abs(np.diff(P_rx[sidx:])).sum()),
                gen_Wh=gen_Wh, surplus_Wh=surplus_Wh,
                served_it_Wh=served_it_Wh,
                water_L=water_L, water_surplus_L=water_surplus_L,
                water_grid_L=water_grid_L, water_dc_L=water_dc_L,
                water_sys_L=water_sys_L,
                water_L_per_kWh_served=water_L_per_kWh_served,
                grid_cost=grid_cost,
                grid_cost_per_MWh_served=grid_cost_per_MWh_served,
                residual=residual, shed_in_settle=shed_in_settle,
                trip_log=trip_log)
    if rx_mode == 'reactive':
        out["margin_guard_steps"] = margin_guard_steps
    if rx_mode == 'hysteresis':
        out["hysteresis_h"] = hysteresis_h
    if timing != 'off':
        us = pol_ns[1:] / 1e3
        out.update(timing_mode=timing, policy_decisions=int(n - 1),
                   policy_total_us=float(us.sum()),
                   policy_mean_us=float(us.mean()),
                   policy_p95_us=float(np.percentile(us, 95)),
                   policy_p99_us=float(np.percentile(us, 99)),
                   policy_max_us=float(us.max()))
    if timing == 'detailed':
        pj = np.asarray(proj_ns, dtype=float) / 1e3
        nan = float('nan')
        out.update(projection_calls=int(len(pj)),
                   projection_total_us=float(pj.sum()),
                   projection_mean_us=float(pj.mean()) if len(pj) else nan,
                   projection_p95_us=(float(np.percentile(pj, 95))
                                      if len(pj) else nan),
                   projection_p99_us=(float(np.percentile(pj, 99))
                                      if len(pj) else nan),
                   projection_max_us=float(pj.max()) if len(pj) else nan)
    if return_series:
        # full-resolution state (ceiling is per-step)
        ceil_ts = R_cap_ts - sigma_m
        out["series"] = dict(t=t_grid, time_s=t_grid,
                             P_rx=P_rx, reactor_power=P_rx,
                             Pd=Pd, demand_power=Pd,
                             rho=rho, iodine=I_ts, xenon=X_ts,
                             R_cap=R_cap_ts, ceiling=ceil_ts,
                             usable_ceiling=ceil_ts,
                             headroom=ceil_ts - rho,
                             grid=grid_ts, tripped=tripped_ts)
        if return_series == 'full':
            # opt-in trace extras: arrays run() already computes (energy
            # arrays zeroed over the settle window exactly as in the
            # metrics). true_unserved == (unmet_W.sum()*dt/3600
            # + dropped_Wh.sum() + leftover).
            out["series"].update(
                unmet_W=unmet, dropped_Wh=drop_ts * Ws, shed_W=shed_ts,
                drained_W=drained_ts, served_it_W=served_ts,
                surplus_W=surplus_ts, unmet_xen_W=sc_xen,
                unmet_trip_W=sc_trip, unmet_ramp_W=sc_ramp)
    return out


def _rolling_max_forward(a, w):
    """out[i] = max(a[i : i + w]) (window truncated at the end)."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([a, np.full(w - 1, -np.inf)])
    return sliding_window_view(pad, w).max(axis=1)


def run_controller(fb_at, fm_at, mode, t_grid, u_at, P_rated, dt, R_cap,
                   controller_mode, mid_horizon_s=3600.0, tau_cool=300.0,
                   settle_h=6.0, force_cap=None, price=None,
                   cheap_thresh=None, G_max_frac=0.0,
                   predict_horizon_s=10*3600.0, ewif=None,
                   return_series=False, eps_buffer=0.0, trip=False,
                   trip_margin=50.0, emergency_grid=True,
                   grid_availability=None, timing='off',
                   controller_cfg=None, known_outages=(),
                   deadline_shock=None, decision_log=None, event_meta=None,
                   project_path=None, step_hook=None, plant_xe=None,
                   model_xe=None, partial_grid_audit=0,
                   price_diagnostics=False):
    """Plant for the opt-in controller arms (computing_only, consolidated).

    Physics, constants, cooling lag, trip rule and metric definitions are
    those of run(); the xenon step uses xe_advance (same arithmetic). All
    scheduling decisions come from consolidated_scheduler. The controller
    sees only values at the current step; computing_only never receives
    I, X, rho, R_cap, headroom, a ceiling or a projection callback.

    known_outages : ((k0, k1),) explicitly scheduled outages only.
    deadline_shock: (k, fraction, deadline_steps) or None.
    decision_log  : None | 'episodes' | 'minute'.
    project_path  : consolidated only; default project_peak_rho_path.
    step_hook     : test hook called with k before each step's samples are
                    read (0 = initial state), and with None after the loop.
    partial_grid_audit : consolidated_partial_grid only; diagnostic audit
                    points per partial-import search (0 = off).
    price_diagnostics : add the price-import diagnostic columns to the
                    output (off by default so canonical rows keep their
                    exact column set).
    """
    import dataclasses
    import consolidated_scheduler as CS
    if controller_mode not in CS.ALL_CONTROLLER_MODES:
        raise ValueError(f"unknown controller_mode {controller_mode!r}")
    if mode != 'tiered':
        raise ValueError("controller arms use their own least-slack-first "
                         "queue; sched must be 'tiered'")
    if timing not in ('off', 'coarse', 'detailed'):
        raise ValueError(f"timing {timing!r}")
    if decision_log not in (None, 'episodes', 'minute'):
        raise ValueError(f"decision_log {decision_log!r}")
    consolidated = controller_mode in CS.REACTOR_AWARE_MODES
    _check_xe_pair(plant_xe, model_xe)
    if model_xe is not None and (not consolidated or project_path is not None):
        raise ValueError("model_xe applies only to consolidated's default "
                         "projection (computing_only never projects; an "
                         "explicit project_path already fixes the model)")
    if not consolidated and project_path is not None:
        raise ValueError("computing_only must not be given a projection "
                         "callback")
    n = len(t_grid)
    R_cap_ts = _per_step(R_cap, n, "R_cap", positive=True)
    if grid_availability is not None:
        grid_availability = _per_step(grid_availability, n,
                                      "grid_availability", lo=0.0, hi=1.0)
    cfg = dataclasses.replace(controller_cfg or CS.ControllerConfig(),
                              horizon_s=predict_horizon_s)
    hook = step_hook or (lambda _k: None)

    proj_ns = []
    if consolidated:
        base_proj = project_path or project_peak_rho_path
        if model_xe is not None:        # controller's (mismatched) model
            def base_proj(I_, X_, path_, pdt_):
                return _project_peak_rho_path_p(I_, X_, path_, pdt_,
                                                model_xe)
        if timing == 'detailed':
            def proj(I_, X_, path_, pdt_):
                a_ = perf_counter_ns()
                v_ = base_proj(I_, X_, path_, pdt_)
                proj_ns.append(perf_counter_ns() - a_)
                return v_
        else:
            proj = base_proj
    else:
        proj = None
    pol_ns = np.zeros(n, dtype=np.int64) if timing != 'off' else None

    mw_b = int(24*3600 / dt)
    mw_m = int(mid_horizon_s / dt)
    ramp_step = ramp_lim/100.0 * (dt/60.0)
    a_cool = 1.0 - np.exp(-dt / tau_cool) if tau_cool > 0 else 1.0
    it_cost = 1.0 + a_cool * (PUE - 1.0)
    ctrl = CS.WorkloadController(controller_mode, dt, P_rated, mw_b, mw_m,
                                 cfg, project_path=proj,
                                 known_outages=known_outages,
                                 partial_audit_points=partial_grid_audit)

    P_rx = np.zeros(n); Pd = np.zeros(n); rho = np.zeros(n)
    I_ts = np.zeros(n); X_ts = np.zeros(n)
    grid_ts = np.zeros(n); g_pr_ts = np.zeros(n); g_br_ts = np.zeros(n)
    g_em_ts = np.zeros(n); surplus_ts = np.zeros(n); served_ts = np.zeros(n)
    rigid_un = np.zeros(n); fac_un = np.zeros(n)
    shed_ts = np.zeros(n); drained_ts = np.zeros(n)
    gridcost_ts = np.zeros(n); gridwater_ts = np.zeros(n)
    tripped_ts = np.zeros(n, dtype=bool)
    state_ts = np.zeros(n, dtype=np.int8)
    spike_ts = np.zeros(n, dtype=bool); bind_ts = np.zeros(n, dtype=np.int8)
    excess_ts = np.full(n, np.nan); req_W = np.zeros(n)
    fc_ts = np.full(n, np.nan); pred_miss = np.zeros(n)
    dropped_ts = np.zeros(n)
    pk_req = np.full(n, np.nan); pk_acc = np.full(n, np.nan)
    ceil_ts = np.full(n, np.nan); modified = np.zeros(n, dtype=bool)
    nofeas_ts = np.zeros(n, dtype=bool); cancel_ts = np.zeros(n, dtype=bool)
    cool_ts = np.zeros(n); offered_ts = np.zeros(n)
    p_cand = np.full(n, np.nan); g_cand = np.zeros(n)
    floor_raise_W = np.zeros(n)
    min_slack_ts = np.full(n, np.nan)
    unmet_parts = {b: np.zeros(n) for b in ('cap', 'force', 'trip', 'ramp')}
    pr_req_ts = np.zeros(n); pr_appr_ts = np.zeros(n)
    pr_part_ts = np.zeros(n, dtype=bool); aud_miss_ts = np.zeros(n, dtype=bool)
    aud_gap_ts = np.zeros(n)
    STATE_CODE = {CS.NORMAL: 0, CS.STRESSED: 1, CS.ISLANDED: 2}

    tripped = False; n_trips = 0; trip_steps = 0; trip_log = []
    acc = dict(def_b=0.0, def_m=0.0, dr_b=0.0, dr_m=0.0, age_b=0.0,
               age_m=0.0, drop_b=0.0, drop_m=0.0, drop_nat=0.0, drop_syn=0.0,
               work_resid=0.0, supply_resid=0.0, projections=0,
               age_wsum=0.0, age_esum=0.0)
    max_age = 0; promoted_W = 0.0
    episodes = CS.EpisodeLog(); minute_rows = []
    meta = dict(event_meta or {})
    meta_log = {k_: meta.get(k_) for k_ in (
        'scenario', 'synthetic_deadline', 'deadline_source',
        'deadline_shock_fraction', 'deadline_shock_minutes',
        'deadline_shock_start_h', 'blackout_type', 'spike_multiplier')}

    hook(0)
    u0 = u_at(t_grid[0]); Pw0 = work_power(u0)
    P_cool = (PUE - 1.0) * Pw0
    p0 = min(1.0, (P_FIXED + Pw0 + P_cool) / P_rated)
    if plant_xe is None:
        I, X = xe_equilibrium(p0)
        P_rx[0] = p0; rho[0] = xe_pcm(X)
    else:                               # true plant parameters
        I, X = _xe_equilibrium_p(p0, plant_xe)
        P_rx[0] = p0; rho[0] = _xe_pcm_p(X, plant_xe)
    I_ts[0] = I; X_ts[0] = X

    for k in range(1, n):
        hook(k)
        tk = t_grid[k]
        u = u_at(tk); P_work = work_power(u)
        fb = fb_at(tk); fm = fm_at(tk)
        CS.validate_fractions(fb, fm, f" at t={tk}")

        if trip:
            hd_prev = R_cap_ts[k-1] - sigma_m - rho[k-1]
            if (not tripped) and hd_prev < 0.0:
                tripped = True; n_trips += 1
                trip_log.append(dict(
                    start_h=(tk - t_grid[0]) / 3600.0,
                    end_h=None, I0=I, X0=X,
                    rho0=rho[k-1], censored=False))
            elif tripped and hd_prev >= trip_margin:
                tripped = False
                if trip_log and trip_log[-1]["end_h"] is None:
                    trip_log[-1]["end_h"] = (tk - t_grid[0]) / 3600.0
        if tripped:
            trip_steps += 1
            tripped_ts[k] = True

        if deadline_shock is not None and k == deadline_shock[0]:
            promoted_W += ctrl.apply_deadline_shock(k, deadline_shock[1],
                                                    deadline_shock[2])

        avail = None if grid_availability is None else grid_availability[k]
        G_eff = G_max_frac if avail is None else G_max_frac * avail
        cheap = (price is not None and G_max_frac > 0.0 and
                 price[k] <= cheap_thresh)
        lim_force = force_cap[k] if force_cap is not None else 1.0
        lim_trip = 0.0 if tripped else 1.0
        cap = min(1.0, lim_force, lim_trip)
        hi = min(cap, P_rx[k-1] + ramp_step)
        lo = 0.0 if tripped else (P_rx[k-1] - ramp_step)
        binding = ('trip' if tripped else 'ramp' if P_rx[k-1] + ramp_step < cap
                   else 'force' if lim_force < 1.0 else 'cap')
        req_W[k] = P_FIXED + PUE * P_work
        robs = (CS.ReactorObs(I, X, rho[k-1],
                              R_cap_ts[k] - sigma_m - eps_buffer)
                if consolidated else None)
        obs = CS.Observation(
            k=k, t_s=tk, P_work_W=P_work, fb=fb, fm=fm,
            requested_W=req_W[k],
            base_facility_W=P_FIXED + (1.0 - a_cool) * P_cool,
            it_cost=it_cost, P_rated_W=P_rated, reactor_prev=P_rx[k-1],
            reactor_hi=hi, reactor_lo=lo, reactor_cap=cap,
            hi_binding=binding, ramp_step=ramp_step,
            grid_cap_W=G_eff * P_rated, grid_avail=avail,
            grid_connected=G_max_frac > 0.0, price_cheap=cheap,
            tripped=tripped, emergency_grid=emergency_grid, reactor=robs)
        if pol_ns is not None:
            _t_pol = perf_counter_ns()
        d = ctrl.step(obs)
        if pol_ns is not None:
            pol_ns[k] = perf_counter_ns() - _t_pol

        grid_use = d.grid_W
        if grid_use > G_eff * P_rated * (1 + 1e-12) + 1e-9 or grid_use < 0:
            raise RuntimeError(f"step {k}: controller grid {grid_use} W "
                               f"exceeds available {G_eff * P_rated} W")
        P_rx[k] = d.p_target
        if P_rx[k] > hi + 1e-12 or (P_rx[k] < min(lo, hi) - 1e-12):
            raise RuntimeError(f"step {k}: controller power {P_rx[k]} "
                               f"outside reachable [{lo}, {hi}]")
        it_served = d.it_served_W
        # Cooling convention identical to run(): cooling follows all
        # non-deferred IT work (P_work - deferred + drained), which includes
        # rigid work left unserved. The cooling owed to unserved rigid work
        # finds no supply and is booked as unserved facility demand, exactly
        # as run() books it inside `unmet`.
        cool_unserved = a_cool * (PUE - 1.0) * d.rigid_unserved_W
        P_cool += a_cool * ((PUE - 1.0) * (it_served + d.rigid_unserved_W)
                            - P_cool)
        fac_short_k = d.facility_unserved_W + cool_unserved
        served_dem = P_FIXED + P_cool + it_served - fac_short_k
        cool_ts[k] = P_cool
        offered_ts[k] = P_work
        Pd[k] = max((served_dem - grid_use) / P_rated, 0.0)
        supply = P_rx[k] * P_rated + grid_use
        surplus_ts[k] = max(0.0, supply - served_dem)
        acc['supply_resid'] += supply - served_dem - surplus_ts[k]
        acc['work_resid'] += P_work - (d.new_served_W + d.rigid_served_W +
                                       d.deferred_b_W + d.deferred_m_W +
                                       d.rigid_unserved_W)

        grid_ts[k] = grid_use; g_pr_ts[k] = d.grid_price_W
        g_br_ts[k] = d.grid_bridge_W; g_em_ts[k] = d.grid_emergency_W
        if grid_use > 0.0 and price is not None:
            gridcost_ts[k] = price[k] * (grid_use * dt / 3600.0) / 1e6
        if grid_use > 0.0 and ewif is not None:
            gridwater_ts[k] = ewif[k] * (grid_use * dt / 3600.0) / 1e6
        served_ts[k] = it_served
        rigid_un[k] = d.rigid_unserved_W; fac_un[k] = fac_short_k
        if rigid_un[k] + fac_un[k] > 0.0:
            unmet_parts[binding][k] = rigid_un[k] + fac_un[k]
        shed_ts[k] = d.deferred_b_W + d.deferred_m_W
        drained_ts[k] = d.drained_W
        dropped_ts[k] = d.dropped_b_W + d.dropped_m_W
        for key_, v_ in (('def_b', d.deferred_b_W), ('def_m', d.deferred_m_W),
                         ('dr_b', d.drained_b_W), ('dr_m', d.drained_m_W),
                         ('age_b', d.drained_age_b), ('age_m', d.drained_age_m),
                         ('drop_b', d.dropped_b_W), ('drop_m', d.dropped_m_W),
                         ('drop_nat', d.dropped_natural_W),
                         ('drop_syn', d.dropped_synthetic_W),
                         ('projections', d.projections)):
            acc[key_] += v_
        state_ts[k] = STATE_CODE[d.state]
        spike_ts[k] = d.spike
        bind_ts[k] = (1 if d.spike_binding == 'sigma' else
                      2 if d.spike_binding == 'rated_floor' else 0)
        excess_ts[k] = d.spike_excess_W
        fc_ts[k] = d.forecast_W; pred_miss[k] = d.pred_miss_W
        pk_req[k] = d.proj_peak_req; pk_acc[k] = d.proj_peak_acc
        ceil_ts[k] = d.ceiling_pcm
        modified[k] = CS.R_PROJECTED_CEILING in d.reasons
        nofeas_ts[k] = d.no_feasible_action
        cancel_ts[k] = d.price_import_canceled
        pr_req_ts[k] = d.price_import_requested_W
        pr_appr_ts[k] = d.price_import_approved_W
        pr_part_ts[k] = d.price_import_partial
        aud_miss_ts[k] = d.partial_audit_missed
        aud_gap_ts[k] = d.partial_audit_gap_W
        p_cand[k] = d.p_candidate; g_cand[k] = d.grid_candidate_W
        floor_raise_W[k] = max(0.0, d.p_floor - d.p_candidate) * P_rated \
            if modified[k] and d.p_floor == d.p_floor else 0.0
        if d.min_slack is not None:
            min_slack_ts[k] = d.min_slack
        if d.queue_max_age > max_age:
            max_age = d.queue_max_age
        acc['age_wsum'] += d.queue_age_wsum
        acc['age_esum'] += d.queue_W

        if d.reasons:
            t_h = (tk - t_grid[0]) / 3600.0
            row = dict(
                t_h=t_h, policy=controller_mode, state=d.state,
                pressure_tier=d.pressure_tier,
                req_p0=d.req_path[0], req_min=d.req_path[1],
                req_mean=d.req_path[2], req_max=d.req_path[3],
                acc_p0=d.acc_path[0], acc_min=d.acc_path[1],
                acc_mean=d.acc_path[2], acc_max=d.acc_path[3],
                reactor_power=P_rx[k-1], accepted_power=P_rx[k],
                projected_peak_pcm=d.proj_peak_req,
                accepted_peak_pcm=d.proj_peak_acc,
                # plant diagnostics (never shown to computing_only):
                applicable_ceiling_pcm=R_cap_ts[k] - sigma_m - eps_buffer,
                headroom_pcm=R_cap_ts[k-1] - sigma_m - rho[k-1],
                ramp_limit=ramp_step, requested_grid_W=d.grid_requested_W,
                available_grid_W=G_eff * P_rated, grid_W=grid_use,
                min_queue_slack_min=(d.min_slack * dt / 60.0
                                     if d.min_slack is not None
                                     else float('nan')),
                no_feasible_action=d.no_feasible_action,
                reason_code=";".join(d.reasons), **meta_log)
            episodes.add(k, t_h, d.reasons, row)
            if decision_log == 'minute':
                minute_rows.append(row)

        if plant_xe is None:
            I, X = xe_advance(I, X, P_rx[k], dt)
            rho[k] = xe_pcm(X)
        else:                           # true plant parameters
            I, X = _xe_advance_p(I, X, P_rx[k], dt, plant_xe)
            rho[k] = _xe_pcm_p(X, plant_xe)
        I_ts[k] = I; X_ts[k] = X

    hook(None)
    episodes.close()
    if trip and trip_log and trip_log[-1]["end_h"] is None:
        trip_log[-1]["end_h"] = (t_grid[-1] - t_grid[0]) / 3600.0
        trip_log[-1]["censored"] = True

    Ws = dt / 3600.0
    leftover = sum(it.amount for it in ctrl.queue) * Ws
    dropped = (acc['drop_b'] + acc['drop_m']) * Ws
    shed_all = shed_ts.sum() * Ws
    residual = shed_all - (drained_ts.sum() * Ws + dropped + leftover)
    sidx = int(settle_h*3600/dt)
    shed_in_settle = shed_ts[:sidx].sum() * Ws
    shed_full = shed_ts.copy(); drained_full = drained_ts.copy()
    unmet = rigid_un + fac_un
    for a in (unmet, rigid_un, fac_un, shed_ts, drained_ts, grid_ts, g_pr_ts,
              g_br_ts, g_em_ts, surplus_ts, gridcost_ts, gridwater_ts,
              served_ts, dropped_ts, floor_raise_W, *unmet_parts.values()):
        a[:sidx] = 0
    Wh = lambda v: v.sum() * dt / 3600.0

    gen_Wh     = float((P_rx[sidx:] * P_rated).sum() * dt / 3600.0)
    surplus_Wh = Wh(surplus_ts)
    served_it_Wh = Wh(served_ts)
    water_L         = gen_Wh     / 1e6 * WATER_L_PER_MWH
    water_surplus_L = surplus_Wh / 1e6 * WATER_L_PER_MWH
    water_grid_L    = float(gridwater_ts.sum()) if ewif is not None else None
    water_dc_L      = served_it_Wh / 1e3 * WUE_L_PER_KWH
    grid_cost = float(gridcost_ts.sum())
    _wgrid = water_grid_L if water_grid_L is not None else 0.0
    water_sys_L = water_L + _wgrid + water_dc_L
    headroom = R_cap_ts[sidx:] - sigma_m - rho[sidx:]
    viol = headroom < -VIOL_TOL_PCM
    cross_episodes = int((viol[1:] & ~viol[:-1]).sum() + (1 if viol[0] else 0))
    _longest = 0; _cur = 0
    for _v in viol:
        _cur = _cur + 1 if _v else 0
        if _cur > _longest: _longest = _cur
    outage = tripped_ts[sidx:]
    viol_op = viol & ~outage
    _pw = np.array([work_power(u_at(tt)) for tt in t_grid[sidx:]])
    demand_Wh = float((P_FIXED + _pw * PUE).sum() * dt / 3600.0)
    sc = state_ts[sidx:]; sp = spike_ts[sidx:]; bd = bind_ts[sidx:]
    n_det = int(sp.sum())
    out = dict(
        controller_mode=controller_mode,
        unmet=Wh(unmet), shed=Wh(shed_ts), drained=Wh(drained_ts),
        dropped=dropped, leftover=leftover,
        true_unserved=Wh(unmet) + dropped + leftover,
        dropped_b=acc['drop_b']*Ws, dropped_m=acc['drop_m']*Ws,
        delay_b=(acc['age_b']/acc['dr_b']*dt/3600.0 if acc['dr_b'] > 0
                 else 0.0),
        delay_m=(acc['age_m']/acc['dr_m']*dt/3600.0 if acc['dr_m'] > 0
                 else 0.0),
        unmet_cap=Wh(unmet_parts['cap']), unmet_force=Wh(unmet_parts['force']),
        unmet_trip=Wh(unmet_parts['trip']), unmet_ramp=Wh(unmet_parts['ramp']),
        unmet_xen=0.0,
        n_trips=n_trips, deadtime_h=trip_steps*dt/3600.0,
        grid_Wh=Wh(grid_ts), cheap_steps=int(
            0 if price is None or not G_max_frac > 0.0
            else (price[1:] <= cheap_thresh).sum()),
        peak_rho=float(rho[sidx:].max()),
        min_headroom=float(headroom.min()),
        crossings=int(viol.sum()),
        crossings_op=int(viol_op.sum()),
        crossings_outage=int((viol & outage).sum()),
        cross_episodes=cross_episodes,
        cross_episodes_op=int((viol_op[1:] & ~viol_op[:-1]).sum()
                              + (1 if viol_op[0] else 0)),
        longest_viol_min=_longest * dt / 60.0,
        demand_Wh=demand_Wh,
        unserved_pct=100.0*(Wh(unmet)+dropped+leftover)/demand_Wh,
        max_exceed=float(max(0.0, -headroom.min())),
        ramp_binds=int((unmet_parts['ramp'][sidx:] > 1e-6).sum()),
        refused_steps=int(modified[sidx:].sum()),
        floor_raise_Wh=Wh(floor_raise_W),
        p_lo=float(P_rx[sidx:].min()), p_hi=float(P_rx[sidx:].max()),
        p_mean=float(P_rx[sidx:].mean()),
        ramp_distance=float(np.abs(np.diff(P_rx[sidx:])).sum()),
        gen_Wh=gen_Wh, surplus_Wh=surplus_Wh, served_it_Wh=served_it_Wh,
        water_L=water_L, water_surplus_L=water_surplus_L,
        water_grid_L=water_grid_L, water_dc_L=water_dc_L,
        water_sys_L=water_sys_L,
        water_L_per_kWh_served=water_sys_L / max(served_it_Wh / 1e3, 1e-9),
        grid_cost=grid_cost,
        grid_cost_per_MWh_served=grid_cost / max(served_it_Wh / 1e6, 1e-9),
        residual=residual, shed_in_settle=shed_in_settle,
        # ---- controller-specific ----
        minutes_NORMAL=float((sc == 0).sum() * dt / 60.0),
        minutes_STRESSED=float((sc == 1).sum() * dt / 60.0),
        minutes_ISLANDED=float((sc == 2).sum() * dt / 60.0),
        state_transitions=int((np.diff(sc) != 0).sum()),
        spike_detections=n_det,
        spike_episodes=int((sp[1:] & ~sp[:-1]).sum() + (1 if sp[0] else 0)),
        spike_bind_sigma=int((bd == 1).sum()),
        spike_bind_rated_floor=int((bd == 2).sum()),
        spike_frac_sigma=float((bd == 1).sum() / n_det) if n_det else
        float('nan'),
        spike_frac_rated_floor=float((bd == 2).sum() / n_det) if n_det else
        float('nan'),
        spike_energy_Wh=float(np.nansum(np.where(sp, excess_ts[sidx:], 0.0))
                              * Ws),
        deadline_promoted_Wh=promoted_W * Ws,
        deadline_miss_b_Wh=acc['drop_b'] * Ws,
        deadline_miss_m_Wh=acc['drop_m'] * Ws,
        dropped_natural_Wh=acc['drop_nat'] * Ws,
        dropped_synthetic_Wh=acc['drop_syn'] * Ws,
        deadline_miss_steps=int((dropped_ts[sidx:] > 0).sum()),
        deferred_b_Wh=acc['def_b'] * Ws, deferred_m_Wh=acc['def_m'] * Ws,
        max_queue_age_min=max_age * dt / 60.0,
        mean_queue_age_min=(acc['age_wsum'] / acc['age_esum'] * dt / 60.0
                            if acc['age_esum'] > 0 else 0.0),
        min_queue_slack_min=(float(np.nanmin(min_slack_ts)) * dt / 60.0
                             if np.isfinite(min_slack_ts).any()
                             else float('nan')),
        mean_min_queue_slack_min=(float(np.nanmean(min_slack_ts)) * dt / 60.0
                                  if np.isfinite(min_slack_ts).any()
                                  else float('nan')),
        rigid_unserved_Wh=Wh(rigid_un),
        facility_unserved_Wh=Wh(fac_un),
        rigid_shortfall_steps=int(((rigid_un + fac_un)[sidx:] > 0).sum()),
        rigid_shortfall_reason=(CS.RIGID_SHORTFALL_CAUSE
                                if (rigid_un + fac_un)[sidx:].any()
                                else 'none'),
        grid_price_Wh=Wh(g_pr_ts), grid_bridge_Wh=Wh(g_br_ts),
        grid_emergency_Wh=Wh(g_em_ts),
        grid_unavailable_min=float(
            0 if grid_availability is None or not G_max_frac > 0.0
            else (grid_availability[sidx:] == 0.0).sum() * dt / 60.0),
        projections=int(acc['projections']),
        no_feasible_action_steps=int(nofeas_ts[sidx:].sum()),
        price_import_canceled_steps=int(cancel_ts[sidx:].sum()),
        energy_queue_residual_Wh=residual,
        energy_work_residual_Wh=acc['work_resid'] * Ws,
        energy_supply_residual_Wh=acc['supply_resid'] * Ws,
        trip_log=trip_log)
    if price_diagnostics:
        # cheap-price import accounting after settling: requested (after the
        # G cap), approved (after reactor feasibility) and delivered
        # (= grid_price_Wh, after surplus displacement)
        rq, ap = pr_req_ts[sidx:], pr_appr_ts[sidx:]
        dl = g_pr_ts[sidx:]
        opp = rq > CS.EPS_W
        out.update(
            price_requested_Wh=float(rq.sum() * Ws),
            price_approved_Wh=float(ap.sum() * Ws),
            price_opportunity_steps=int(opp.sum()),
            price_full_accept_steps=int((opp & (ap >= rq * (1 - 1e-12)))
                                        .sum()),
            price_partial_steps=int(pr_part_ts[sidx:].sum()),
            price_full_cancel_steps=int((opp & (ap <= CS.EPS_W)).sum()),
            price_approved_mean_frac=(float(ap[ap > CS.EPS_W].mean()
                                            / P_rated)
                                      if (ap > CS.EPS_W).any()
                                      else float('nan')),
            price_delivered_mean_frac=(float(dl[dl > CS.EPS_W].mean()
                                             / P_rated)
                                       if (dl > CS.EPS_W).any()
                                       else float('nan')),
            price_opp_mean_proj_slack_pcm=(
                float(np.nanmean((ceil_ts - pk_acc)[sidx:][opp]))
                if consolidated and opp.any() else float('nan')),
            partial_audit_missed_steps=int(aud_miss_ts[sidx:].sum()),
            partial_audit_max_gap_frac=float(aud_gap_ts[sidx:].max()
                                             / P_rated))
    out['energy_residual_Wh'] = max(abs(out['energy_queue_residual_Wh']),
                                    abs(out['energy_work_residual_Wh']),
                                    abs(out['energy_supply_residual_Wh']))
    for r_ in CS.REASON_CODES:
        out[f'reason_{r_}_steps'] = episodes.counts[r_]
    out['decision_episodes_n'] = len(episodes.episodes)

    # ---- post-simulation forecast validation (realized future only here)
    H = int(predict_horizon_s / dt)
    ks = np.arange(max(sidx, 1), n)
    k60 = ks[ks + 60 < n]
    err = fc_ts[k60] - req_W[k60 + 60]
    out['fc_mae_1h_W'] = float(np.mean(np.abs(err))) if len(err) else np.nan
    out['fc_bias_1h_W'] = float(np.mean(err)) if len(err) else np.nan
    cs_ = np.concatenate([[0.0], np.cumsum(req_W)])
    kh = ks[ks + H < n]
    realized_mean = (cs_[kh + H + 1] - cs_[kh + 1]) / H
    errh = fc_ts[kh] - realized_mean
    out['fc_mae_horizon_W'] = float(np.mean(np.abs(errh))) if len(errh) \
        else np.nan
    realized_pk = _rolling_max_forward(rho, H + 1)       # from rho[k-1]
    fin = np.isfinite(pk_acc) & (np.arange(n) >= sidx)
    fk = np.nonzero(fin)[0]
    fk = fk[fk - 1 + H < n]
    if len(fk):
        real = realized_pk[fk - 1]
        diff_ = pk_acc[fk] - real
        acc_ok = pk_acc[fk] <= ceil_ts[fk]
        true_ceiling = R_cap_ts[fk] - sigma_m
        out.update(
            proj_vs_realized_mean_pcm=float(diff_.mean()),
            proj_vs_realized_min_pcm=float(diff_.min()),
            proj_underpredict_frac=float((diff_ < 0).mean()),
            false_safe_decisions=int((acc_ok & (real > true_ceiling
                                                + VIOL_TOL_PCM)).sum()),
            false_safe_eps_decisions=int((acc_ok & (real > ceil_ts[fk])).sum()))
    else:
        out.update(proj_vs_realized_mean_pcm=np.nan,
                   proj_vs_realized_min_pcm=np.nan,
                   proj_underpredict_frac=np.nan, false_safe_decisions=0,
                   false_safe_eps_decisions=0)
    # unnecessarily conservative: a modified decision whose rejected
    # candidate, rebuilt from the REALIZED requested demand (same grid
    # persistence, no re-simulated queue response), projects under the
    # ceiling. Post-hoc counterfactual; never available to the controller.
    uc = 0
    mk = np.nonzero(modified & (np.arange(n) >= sidx))[0]
    seg = int(round(cfg.path_dt_s / dt)); nseg = int(cfg.horizon_s /
                                                    cfg.path_dt_s)
    for k in mk:
        if k + nseg * seg >= n:
            continue
        path = [p_cand[k]]
        for j in range(1, nseg):
            a_, b_ = k + j*seg, k + (j+1)*seg
            dbar = (cs_[b_] - cs_[a_]) / seg
            path.append(min(max((dbar - g_cand[k]) / P_rated, 0.0), 1.0))
        path = [max(p_, cfg.p_min_proj) for p_ in path]
        if (project_peak_rho_path(I_ts[k-1], X_ts[k-1], path,
                                  cfg.path_dt_s) if model_xe is None
                else _project_peak_rho_path_p(I_ts[k-1], X_ts[k-1], path,
                                              cfg.path_dt_s, model_xe)) \
                <= ceil_ts[k]:
            uc += 1
    out['unnecessarily_conservative_decisions'] = uc
    out['modified_decisions_evaluated'] = int(len(mk))
    # deadline prediction error (hourly samples): predicted misses of the
    # queue at k vs realized drops over (k, k + H]
    cd = np.concatenate([[0.0], np.cumsum(dropped_ts)])
    kd = ks[(ks % 60 == 0) & (ks + H < n)]
    if len(kd):
        realized_drop = (cd[kd + H + 1] - cd[kd + 1]) * Ws
        pe = pred_miss[kd] * Ws - realized_drop
        out['deadline_pred_mae_Wh'] = float(np.mean(np.abs(pe)))
        out['deadline_pred_bias_Wh'] = float(np.mean(pe))
    else:
        out['deadline_pred_mae_Wh'] = out['deadline_pred_bias_Wh'] = np.nan

    if timing != 'off':
        us = pol_ns[1:] / 1e3
        out.update(timing_mode=timing, policy_decisions=int(n - 1),
                   policy_total_us=float(us.sum()),
                   policy_mean_us=float(us.mean()),
                   policy_p95_us=float(np.percentile(us, 95)),
                   policy_p99_us=float(np.percentile(us, 99)),
                   policy_max_us=float(us.max()))
    if timing == 'detailed':
        pj = np.asarray(proj_ns, dtype=float) / 1e3
        nan = float('nan')
        out.update(projection_calls=int(len(pj)),
                   projection_total_us=float(pj.sum()),
                   projection_mean_us=float(pj.mean()) if len(pj) else nan,
                   projection_p95_us=(float(np.percentile(pj, 95))
                                      if len(pj) else nan),
                   projection_p99_us=(float(np.percentile(pj, 99))
                                      if len(pj) else nan),
                   projection_max_us=float(pj.max()) if len(pj) else nan)
    if decision_log is not None:
        out['decision_episodes'] = episodes.episodes
        out['decision_rows'] = minute_rows
    if return_series:
        ceil_s = R_cap_ts - sigma_m
        out["series"] = dict(t=t_grid, time_s=t_grid,
                             P_rx=P_rx, reactor_power=P_rx,
                             Pd=Pd, demand_power=Pd,
                             rho=rho, iodine=I_ts, xenon=X_ts,
                             R_cap=R_cap_ts, ceiling=ceil_s,
                             usable_ceiling=ceil_s, headroom=ceil_s - rho,
                             grid=grid_ts, tripped=tripped_ts,
                             state=state_ts, spike=spike_ts,
                             requested_W=req_W, forecast_W=fc_ts,
                             projected_peak_req=pk_req,
                             projected_peak_acc=pk_acc,
                             controller_ceiling=ceil_ts,
                             no_feasible_action=nofeas_ts,
                             cooling=cool_ts, offered_it=offered_ts,
                             deferred=shed_full, drained=drained_full,
                             rigid_unserved=rigid_un,
                             facility_unserved=fac_un)
    return out


def run_continuous(fb_at, fm_at, sched, t_grid, u_at, P_rated, dt, R_cap,
                   **kw):
    """Year orchestration: exactly ONE run() over the complete grid. Cell/
    segment/provenance labels never reach this call, so no boundary can
    re-initialize iodine, xenon, reactor power, queues, cooling or trip
    state. Verified by --self-test."""
    return run(fb_at, fm_at, sched, t_grid, u_at, P_rated, dt, R_cap, **kw)


def self_test():
    """--self-test: checks that the year path keeps reactor/queue state
    continuous across workload-segment boundaries. Uses a small in-memory
    two-segment fixture (TEST ONLY; not data, not an execution mode)."""
    dt = 60.0; n0 = int(30 * 3600 / dt)            # two 30 h segments
    t_grid = dt * np.arange(2 * n0)
    seg = np.repeat([0, 1], n0)                    # provenance label only
    u_arr = np.where(seg == 0, 0.85, 0.30)         # step down -> Xe transient
    u_at = lambda tt: float(np.interp(tt, t_grid, u_arr))
    fb_at = lambda tt: 0.30
    fm_at = lambda tt: 0.10
    P_rated = (P_FIXED + work_power(0.85) * PUE) / 0.95
    kw = dict(rx_mode='free', return_series=True)
    b = int(np.argmax(np.diff(seg) != 0)) + 1      # first index of segment 1
    ok = True

    g = globals(); real_run, real_eq = g['run'], g['xe_equilibrium']
    calls = {'run': 0, 'init': 0}
    def c_run(*a, **k):
        calls['run'] += 1; return real_run(*a, **k)
    def c_eq(p):
        calls['init'] += 1; return real_eq(p)
    g['run'], g['xe_equilibrium'] = c_run, c_eq
    try:
        r_year = run_continuous(fb_at, fm_at, 'tiered', t_grid, u_at,
                                P_rated, dt, 8000.0, **kw)
    finally:
        g['run'], g['xe_equilibrium'] = real_run, real_eq
    resets = calls['init'] - calls['run']          # inits beyond run() entry
    ok &= calls['run'] == 1 and resets == 0

    r_dir = run(fb_at, fm_at, 'tiered', t_grid, u_at, P_rated, dt, 8000.0,
                **kw)
    same = all(r_year[k] == r_dir[k] or
               (isinstance(r_year[k], float) and np.isnan(r_year[k])
                and np.isnan(r_dir[k]))
               for k in r_year if k not in ('series', 'trip_log'))
    same &= all(np.array_equal(r_year['series'][k], r_dir['series'][k])
                for k in r_year['series'])
    ok &= same

    S = r_year['series']
    fin = all(np.all(np.isfinite(S[k])) for k in ('iodine', 'xenon', 'rho',
                                                  'P_rx'))
    cont = True
    for k in ('iodine', 'xenon', 'P_rx'):
        dv = np.abs(np.diff(S[k]))
        other = np.delete(dv, b - 1)
        cont &= dv[b - 1] <= other.max()           # boundary step not an outlier
    ok &= fin and cont

    r0 = run(fb_at, fm_at, 'tiered', t_grid[:b], u_at, P_rated, dt, 8000.0,
             **kw)
    r1 = run(fb_at, fm_at, 'tiered', t_grid[b:], u_at, P_rated, dt, 8000.0,
             **kw)
    x_jump_reset = abs(r1['series']['xenon'][0] - r0['series']['xenon'][-1])
    x_jump_cont = abs(S['xenon'][b] - S['xenon'][b - 1])
    rho_diff = float(np.max(np.abs(r1['series']['rho'] - S['rho'][b:])))
    differs = rho_diff > 1.0
    ok &= differs

    print("SELF-TEST: year-path state continuity (two-segment test fixture)")
    print(f"  continuous year orchestration calls to run(): {calls['run']}")
    print(f"  segment-boundary state resets detected: {resets}")
    print(f"  continuous-vs-direct one-call equivalence: "
          f"{'PASS' if same else 'FAIL'}")
    print(f"  I_ts/X_ts finite and continuous across boundary (step {b}): "
          f"{'PASS' if fin and cont else 'FAIL'}  "
          f"(|dX| at boundary {x_jump_cont:.3e} vs reset jump "
          f"{x_jump_reset:.3e})")
    print(f"  continuous-vs-deliberately-reset comparison differs: "
          f"{'YES' if differs else 'NO'}  (max |rho diff| over segment 2 = "
          f"{rho_diff:.1f} pcm; split-and-reset is a test only)")
    if not ok:
        sys.exit("SELF-TEST FAILED")
    print("SELF-TEST: PASS")


RXLABEL = {'free': 'free', 'reactive': 'reac', 'predictive': 'pred',
           'hysteresis': 'hyst',
           # --year with --computing-only / --consolidated only
           'computing_only': 'comp', 'consolidated': 'cons'}
ROW = ("{tag:>20} | {rx:>4} | {tu:9.1f} | {up:5.2f} | {cr:5d} | {ep:4d} | "
       "{mh:7.0f} | {ref:5d} | {wk:8.1f} | {wsk:7.1f} | {wg:>7} | "
       "{gc:8.2f} | {rd:6.1f} | {pm:5.2f} | {res:5.2f}{flag}")
HDR = (f"{'experiment':>20} | {'rx':>4} | {'true uns':>9} | {'uns%':>5} | "
       f"{'cross':>5} | {'epis':>4} | {'min_hd':>7} | {'refus':>5} | "
       f"{'water_kL':>8} | {'wst_kL':>7} | {'wgrd_kL':>7} | {'grid_$':>8} | "
       f"{'rampD':>6} | {'p_avg':>5} | {'resid':>5}")

CSV_ROWS = []
TRIP_ROWS = []

def show(tag, mode, rx_mode, r, bad, extra=None):
    flag = ""
    if abs(r['residual']) > 1.0:
        flag = "  <-- CONSERVATION BROKEN"; bad[0] += 1
    if r['shed_in_settle'] > 1.0:
        flag += f"  [warn: {r['shed_in_settle']:.0f} Wh shed in settle]"
    wg = (f"{r['water_grid_L']/1e3:7.1f}" if r.get('water_grid_L')
          is not None else "    n/a")
    print(ROW.format(tag=tag, rx=RXLABEL[rx_mode],
                     tu=r['true_unserved']/1e3, up=r['unserved_pct'],
                     cr=r['crossings'], ep=r['cross_episodes'],
                     mh=r['min_headroom'], ref=r['refused_steps'],
                     wk=r['water_L']/1e3, wsk=r['water_surplus_L']/1e3,
                     wg=wg, gc=r['grid_cost'], rd=r['ramp_distance'],
                     pm=r['p_mean'], res=r['residual'], flag=flag))
    if r.get('n_trips', 0):
        print(f"{'':>20}   -> {r['n_trips']} trip(s), "
              f"{r['deadtime_h']:.1f} h dead-time, "
              f"{r['unmet_trip']/1e3:.0f} kWh lost to outage")

        print(f"{'':>20}      cross split: {r['crossings_op']} min "
              f"OPERATING (cause) + {r['crossings_outage']} min in-outage "
              f"transient (consequence, already priced as dead-time)")

        spans = ", ".join(
            f"{d['start_h']:.1f}->{d['end_h']:.1f}h"
            f"{'*' if d.get('censored') else ''}"
            for d in r.get('trip_log', []))
        print(f"{'':>20}      trips: {spans}")
    rec = dict(experiment=tag, sched=mode, rx=rx_mode)

    rec.update({k: v for k, v in r.items()
                if k not in ("series", "trip_log")})
    rec["trip_starts_h"] = ";".join(f"{d['start_h']:.3f}"
                                    for d in r.get("trip_log", []))
    if extra: rec.update(extra)
    CSV_ROWS.append(rec)
    for d in r.get("trip_log", []):
        TRIP_ROWS.append(dict(
            experiment=tag, sched=mode, rx=rx_mode,
            R_cap=(extra or {}).get("R_cap"),
            G=(extra or {}).get("G"),
            start_h=d["start_h"], end_h=d["end_h"],
            dur_h=(d["end_h"] - d["start_h"]
                   if d["end_h"] is not None else None),
            I0=repr(d["I0"]), X0=repr(d["X0"]),
            rho0=d["rho0"], censored=d.get("censored", False)))

ARMS = ('free', 'reactive', 'predictive')
SCHEDS = ('tiered', 'blind')
HYST_H = (2.0, 4.0, 6.0)   # --hysteresis only; never part of ARMS
MONTHLY_FILES = ("demand_curve_cell_a.csv", "pjm_dom_34885183.csv",
                 "load_pjm_genmix.csv")

# ================= opt-in workload-aware controller experiments =============
# Reached only through --computing-only / --consolidated (see
# _controller_cli). Nothing below is used by the legacy experiment grid.
CONTROLLER_RCAPS = (8000.0, 3500.0, 2700.0)  # BOC / intermediate / EOC-like
CONTROLLER_ARMS = (('free', None), ('computing_only', None),
                   ('reactive', None), ('hysteresis', 2.0),
                   ('hysteresis', 4.0), ('hysteresis', 6.0),
                   ('predictive', None), ('consolidated', None))
STRESS_CELLS = tuple("abcdefgh")
PROTECTED_OUTPUTS = {"v6_results.csv", "v6_results_baseline.csv",
                     "v6_results_pre_tol.csv", "hysteresis_results.csv",
                     "quick_hysteresis_results.csv", "trip_log.csv",
                     "trip_log_hysteresis.csv", "trip_log_year.csv",
                     "year_results.csv", "oracle_results.csv",
                     "deadtime_vs_rcap.csv", "v5_results.csv"}
# The completed canonical controller evaluation (2,112 runs). No run may
# write any file whose name starts with one of these prefixes.
CANONICAL_RUN_PREFIXES = ("poster_final_20260929_020829",)

# ---- opt-in one-factor sensitivity sweeps (--predict-horizon-h or
# --eps-buffer-pcm). Each varies ONE existing run() argument, and only for
# the arms that consume it; every other setting keeps the canonical
# controller-suite value. Nothing here is reached without those flags.
#   horizon: predict_horizon_s. predictive -> project_peak_rho constant-hold
#            horizon (int(h / 300 s) steps; h = 0 is a current-state check:
#            the "projection" returns the present xenon reactivity, so a
#            down-ramp is refused only if rho already exceeds the ceiling --
#            NOT the reactive headroom gate). consolidated ->
#            ControllerConfig.horizon_s = length of the requested power path
#            that is projected (the same length bounds which pending
#            deadlines enter that path); it needs >= 2 path segments, so
#            h < 2 * path_dt_s is undefined and is skipped, never
#            reinterpreted.
#   eps:     eps_buffer (pcm) subtracted from the ceiling R_cap - sigma_m
#            by predictive and consolidated only. sigma_m, the trip rule and
#            the crossing metric (R_cap - sigma_m - rho < -VIOL_TOL_PCM) do
#            not depend on it, so it changes conservatism only.
SWEEP_ARMS = ('predictive', 'consolidated')
SWEEP_SUITES = {'horizon': ('horizon_sensitivity',
                            'sensitivity_prediction_horizon'),
                'eps': ('margin_sensitivity', 'sensitivity_eps_buffer'),
                # model_mismatch_experiment.py only (no CLI sweep kind)
                'model': ('model_mismatch', 'sensitivity_model_mismatch')}
CANONICAL_EPS_PCM = 20.0          # predictive / consolidated suite value
CANONICAL_HORIZON_H = 10.0        # run() / ControllerConfig default
PROJ_SEG_S = 300.0                # project_peak_rho dt_proj == path_dt_s


def _sweep_undefined(arm, kind, value):
    """Reason string if (arm, sweep value) has no defined meaning, else
    None. Undefined combinations are skipped and listed in the manifest."""
    import consolidated_scheduler as CS
    if kind != 'horizon':
        return None
    s = value * 3600.0
    if arm == 'consolidated' and s < 2 * CS.ControllerConfig().path_dt_s:
        return ("consolidated projects a requested path of >= 2 segments "
                f"({2 * CS.ControllerConfig().path_dt_s:.0f} s); a "
                f"{value:g} h horizon is not a projection")
    return None


def _is_canonical_output(p):
    import os
    b = os.path.basename(p)
    return b in PROTECTED_OUTPUTS or b.startswith(CANONICAL_RUN_PREFIXES)


_CTX = {}      # per-suite context, inherited by forked workers


def _load_cell_ctx(cell, path, price_path, genmix_path, dt=60.0):
    import consolidated_scheduler as CS
    d = pd.read_csv(path)
    tot = d["total_cpu"].values
    fb_s = d["defer_low"].values / tot
    fm_s = (d["defer_high"].values - d["defer_low"].values) / tot
    for i in range(len(d)):                    # every input sample
        CS.validate_fractions(fb_s[i], fm_s[i], f" in {path} row {i}")
    t, u_at, fb_at, fm_at = load_borg(path)
    t_grid = np.arange(t[0], t[-1], dt)
    Pdcs = np.array([P_FIXED + work_power(u_at(tt))*PUE for tt in t_grid])
    price, thresh, tiled = load_pjm(price_path, t_grid)
    ewif = load_pjm_genmix(genmix_path, t_grid)
    u = 0.85 * tot / tot.max()
    chars = dict(cell=cell, source=path, samples_5min=len(d),
                 days=float(d["hours"].iloc[-1] / 24.0),
                 u_mean=float(u.mean()), u_peak=float(u.max()),
                 u_cv=float(u.std() / u.mean()),
                 u_p95_abs_step=float(np.percentile(np.abs(np.diff(u)), 95)),
                 batch_frac_mean=float(fb_s.mean()),
                 mid_frac_mean=float(fm_s.mean()),
                 rigid_frac_mean=float((1 - fb_s - fm_s).mean()),
                 flexible_frac_mean=float((fb_s + fm_s).mean()),
                 P_rated_W=float(Pdcs.mean() / 0.95),
                 price_tiled=bool(tiled), genmix_available=ewif is not None)
    return dict(path=path, t_grid=t_grid, u_at=u_at, fb_at=fb_at,
                fm_at=fm_at, P_rated=Pdcs.mean() / 0.95, price=price,
                thresh=thresh, ewif=ewif, chars=chars)


def _controller_tag(t):
    tag = (f"{t['prefix']} {t['cell']} R={t['R_cap']:.0f} G={t['G']:.1f} "
           f"{t['scenario']}")
    if t['h'] is not None:
        tag += f" h={t['h']:.0f}"
    if t['suite'] == 'sensitivity':
        tag += f" k={t['k_sigma']:g} f={t['floor']:g}"
    if t.get('horizon_h') is not None:          # sweep tasks only
        tag += f" H={t['horizon_h']:g}h"
    if t.get('eps') is not None:
        tag += f" eps={t['eps']:g}"
    if t.get('mismatch_case') is not None:     # model-mismatch tasks only
        tag += f" M={t['mismatch_case']}/{t['model_condition']}"
    if t['synthetic']:
        tag += " [synthetic_deadline]"
    return tag


PRIMARY_EVIDENCE = 'primary_cross_arm'
DIAGNOSTIC_DEADLINE = 'controller_diagnostic_endogenous_deadline_shock'


def _evidence_class(suite, scn):
    """The deadline shock promotes a fraction of each policy's OWN queue,
    so its size differs between policies (endogenous). Those rows are
    controller diagnostics only, never primary cross-policy evidence."""
    if scn.deadline is not None:
        return DIAGNOSTIC_DEADLINE
    for suite_, evidence_ in SWEEP_SUITES.values():
        if suite == suite_:
            return evidence_
    return PRIMARY_EVIDENCE if suite == 'primary' else 'sensitivity'


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _write_manifest(path, man):
    """Write-once, read-only run manifest (refuses to replace a file)."""
    import json, os, stat
    with open(path, 'x') as f:
        json.dump(man, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _controller_task(t):
    import consolidated_scheduler as CS
    C = _CTX['cells'][t['cell']]
    scn = _CTX['scenarios'][t['scenario']]
    dt = 60.0
    tg = C['t_grid']; n = len(tg)
    ev = CS.build_event_inputs(scn, n, dt, _CTX['stress_cfg'])
    u_eff = CS.wrap_u_at(C['u_at'], ev, tg[0])
    arm, h, R, G = t['arm'], t['h'], t['R_cap'], t['G']
    eps = 20.0 if arm in ('predictive', 'consolidated',
                          'consolidated_partial_grid') else 0.0
    if t.get('eps') is not None:                # --eps-buffer-pcm sweep
        eps = t['eps']
    md = scn.metadata()
    kw = dict(price=C['price'], cheap_thresh=C['thresh'], G_max_frac=G,
              trip=True, ewif=C['ewif'], eps_buffer=eps, return_series=True,
              grid_availability=ev.grid_availability,
              timing=_CTX['timing'])
    if t.get('horizon_h') is not None:          # --predict-horizon-h sweep
        kw['predict_horizon_s'] = t['horizon_h'] * 3600.0
    if t.get('plant_xe') is not None:           # model-mismatch tasks only
        kw['plant_xe'] = t['plant_xe']
    if t.get('model_xe') is not None:
        kw['model_xe'] = t['model_xe']
    if t.get('price_diagnostics'):              # partial-grid experiment
        kw['price_diagnostics'] = True
    if t.get('partial_grid_audit'):
        kw['partial_grid_audit'] = t['partial_grid_audit']
    is_ctrl = arm in CS.ALL_CONTROLLER_MODES
    if is_ctrl:
        cfg = CS.ControllerConfig(detector=CS.DetectorConfig(
            k_sigma=t['k_sigma'], rated_floor_fraction=t['floor']))
        r = run(C['fb_at'], C['fm_at'], 'tiered', tg, u_eff, C['P_rated'],
                dt, R, controller_mode=arm, controller_cfg=cfg,
                known_outages=ev.known_outages,
                deadline_shock=ev.deadline_shock,
                decision_log=_CTX['decision_log'], event_meta=md, **kw)
    else:
        hk = {} if h is None else dict(hysteresis_h=h)
        r = run(C['fb_at'], C['fm_at'], 'tiered', tg, u_eff, C['P_rated'],
                dt, R, rx_mode=arm, **hk, **kw)
    ser = r.pop('series'); trips = r.pop('trip_log')
    episodes = r.pop('decision_episodes', []); minutes = r.pop('decision_rows',
                                                              [])
    sidx = int(6 * 3600 / dt)
    Ws = dt / 3600.0
    tag = _controller_tag(t)
    k_ev = None if scn.name == 'none' else \
        int(round(md['event_start_h'] * 3600.0 / dt))
    extra = dict(
        experiment=tag, sched='tiered', rx=arm, suite=t['suite'],
        cell=t['cell'], R_cap=R, G=G, eps=eps, hyst_h=h,
        controller_k_sigma=t['k_sigma'] if is_ctrl else np.nan,
        controller_rated_floor_fraction=t['floor'] if is_ctrl else np.nan,
        R_cap_at_event_pcm=R if k_ev is not None else np.nan,
        pre_event_headroom_pcm=(float(ser['headroom'][k_ev - 1])
                                if k_ev is not None else np.nan),
        pre_event_reactor_power=(float(ser['P_rx'][k_ev - 1])
                                 if k_ev is not None else np.nan),
        deadline_shock_applied=bool(is_ctrl and ev.deadline_shock
                                    is not None),
        deadline_shock_vacuous=(bool(r['deadline_promoted_Wh'] <= 0.0)
                                if is_ctrl and ev.deadline_shock is not None
                                else np.nan),
        controller_mode=arm if is_ctrl else 'legacy',
        evidence_class=_evidence_class(t['suite'], scn))
    if t.get('horizon_h') is not None or t.get('eps') is not None:
        # sweep rows only (default rows keep the canonical column set)
        extra['sweep_parameter'] = ('predict_horizon_h'
                                    if t.get('horizon_h') is not None
                                    else 'eps_buffer_pcm')
        extra['predict_horizon_h'] = (t['horizon_h']
                                      if t.get('horizon_h') is not None
                                      else CANONICAL_HORIZON_H)
    if t.get('mismatch_case') is not None:
        pq = t.get('plant_xe') or NOMINAL_XE
        mq = t.get('model_xe') or NOMINAL_XE
        extra.update(mismatch_case=t['mismatch_case'],
                     model_condition=t['model_condition'],
                     range_label=t['range_label'],
                     predict_horizon_h=CANONICAL_HORIZON_H,
                     **{f"plant_{f.name}": getattr(pq, f.name)
                        for f in _fields(pq)},
                     **{f"model_{f.name}": getattr(mq, f.name)
                        for f in _fields(mq)})
    if not is_ctrl:
        g = ser['grid'].copy(); g[:sidx] = 0
        extra['grid_emergency_Wh'] = float(g[ser['tripped']].sum() * Ws)
        extra['grid_unavailable_min'] = float(
            0 if ev.grid_availability is None or not G > 0.0 else
            (np.asarray(ev.grid_availability)[sidx:] == 0.0).sum()
            * dt / 60.0)
    if ev.blackout_window is not None:
        k0, k1 = ev.blackout_window
        g = ser['grid'][k0:k1]
        extra['grid_Wh_during_blackout'] = float(g.sum() * Ws)
    row = {**extra, **md}
    row.update({k_: v_ for k_, v_ in r.items() if k_ not in row})
    row['controller_mode'] = extra['controller_mode']
    trip_rows = [dict(experiment=tag, sched='tiered', rx=arm, R_cap=R, G=G,
                      cell=t['cell'], scenario=scn.name, suite=t['suite'],
                      start_h=d_['start_h'], end_h=d_['end_h'],
                      dur_h=(d_['end_h'] - d_['start_h']
                             if d_['end_h'] is not None else None),
                      I0=repr(d_['I0']), X0=repr(d_['X0']),
                      rho0=d_['rho0'], censored=d_.get('censored', False))
                 for d_ in trips]
    for e in episodes:
        e.update(experiment=tag, cell=t['cell'], R_cap=R, suite=t['suite'])
    for m in minutes:
        m.update(experiment=tag, cell=t['cell'], R_cap=R, suite=t['suite'])
    return t['idx'], row, trip_rows, episodes, minutes


def _controller_tasks(cells, scen_names, ctrl_arms, scenarios, G, prefix,
                      sensitivity, sweep=None):
    """Deterministic run list: primary comparisons for every cell x R_cap
    snapshot x scenario x arm, plus (stress suite) the unconditional spike
    sensitivity grid: every (k_sigma, rated_floor_fraction) pair for every
    cell, R_cap and controller arm on the natural and injected-spike
    traces. Primary-pair sensitivity rows are the primary rows.

    sweep (opt-in, see SWEEP_SUITES): dict(kind, values, arms, rcaps).
    Returns ONLY the one-factor sweep tasks (primary detector settings);
    undefined (arm, value) pairs are skipped, see _sweep_undefined."""
    import consolidated_scheduler as CS
    arms = [(a, h) for a, h in CONTROLLER_ARMS
            if a not in CS.CONTROLLER_MODES or a in ctrl_arms]
    k0 = CS.PRIMARY_DETECTOR.k_sigma
    f0 = CS.PRIMARY_DETECTOR.rated_floor_fraction
    tasks = []

    def add(suite, c, R, s, a, h, k_, f_, **sweep_kw):
        tasks.append(dict(idx=len(tasks), suite=suite, prefix=prefix,
                          cell=c, R_cap=R, G=G, scenario=s, arm=a, h=h,
                          k_sigma=k_, floor=f_,
                          synthetic=scenarios[s].synthetic_deadline,
                          **sweep_kw))
    if sweep is not None:
        suite = SWEEP_SUITES[sweep['kind']][0]
        key = 'horizon_h' if sweep['kind'] == 'horizon' else 'eps'
        for c in cells:
            for R in sweep['rcaps']:
                for s in scen_names:
                    for a in sweep['arms']:
                        for v in sweep['values']:
                            if _sweep_undefined(a, sweep['kind'], v):
                                continue
                            add(suite, c, R, s, a, None, k0, f0,
                                **{key: float(v)})
        return tasks
    for c in cells:
        for R in CONTROLLER_RCAPS:
            for s in scen_names:
                for a, h in arms:
                    add('primary', c, R, s, a, h, k0, f0)
    if sensitivity:
        for c in cells:
            for R in CONTROLLER_RCAPS:
                for s in ("none", "spike"):
                    for k_, f_ in CS.sensitivity_grid():
                        if (k_, f_) == (k0, f0) and s in scen_names:
                            continue        # identical to a primary row
                        for a in ctrl_arms:
                            add('sensitivity', c, R, s, a, None, k_, f_)
    return tasks


def _float_list(s, flag):
    try:
        v = [float(x) for x in s.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"{flag}: need comma-separated numbers, got {s!r}")
    if not v or len(set(v)) != len(v) or not all(np.isfinite(v)):
        sys.exit(f"{flag}: need distinct finite comma-separated numbers")
    return v


def _sweep_from_args(args, ctrl_arms):
    """Validate --predict-horizon-h / --eps-buffer-pcm / --arms / --rcaps
    and return the sweep dict used by _controller_tasks."""
    if not args.stress_suite:
        sys.exit("--predict-horizon-h / --eps-buffer-pcm / --arms / --rcaps "
                 "need --stress-suite (cells A-H).")
    if (args.predict_horizon_h is None) == (args.eps_buffer_pcm is None):
        sys.exit("give exactly one of --predict-horizon-h or "
                 "--eps-buffer-pcm (one factor per sweep); --arms and "
                 "--rcaps only apply to such a sweep.")
    kind = 'horizon' if args.predict_horizon_h is not None else 'eps'
    if kind == 'horizon':
        values = _float_list(args.predict_horizon_h, "--predict-horizon-h")
        for v in values:
            q = v * 3600.0 / PROJ_SEG_S
            if v < 0 or abs(q - round(q)) > 1e-9:
                sys.exit(f"--predict-horizon-h {v:g}: horizons must be >= 0 "
                         f"and a whole number of {PROJ_SEG_S:.0f} s "
                         f"projection segments.")
    else:
        values = _float_list(args.eps_buffer_pcm, "--eps-buffer-pcm")
        if any(v < 0 for v in values):
            sys.exit("--eps-buffer-pcm: buffers must be >= 0 pcm (a negative "
                     "buffer would move the enforced ceiling above "
                     "R_cap - sigma_m).")
    if 'computing_only' in ctrl_arms:
        sys.exit("computing_only consumes neither the reactor horizon nor "
                 "eps_buffer; its rows already exist in the canonical suite. "
                 "Drop --computing-only for a sweep.")
    arms = (args.arms.split(",") if args.arms else
            ['predictive'] + [a for a in ctrl_arms if a in SWEEP_ARMS])
    badarm = [a for a in arms if a not in SWEEP_ARMS]
    if badarm or len(set(arms)) != len(arms):
        sys.exit(f"--arms: distinct names from {','.join(SWEEP_ARMS)} only "
                 f"(other arms do not consume this parameter; their "
                 f"canonical rows already exist); got {args.arms!r}")
    if 'consolidated' in arms and 'consolidated' not in ctrl_arms:
        sys.exit("--arms consolidated needs --consolidated.")
    rcaps = (_float_list(args.rcaps, "--rcaps") if args.rcaps
             else list(CONTROLLER_RCAPS))
    if any(R not in CONTROLLER_RCAPS for R in rcaps):
        sys.exit(f"--rcaps must be a subset of the fixed-margin snapshots "
                 f"{CONTROLLER_RCAPS}; got {args.rcaps!r}")
    undefined = [dict(arm=a, value=v, reason=_sweep_undefined(a, kind, v))
                 for a in arms for v in values if _sweep_undefined(a, kind, v)]
    return dict(kind=kind, values=values, arms=arms, rcaps=rcaps,
                undefined=undefined)


def _controller_cli(args):
    """Validate and dispatch the opt-in controller flags. Returns False when
    none are present (the legacy path then runs exactly as before)."""
    ctrl_arms = [a for a, on in (('computing_only', args.computing_only),
                                 ('consolidated', args.consolidated)) if on]
    extra = {k: getattr(args, k) for k in (
        'stress_suite', 'decision_log', 'decision_log_level', 'cells',
        'scenarios', 'stress_G', 'stress_start_h', 'spike_mult',
        'spike_duration_h', 'deadline_shock_fraction',
        'deadline_shock_minutes', 'blackout_duration_h',
        'compound_offset_h', 'workers', 'overwrite')}
    given = [k for k, v in extra.items() if v not in (None, False)]
    sweep_given = [k for k in ('predict_horizon_h', 'eps_buffer_pcm', 'arms',
                               'rcaps') if getattr(args, k) is not None]
    args._sweep = None
    if not ctrl_arms:
        if given or sweep_given:
            sys.exit(f"--{(given + sweep_given)[0].replace('_', '-')} needs "
                     f"--computing-only and/or --consolidated.")
        return False
    if args.year:
        # Synthetic continuous year: main() runs these arms through
        # run_continuous (one run() per config), same controller code as the
        # monthly suite. Suite/sweep flags belong to the fixed-margin suite.
        other = [f for f, v in (('--frontier', args.frontier),
                                ('--quick', args.quick),
                                ('--self-test', args.self_test),
                                ('--physics-audit', args.physics_audit)) if v]
        other += ["--" + k.replace('_', '-') for k in given + sweep_given]
        if other:
            sys.exit(f"--year with controller arms cannot be combined with "
                     f"{', '.join(other)}.")
        return False
    if sweep_given:
        args._sweep = _sweep_from_args(args, ctrl_arms)
    bad = [f for f, v in (('--year', args.year), ('--r-cap-file',
                                                  args.r_cap_file),
                          ('--frontier', args.frontier),
                          ('--hysteresis', args.hysteresis),
                          ('--quick', args.quick),
                          ('--self-test', args.self_test),
                          ('--physics-audit', args.physics_audit)) if v]
    if bad:
        sys.exit(f"controller arms cannot be combined with {', '.join(bad)} "
                 f"(year-ordering and legacy grids stay separate from "
                 f"fixed-margin controller comparisons).")
    stress_only = ('cells', 'scenarios', 'stress_start_h', 'spike_mult',
                   'spike_duration_h', 'deadline_shock_fraction',
                   'deadline_shock_minutes', 'blackout_duration_h',
                   'compound_offset_h')
    if not args.stress_suite:
        s = [k for k in stress_only if getattr(args, k) is not None]
        if s:
            sys.exit(f"--{s[0].replace('_', '-')} needs --stress-suite.")
    elif args.borg:
        sys.exit("--stress-suite reads demand_curve_cell_<x>.csv for --cells;"
                 " --borg is ambiguous here.")
    if args.decision_log_level and not args.decision_log:
        sys.exit("--decision-log-level needs --decision-log.")
    controller_main(args, ctrl_arms)
    return True


def controller_main(args, ctrl_arms):
    import os, time, multiprocessing as mp
    import consolidated_scheduler as CS
    dt = 60.0
    sweep = getattr(args, '_sweep', None)       # set by _controller_cli
    scfg = CS.StressConfig(**{k: v for k, v in dict(
        event_start_h=args.stress_start_h, spike_multiplier=args.spike_mult,
        spike_duration_h=args.spike_duration_h,
        deadline_shock_fraction=args.deadline_shock_fraction,
        deadline_shock_minutes=args.deadline_shock_minutes,
        blackout_duration_h=args.blackout_duration_h,
        compound_offset_h=args.compound_offset_h).items() if v is not None})
    try:
        scfg.validate()
    except ValueError as e:
        sys.exit(f"[stress] {e}")
    G = 0.4 if args.stress_G is None else args.stress_G
    if not 0.0 <= G <= 1.0:
        sys.exit("--stress-G must lie in [0, 1]")
    price_path = args.price_file or MONTHLY_FILES[1]
    genmix_path = args.genmix_file or MONTHLY_FILES[2]
    if args.stress_suite:
        cells = list(args.cells or "".join(STRESS_CELLS))
        badc = [c for c in cells if c not in STRESS_CELLS]
        if badc or len(set(cells)) != len(cells):
            sys.exit(f"--cells must be distinct letters from "
                     f"{''.join(STRESS_CELLS)}; got {args.cells!r}")
        cell_paths = {c: f"demand_curve_cell_{c}.csv" for c in cells}
        scen_names = (args.scenarios.split(",") if args.scenarios
                      else ["none"] if sweep is not None   # nominal sweeps
                      else list(CS.SCENARIOS))
        prefix, default_out = "ST", "stress_results.csv"
    else:
        path = args.borg or MONTHLY_FILES[0]
        cells = [os.path.splitext(os.path.basename(path))[0]]
        cell_paths = {cells[0]: path}
        scen_names = ["none"]
        prefix, default_out = "CN", "controller_results.csv"
    try:
        scenarios = {s.name: s for s in CS.build_scenarios(
            scfg, scen_names + [x for x in ("none", "spike")
                                if args.stress_suite and x not in scen_names])}
    except ValueError as e:
        sys.exit(f"[stress] {e}")
    out = args.out or default_out
    stem = out[:-4] if out.endswith(".csv") else out
    outputs = dict(results=out, trips=f"{stem}_trip_log.csv",
                   comparisons=f"{stem}_comparisons.csv",
                   cells=f"{stem}_cell_characteristics.csv")
    if sweep is not None:
        # a one-factor sweep has no primary cross-arm rows and no detector
        # grid; it gets a per-value summary instead
        del outputs['comparisons']
        outputs['sweep_summary'] = f"{stem}_sweep_summary.csv"
    elif args.stress_suite:
        outputs.update(sensitivity=f"{stem}_spike_sensitivity.csv",
                       detection=f"{stem}_detection.csv")
    outputs['manifest'] = f"{stem}_manifest.json"
    if args.decision_log:
        outputs['episodes'] = f"{stem}_decision_episodes.csv"
        if args.decision_log_level == 'minute':
            outputs['minutes'] = f"{stem}_decisions_minute.csv"
    for p in outputs.values():
        if _is_canonical_output(p):
            sys.exit(f"refusing to write canonical baseline file {p}")
        if os.path.exists(p) and (not args.overwrite or
                                  p.endswith("_manifest.json")):
            sys.exit(f"{p} exists; choose another --out"
                     + ("" if p.endswith("_manifest.json") else
                        " or pass --overwrite")
                     + " (run manifests are immutable)")

    import datetime, dataclasses as _dc, platform
    code_files = [os.path.abspath(__file__),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "consolidated_scheduler.py")]
    code_hash0 = {os.path.basename(f): _sha256(f) for f in code_files}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[controller] arms: legacy free/reactive/hysteresis 2-4-6 h/"
          f"predictive + {', '.join(ctrl_arms)} | R_cap snapshots "
          f"{CONTROLLER_RCAPS} pcm (fixed-margin, not a fuel cycle) | "
          f"G={G} | tiered, price signal, trips on, eps=20 for predictive "
          f"and consolidated")
    if sweep is not None:
        print(f"[controller] ONE-FACTOR SWEEP ({SWEEP_SUITES[sweep['kind']][1]}"
              f"): {'predict_horizon_h' if sweep['kind'] == 'horizon' else 'eps_buffer_pcm'}"
              f" = {sweep['values']} for arms {sweep['arms']} at R_cap "
              f"{sweep['rcaps']} pcm; every other setting canonical; no "
              f"detector-sensitivity grid")
        for u_ in sweep['undefined']:
            print(f"[controller] skipped (undefined): {u_['arm']} at "
                  f"{u_['value']:g}: {u_['reason']}")
    print(f"[controller] price={price_path} genmix={genmix_path}")
    ctx_cells = {}
    for c in cells:
        ctx_cells[c] = _load_cell_ctx(c, cell_paths[c], price_path,
                                      genmix_path, dt)
    tasks = _controller_tasks(cells, scen_names, ctrl_arms, scenarios, G,
                              prefix, args.stress_suite and sweep is None,
                              sweep=sweep)
    _CTX.update(cells=ctx_cells, scenarios=scenarios, stress_cfg=scfg,
                timing=args.timing_mode,
                decision_log=(None if not args.decision_log else
                              args.decision_log_level or 'episodes'))
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[controller] {len(tasks)} runs on {workers} worker(s); cells "
          f"{''.join(cells) if args.stress_suite else cells[0]}; scenarios "
          f"{scen_names}")
    t0 = time.time()
    rows = [None] * len(tasks); trips, eps_rows, minute_rows = [], [], []
    bad = 0

    def flush():
        pd.DataFrame([r_ for r_ in rows if r_ is not None]).to_csv(
            outputs['results'], index=False)
        if trips:
            pd.DataFrame(trips).to_csv(outputs['trips'], index=False)

    if workers > 1:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap_unordered(_controller_task, tasks, chunksize=1)
    else:
        pool = None
        it = map(_controller_task, tasks)
    try:
        for done, (i, row, tr, ep, mn) in enumerate(it, 1):
            rows[i] = row; trips.extend(tr); eps_rows.extend(ep)
            minute_rows.extend(mn)
            flag = ""
            if abs(row['residual']) > 1.0 or \
               abs(row.get('energy_residual_Wh', 0.0) or 0.0) > 1.0:
                flag = "  <-- CONSERVATION BROKEN"; bad += 1
            print(f"  [{done:4d}/{len(tasks)}] {row['experiment']:>60} | "
                  f"{row['rx']:>14} | uns {row['true_unserved']/1e3:10.1f} "
                  f"kWh | cross {row['crossings']:5d} | trips "
                  f"{row['n_trips']:2d} | {time.time()-t0:7.0f} s{flag}",
                  flush=True)
            if done % 25 == 0:
                flush()
    finally:
        if pool:
            pool.close(); pool.join()
    rows = [r_ for r_ in rows if r_ is not None]
    trips.sort(key=lambda d_: (d_['experiment'], d_['rx'], d_['start_h']))
    flush()
    df = pd.DataFrame(rows)
    pd.DataFrame([ctx_cells[c]['chars'] for c in cells]).to_csv(
        outputs['cells'], index=False)
    if 'episodes' in outputs:
        pd.DataFrame(eps_rows).to_csv(outputs['episodes'], index=False)
    if 'minutes' in outputs:
        pd.DataFrame(minute_rows).to_csv(outputs['minutes'], index=False)
    if 'comparisons' in outputs:
        _controller_comparisons(df, outputs['comparisons'])
    if 'sensitivity' in outputs:
        _controller_sensitivity(df, ctx_cells, scenarios, scfg, dt,
                                outputs['sensitivity'], outputs['detection'])
    if sweep is not None:
        _sweep_summary(df, sweep, outputs['sweep_summary'])
    else:
        _controller_print(df, ctrl_arms)
    code_hash1 = {os.path.basename(f): _sha256(f) for f in code_files}
    inputs = {}
    for pth in list(cell_paths.values()) + [price_path, genmix_path]:
        inputs[pth] = _sha256(pth) if os.path.exists(pth) else "MISSING"
    man = dict(
        manifest_version=1, started_utc=started,
        finished_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        argv=sys.argv, cwd=os.getcwd(),
        python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__,
        code_sha256_at_start=code_hash0, code_sha256_at_end=code_hash1,
        code_unchanged_during_run=code_hash0 == code_hash1,
        input_sha256=inputs,
        R_cap_snapshots_pcm=list(CONTROLLER_RCAPS),
        R_cap_note="fixed-margin snapshots; no fuel-cycle R_cap(t)",
        G_max_frac=G, dt_s=dt, settle_h=6.0, sched='tiered',
        trip=True, emergency_grid=True,
        eps_buffer_pcm={a_: (20.0 if a_ in ('predictive', 'consolidated')
                             else 0.0) for a_, _ in CONTROLLER_ARMS},
        arms=[dict(arm=a_, hysteresis_h=h_) for a_, h_ in CONTROLLER_ARMS
              if a_ not in CS.CONTROLLER_MODES or a_ in ctrl_arms],
        controller_config=_dc.asdict(CS.ControllerConfig()),
        controller_horizon_s=CS.ControllerConfig().horizon_s,
        legacy_predict_horizon_s=10 * 3600.0,
        sensitivity_grid=CS.sensitivity_grid() if args.stress_suite else [],
        stress_config=_dc.asdict(scfg),
        scenarios={n_: s_.metadata() for n_, s_ in scenarios.items()},
        evidence_classes={PRIMARY_EVIDENCE: "primary cross-arm comparison",
                          DIAGNOSTIC_DEADLINE: "endogenous shock size per "
                          "policy; diagnostic only",
                          "sensitivity": "detector threshold sensitivity"},
        cooling_convention="all arms: cooling follows non-deferred IT work "
                           "(P_work - deferred + drained), as in run()",
        timing_mode=args.timing_mode, workers=workers, runs=len(tasks),
        conservation_failures=bad,
        synthetic_deadline_note=CS.SYNTHETIC_DEADLINE_NOTE,
        outputs={k_: dict(path=v_, sha256=_sha256(v_))
                 for k_, v_ in outputs.items()
                 if k_ != 'manifest' and os.path.exists(v_)})
    man.update(_git_state(os.path.dirname(os.path.abspath(__file__))))
    if sweep is not None:
        kind = sweep['kind']
        man.update(
            R_cap_snapshots_pcm=list(sweep['rcaps']),
            arms=[dict(arm=a_, hysteresis_h=None) for a_ in sweep['arms']],
            sensitivity_grid=[],
            evidence_classes={SWEEP_SUITES[kind][1]:
                              f"one-factor {kind} sensitivity; never "
                              f"primary cross-arm evidence"},
            sweep=dict(
                parameter=('predict_horizon_h' if kind == 'horizon'
                           else 'eps_buffer_pcm'),
                values=sweep['values'], arms=sweep['arms'],
                rcaps_pcm=sweep['rcaps'], scenarios=scen_names,
                cells=cells, suite=SWEEP_SUITES[kind][0],
                evidence_class=SWEEP_SUITES[kind][1],
                skipped_undefined=sweep['undefined'],
                canonical_value=(CANONICAL_HORIZON_H if kind == 'horizon'
                                 else CANONICAL_EPS_PCM),
                fixed=dict(eps_buffer_pcm=CANONICAL_EPS_PCM)
                if kind == 'horizon' else
                dict(predict_horizon_h=CANONICAL_HORIZON_H),
                detector=dict(k_sigma=CS.PRIMARY_DETECTOR.k_sigma,
                              rated_floor_fraction=CS.PRIMARY_DETECTOR
                              .rated_floor_fraction),
                semantics=_SWEEP_SEMANTICS[kind]))
        if kind == 'horizon':
            man.update(controller_horizon_s='swept (see sweep.values, h)',
                       legacy_predict_horizon_s='swept (see sweep.values, h)')
        else:
            man.update(eps_buffer_pcm={a_: 'swept (see sweep.values)'
                                       for a_ in sweep['arms']})
    _write_manifest(outputs['manifest'], man)
    if code_hash0 != code_hash1:
        print("WARNING: code changed during the run (see manifest).")
    if sweep is None:
        _controller_print(df, ctrl_arms)
    print(f"\n[controller] wrote " + ", ".join(outputs.values()) +
          f" in {(time.time() - t0)/60:.1f} min.")
    if bad:
        print(f"WARNING: {bad} row(s) FAILED CONSERVATION.")
    else:
        print("All controller rows passed conservation (|residual| <= 1 Wh).")
    print(f"NOTE: {CS.SYNTHETIC_DEADLINE_NOTE}")


def _git_state(path):
    """Manifest provenance: git HEAD and whether the code files differ from
    it (the code sha256 fields stay the authoritative identity)."""
    import subprocess
    try:
        run_ = lambda *a: subprocess.run(["git", *a], cwd=path, check=True,
                                         capture_output=True,
                                         text=True).stdout.strip()
        head = run_("rev-parse", "HEAD")
        dirty = run_("status", "--porcelain", "--", "sim.py",
                     "consolidated_scheduler.py")
        return dict(git_head=head, git_code_files_modified_vs_head=bool(dirty))
    except (OSError, subprocess.CalledProcessError) as e:
        return dict(git_head=None, git_error=str(e))


_SWEEP_SEMANTICS = {
    'horizon': (
        "predict_horizon_s = h * 3600. predictive: constant-hold "
        "project_peak_rho over int(h*3600/300) steps; h=0 checks the present "
        "xenon reactivity only (refuses a down-ramp only when rho already "
        "exceeds R_cap - sigma_m - eps); this is NOT the reactive headroom "
        "gate. consolidated: ControllerConfig.horizon_s = length of the "
        "projected requested power path (300 s segments), which also bounds "
        "the pending deadlines that enter that path; needs >= 2 segments. "
        "Distinct from hysteresis (wait time after a maneuver): this is how "
        "far forward reactor state is projected BEFORE accepting one."),
    'eps': (
        "eps_buffer (pcm) lowers the enforced ceiling to R_cap - sigma_m - "
        "eps for predictive and consolidated only. sigma_m (100 pcm), the "
        "trip rule (R_cap - sigma_m - rho < 0) and the crossing metric are "
        "unchanged, so eps changes conservatism only. It is a deterministic "
        "buffer, not a measurement-noise model."),
}
_SWEEP_METRICS = ('true_unserved', 'dropped', 'n_trips', 'deadtime_h',
                  'crossings', 'cross_episodes', 'max_exceed', 'min_headroom',
                  'grid_cost', 'refused_steps', 'false_safe_decisions',
                  'false_safe_eps_decisions', 'proj_vs_realized_min_pcm',
                  'policy_mean_us', 'policy_p95_us', 'projection_mean_us',
                  'projection_calls')


def _sweep_summary(df, sweep, path):
    """Per (R_cap, scenario, arm, swept value): mean/std/min/max over
    cells. Cell-level values stay in the results CSV."""
    col = 'predict_horizon_h' if sweep['kind'] == 'horizon' else 'eps'
    ms = [m for m in _SWEEP_METRICS if m in df.columns]
    g = df.groupby(['R_cap', 'scenario', 'rx', col], sort=True)
    s = g[ms].agg(['mean', 'std', 'min', 'max'])
    s.columns = [f"{m}_{a}" for m, a in s.columns]
    s.insert(0, 'cells', g.cell.agg(lambda x: "".join(sorted(x))))
    s.insert(0, 'n_cells', g.size())
    s = s.reset_index()
    s.to_csv(path, index=False)
    print(f"\nSWEEP SUMMARY ({col}; mean over cells; true_unserved in kWh)")
    for (R, sc), x in s.groupby(['R_cap', 'scenario'], sort=False):
        print(f"\n  R_cap={R:.0f} pcm | scenario={sc}")
        print(f"    {'arm':>13} | {col:>17} | {'n':>2} | {'uns kWh':>10} | "
              f"{'trips':>5} | {'dead h':>6} | {'cross':>6} | "
              f"{'refused':>8} | {'grid $':>8}")
        for _, r_ in x.iterrows():
            print(f"    {r_['rx']:>13} | {r_[col]:17g} | {r_['n_cells']:2d} | "
                  f"{r_['true_unserved_mean']/1e3:10.1f} | "
                  f"{r_['n_trips_mean']:5.2f} | {r_['deadtime_h_mean']:6.1f} "
                  f"| {r_['crossings_mean']:6.0f} | "
                  f"{r_['refused_steps_mean']:8.0f} | "
                  f"{r_['grid_cost_mean']:8.2f}")


_CMP_METRICS = ('true_unserved', 'dropped', 'unmet', 'crossings',
                'cross_episodes', 'max_exceed', 'n_trips', 'deadtime_h',
                'grid_Wh', 'grid_cost', 'surplus_Wh', 'water_sys_L',
                'rigid_unserved_Wh', 'deadline_miss_b_Wh',
                'deadline_miss_m_Wh')
_COMPARISONS = (('computing_only', 'free',
                 'computing scheduling alone'),
                ('consolidated', 'computing_only',
                 'incremental reactor/xenon awareness'),
                ('consolidated', 'predictive',
                 'joint coordination vs constant-hold gate'))


def _controller_comparisons(df, path):
    key = ['suite', 'cell', 'R_cap', 'scenario']
    prim = df[(df.evidence_class == PRIMARY_EVIDENCE) & (df.hyst_h.isna())]
    rows = []
    for gk, g in prim.groupby(key, sort=False):
        by = {r_['rx']: r_ for _, r_ in g.iterrows()}
        for a, b, what in _COMPARISONS:
            if a not in by or b not in by:
                continue
            A, B = by[a], by[b]
            rec = dict(zip(key, gk), comparison=f"{a} - {b}", isolates=what,
                       synthetic_deadline=A['synthetic_deadline'],
                       deadline_source=A['deadline_source'],
                       blackout_type=A['blackout_type'])
            for m in _CMP_METRICS:
                va, vb = A.get(m, np.nan), B.get(m, np.nan)
                rec[f"{m}_{a}"] = va; rec[f"{m}_{b}"] = vb
                try:
                    rec[f"delta_{m}"] = float(va) - float(vb)
                except (TypeError, ValueError):
                    rec[f"delta_{m}"] = np.nan
            rows.append(rec)
    pd.DataFrame(rows).to_csv(path, index=False)


def _controller_sensitivity(df, ctx_cells, scenarios, scfg, dt, path_s,
                            path_d):
    import consolidated_scheduler as CS
    cols = ['cell', 'controller_k_sigma', 'controller_rated_floor_fraction',
            'R_cap', 'scenario', 'rx', 'suite', 'spike_detections',
            'spike_episodes', 'minutes_STRESSED', 'spike_energy_Wh',
            'spike_frac_sigma', 'spike_frac_rated_floor', 'true_unserved',
            'unmet', 'dropped', 'dropped_b', 'dropped_m',
            'deadline_miss_b_Wh', 'deadline_miss_m_Wh', 'rigid_unserved_Wh',
            'crossings', 'cross_episodes', 'n_trips', 'deadtime_h']
    s = df[df.rx.isin(CS.CONTROLLER_MODES) & df.scenario.isin(
        ('none', 'spike'))]
    s[cols].sort_values(cols[:6]).to_csv(path_s, index=False)
    # detector-only passes (request series is exogenous -> arm-independent)
    sidx = int(6 * 3600 / dt)
    hold = int(round(CS.ControllerConfig().stress_hold_s / dt))
    rows = []
    for c, C in ctx_cells.items():
        tg = C['t_grid']
        for k_, f_ in CS.sensitivity_grid():
            cfg = CS.DetectorConfig(k_sigma=k_, rated_floor_fraction=f_)
            nat = None
            for name in ('none',) + tuple(n_ for n_ in scenarios
                                          if scenarios[n_].spike):
                ev = CS.build_event_inputs(scenarios[name], len(tg), dt, scfg)
                u_eff = CS.wrap_u_at(C['u_at'], ev, tg[0])
                req = [P_FIXED + PUE * work_power(u_eff(tt)) for tt in tg]
                spk, bind, exc = CS.detector_pass(req, dt, C['P_rated'], cfg)
                if name == 'none':
                    nat = spk
                sc = spk[sidx:]; bd = bind[sidx:]
                nd = sum(sc)
                rec = dict(cell=c, k_sigma=k_, rated_floor_fraction=f_,
                           scenario=name, detections=nd,
                           frac_sigma=(sum(b_ == 'sigma' for b_ in bd) / nd
                                       if nd else np.nan),
                           frac_rated_floor=(sum(b_ == 'rated_floor'
                                                 for b_ in bd) / nd
                                             if nd else np.nan),
                           spike_energy_Wh=float(sum(
                               e_ for e_, s_ in zip(exc[sidx:], sc) if s_))
                           * dt / 3600.0,
                           ground_truth=('none (natural trace; detections '
                                         'not labelled true/false)'
                                         if name == 'none' else
                                         'injected spike window'))
                if ev.spike_window is not None:
                    rec.update(CS.injected_detection_metrics(
                        spk, nat, ev.spike_window, hold, dt))
                rows.append(rec)
    d = pd.DataFrame(rows)
    inj = d[d.scenario != 'none']
    if len(inj):
        miss = inj.groupby(['k_sigma', 'rated_floor_fraction'])[
            'injected_detected'].apply(lambda x: 1.0 - x.mean())
        d = d.merge(miss.rename('missed_event_rate_all_cells').reset_index(),
                    on=['k_sigma', 'rated_floor_fraction'], how='left')
    d.to_csv(path_d, index=False)


def _controller_print(df, ctrl_arms):
    import consolidated_scheduler as CS
    prim = df[df.evidence_class == PRIMARY_EVIDENCE]
    order = list(dict.fromkeys(a for a, _ in CONTROLLER_ARMS))
    n_diag = int((df.evidence_class == DIAGNOSTIC_DEADLINE).sum())
    if n_diag:
        print(f"\n{n_diag} deadline-shock rows are controller diagnostics "
              f"(endogenous shock size per policy) and are excluded from "
              f"the primary tables. {CS.SYNTHETIC_DEADLINE_NOTE}")
    print("\nPRIMARY COMPARISON -- mean over cells per (R_cap, scenario); "
          "rows never mix scenarios. Borg cells differ mainly in "
          "flexibility (batch/mid fraction), moderately in utilisation and "
          "burstiness; they are not eight distinct workload types.")
    for (R, s), g in prim.groupby(['R_cap', 'scenario'], sort=False):
        syn = bool(g['synthetic_deadline'].iloc[0])
        print(f"\n  R_cap={R:.0f} pcm | scenario={s}"
              + (" [SYNTHETIC DEADLINE]" if syn else ""))
        print(f"    {'arm':>16} | {'uns kWh':>10} | {'drop kWh':>9} | "
              f"{'cross':>7} | {'trips':>5} | {'dead h':>7} | "
              f"{'grid $':>8} | {'rigid kWh':>9} | {'STRESS min':>10}")
        for a in order:
            for h in sorted(set(g[g.rx == a]['hyst_h'].dropna())) or [None]:
                x = g[(g.rx == a) & ((g.hyst_h == h) if h is not None
                                     else g.hyst_h.isna())]
                if not len(x):
                    continue
                lab = a if h is None else f"hyst {h:.0f}h"
                print(f"    {lab:>16} | {x.true_unserved.mean()/1e3:10.1f} | "
                      f"{x.dropped.mean()/1e3:9.1f} | "
                      f"{x.crossings.mean():7.0f} | {x.n_trips.mean():5.1f} | "
                      f"{x.deadtime_h.mean():7.1f} | {x.grid_cost.mean():8.2f} "
                      f"| {x.get('rigid_unserved_Wh', pd.Series([np.nan])).mean()/1e3:9.1f} "
                      f"| {x.get('minutes_STRESSED', pd.Series([np.nan])).mean():10.0f}")
        if syn:
            print(f"    NOTE: {CS.SYNTHETIC_DEADLINE_NOTE}")


def main():
    import argparse, time, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="trimmed grid for smoke-testing (NOT for results)")
    ap.add_argument("--frontier", action="store_true",
                    help="[NEW] also run EXP H: dip-depth x fuel-age grid "
                         "(free arm) -> feasibility phase-diagram data. "
                         "OFF by default so default runs stay row-for-row "
                         "comparable to the baseline CSV.")
    ap.add_argument("--out", default=None,
                    help="results CSV (default v6_results.csv; "
                         "year_results.csv with --year)")
    ap.add_argument("--borg", help="workload demand curve CSV")
    ap.add_argument("--price-file", help="PJM hourly LMP CSV")
    ap.add_argument("--genmix-file", help="PJM hourly generation-by-fuel CSV")
    ap.add_argument("--year", action="store_true",
                    help="synthetic 365-day path: strict inputs, one "
                         "continuous run() per config; needs --borg, "
                         "--price-file, --genmix-file")
    ap.add_argument("--r-cap-file",
                    help="--year only: one R_cap (pcm) per simulation "
                         "minute, one per line, '#' comments, no header")
    ap.add_argument("--hysteresis", action="store_true",
                    help="also run the fixed-delay hysteresis baseline "
                         "(2/4/6 h). OFF by default.")
    ap.add_argument("--physics-audit", action="store_true",
                    help="print physics reference checks and exit")
    ap.add_argument("--timing-mode", default="off",
                    choices=("off", "coarse", "detailed"),
                    help="policy-decision timing (default off)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the year-path state-continuity test and exit")
    g_ = ap.add_argument_group("opt-in workload-aware controller "
                               "(never part of the default run)")
    g_.add_argument("--computing-only", action="store_true",
                    help="add the causal least-slack-first controller arm "
                         "(no reactor state)")
    g_.add_argument("--consolidated", action="store_true",
                    help="add the workload + xenon-path controller arm")
    g_.add_argument("--stress-suite", action="store_true",
                    help="deterministic stress suite over Borg cells A-H at "
                         "R_cap 8000/3500/2700 plus the spike-threshold "
                         "sensitivity grid")
    g_.add_argument("--decision-log", action="store_true",
                    help="write refusal/modification episodes")
    g_.add_argument("--decision-log-level", choices=("episodes", "minute"),
                    default=None, help="'minute' also writes every "
                                       "modified minute (large)")
    g_.add_argument("--cells", default=None,
                    help="stress suite cell letters (default abcdefgh)")
    g_.add_argument("--scenarios", default=None,
                    help="stress suite scenarios, comma-separated "
                         "(default all)")
    g_.add_argument("--stress-G", type=float, default=None,
                    help="interconnection fraction (default 0.4)")
    g_.add_argument("--stress-start-h", type=float, default=None)
    g_.add_argument("--spike-mult", type=float, default=None)
    g_.add_argument("--spike-duration-h", type=float, default=None)
    g_.add_argument("--deadline-shock-fraction", type=float, default=None)
    g_.add_argument("--deadline-shock-minutes", type=float, default=None)
    g_.add_argument("--blackout-duration-h", type=float, default=None)
    g_.add_argument("--compound-offset-h", type=float, default=None)
    g_.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default cpu_count - 1)")
    g_.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing controller outputs "
                         "(canonical baselines are always refused)")
    s_ = ap.add_argument_group(
        "opt-in one-factor sweeps (with --stress-suite; never part of the "
        "default or canonical runs; nominal scenario unless --scenarios)")
    s_.add_argument("--predict-horizon-h", default=None,
                    help="comma-separated reactor projection horizons (h), "
                         "e.g. 0,2,4,6,10,15 (canonical 10); multiples of "
                         "300 s")
    s_.add_argument("--eps-buffer-pcm", default=None,
                    help="comma-separated ceiling buffers (pcm) for "
                         "predictive/consolidated (canonical 20)")
    s_.add_argument("--arms", default=None,
                    help="sweep arms: predictive and/or consolidated "
                         "(default predictive + the enabled controller arm)")
    s_.add_argument("--rcaps", default=None,
                    help="sweep R_cap snapshots, subset of 8000,3500,2700 "
                         "(default all three)")
    args = ap.parse_args()

    if _controller_cli(args):
        return

    if args.physics_audit:
        physics_audit(); return
    if args.self_test:
        self_test(); return
    if args.r_cap_file and not args.year:
        sys.exit("--r-cap-file is only used with --year.")

    if args.year:
        need = [f for f, v in (("--borg", args.borg),
                               ("--price-file", args.price_file),
                               ("--genmix-file", args.genmix_file)) if not v]
        if need:
            sys.exit(f"--year requires explicit {', '.join(need)} (monthly "
                     f"defaults are never substituted).")
        borg_path, price_path, genmix_path = \
            args.borg, args.price_file, args.genmix_file
    else:
        borg_path = args.borg or MONTHLY_FILES[0]
        price_path = args.price_file or MONTHLY_FILES[1]
        genmix_path = args.genmix_file or MONTHLY_FILES[2]
    out_path = args.out or ("year_results.csv" if args.year
                            else "v6_results.csv")
    trip_path = "trip_log_year.csv" if args.year else "trip_log.csv"
    # --year --computing-only/--consolidated (admitted by _controller_cli):
    # same controller code as the monthly suite, via run_continuous.
    year_ctrl = [a for a, on in (('computing_only', args.computing_only),
                                 ('consolidated', args.consolidated)) if on]
    if year_ctrl:
        if not args.out or _is_canonical_output(args.out) or \
                os.path.exists(args.out):
            sys.exit("--year with controller arms needs a NEW --out file "
                     "(not an existing or canonical baseline name).")
        trip_path = (args.out[:-4] if args.out.endswith(".csv")
                     else args.out) + "_trip_log.csv"
        if os.path.exists(trip_path):
            sys.exit(f"{trip_path} exists; choose another --out.")
    print(f"Inputs: workload = {borg_path} | price = {price_path} | "
          f"genmix = {genmix_path}")

    dt = 60.0
    if args.year:
        print("Synthetic 365-day workload assembled from repeated real Borg "
              "Cell A-H traces (not a recorded year).")
        validate_year_workload(borg_path)
    t, u_at, fb_at, fm_at = load_borg(borg_path)
    if args.year:
        # rows are 5-min window STARTS; the last (day 364 23:55) covers one
        # more window, so coverage ends at t[-1] + spacing. The final 4 min
        # past t[-1] take that window's value (np.interp end-hold).
        sp = np.diff(t)
        t_end = t[-1] + float(np.median(sp[sp > 0]))
        n_steps = int(round((t_end - t[0]) / dt))
        t_grid = t[0] + dt * np.arange(n_steps)
        days = n_steps * dt / 86400.0
        if n_steps != YEAR_STEPS or days != 365.0:
            sys.exit(f"[YEAR grid] {n_steps} steps / {days} days, expected "
                     f"{YEAR_STEPS} / 365.")
        umax = max(u_at(tt) for tt in t)
        print(f"[YEAR grid] {n_steps} one-minute steps, {days:.3f} days; "
              f"existing u = 0.85*tot/tot.max() leaves the peak-mode year "
              f"unchanged (max u = {umax:.12g})")
    else:
        t_grid = np.arange(t[0], t[-1], dt)

    Pdcs = np.array([P_FIXED + work_power(u_at(tt))*PUE for tt in t_grid])
    rated_mean = Pdcs.mean() / 0.95
    rated_peak = Pdcs.max()  / 0.95

    print(f"Simulating {t_grid[-1]/86400:.1f} days, {len(t_grid)} steps, "
          f"rho_eq = {rho_eq:.0f} pcm")
    print(f"peak={Pdcs.max()/1e6:.2f} MW  mean={Pdcs.mean()/1e6:.2f} MW  "
          f"mean-sized rated={rated_mean/1e6:.2f} MW  "
          f"peak-sized rated={rated_peak/1e6:.2f} MW")
    if not args.year:
        print(f"Fuel-age parameter: R_cap pcm, cited endpoints "
              f"{R_CAP_GRID[0]:.0f} (BOC) ... {R_CAP_GRID[-1]:.0f} (EOC). "
              f"A 31-day workload month is simulated at each fixed ceiling; "
              f"no fuel-cycle time is simulated.")
    print("Reactor arms: free = limit-agnostic (counterfactual) | reactive = "
          "up-ramp gate | predictive = forward-projected dip refusal.")
    print("Scheduler arms: tiered = type-aware | blind = type-agnostic.")
    print(f"Water: {WATER_GAL_PER_MWH:.0f} gal/MWh (nuclear wet tower) "
          f"[Macknick/NREL Table 2, range {WATER_GAL_PER_MWH_MIN:.0f}-"
          f"{WATER_GAL_PER_MWH_MAX:.0f}] = {WATER_L_PER_MWH:.0f} L/MWh; "
          f"DC facility WUE {WUE_L_PER_KWH} L/kWh [VERIFY source]. "
          f"Linear proxies [disclosed].\n")

    if args.year:
        try:
            p_start = _utc_series(pd.read_csv(price_path), price_path,
                                  "[YEAR price]")[0].min()
        except (OSError, ValueError) as e:
            sys.exit(f"[YEAR price] could not read {price_path}: {e}")
        price, thresh, tiled = load_pjm(price_path, t_grid, strict=True,
                                        start_utc=p_start)
    else:
        price, thresh, tiled = load_pjm(price_path, t_grid)
    print("[genmix] fuel -> gal/MWh map: " +
          ", ".join(f"{k}:{v:.0f}" for k, v in FUEL_WATER_GAL_MWH.items()))
    if args.year:
        ewif = load_pjm_genmix(genmix_path, t_grid, strict=True,
                               start_utc=p_start)
    else:
        ewif = load_pjm_genmix(genmix_path, t_grid)
    print()

    R_cap_ts = None
    if args.year:
        if not args.r_cap_file:
            print("Year inputs validated; no cited R_cap trajectory was "
                  "supplied, so no year reactor run was executed.")
            return
        R_cap_ts = load_r_cap_file(args.r_cap_file, len(t_grid))


    if args.quick:
        RCAPS   = (8000.0, 2700.0)
        MIDHZ   = (3600.0,)
        DIPS    = (0.60,)
        SPACING = (5.0,)
        GVALS   = (0.4,)
        EPSES   = (0.0, 20.0)
        print("*** --quick ***\n")
    else:
        RCAPS   = R_CAP_GRID
        MIDHZ   = (1800.0, 3600.0, 14400.0)
        DIPS    = (0.78, 0.60, 0.50)
        SPACING = (3.0, 5.0, 8.0)
        GVALS   = (0.2, 0.4)
        EPSES   = (0.0, 20.0, 50.0)

    bad = [0]
    t0 = time.time()
    done = [0]

    def flush_csv():
        pd.DataFrame(CSV_ROWS).to_csv(out_path, index=False)
        if TRIP_ROWS:
            pd.DataFrame(TRIP_ROWS).to_csv(trip_path, index=False)

    def go(tag, sched, arm, extra, **kw):
        """run one config, print, record, checkpoint csv"""
        if args.timing_mode != 'off':
            kw['timing'] = args.timing_mode
        # controller arms (--year only) dispatch through run()'s
        # controller_mode, exactly as in the monthly controller suite
        arm_kw = (dict(controller_mode=arm) if arm in year_ctrl
                  else dict(rx_mode=arm))
        r = (run_continuous if args.year else run)(
                fb_at, fm_at, sched, t_grid, u_at,
                kw.pop('P_rated', rated_mean), dt, kw.pop('R_cap'),
                ewif=ewif, **arm_kw, **kw)
        show(tag, sched, arm, r, bad, extra=extra)
        done[0] += 1
        if done[0] % 10 == 0:
            flush_csv()
            el = time.time() - t0
            print(f"    [{done[0]} runs, {el/60:.1f} min elapsed, "
                  f"{el/done[0]:.1f} s/run, checkpointed]")


    def hyst_table(exp):
        rows = [r_ for r_ in CSV_ROWS if r_.get('exp') == exp]
        print(f"\n{exp} metrics: {'experiment':>22} | {'rx':>4} | "
              f"{'true_uns kWh':>12} | {'uns%':>5} | {'cross':>5} | "
              f"{'epis':>4} | {'max_exc':>7} | {'trips':>5} | "
              f"{'dead_h':>6} | {'grid kWh':>9} | {'grid $':>8}")
        for r_ in rows:
            print(f"{'':>{len(exp) + 10}}{r_['experiment']:>22} | "
                  f"{RXLABEL[r_['rx']]:>4} | {r_['true_unserved']/1e3:12.1f} | "
                  f"{r_['unserved_pct']:5.2f} | {r_['crossings']:5d} | "
                  f"{r_['cross_episodes']:4d} | {r_['max_exceed']:7.1f} | "
                  f"{r_['n_trips']:5d} | {r_['deadtime_h']:6.1f} | "
                  f"{r_['grid_Wh']/1e3:9.1f} | {r_['grid_cost']:8.2f}")

    if args.year:
        arms_y = [(a, None) for a in ARMS]
        if args.hysteresis:
            arms_y += [('hysteresis', h) for h in HYST_H]
        arms_y += [(a, None) for a in year_ctrl]
        print("EXP Y -- synthetic 365-day workload, ONE continuous run() per "
              f"config, R_cap(t) from {args.r_cap_file}; EXP G settings "
              "(tiered, price signal, trips on, eps=20 for predictive"
              + (" and consolidated" if 'consolidated' in year_ctrl else "")
              + ")")
        print(HDR)
        for G in GVALS:
            for arm, h in arms_y:
                eps = 20.0 if arm in ("predictive", "consolidated") else 0.0
                hk = {} if h is None else dict(hysteresis_h=h)
                go(f"Y G={G:.1f}" + ("" if h is None else f" h={h:.0f}"),
                   "tiered", arm,
                   dict(exp='Y', R_cap=float('nan'),
                        r_cap_file=args.r_cap_file, mid_h=1.0, dip_to=None,
                        spacing_h=None, G=G, sizing='mean', eps=eps,
                        hyst_h=h),
                   R_cap=R_cap_ts, price=price, cheap_thresh=thresh,
                   G_max_frac=G, trip=True, eps_buffer=eps, **hk)
            print()
        if args.hysteresis:
            hyst_table('Y')
        flush_csv()
        print(f"\nWrote {out_path} ({len(CSV_ROWS)} rows) and "
              f"{trip_path if TRIP_ROWS else 'no trip log'} in "
              f"{(time.time() - t0)/60:.1f} min.")
        if bad[0]:
            print(f"WARNING: {bad[0]} row(s) FAILED CONSERVATION.")
        return

    print("EXP A -- CONTROL: mean-sized, no forced dips, no price signal")
    print(HDR)
    for R_cap in RCAPS:
        for mid_h in MIDHZ:
            for sched in SCHEDS:
                for arm in ARMS:
                    go(f"A R={R_cap:.0f} m={mid_h/3600:.1f}h", sched, arm,
                       dict(exp='A', R_cap=R_cap, mid_h=mid_h/3600,
                            dip_to=None, spacing_h=None, G=0.0,
                            sizing='mean', eps=0.0),
                       R_cap=R_cap, mid_horizon_s=mid_h)
        print()


    print("\nEXP B -- PEAK-sized reactor, natural ramps from the real trace")
    print(HDR)
    for R_cap in RCAPS:
        for sched in SCHEDS:
            for arm in ARMS:
                go(f"B peak R={R_cap:.0f}", sched, arm,
                   dict(exp='B', R_cap=R_cap, mid_h=1.0, dip_to=None,
                        spacing_h=None, G=0.0, sizing='peak', eps=0.0),
                   R_cap=R_cap, P_rated=rated_peak)
        print()


    print("\nEXP C -- forced stacked dips at EOC (R_cap=2700), mean-sized")
    print(HDR)
    for dip_to in DIPS:
        for sp in SPACING:
            fc = make_dip_cap(t_grid, day=10.0, hour=4.0, dip_to=dip_to,
                              dip_len_h=2.0, spacing_h=sp)
            for sched in SCHEDS:
                for arm in ARMS:
                    go(f"C d={dip_to:.2f} sp={sp:.0f}h", sched, arm,
                       dict(exp='C', R_cap=2700.0, mid_h=1.0, dip_to=dip_to,
                            spacing_h=sp, G=0.0, sizing='mean', eps=0.0),
                       R_cap=2700.0, force_cap=fc)
            print()

    print("\nEXP E -- price-signal dips across fuel age")
    print(HDR)
    for R_cap in RCAPS:
        for G in GVALS:
            for sched in SCHEDS:
                for arm in ARMS:
                    go(f"E R={R_cap:.0f} G={G:.1f}", sched, arm,
                       dict(exp='E', R_cap=R_cap, mid_h=1.0, dip_to=None,
                            spacing_h=None, G=G, sizing='mean', eps=0.0),
                       R_cap=R_cap, price=price, cheap_thresh=thresh,
                       G_max_frac=G)
            print()


    print("\nEXP F  predictive eps-buffer sweep (EOC, worst cases)")
    print(HDR)
    for eps in EPSES:
        for G in GVALS:
            go(f"F price G={G:.1f} eps={eps:.0f}", "tiered", "predictive",
               dict(exp='F', R_cap=2700.0, mid_h=1.0, dip_to=None,
                    spacing_h=None, G=G, sizing='mean', eps=eps),
               R_cap=2700.0, price=price, cheap_thresh=thresh,
               G_max_frac=G, eps_buffer=eps)
        for dip_to in DIPS:
            fc = make_dip_cap(t_grid, day=10.0, hour=4.0, dip_to=dip_to,
                              dip_len_h=2.0, spacing_h=5.0)
            go(f"F dip={dip_to:.2f} eps={eps:.0f}", "tiered", "predictive",
               dict(exp='F', R_cap=2700.0, mid_h=1.0, dip_to=dip_to,
                    spacing_h=5.0, G=0.0, sizing='mean', eps=eps),
               R_cap=2700.0, force_cap=fc, eps_buffer=eps)
        print()

    print("\nEXP G -- consequence model at EOC (R_cap=2700): trips + "
          "xenon dead-time priced in (tiered, mid_hz=1h)")
    print(HDR)
    for G in GVALS:
        for arm in ARMS:
            eps = 20.0 if arm == "predictive" else 0.0
            go(f"G price G={G:.1f}", "tiered", arm,
               dict(exp='G', R_cap=2700.0, mid_h=1.0, dip_to=None,
                    spacing_h=None, G=G, sizing='mean', eps=eps),
               R_cap=2700.0, price=price, cheap_thresh=thresh,
               G_max_frac=G, trip=True, eps_buffer=eps)
        print()

    for G in GVALS:
        for arm in ARMS:
            go(f"G ctrl R=3500 G={G:.1f}", "tiered", arm,
               dict(exp='G', R_cap=3500.0, mid_h=1.0, dip_to=None,
                    spacing_h=None, G=G, sizing='mean', eps=0.0),
               R_cap=3500.0, price=price, cheap_thresh=thresh,
               G_max_frac=G, trip=True)
    print()


    if args.frontier:
        print("\nEXP H -- feasibility frontier: dip depth x fuel age "
              "(free arm, tiered, 2h dips x2, sp=5h)")
        print(HDR)
        DEPTHS_H = (0.85, 0.78, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40)
        for R_cap in RCAPS:
            for dip_to in DEPTHS_H:
                fc = make_dip_cap(t_grid, day=10.0, hour=4.0, dip_to=dip_to,
                                  dip_len_h=2.0, spacing_h=5.0)
                go(f"H R={R_cap:.0f} d={dip_to:.2f}", "tiered", "free",
                   dict(exp='H', R_cap=R_cap, mid_h=1.0, dip_to=dip_to,
                        spacing_h=5.0, G=0.0, sizing='mean', eps=0.0),
                   R_cap=R_cap, force_cap=fc)
            print()

    if args.hysteresis:
        print("\nEXP HY -- fixed-delay hysteresis baseline (Reviewer 4 "
              "comparison; not Slurm): after a real down-ramp, up-ramps "
              "blocked for h hours. EXP G settings (tiered, price signal, "
              "trips on, eps=20 for predictive); free/reactive/predictive "
              "rerun here under identical settings for comparison.")
        print(HDR)
        for R_cap in RCAPS:
            for G in GVALS:
                for arm, h in [(a, None) for a in ARMS] + \
                              [('hysteresis', h) for h in HYST_H]:
                    eps = 20.0 if arm == "predictive" else 0.0
                    hk = {} if h is None else dict(hysteresis_h=h)
                    go(f"HY R={R_cap:.0f} G={G:.1f}"
                       + ("" if h is None else f" h={h:.0f}"),
                       "tiered", arm,
                       dict(exp='HY', R_cap=R_cap, mid_h=1.0, dip_to=None,
                            spacing_h=None, G=G, sizing='mean', eps=eps,
                            hyst_h=h),
                       R_cap=R_cap, price=price, cheap_thresh=thresh,
                       G_max_frac=G, trip=True, eps_buffer=eps, **hk)
                print()
        hyst_table('HY')

    flush_csv()
    el = time.time() - t0
    print(f"\nWrote {out_path} ({len(CSV_ROWS)} rows, "
          f"{len(pd.DataFrame(CSV_ROWS).columns)} columns) in "
          f"{el/60:.1f} min.")
    if TRIP_ROWS:
        print(f"Wrote {trip_path} ({len(TRIP_ROWS)} SCRAM events, exact "
              f"(I,X) at each trip) -- run validate_deadtime.py next.")
    if bad[0]:
        print(f"WARNING: {bad[0]} row(s) FAILED CONSERVATION -- fix before "
              f"trusting ANY number above.")
    else:
        print("All rows passed conservation on the FULL unzeroed run.")

    df = pd.DataFrame(CSV_ROWS)
    try:
        base = df[(df.exp == 'A') & (df.R_cap == 2700.0) &
                  (df.sched == 'tiered') & (df.rx == 'free') &
                  (df.mid_h == 1.0)]
        if len(base) and 'water_grid_L' in df:
            b = float(base.iloc[0]['water_L'])
            print("\nWater relocation vs the EXP A control "
                  "(no dips, no imports):")
            print(f"  {'case':>16} | {'SMR kL':>8} | {'grid kL':>8} | "
                  f"{'total kL':>8} | {'apparent':>8} | {'real':>7} | "
                  f"{'reloc %':>7}")
            for G in GVALS:
                row = df[(df.exp == 'E') & (df.R_cap == 2700.0) &
                         (df.sched == 'tiered') & (df.rx == 'free') &
                         (df.G == G)]
                if not len(row): continue
                row = row.iloc[0]
                smr = float(row['water_L']); grid = row['water_grid_L']
                if pd.isna(grid): continue
                grid = float(grid)
                app = (b - smr)/1e3
                real = (b - smr - grid)/1e3
                pct = 100.0*(1 - real/app) if app > 0 else float('nan')
                print(f"  {'G=%.1f' % G:>16} | {smr/1e3:8.1f} | "
                      f"{grid/1e3:8.1f} | {(smr+grid)/1e3:8.1f} | "
                      f"{app:8.1f} | {real:7.1f} | {pct:6.1f}%")
            print("  (apparent = on-site saving only; real = after counting "
                  "water embedded in imported grid power)")
    except Exception as e:
        print(f"[summary] skipped: {e}")


    try:
        g = df[(df.exp == 'G')]
        if len(g):
            print("\nEXP G intensity per unit of IT work SERVED "
                  "(the fair basis in the consequence regime):")
            print(f"  {'row':>22} | {'rx':>4} | {'served MWh':>10} | "
                  f"{'L/kWh served':>12} | {'grid $/MWh served':>17}")
            for _, row in g.iterrows():
                print(f"  {row['experiment']:>22} | "
                      f"{RXLABEL[row['rx']]:>4} | "
                      f"{row['served_it_Wh']/1e6:10.1f} | "
                      f"{row['water_L_per_kWh_served']:12.3f} | "
                      f"{row['grid_cost_per_MWh_served']:17.2f}")
    except Exception as e:
        print(f"[G summary] skipped: {e}")

if __name__ == "__main__":
    main()