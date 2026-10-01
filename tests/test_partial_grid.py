#!/usr/bin/env python3
"""
test_partial_grid.py -- tests for the opt-in consolidated_partial_grid arm
(largest projected-feasible cheap-price import instead of all-or-nothing
cancellation) and for the default controllers being unchanged.

Run from data/:   python3 -m pytest -q test_partial_grid.py
"""
import math
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
HERE = os.path.join(ROOT, "data")      # working directory: inputs, outputs
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import consolidated_scheduler as CS  # noqa: E402


def _require(*names):
    """Skip unless every named file is present in data/: the external
    inputs (see data/README.md) or a user-generated reference run."""
    miss = [n for n in names if not os.path.exists(os.path.join(HERE, n))]
    if miss:
        pytest.skip("not present in data/: " + ", ".join(miss))
import sim as S  # noqa: E402

DT = 60.0
PARTIAL = CS.CONSOLIDATED_PARTIAL_GRID
P_RATED = 2.0e6
G_CAP = 0.8e6                    # 0.4 x rating


def _obs(ceiling, **kw):
    """NORMAL, cheap price, empty queue. need_all = 1e6 + 1.036 * 0.9e6 =
    1.9324e6 W <= reactor_hi * P_RATED, so the reactor alone can serve all
    work; the requested price import is min(need_all, G_CAP) = G_CAP. With
    import x the step-0 reactor request is (1.9324e6 - x) / P_RATED, and
    later path segments sit at (2.2e6 - x) / P_RATED (clipped to 1), so the
    path minimum is the step-0 request."""
    base = dict(k=1, t_s=60.0, P_work_W=0.9e6, fb=0.3, fm=0.1,
                requested_W=2.2e6, base_facility_W=1.0e6, it_cost=1.036,
                P_rated_W=P_RATED, reactor_prev=0.8, reactor_hi=1.0,
                reactor_lo=0.0, reactor_cap=1.0, hi_binding="cap",
                ramp_step=0.05, grid_cap_W=G_CAP, grid_avail=None,
                grid_connected=True, price_cheap=True, tripped=False,
                emergency_grid=True,
                reactor=CS.ReactorObs(1.0, 1.0, 2000.0, ceiling))
    base.update(kw)
    return CS.Observation(**base)


def proj_monotone(I, X, path, pdt):
    """Deeper reactor dip -> higher projected peak (monotone in import)."""
    return 3000.0 - 2000.0 * min(path)


def _ctrl(mode, proj=proj_monotone, **kw):
    return CS.WorkloadController(mode, DT, P_RATED, 1440, 60,
                                 project_path=proj, **kw)


# peak(x) = 3000 - 2000 * (1.9324e6 - x) / 2e6
#   x = 0 -> 1067.6 pcm ; x = G_CAP -> 1867.6 pcm
NEED = 1.0e6 + 1.036 * 0.9e6
PEAK0 = 3000.0 - 2000.0 * NEED / P_RATED
X_MAX_1500 = NEED - 0.75 * P_RATED             # largest x with peak <= 1500
RES = G_CAP / 2 ** CS.ControllerConfig().bisect_iters


# --------------------------------------------------- controller level ----
def test_mode_is_opt_in_only():
    assert CS.CONTROLLER_MODES == ("computing_only", "consolidated")
    assert PARTIAL not in CS.CONTROLLER_MODES
    assert PARTIAL in CS.ALL_CONTROLLER_MODES
    with pytest.raises(ValueError):
        _ctrl("consolidated", partial_audit_points=8)


def test_fully_feasible_request_accepts_full_import():
    for mode in ("consolidated", PARTIAL):
        d = _ctrl(mode).step(_obs(2000.0))
        assert CS.R_PROJECTED_CEILING not in d.reasons
        assert d.price_import_requested_W == pytest.approx(G_CAP)
        assert d.price_import_approved_W == pytest.approx(G_CAP)
        assert d.grid_price_W == pytest.approx(G_CAP)
        assert not d.price_import_partial and not d.price_import_canceled


def test_partially_feasible_request_accepts_largest_safe_import():
    d = _ctrl(PARTIAL).step(_obs(1500.0))
    assert CS.R_PROJECTED_CEILING in d.reasons
    assert d.price_import_partial and not d.price_import_canceled
    assert 0.0 < d.price_import_approved_W < d.price_import_requested_W
    # within one bisection resolution of the analytic maximum, never above
    assert X_MAX_1500 - RES <= d.price_import_approved_W <= X_MAX_1500
    # the accepted partial import's own projected path is under the ceiling
    assert d.proj_peak_acc <= 1500.0
    assert d.proj_peak_req > 1500.0
    # the canonical arm cancels the same request completely
    c = _ctrl("consolidated").step(_obs(1500.0))
    assert c.price_import_canceled and c.price_import_approved_W == 0.0
    assert c.grid_price_W == 0.0


def test_partial_import_displaces_only_its_own_reactor_output():
    d = _ctrl(PARTIAL).step(_obs(1500.0))
    served = 1.0e6 + 1.036 * d.it_served_W
    assert d.it_served_W == pytest.approx(0.9e6)          # all work served
    # supply = reactor + grid exactly covers the served demand (no surplus)
    assert d.p_target * P_RATED + d.grid_W == pytest.approx(served,
                                                            rel=1e-12)
    assert d.grid_W == pytest.approx(d.price_import_approved_W)
    c = _ctrl("consolidated").step(_obs(1500.0))
    assert c.it_served_W == pytest.approx(d.it_served_W)   # same service
    # reactor output drops by exactly the accepted import
    assert (c.p_target - d.p_target) * P_RATED == pytest.approx(
        d.grid_W - c.grid_W, rel=1e-9)


def test_no_feasible_import_gives_zero_import():
    # x = 0 is feasible, but even the smallest bisection step (~195 W ->
    # +0.195 pcm) is above the ceiling
    d = _ctrl(PARTIAL).step(_obs(PEAK0 + 0.05))
    assert d.price_import_approved_W == 0.0 and d.grid_price_W == 0.0
    assert d.price_import_canceled and not d.price_import_partial
    assert d.proj_peak_acc <= PEAK0 + 0.05
    # even zero import at maximum power infeasible: existing no-feasible
    # logic, and no price import is ever accepted
    d2 = _ctrl(PARTIAL).step(_obs(900.0))
    assert d2.price_import_approved_W == 0.0 and d2.grid_price_W == 0.0
    assert d2.no_feasible_action


def test_nonmonotone_projection_never_accepts_unsafe_import():
    """Feasible 'island' above an infeasible band: bisection may miss it
    (the audit flags that) but must never accept an unsafe import."""
    def proj_island(I, X, path, pdt):
        x = NEED - min(path) * P_RATED            # implied price import
        return 1000.0 if (x < 0.2e6 or 0.55e6 < x < 0.65e6) else 2000.0
    d = _ctrl(PARTIAL, proj=proj_island, partial_audit_points=8).step(
        _obs(1500.0))
    assert d.proj_peak_acc <= 1500.0
    assert d.partial_audit_missed and d.partial_audit_gap_W > 0.0
    d2 = _ctrl(PARTIAL, partial_audit_points=8).step(_obs(1500.0))
    assert not d2.partial_audit_missed               # monotone: none missed


def test_nonbinding_projection_partial_equals_consolidated():
    import dataclasses
    never = lambda *a: -math.inf
    for ceil in (1500.0, 2000.0):
        a = _ctrl("consolidated", proj=never).step(_obs(ceil))
        b = _ctrl(PARTIAL, proj=never).step(_obs(ceil))
        for f in dataclasses.fields(a):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            assert va == vb or (isinstance(va, float) and math.isnan(va)
                                and math.isnan(vb)), f.name


# -------------------------------------------------------- plant level ----
@pytest.fixture(scope="module")
def cell():
    _require("demand_curve_cell_a.csv", "pjm_dom_34885183.csv")
    t, u_at, fb_at, fm_at = S.load_borg(os.path.join(HERE,
                                                     "demand_curve_cell_a.csv"))
    tg_full = np.arange(t[0], t[-1], DT)
    Pd = np.array([S.P_FIXED + S.work_power(u_at(tt)) * S.PUE
                   for tt in tg_full])
    t_grid = np.arange(t[0], t[0] + 3 * 86400.0, DT)     # 3-day slice
    price, thresh, _ = S.load_pjm(os.path.join(HERE,
                                               "pjm_dom_34885183.csv"),
                                  t_grid)
    return types.SimpleNamespace(u_at=u_at, fb_at=fb_at, fm_at=fm_at,
                                 t_grid=t_grid, P_rated=Pd.mean() / 0.95,
                                 price=price, thresh=thresh)


def _run(cell, mode, **kw):
    return S.run(cell.fb_at, cell.fm_at, "tiered", cell.t_grid, cell.u_at,
                 cell.P_rated, DT, 2700.0, controller_mode=mode,
                 price=cell.price, cheap_thresh=cell.thresh,
                 G_max_frac=0.4, trip=True, eps_buffer=20.0,
                 return_series=True, **kw)


@pytest.fixture(scope="module")
def runs(cell):
    return {m: _run(cell, m, price_diagnostics=True,
                    **({"partial_grid_audit": 8} if m == PARTIAL else {}))
            for m in ("consolidated", PARTIAL)}


def test_plant_energy_conservation(runs):
    for r in runs.values():
        assert abs(r["energy_residual_Wh"]) <= 1.0
        assert abs(r["residual"]) <= 1.0


def test_plant_accepts_only_projected_feasible(runs):
    s = runs[PARTIAL]["series"]
    pk, ce, forced = (s["projected_peak_acc"], s["controller_ceiling"],
                      s["no_feasible_action"])
    m = np.isfinite(pk)
    assert m.sum() > 0 and ((pk[m] <= ce[m]) | forced[m]).all()


def test_plant_price_accounting(runs):
    for m, r in runs.items():
        assert r["price_approved_Wh"] <= r["price_requested_Wh"] + 1e-6
        assert r["grid_price_Wh"] <= r["price_approved_Wh"] + 1e-6
        assert r["price_opportunity_steps"] == (
            r["price_full_accept_steps"] + r["price_partial_steps"]
            + r["price_full_cancel_steps"])
    assert runs["consolidated"]["price_partial_steps"] == 0
    assert runs[PARTIAL]["price_approved_Wh"] >= \
        runs["consolidated"]["price_approved_Wh"]


def test_default_columns_unchanged(cell):
    r = _run(cell, "consolidated")
    assert not any(k.startswith(("price_requested", "price_approved",
                                 "price_partial", "partial_audit"))
                   for k in r)


def test_plant_deterministic(cell, runs):
    r = _run(cell, PARTIAL, price_diagnostics=True, partial_grid_audit=8)
    a, b = runs[PARTIAL], r
    for k in a:
        if k in ("series", "trip_log", "decision_episodes",
                 "decision_rows") or k.startswith(("policy_",
                                                   "projection_")):
            continue
        va, vb = a[k], b[k]
        assert (va == vb) or (isinstance(va, float) and math.isnan(va)
                              and math.isnan(vb)), k


# ------------------------------------------- default controller: canonical
def test_canonical_consolidated_row_bit_identical():
    """Cell A, consolidated, R_cap 2700, scenario none re-simulated with the
    current code equals the stored canonical row (timing excluded)."""
    _require("controller_runs/main_suite.csv", "demand_curve_cell_a.csv",
             "pjm_dom_34885183.csv", "load_pjm_genmix.csv")
    os.chdir(HERE)
    canon = pd.read_csv("controller_runs/main_suite.csv")
    row = canon[(canon.cell == "a") & (canon.rx == "consolidated") &
                (canon.R_cap == 2700.0) & (canon.scenario == "none") &
                (canon.suite == "primary")].iloc[0]
    scfg = CS.StressConfig()
    scen = {s.name: s for s in CS.build_scenarios(scfg, ["none", "spike"])}
    ctx = {"a": S._load_cell_ctx("a", "demand_curve_cell_a.csv",
                                 "pjm_dom_34885183.csv",
                                 "load_pjm_genmix.csv", 60.0)}
    S._CTX.update(cells=ctx, scenarios=scen, stress_cfg=scfg,
                  timing="detailed", decision_log="episodes")
    t = [t for t in S._controller_tasks(["a"], ["none"], ["consolidated"],
                                        scen, 0.4, "ST", False)
         if t["arm"] == "consolidated" and t["R_cap"] == 2700.0][0]
    _, new, *_ = S._controller_task(t)
    import io
    buf = io.StringIO()
    pd.DataFrame([new]).to_csv(buf, index=False)
    buf.seek(0)
    new = pd.read_csv(buf).iloc[0]
    assert set(new.index) <= set(canon.columns)      # no new columns
    for c in canon.columns:
        if c.startswith(("policy_", "projection_")):
            continue
        a = row[c]
        b = new[c] if c in new.index else float("nan")
        assert (pd.isna(a) and pd.isna(b)) or a == b, c
