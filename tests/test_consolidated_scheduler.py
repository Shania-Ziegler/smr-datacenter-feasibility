#!/usr/bin/env python3
"""
test_consolidated_scheduler.py -- tests for the opt-in workload-aware
controller and for the untouched legacy path.

Run from data/:   python -m pytest -q test_consolidated_scheduler.py

Fast tests use short (2-3 day) slices of the real Borg Cell A trace and the
monthly PJM price file. The full-length legacy regression (the 321-row
--hysteresis grid, ~50 min) is opt-in:

    RUN_SLOW_REGRESSION=1 python -m pytest -q test_consolidated_scheduler.py \
        -k full_cli_regression

or run it yourself and compare with
    python test_consolidated_scheduler.py --compare-legacy NEW.csv NEW_TRIPS.csv

Frozen legacy reference: git commit 742203f (data/sim.py sha256
d8adc859...), the version that produced hysteresis_results.csv and
trip_log_hysteresis.csv.
"""
import dataclasses
import hashlib
import math
import os
import subprocess
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

FROZEN_COMMIT = "742203f"
FROZEN_SHA256 = ("d8adc859c5a250a0cd18a323ea49b0143d2bbd91a719aa3e1f6702724"
                 "ef02e2a")
TIMING_KEYS = ("timing_mode", "policy_decisions", "policy_total_us",
               "policy_mean_us", "policy_p95_us", "policy_p99_us",
               "policy_max_us", "projection_calls", "projection_total_us",
               "projection_mean_us", "projection_p95_us",
               "projection_p99_us", "projection_max_us")
# Declared strict tolerance for the projection-equivalence test ONLY
# (never used for legacy-result regression, which is exact).
PROJ_TOL_PCM = 1e-9
DT = 60.0


# ------------------------------------------------------------ fixtures --
@pytest.fixture(scope="module", autouse=True)
def _cwd():
    old = os.getcwd()
    os.chdir(HERE)
    yield
    os.chdir(old)


@pytest.fixture(scope="module")
def cell():
    _require("demand_curve_cell_a.csv", "pjm_dom_34885183.csv")
    t, u_at, fb_at, fm_at = S.load_borg(os.path.join(HERE,
                                                     "demand_curve_cell_a.csv"))
    tg_full = np.arange(t[0], t[-1], DT)
    Pd = np.array([S.P_FIXED + S.work_power(u_at(tt)) * S.PUE
                   for tt in tg_full])
    P_rated = Pd.mean() / 0.95
    t_grid = np.arange(t[0], t[0] + 3 * 86400.0, DT)     # 3-day slice
    price, thresh, _ = S.load_pjm(os.path.join(HERE,
                                               "pjm_dom_34885183.csv"),
                                  t_grid)
    return types.SimpleNamespace(u_at=u_at, fb_at=fb_at, fm_at=fm_at,
                                 t_grid=t_grid, P_rated=P_rated, price=price,
                                 thresh=thresh)


@pytest.fixture(scope="module")
def frozen():
    """The frozen legacy sim.py, loaded from git as its own module."""
    try:
        src = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:data/sim.py"],
                             cwd=HERE, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"frozen legacy sim.py not available from git: {e}")
    assert hashlib.sha256(src).hexdigest() == FROZEN_SHA256
    mod = types.ModuleType("sim_frozen")
    mod.__file__ = os.path.join(HERE, "sim_frozen.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


# short-grid stress configuration: same event definitions, placed inside
# the 3-day slice (warm-up / recovery bounds scaled to the slice)
SHORT_STRESS = CS.StressConfig(event_start_h=30.0, spike_duration_h=2.0,
                               blackout_duration_h=6.0, compound_offset_h=2.0,
                               min_pre_event_h=24.0, min_post_event_h=12.0)
BASE_KW = dict(G_max_frac=0.4, trip=True)


def _kw(cell, **extra):
    kw = dict(price=cell.price, cheap_thresh=cell.thresh, **BASE_KW)
    kw.update(extra)
    return kw


def _run(cell, arm, R=2700.0, u_at=None, h=None, **kw):
    kw = _kw(cell, **kw)
    if arm in CS.CONTROLLER_MODES:
        if arm == "consolidated":
            kw.setdefault("eps_buffer", 20.0)
        return S.run(cell.fb_at, cell.fm_at, "tiered", cell.t_grid,
                     u_at or cell.u_at, cell.P_rated, DT, R,
                     controller_mode=arm, **kw)
    if arm == "predictive":
        kw.setdefault("eps_buffer", 20.0)
    if h is not None:
        kw["hysteresis_h"] = h
    return S.run(cell.fb_at, cell.fm_at, "tiered", cell.t_grid,
                 u_at or cell.u_at, cell.P_rated, DT, R, rx_mode=arm, **kw)


def _eq(x, y):
    """Structural equality with NaN == NaN (floats compared bitwise)."""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and \
            all(_eq(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and \
            all(_eq(a, b) for a, b in zip(x, y))
    if isinstance(x, float) and isinstance(y, float) and \
            math.isnan(x) and math.isnan(y):
        return True
    return bool(x == y)


def _same(a, b, skip=TIMING_KEYS):
    """Exact equality of run() outputs (floats bitwise, NaN == NaN)."""
    assert set(a) - set(skip) == set(b) - set(skip), \
        set(a) ^ set(b)
    for k in a:
        if k in skip:
            continue
        x, y = a[k], b[k]
        if k == "series":
            assert set(x) == set(y)
            for s in x:
                assert np.array_equal(x[s], y[s], equal_nan=True), s
        elif isinstance(x, (list, tuple)):
            assert _eq(x, y), k
        elif isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


ALL_ARMS = (("free", None), ("computing_only", None), ("reactive", None),
            ("hysteresis", 2.0), ("hysteresis", 4.0), ("hysteresis", 6.0),
            ("predictive", None), ("consolidated", None))


# ================================================================ tests ==
# 1. legacy outputs bit-identical to the frozen implementation
LEGACY_CASES = [
    dict(sched="tiered", rx_mode="free"),
    dict(sched="blind", rx_mode="reactive"),
    dict(sched="single", rx_mode="predictive", eps_buffer=20.0),
    dict(sched="tiered", rx_mode="hysteresis", hysteresis_h=2.0),
    dict(sched="tiered", rx_mode="predictive", price=True, G_max_frac=0.4,
         trip=True, eps_buffer=20.0, return_series=True),
    dict(sched="tiered", rx_mode="free", price=True, G_max_frac=0.2,
         trip=True, timing="detailed"),
    dict(sched="tiered", rx_mode="reactive", force_cap=True, R=8000.0),
    dict(sched="blind", rx_mode="predictive", grid_avail=True, price=True,
         G_max_frac=0.4, R="array", mid_horizon_s=1800.0),
    dict(sched="tiered", rx_mode="hysteresis", hysteresis_h=6.0, price=True,
         G_max_frac=0.4, trip=True, grid_override=True),
]


@pytest.mark.parametrize("case", LEGACY_CASES,
                         ids=[f"{c['sched']}-{c['rx_mode']}-{i}"
                              for i, c in enumerate(LEGACY_CASES)])
def test_legacy_bit_identical_to_frozen(cell, frozen, case):
    c = dict(case)
    n = len(cell.t_grid)
    R = c.pop("R", 2700.0)
    if R == "array":
        R = np.linspace(3500.0, 2700.0, n)
    kw = {}
    if c.pop("price", False):
        kw.update(price=cell.price, cheap_thresh=cell.thresh)
    if c.pop("force_cap", False):
        kw["force_cap"] = S.make_dip_cap(cell.t_grid, day=1.0, hour=4.0,
                                         dip_to=0.5)
    if c.pop("grid_avail", False):
        a = np.ones(n); a[1500:1800] = 0.0; a[2500:2600] = 0.5
        kw["grid_availability"] = a
    if c.pop("grid_override", False):
        ov = np.zeros(n); ov[1000:1400] = 0.4
        kw["grid_override"] = ov
    sched = c.pop("sched")
    kw.update(c)
    args = (cell.fb_at, cell.fm_at, sched, cell.t_grid, cell.u_at,
            cell.P_rated, DT, R)
    _same(frozen.run(*args, **kw), S.run(*args, **kw))


# 2. legacy row count, ordering, names and settings unchanged
def _stub_run_factory(calls):
    def stub(fb_at, fm_at, mode, t_grid, u_at, P_rated, dt, R_cap, **kw):
        calls.append((mode, kw.get("rx_mode"), float(np.mean(R_cap)),
                      P_rated, tuple(sorted(
                          (k, v if np.isscalar(v) or v is None
                           else ("arr", float(np.nansum(v))))
                          for k, v in kw.items()))))
        return dict(residual=0.0, shed_in_settle=0.0, water_grid_L=None,
                    true_unserved=1.0, unserved_pct=0.0, crossings=0,
                    cross_episodes=0, min_headroom=1.0, refused_steps=0,
                    water_L=1.0, water_surplus_L=0.0, grid_cost=0.0,
                    ramp_distance=0.0, p_mean=0.5, n_trips=0,
                    max_exceed=0.0, deadtime_h=0.0, grid_Wh=0.0,
                    served_it_Wh=1.0, water_L_per_kWh_served=0.0,
                    grid_cost_per_MWh_served=0.0, trip_log=[])
    return stub


@pytest.mark.parametrize("argv", [[], ["--hysteresis"],
                                  ["--frontier", "--quick"]])
def test_legacy_row_enumeration_unchanged(frozen, tmp_path, argv,
                                          monkeypatch):
    results = []
    for mod in (frozen, S):
        calls = []
        monkeypatch.setattr(mod, "run", _stub_run_factory(calls))
        mod.CSV_ROWS.clear(); mod.TRIP_ROWS.clear()
        out = tmp_path / f"{mod.__name__}.csv"
        monkeypatch.setattr(sys, "argv", ["sim.py", *argv, "--out",
                                          str(out)])
        mod.main()
        rows = [dict(r) for r in mod.CSV_ROWS]
        results.append((calls, rows, pd.read_csv(out)))
        mod.CSV_ROWS.clear()
    (c0, r0, d0), (c1, r1, d1) = results
    assert c0 == c1                       # same run() calls, same order
    assert r0 == r1                       # same rows and fields
    assert list(d0.columns) == list(d1.columns)
    pd.testing.assert_frame_equal(d0, d1)
    assert not (tmp_path / "trip_log.csv").exists()


# 3. computing_only runs without reactor state; projection never invoked
def test_computing_only_without_reactor_state(cell, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("projection invoked by computing_only")
    monkeypatch.setattr(S, "project_peak_rho", boom)
    monkeypatch.setattr(S, "project_peak_rho_path", boom)
    seen = []
    real_step = CS.WorkloadController.step

    def spy(self, obs):
        seen.append(obs.reactor)
        return real_step(self, obs)
    monkeypatch.setattr(CS.WorkloadController, "step", spy)
    r = _run(cell, "computing_only", R=2700.0)
    assert r["projections"] == 0 and seen and all(x is None for x in seen)
    fields = {f.name for f in dataclasses.fields(CS.Observation)}
    assert not fields & {"I", "X", "rho", "R_cap", "headroom", "ceiling"}
    with pytest.raises(ValueError):
        S.run(cell.fb_at, cell.fm_at, "tiered", cell.t_grid, cell.u_at,
              cell.P_rated, DT, 2700.0, controller_mode="computing_only",
              project_path=boom)
    ctl = CS.WorkloadController("computing_only", DT, 1e6, 1440, 60)
    with pytest.raises(RuntimeError):
        ctl._project(0.0, 0.0, [0.5], 300.0)
    obs = _obs(reactor=CS.ReactorObs(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        ctl.step(obs)


# 4. no controller can read future Borg demand or workload fractions
class FutureGuard:
    def __init__(self, t_grid):
        self.t_grid, self.now = t_grid, None

    def hook(self, k):
        self.now = None if k is None else self.t_grid[k]

    def wrap(self, f):
        def g(tt):
            if self.now is not None and tt > self.now:
                raise AssertionError(f"future sample requested: {tt} > "
                                     f"{self.now}")
            return f(tt)
        return g


@pytest.mark.parametrize("arm", CS.CONTROLLER_MODES)
def test_no_future_access(cell, arm):
    g = FutureGuard(cell.t_grid)
    ev = CS.build_event_inputs(CS.build_scenarios(SHORT_STRESS, ["spike"])[0],
                               len(cell.t_grid), DT, SHORT_STRESS)
    u = CS.wrap_u_at(g.wrap(cell.u_at), ev, cell.t_grid[0])
    kw = _kw(cell, eps_buffer=20.0)
    S.run(g.wrap(cell.fb_at), g.wrap(cell.fm_at), "tiered", cell.t_grid, u,
          cell.P_rated, DT, 2700.0, controller_mode=arm, step_hook=g.hook,
          **kw)
    g.hook(10)                                   # the guard itself works
    with pytest.raises(AssertionError):
        g.wrap(cell.u_at)(cell.t_grid[11])


# 5. constant path == constant hold within the declared tolerance
@pytest.mark.parametrize("p0,p1,hz", [(1.0, 0.3, 36000.0), (0.9, 0.55, 36000.0),
                                      (0.6, 0.95, 18000.0), (0.8, 1e-3, 36000.0),
                                      (0.95, 0.7, 7200.0)])
def test_constant_path_matches_constant_hold(p0, p1, hz):
    I, X = S.xe_equilibrium(p0)
    a = S.project_peak_rho(I, X, p1, hz)
    b = S.project_peak_rho_path(I, X, [p1] * int(hz / 300.0), 300.0)
    assert abs(a - b) <= PROJ_TOL_PCM
    assert a == b                              # observed: exactly equal
    # a genuinely time-varying path differs (the path is really used)
    c = S.project_peak_rho_path(I, X, [p1] * 12 + [1.0] * (int(hz / 300) - 12),
                                300.0)
    assert c <= b


def test_xe_advance_matches_legacy_integration(cell):
    r = S.run(cell.fb_at, cell.fm_at, "tiered", cell.t_grid, cell.u_at,
              cell.P_rated, DT, 2700.0, rx_mode="free", return_series=True,
              **_kw(cell))
    s = r["series"]
    I, X = s["iodine"][0], s["xenon"][0]
    for k in range(1, len(cell.t_grid)):
        I, X = S.xe_advance(I, X, s["P_rx"][k], DT)
        assert I == s["iodine"][k] and X == s["xenon"][k]


# 6. xenon_guard parity and distinct responsibilities
def test_xenon_guard_parity():
    import xenon_guard as XG
    XG.test_consistency()                      # constants identical
    for p0, p1, hz in [(1.0, 0.3, 36000.0), (0.9, 0.55, 36000.0),
                       (0.5, 0.0, 18000.0)]:
        st = XG.equilibrium_state(p0)
        assert (st.I, st.X) == S.xe_equilibrium(p0)
        assert XG.project_peak_rho(st, p1, hz) == \
            S.project_peak_rho(st.I, st.X, p1, hz)
        # same ceiling definition as sim's predictive gate
        R, eps = 2700.0, 20.0
        safe, pk = XG.is_dip_safe(st, p1, R, eps_buffer=eps, horizon_s=hz)
        assert safe == (S.project_peak_rho(st.I, st.X, max(p1, 1e-3), hz)
                        <= R - S.sigma_m - eps)
    # Documented split: xenon_guard is a dependency-free constant-hold API
    # (its deadtime_if_tripped uses permanent clearance, sim uses first
    # clearance); it has no time-varying path, so the controller uses
    # sim.project_peak_rho_path through a callback.
    assert not hasattr(XG, "project_peak_rho_path")


# 7. grid_availability=None == all-ones
@pytest.mark.parametrize("arm,h", [("free", None), ("predictive", None),
                                   ("hysteresis", 4.0),
                                   ("computing_only", None),
                                   ("consolidated", None)])
def test_grid_none_equals_ones(cell, arm, h):
    a = _run(cell, arm, h=h, return_series=True)
    b = _run(cell, arm, h=h, return_series=True,
             grid_availability=np.ones(len(cell.t_grid)))
    _same(a, b)


def _scenario_run(cell, name, arm, h=None, R=2700.0, **kw):
    scn = CS.build_scenarios(SHORT_STRESS, [name])[0]
    ev = CS.build_event_inputs(scn, len(cell.t_grid), DT, SHORT_STRESS)
    u = CS.wrap_u_at(cell.u_at, ev, cell.t_grid[0])
    extra = dict(grid_availability=ev.grid_availability, return_series=True)
    if arm in CS.CONTROLLER_MODES:
        extra.update(known_outages=ev.known_outages,
                     deadline_shock=ev.deadline_shock,
                     event_meta=scn.metadata())
    extra.update(kw)
    return scn, ev, _run(cell, arm, R=R, u_at=u, h=h, **extra)


# 8. a surprise blackout gives exactly zero grid use after onset
@pytest.mark.parametrize("R", [3500.0, 2700.0])
@pytest.mark.parametrize("arm,h", ALL_ARMS)
def test_surprise_blackout_zero_grid(cell, arm, h, R):
    scn, ev, r = _scenario_run(cell, "surprise_blackout", arm, h, R=R)
    k0, k1 = ev.blackout_window
    assert ev.known_outages == ()
    assert np.all(r["series"]["grid"][k0:k1] == 0.0)
    if R == 3500.0:       # non-vacuous: grid was in use before onset
        assert r["series"]["grid"][:k0].sum() > 0.0     # (at EOC the
        # consolidated arm legitimately cancels every price import)


# 9. scheduled outage is labelled differently from a surprise blackout
def test_scheduled_vs_surprise_labels():
    sur, sch = CS.build_scenarios(SHORT_STRESS, ["surprise_blackout",
                                                 "scheduled_outage"])
    ms, mc = sur.metadata(), sch.metadata()
    assert ms["blackout_type"] == "surprise" and not \
        ms["blackout_known_in_advance"]
    assert mc["blackout_type"] == "scheduled" and \
        mc["blackout_known_in_advance"]
    es = CS.build_event_inputs(sur, 4320, DT, SHORT_STRESS)
    ec = CS.build_event_inputs(sch, 4320, DT, SHORT_STRESS)
    assert es.known_outages == () and ec.known_outages == (ec.blackout_window,)
    assert es.grid_availability == ec.grid_availability
    t = dict(prefix="ST", cell="a", R_cap=2700.0, G=0.4, h=None,
             suite="primary", synthetic=False)
    assert S._controller_tag({**t, "scenario": sur.name}) != \
        S._controller_tag({**t, "scenario": sch.name})


# 10. synthetic stress cases conserve energy
@pytest.mark.parametrize("name", CS.SCENARIOS)
@pytest.mark.parametrize("arm,h", ALL_ARMS)
def test_conservation_stress(cell, name, arm, h):
    _, _, r = _scenario_run(cell, name, arm, h)
    assert abs(r["residual"]) <= 1.0
    if arm in CS.CONTROLLER_MODES:
        assert r["energy_residual_Wh"] <= 1.0


# 11. synthetic-deadline provenance in every applicable output
def _short_ctx(cell):
    return {"a": dict(path="demand_curve_cell_a.csv", t_grid=cell.t_grid,
                      u_at=cell.u_at, fb_at=cell.fb_at, fm_at=cell.fm_at,
                      P_rated=cell.P_rated, price=cell.price,
                      thresh=cell.thresh, ewif=None, chars={})}


PROV = ("synthetic_deadline", "deadline_source", "deadline_shock_fraction",
        "deadline_shock_minutes", "deadline_shock_start_h")


def test_synthetic_deadline_provenance(cell, tmp_path, capsys):
    scen = {s.name: s for s in CS.build_scenarios(
        SHORT_STRESS, ["none", "deadline_shock", "compound_blackout_deadline"])}
    S._CTX.clear()
    S._CTX.update(cells=_short_ctx(cell), scenarios=scen,
                  stress_cfg=SHORT_STRESS, timing="off",
                  decision_log="minute")
    tasks = S._controller_tasks(["a"], list(scen), list(CS.CONTROLLER_MODES),
                                scen, 0.4, "ST", False)
    rows, eps, mins = [], [], []
    for t in tasks:
        if t["R_cap"] != 2700.0:
            continue
        _, row, _, ep, mn = S._controller_task(t)
        rows.append(row); eps.extend(ep); mins.extend(mn)
    for row in rows:
        for f in PROV:
            assert f in row
        syn = row["scenario"] != "none"
        assert row["synthetic_deadline"] == syn
        assert row["deadline_source"] == (CS.SRC_SYNTHETIC if syn
                                          else CS.SRC_NATURAL)
        assert ("[synthetic_deadline]" in row["experiment"]) == syn
        if syn:
            assert row["deadline_shock_fraction"] == 0.5
            assert row["deadline_shock_minutes"] == 60.0
    for e in eps:
        assert "first_synthetic_deadline" in e and \
            "first_deadline_source" in e
    for m in mins:
        assert "synthetic_deadline" in m and "deadline_source" in m
    df = pd.DataFrame(rows)
    S._controller_comparisons(df, str(tmp_path / "cmp.csv"))
    cmp_ = pd.read_csv(tmp_path / "cmp.csv")
    assert {"synthetic_deadline", "deadline_source"} <= set(cmp_.columns)
    # endogenous deadline-shock rows are diagnostics, never primary
    assert set(df[df.synthetic_deadline].evidence_class) == \
        {S.DIAGNOSTIC_DEADLINE}
    assert set(cmp_.scenario) == {"none"}
    S._controller_print(df, list(CS.CONTROLLER_MODES))
    out = capsys.readouterr().out
    assert CS.SYNTHETIC_DEADLINE_NOTE in out
    assert "scenario=deadline_shock" not in out


def test_deadline_shock_mechanics():
    c = CS.WorkloadController("computing_only", DT, 2e6, 1440, 60)
    obs = _obs(reactor_hi=0.0, reactor_cap=0.0, fb=0.5, fm=0.2,
               grid_connected=False)
    for k in range(1, 6):
        c.step(dataclasses.replace(obs, k=k))
    queued_b = sum(it.amount for it in c.queue if it.tier == CS.TIER_BATCH)
    promoted = c.apply_deadline_shock(6, 0.5, 3)
    assert promoted == pytest.approx(0.5 * queued_b)
    syn = [it for it in c.queue if it.synthetic_deadline]
    assert syn and all(it.deadline_source == CS.SRC_SYNTHETIC and
                       it.slack == 3 for it in syn)
    drops = [c.step(dataclasses.replace(obs, k=k)).dropped_synthetic_W
             for k in range(6, 12)]
    assert sum(drops) == pytest.approx(promoted)


# 12. rigid shortfall is explicit, never hidden in flexible deferral
def _obs(**kw):
    base = dict(k=1, t_s=60.0, P_work_W=1.0e6, fb=0.3, fm=0.1,
                requested_W=2.2e6, base_facility_W=1.0e6, it_cost=1.036,
                P_rated_W=2.0e6, reactor_prev=0.8, reactor_hi=0.85,
                reactor_lo=0.75, reactor_cap=1.0, hi_binding="ramp",
                ramp_step=0.05, grid_cap_W=0.0, grid_avail=None,
                grid_connected=True, price_cheap=False, tripped=False,
                emergency_grid=True, reactor=None)
    base.update(kw)
    return CS.Observation(**base)


def test_rigid_shortfall_explicit(cell):
    c = CS.WorkloadController("computing_only", DT, 2e6, 1440, 60)
    d = c.step(_obs(reactor_hi=0.6, reactor_lo=0.0, tripped=False,
                    grid_connected=False))
    P_b, P_m, P_r = CS.split_work(1.0e6, 0.3, 0.1)
    cap_it = (0.6 * 2e6 - 1.0e6) / 1.036
    assert d.rigid_unserved_W == pytest.approx(P_r - cap_it)
    assert d.deferred_b_W == pytest.approx(P_b)
    assert d.deferred_m_W == pytest.approx(P_m)
    assert CS.R_RIGID_SHORTFALL in d.reasons
    assert all(it.tier != CS.TIER_RIGID for it in c.queue)
    # plant level: a forced outage produces booked rigid shortfall
    fc = S.make_dip_cap(cell.t_grid, day=1.0, hour=4.0, dip_to=0.0,
                        dip_len_h=1.0, n_dips=1)
    r = S.run(cell.fb_at, cell.fm_at, "tiered", cell.t_grid, cell.u_at,
              cell.P_rated, DT, 8000.0, controller_mode="computing_only",
              force_cap=fc)
    assert r["rigid_unserved_Wh"] > 0.0
    assert r["rigid_shortfall_reason"] == CS.RIGID_SHORTFALL_CAUSE
    assert r["rigid_shortfall_steps"] > 0
    # deferral holds only flexible tiers (shed is settle-trimmed)
    assert r["shed"] <= r["deferred_b_Wh"] + r["deferred_m_Wh"] + 1e-6
    assert r["unmet"] == pytest.approx(r["rigid_unserved_Wh"] +
                                       r["facility_unserved_Wh"])


# 13. consolidated never deliberately accepts a projected-unsafe path
@pytest.mark.parametrize("name", ["none", "surprise_blackout",
                                  "compound_blackout_spike"])
def test_consolidated_accepts_only_feasible(cell, name):
    _, _, r = _scenario_run(cell, name, "consolidated")
    s = r["series"]
    pk, ce, forced = (s["projected_peak_acc"], s["controller_ceiling"],
                      s["no_feasible_action"])
    m = np.isfinite(pk)
    assert m.sum() > 0
    ok = (pk[m] <= ce[m]) | forced[m]
    assert ok.all()
    assert (pk[m][forced[m]] > ce[m][forced[m]]).all()


def test_consolidated_bisection_and_forced_flag():
    calls = []

    def proj_monotone(I, X, path, pdt):        # higher floor -> lower peak
        calls.append(min(path))
        return 3000.0 - 2000.0 * min(path)
    c = CS.WorkloadController("consolidated", DT, 2e6, 1440, 60,
                              project_path=proj_monotone)
    o = _obs(reactor=CS.ReactorObs(1.0, 1.0, 2300.0, 2000.0),
             reactor_hi=0.9, reactor_lo=0.0, price_cheap=True,
             grid_cap_W=1.5e6, hi_binding="cap")
    d = c.step(o)
    assert CS.R_PROJECTED_CEILING in d.reasons
    assert d.price_import_canceled and d.proj_peak_acc <= 2000.0
    assert not d.no_feasible_action and d.p_floor == d.p_floor

    c2 = CS.WorkloadController("consolidated", DT, 2e6, 1440, 60,
                               project_path=lambda *a: 9999.0)
    d2 = c2.step(o)
    # no reachable action is projected safe: recorded as such, and the
    # best-effort maximum power is NOT labelled an accepted safe floor
    assert d2.no_feasible_action and CS.R_PROJECTED_CEILING in d2.reasons
    assert math.isnan(d2.p_floor)
    assert d2.proj_peak_acc > d2.ceiling_pcm


# isolation: with a never-binding feasibility callback, consolidated must
# take exactly computing_only's workload, queue, grid and reactor actions
@pytest.mark.parametrize("name", ["none", "spike", "surprise_blackout",
                                  "scheduled_outage",
                                  "compound_blackout_deadline"])
@pytest.mark.parametrize("R", [8000.0, 2700.0])
def test_isolation_nonbinding_callback(cell, name, R):
    calls = []

    def always_safe(I, X, path, pdt):
        calls.append(1)
        return -math.inf
    _, _, a = _scenario_run(cell, name, "computing_only", R=R)
    _, _, b = _scenario_run(cell, name, "consolidated", R=R,
                            project_path=always_safe)
    assert calls and b["projections"] == len(calls)
    for k in ("P_rx", "Pd", "grid", "state", "spike", "rigid_unserved",
              "facility_unserved", "cooling", "deferred", "drained",
              "rho", "tripped"):
        assert np.array_equal(a["series"][k], b["series"][k]), k
    for k in ("true_unserved", "dropped", "leftover", "shed", "drained",
              "grid_Wh", "grid_cost", "deferred_b_Wh", "deferred_m_Wh",
              "rigid_unserved_Wh", "minutes_STRESSED", "n_trips",
              "crossings", "deadline_promoted_Wh", "surplus_Wh"):
        assert a[k] == b[k], k
    assert b["refused_steps"] == 0 and b["no_feasible_action_steps"] == 0


# cooling follows non-deferred IT work (P_work - deferred + drained) in
# the controller arms, the same convention as run()
@pytest.mark.parametrize("arm", CS.CONTROLLER_MODES)
def test_cooling_convention_matches_legacy(cell, arm):
    fc = S.make_dip_cap(cell.t_grid, day=1.0, hour=4.0, dip_to=0.0,
                        dip_len_h=1.0, n_dips=1)       # forces rigid shortfall
    r = _run(cell, arm, R=2700.0, force_cap=fc, return_series=True)
    s = r["series"]
    a = 1.0 - np.exp(-DT / 300.0)
    Pc = (S.PUE - 1.0) * S.work_power(cell.u_at(cell.t_grid[0]))
    for k in range(1, len(cell.t_grid)):
        it = s["offered_it"][k] - s["deferred"][k] + s["drained"][k]
        Pc += a * ((S.PUE - 1.0) * it - Pc)
        assert s["cooling"][k] == pytest.approx(Pc, rel=1e-9, abs=1e-6)
    ru = s["rigid_unserved"]
    assert ru.max() > 0
    # cooling owed to unserved rigid work is booked as unserved facility
    assert np.all(s["facility_unserved"] >=
                  a * (S.PUE - 1.0) * ru - 1e-6)


def test_manifest_and_evidence(tmp_path):
    p = tmp_path / "m.json"
    S._write_manifest(str(p), dict(a=1))
    assert not os.access(p, os.W_OK)
    with pytest.raises(FileExistsError):
        S._write_manifest(str(p), dict(a=2))
    sc = {x.name: x for x in CS.build_scenarios(CS.StressConfig())}
    assert S._evidence_class("primary", sc["none"]) == S.PRIMARY_EVIDENCE
    assert S._evidence_class("primary", sc["surprise_blackout"]) == \
        S.PRIMARY_EVIDENCE
    for n in ("deadline_shock", "compound_blackout_deadline"):
        assert S._evidence_class("primary", sc[n]) == S.DIAGNOSTIC_DEADLINE


# 14. every modified/refused path carries an allowed reason code
@pytest.mark.parametrize("arm", CS.CONTROLLER_MODES)
def test_reason_codes_allowed(cell, arm):
    seen = set()
    for name in CS.SCENARIOS:
        _, _, r = _scenario_run(cell, name, arm, decision_log="minute")
        for row in r["decision_rows"]:
            codes = row["reason_code"].split(";")
            assert codes and set(codes) <= set(CS.REASON_CODES)
            seen.update(codes)
        for e in r["decision_episodes"]:
            assert set(e["reason_code"].split(";")) <= set(CS.REASON_CODES)
        assert sum(r[f"reason_{c}_steps"] for c in CS.REASON_CODES) >= \
            len(r["decision_rows"])
    assert seen


# 15. timing instrumentation does not change scientific results
@pytest.mark.parametrize("arm", ["predictive", "computing_only",
                                 "consolidated"])
def test_timing_invariance(cell, arm):
    a = _run(cell, arm, return_series=True)
    b = _run(cell, arm, return_series=True, timing="detailed")
    _same(a, b)


# 16. stress placement identical across fixed R_cap snapshots
def test_stress_placement_identical_across_rcaps(cell):
    scen = {s.name: s for s in CS.build_scenarios(SHORT_STRESS,
                                                  ["compound_blackout_spike"])}
    S._CTX.clear()
    S._CTX.update(cells=_short_ctx(cell), scenarios=scen,
                  stress_cfg=SHORT_STRESS, timing="off", decision_log=None)
    rows = {}
    for t in S._controller_tasks(["a"], list(scen), ["computing_only"], scen,
                                 0.4, "ST", False):
        if t["arm"] == "free":
            rows[t["R_cap"]] = S._controller_task(t)[1]
    keys = ("event_start_h", "event_duration_h", "event_magnitude",
            "spike_multiplier", "spike_start_h", "spike_duration_h",
            "blackout_type", "blackout_start_h", "blackout_duration_h",
            "demand_Wh", "grid_unavailable_min")
    ref = rows[8000.0]
    for R, row in rows.items():
        assert row["R_cap_at_event_pcm"] == R
        for k in keys:
            assert row[k] == ref[k], k
    assert len({r_["pre_event_headroom_pcm"] for r_ in rows.values()}) == 3


# 17. sensitivity grid: every pair for every cell; binding term recorded
def test_sensitivity_grid_complete(cell):
    scen = {s.name: s for s in CS.build_scenarios(CS.StressConfig())}
    tasks = S._controller_tasks(list("abcdefgh"), list(scen),
                                list(CS.CONTROLLER_MODES), scen, 0.4, "ST",
                                True)
    got = {(t["cell"], t["k_sigma"], t["floor"], t["scenario"], t["arm"],
            t["R_cap"]) for t in tasks if t["arm"] in CS.CONTROLLER_MODES
           and t["scenario"] in ("none", "spike")}
    for c in "abcdefgh":
        for k, f in CS.sensitivity_grid():
            for s in ("none", "spike"):
                for a in CS.CONTROLLER_MODES:
                    for R in S.CONTROLLER_RCAPS:
                        assert (c, k, f, s, a, R) in got
    assert len(CS.sensitivity_grid()) == 9
    ids = [(t["experiment"] if "experiment" in t else S._controller_tag(t),
            t["arm"]) for t in tasks]
    assert len(ids) == len(set(ids))            # unique result keys
    # binding term recorded, and in-run detections == detector-only pass
    req = [S.P_FIXED + S.PUE * S.work_power(cell.u_at(t))
           for t in cell.t_grid]
    for k, f in CS.sensitivity_grid():
        cfg = CS.DetectorConfig(k_sigma=k, rated_floor_fraction=f)
        spk, bind, _ = CS.detector_pass(req, DT, cell.P_rated, cfg)
        assert all(b in ("sigma", "rated_floor") for s_, b in
                   zip(spk, bind) if s_)
    cfg = CS.ControllerConfig(detector=CS.DetectorConfig(k_sigma=2.0,
                                                         rated_floor_fraction=0.025))
    r = _run(cell, "computing_only", controller_cfg=cfg, return_series=True)
    spk, _, _ = CS.detector_pass(req, DT, cell.P_rated, cfg.detector)
    assert list(r["series"]["spike"]) == spk


def test_spike_detector_is_causal():
    d = CS.SpikeDetector(CS.DetectorConfig(), DT, 1e6)
    for _ in range(60):
        assert d.observe(1e6)[0] is False
    spike, binding, excess, thr = d.observe(1.2e6)
    assert spike and binding == "rated_floor" and excess == pytest.approx(0.2e6)
    assert thr == pytest.approx(0.05e6)
    assert d.mean > 1e6                      # folded in only afterwards
    assert d.alpha == pytest.approx(1 - math.exp(-60.0 / 3600.0))


def test_lsf_tie_break_deterministic():
    mk = lambda tier, arr, seq, dl: CS.QueueItem(1.0, 0, dl, tier, False,
                                                 CS.SRC_NATURAL, arr, seq)
    items = [mk(CS.TIER_BATCH, 5, 4, 10), mk(CS.TIER_MID, 5, 3, 10),
             mk(CS.TIER_MID, 3, 2, 12), mk(CS.TIER_BATCH, 1, 1, 30)]
    order = sorted(items, key=CS.lsf_key)
    # due steps: mid(5)=15, mid(3)=15, batch(5)=15, batch(1)=31 -> equal
    # slack: mid before batch, then earlier arrival first
    assert [(i.tier, i.arrival_step) for i in order] == [
        (CS.TIER_MID, 3), (CS.TIER_MID, 5), (CS.TIER_BATCH, 5),
        (CS.TIER_BATCH, 1)]
    # slack order == static key order at any step
    for it in items:
        it.age_steps = 7 - it.arrival_step
    assert sorted(items, key=lambda i: (i.slack, CS._TIER_RANK[i.tier],
                                        i.arrival_step, i.seq)) == order


def test_fraction_validation():
    CS.validate_fractions(0.0, 0.0); CS.validate_fractions(0.6, 0.4)
    for fb, fm in ((-1e-12, 0.1), (0.1, -0.1), (0.7, 0.30001)):
        with pytest.raises(ValueError):
            CS.validate_fractions(fb, fm)


# 18. repeated runs are deterministic
@pytest.mark.parametrize("arm", CS.CONTROLLER_MODES)
def test_determinism(cell, arm):
    a = _scenario_run(cell, "compound_blackout_spike", arm,
                      decision_log="minute")[2]
    b = _scenario_run(cell, "compound_blackout_spike", arm,
                      decision_log="minute")[2]
    _same(a, b)


# 19. continuous-year state test still calls run once, no segment reset
def test_self_test_passes(capsys):
    S.self_test()
    out = capsys.readouterr().out
    assert "continuous year orchestration calls to run(): 1" in out
    assert "segment-boundary state resets detected: 0" in out
    assert "SELF-TEST: PASS" in out


# 20. conservation and validate_deadtime checks still pass
def test_validate_deadtime_on_controller_trips(cell, tmp_path):
    import validate_deadtime as V
    rows = []
    for arm in ("free", "computing_only"):
        r = _run(cell, arm, R=2700.0)
        assert abs(r["residual"]) <= 1.0
        for d in r["trip_log"]:
            rows.append(dict(experiment=f"T {arm}", rx=arm, R_cap=2700.0,
                             start_h=d["start_h"], end_h=d["end_h"],
                             dur_h=d["end_h"] - d["start_h"],
                             I0=repr(d["I0"]), X0=repr(d["X0"]),
                             rho0=d["rho0"], censored=d["censored"]))
    assert rows, "expected SCRAMs at EOC with G=0.4 price dips"
    p = tmp_path / "trip_log.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert V.check_logged_trips(str(p))


# ------------------------------------------ full CLI legacy regression --
def compare_legacy(new_csv, new_trips,
                   base_csv=os.path.join(HERE, "hysteresis_results.csv"),
                   base_trips=os.path.join(HERE, "trip_log_hysteresis.csv")):
    """Every non-timing CSV field must be string-identical to the frozen
    baseline (same float repr => bit-identical); the trip log must be
    byte-identical. Returns a list of problems (empty == clean)."""
    a = pd.read_csv(base_csv, dtype=str, keep_default_na=False)
    b = pd.read_csv(new_csv, dtype=str, keep_default_na=False)
    probs = []
    if list(a.columns) != list(b.columns):
        probs.append(f"columns differ: {set(a.columns) ^ set(b.columns)}")
    if len(a) != len(b):
        probs.append(f"row count {len(b)} != {len(a)}")
    cols = [c for c in a.columns if c in b.columns and c not in TIMING_KEYS]
    for c in cols:
        bad = (a[c].values[:len(b)] != b[c].values[:len(a)])
        if bad.any():
            i = int(np.argmax(bad))
            probs.append(f"{c}: {int(bad.sum())} cell(s) differ, first row "
                         f"{i} ({a.experiment[i]}): {a[c][i]!r} vs "
                         f"{b[c][i]!r}")
    with open(base_trips, "rb") as f1, open(new_trips, "rb") as f2:
        if f1.read() != f2.read():
            probs.append("trip log is not byte-identical")
    return probs


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_REGRESSION"),
                    reason="~50 min; set RUN_SLOW_REGRESSION=1")
def test_full_cli_regression(tmp_path):
    _require("demand_curve_cell_a.csv", "pjm_dom_34885183.csv",
             "hysteresis_results.csv", "trip_log_hysteresis.csv")
    for f in ("demand_curve_cell_a.csv", "pjm_dom_34885183.csv"):
        os.symlink(os.path.join(HERE, f), tmp_path / f)
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "sim.py"),
                    "--hysteresis", "--timing-mode", "coarse", "--out",
                    "hysteresis_results.csv"], cwd=tmp_path, check=True,
                   env={**os.environ, "PYTHONPATH": SRC_DIR})
    probs = compare_legacy(tmp_path / "hysteresis_results.csv",
                           tmp_path / "trip_log.csv")
    assert not probs, "\n".join(probs)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare-legacy":
        p = compare_legacy(sys.argv[2], sys.argv[3])
        print("\n".join(p) if p else
              "LEGACY REGRESSION CLEAN: all non-timing fields identical, "
              "trip log byte-identical.")
        sys.exit(1 if p else 0)
    sys.exit(pytest.main([__file__, "-q"]))
