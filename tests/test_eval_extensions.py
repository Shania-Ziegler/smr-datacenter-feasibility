#!/usr/bin/env python3
"""
test_eval_extensions.py -- tests for the opt-in evaluation extensions:
  * one-factor sweeps: --predict-horizon-h / --eps-buffer-pcm
  * --year with --computing-only / --consolidated (run_continuous)
  * canonical-output protection
The existing controller/legacy tests live in test_consolidated_scheduler.py
and are not modified. Run from data/:
    python3 -m pytest -q test_eval_extensions.py
Fast tests use a 3-day slice of Borg Cell A and a small in-memory
two-segment year fixture (TEST ONLY; not data, not an execution mode).
"""
import math
import os
import subprocess
import sys
import types

import numpy as np
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
TIMING_KEYS = ("timing_mode", "policy_decisions", "policy_total_us",
               "policy_mean_us", "policy_p95_us", "policy_p99_us",
               "policy_max_us", "projection_calls", "projection_total_us",
               "projection_mean_us", "projection_p95_us",
               "projection_p99_us", "projection_max_us")
# decision series that must not move when an unused parameter moves
DECISION_SERIES = ("P_rx", "Pd", "grid", "deferred", "drained",
                   "rigid_unserved", "facility_unserved", "rho", "tripped")


@pytest.fixture(scope="module", autouse=True)
def _cwd():
    old = os.getcwd()
    os.chdir(HERE)
    yield
    os.chdir(old)


@pytest.fixture(scope="module")
def cell():
    _require("demand_curve_cell_a.csv", "pjm_dom_34885183.csv")
    t, u_at, fb_at, fm_at = S.load_borg("demand_curve_cell_a.csv")
    tg_full = np.arange(t[0], t[-1], DT)
    Pd = np.array([S.P_FIXED + S.work_power(u_at(tt)) * S.PUE
                   for tt in tg_full])
    t_grid = np.arange(t[0], t[0] + 3 * 86400.0, DT)     # 3-day slice
    price, thresh, _ = S.load_pjm("pjm_dom_34885183.csv", t_grid)
    return types.SimpleNamespace(u_at=u_at, fb_at=fb_at, fm_at=fm_at,
                                 t_grid=t_grid, P_rated=Pd.mean() / 0.95,
                                 price=price, thresh=thresh)


def _run(c, arm, R=2700.0, h=None, **kw):
    base = dict(price=c.price, cheap_thresh=c.thresh, G_max_frac=0.4,
                trip=True, return_series=True)
    base.update(kw)
    if arm in CS.CONTROLLER_MODES:
        return S.run(c.fb_at, c.fm_at, "tiered", c.t_grid, c.u_at,
                     c.P_rated, DT, R, controller_mode=arm, **base)
    if h is not None:
        base["hysteresis_h"] = h
    return S.run(c.fb_at, c.fm_at, "tiered", c.t_grid, c.u_at, c.P_rated,
                 DT, R, rx_mode=arm, **base)


def _same(a, b, skip=TIMING_KEYS):
    assert set(a) - set(skip) == set(b) - set(skip), set(a) ^ set(b)
    for k in a:
        if k in skip:
            continue
        x, y = a[k], b[k]
        if k == "series":
            assert set(x) == set(y)
            for s in x:
                assert np.array_equal(x[s], y[s], equal_nan=True), s
        elif isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


def _same_decisions(a, b):
    for s in DECISION_SERIES:
        if s in a["series"]:
            assert np.array_equal(a["series"][s], b["series"][s],
                                  equal_nan=True), s
    for k in ("true_unserved", "dropped", "unmet", "n_trips", "grid_Wh",
              "grid_cost", "crossings", "served_it_Wh", "residual"):
        assert a[k] == b[k], (k, a[k], b[k])


# ============================================ 1. sweeps: default unchanged ==
@pytest.mark.parametrize("arm", ["predictive", "consolidated"])
def test_explicit_canonical_values_equal_default(cell, arm):
    """Passing the canonical horizon (10 h) and eps (20 pcm) explicitly is
    identical to the defaults the canonical suite used."""
    a = _run(cell, arm, eps_buffer=20.0)
    b = _run(cell, arm, eps_buffer=S.CANONICAL_EPS_PCM,
             predict_horizon_s=S.CANONICAL_HORIZON_H * 3600.0)
    _same(a, b)


def test_default_tasks_carry_no_sweep_keys():
    """Default suite tasks have no horizon/eps overrides, so
    _controller_task passes exactly the canonical arguments and emits the
    canonical column set and experiment tags."""
    scn = {s.name: s for s in CS.build_scenarios(CS.StressConfig(),
                                                 ["none", "spike"])}
    tasks = S._controller_tasks(list("ab"), ["none"],
                                ["computing_only", "consolidated"], scn, 0.4,
                                "ST", True)
    assert tasks and all("horizon_h" not in t and "eps" not in t
                         for t in tasks)
    t = next(t for t in tasks if t["suite"] == "primary")
    assert S._controller_tag(t) == "ST a R=8000 G=0.4 none"


# ===================================================== 2. horizon semantics ==
def test_horizon_zero_projection_is_current_state():
    I, X = S.xe_equilibrium(0.9)
    for p in (0.2, 0.6, 1.0):
        assert S.project_peak_rho(I, X, p, 0.0) == S.xe_pcm(X)


def test_predictive_horizon_zero_is_not_reactive(cell):
    """h=0 is a current-state ceiling check, documented as distinct from the
    reactive headroom gate -- it must not silently duplicate reactive."""
    p0 = _run(cell, "predictive", eps_buffer=20.0, predict_horizon_s=0.0)
    re = _run(cell, "reactive")
    assert not np.array_equal(p0["series"]["P_rx"], re["series"]["P_rx"])
    assert "margin_guard_steps" in re and "margin_guard_steps" not in p0


def test_consolidated_horizon_needs_two_segments(cell):
    with pytest.raises(ValueError, match="two path segments"):
        _run(cell, "consolidated", eps_buffer=20.0, predict_horizon_s=300.0)
    assert S._sweep_undefined("consolidated", "horizon", 0.0)
    assert S._sweep_undefined("consolidated", "horizon", 600 / 3600) is None
    assert S._sweep_undefined("predictive", "horizon", 0.0) is None
    assert S._sweep_undefined("consolidated", "eps", 0.0) is None


def test_horizon_does_not_change_computing_only_decisions(cell):
    """computing_only never projects reactor state; the horizon only enters
    its logged deadline-miss prediction, never a decision."""
    a = _run(cell, "computing_only", predict_horizon_s=2 * 3600.0)
    b = _run(cell, "computing_only", predict_horizon_s=10 * 3600.0)
    _same_decisions(a, b)


@pytest.mark.parametrize("h", [2.0, 6.0, 15.0])
def test_horizon_changes_predictive_projection(cell, h):
    a = _run(cell, "predictive", eps_buffer=20.0, predict_horizon_s=h * 3600)
    assert a["residual"] == pytest.approx(0.0, abs=1.0)


# ========================================================= 3. eps semantics ==
@pytest.mark.parametrize("arm,h", [("free", None), ("reactive", None),
                                   ("hysteresis", 4.0),
                                   ("computing_only", None)])
def test_eps_ignored_by_arms_that_do_not_consume_it(cell, arm, h):
    _same(_run(cell, arm, h=h, eps_buffer=0.0),
          _run(cell, arm, h=h, eps_buffer=50.0))


def test_eps_moves_only_the_enforced_ceiling(cell):
    """eps lowers only the controller's enforced ceiling; the physical
    ceiling R_cap - sigma_m (trip rule, crossing metric) is unchanged."""
    a = _run(cell, "consolidated", eps_buffer=0.0)
    b = _run(cell, "consolidated", eps_buffer=50.0)
    assert np.array_equal(a["series"]["ceiling"], b["series"]["ceiling"])
    ca, cb = a["series"]["controller_ceiling"], b["series"][
        "controller_ceiling"]
    both = np.isfinite(ca) & np.isfinite(cb)
    assert both.any() and np.allclose(ca[both] - cb[both], 50.0, rtol=0,
                                      atol=1e-9)


# ====================================== 4. sweep task list and evidence ==
def test_sweep_tasks_labels_and_skips():
    scn = {s.name: s for s in CS.build_scenarios(CS.StressConfig(),
                                                 ["none", "spike"])}
    sw = dict(kind="horizon", values=[0.0, 2.0, 10.0],
              arms=["predictive", "consolidated"], rcaps=[2700.0],
              undefined=[])
    t = S._controller_tasks(list("ab"), ["none"], ["consolidated"], scn,
                            0.4, "ST", False, sweep=sw)
    # 2 cells x (predictive 3 values + consolidated 2 defined values)
    assert len(t) == 2 * (3 + 2)
    assert not any(x["arm"] == "consolidated" and x["horizon_h"] == 0.0
                   for x in t)
    assert {x["suite"] for x in t} == {"horizon_sensitivity"}
    assert all(x["R_cap"] == 2700.0 and x["scenario"] == "none" for x in t)
    ev = S._evidence_class("horizon_sensitivity", scn["none"])
    assert ev == "sensitivity_prediction_horizon" != S.PRIMARY_EVIDENCE
    assert S._evidence_class("margin_sensitivity", scn["none"]) == \
        "sensitivity_eps_buffer"
    assert S._evidence_class("primary", scn["none"]) == S.PRIMARY_EVIDENCE
    assert S._controller_tag(t[0]).endswith(" H=0h")


def test_canonical_outputs_are_protected():
    for p in ("controller_runs/poster_final_20260929_020829.csv",
              "poster_final_20260929_020829_manifest.json",
              "x/poster_final_20260929_020829_trip_log.csv",
              "v6_results.csv"):
        assert S._is_canonical_output(p)
    assert not S._is_canonical_output("controller_runs/horizon_2700.csv")


def test_cli_rejects_invalid_sweeps():
    def cli(*a):
        return subprocess.run([sys.executable, os.path.join(SRC_DIR, "sim.py"), *a],
                              cwd=HERE,
                              capture_output=True, text=True, timeout=120)
    cases = [
        (("--consolidated", "--stress-suite", "--predict-horizon-h", "2",
          "--eps-buffer-pcm", "5"), "exactly one"),
        (("--consolidated", "--predict-horizon-h", "2"), "--stress-suite"),
        (("--consolidated", "--stress-suite", "--predict-horizon-h", "0.1"),
         "300 s"),
        (("--consolidated", "--stress-suite", "--eps-buffer-pcm", "-1"),
         ">= 0"),
        (("--consolidated", "--stress-suite", "--eps-buffer-pcm", "5",
          "--arms", "reactive"), "--arms"),
        (("--computing-only", "--consolidated", "--stress-suite",
          "--eps-buffer-pcm", "5"), "computing_only"),
        (("--consolidated", "--stress-suite", "--out",
          "controller_runs/poster_final_20260929_020829.csv"),
         "refusing to write canonical"),
        (("--consolidated", "--year", "--stress-suite"), "--year"),
    ]
    for argv, msg in cases:
        r = cli(*argv)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), \
            (argv, r.stdout[-300:], r.stderr[-300:])


# ============================================= 5. year-mode controller arms ==
@pytest.fixture(scope="module")
def year():
    """Two 30 h workload segments (a synthetic-year stand-in). A forced cap
    straddles the boundary so deferred work is queued when segment 1
    starts."""
    n0 = int(30 * 3600 / DT)
    t_grid = DT * np.arange(2 * n0)
    seg = np.repeat([0, 1], n0)                     # provenance label only
    u_arr = np.where(seg == 0, 0.85, 0.40)
    u_at = lambda tt: float(np.interp(tt, t_grid, u_arr))
    b = n0
    force = np.ones(len(t_grid))
    force[b - 120:b + 30] = 0.55
    P_rated = (S.P_FIXED + S.work_power(0.85) * S.PUE) / 0.95
    return types.SimpleNamespace(t_grid=t_grid, u_at=u_at, b=b, force=force,
                                 P_rated=P_rated, fb_at=lambda tt: 0.30,
                                 fm_at=lambda tt: 0.10)


def _yrun(y, arm, R=2700.0, fn=None, t_grid=None, **kw):
    tg = y.t_grid if t_grid is None else t_grid
    fc = y.force if t_grid is None else y.force[:len(tg)] \
        if tg[0] == y.t_grid[0] else y.force[-len(tg):]
    base = dict(controller_mode=arm, force_cap=fc, trip=True,
                eps_buffer=20.0 if arm == "consolidated" else 0.0,
                return_series=True)
    base.update(kw)
    return (fn or S.run_continuous)(y.fb_at, y.fm_at, "tiered", tg, y.u_at,
                                    y.P_rated, DT, R, **base)


class _Recorder:
    """Wraps CS.WorkloadController to observe (never alter) it."""
    instances = []

    def __init__(self, real):
        self.real = real

    def __call__(self, *a, **k):
        ctrl = self.real(*a, **k)
        ctrl._rec_project = k.get("project_path")
        ctrl._rec_queue = {}
        ctrl._rec_reactor = []
        orig = ctrl.step

        def step(obs):
            ctrl._rec_queue[obs.k] = [(it.seq, it.arrival_step, it.amount)
                                      for it in ctrl.queue]
            ctrl._rec_reactor.append(obs.reactor)
            return orig(obs)
        ctrl.step = step
        _Recorder.instances.append(ctrl)
        return ctrl


@pytest.fixture
def recorder(monkeypatch):
    _Recorder.instances = []
    monkeypatch.setattr(CS, "WorkloadController",
                        _Recorder(CS.WorkloadController))
    return _Recorder


@pytest.mark.parametrize("arm", ["computing_only", "consolidated"])
def test_year_controller_state_persists_across_segments(year, arm,
                                                        recorder,
                                                        monkeypatch):
    """ONE controller and ONE reactor initialisation for the whole path;
    run_continuous == direct run(); a deliberate split-and-reset differs."""
    inits = {"n": 0}
    real_eq = S.xe_equilibrium

    def c_eq(p):
        inits["n"] += 1
        return real_eq(p)
    monkeypatch.setattr(S, "xe_equilibrium", c_eq)
    r_year = _yrun(year, arm)
    assert len(recorder.instances) == 1 and inits["n"] == 1
    monkeypatch.setattr(S, "xe_equilibrium", real_eq)
    r_dir = _yrun(year, arm, fn=S.run)
    _same(r_year, r_dir)
    b = year.b
    r1 = _yrun(year, arm, t_grid=year.t_grid[b:])      # reset at boundary
    S_ = r_year["series"]
    assert np.max(np.abs(r1["series"]["rho"] - S_["rho"][b:])) > 1.0
    for k in ("iodine", "xenon", "P_rx"):
        dv = np.abs(np.diff(S_[k]))
        assert dv[b - 1] <= np.delete(dv, b - 1).max()


@pytest.mark.parametrize("arm", ["computing_only", "consolidated"])
def test_year_queue_persists_across_segments(year, arm, recorder):
    _yrun(year, arm)
    ctrl = recorder.instances[0]
    b = year.b
    q_at_b = ctrl._rec_queue[b]
    assert q_at_b, "fixture must leave deferred work queued at the boundary"
    assert all(arr < b for _, arr, _ in q_at_b)
    # the same items (by seq) are still owned by the controller after the
    # boundary, or were served/dropped by it -- never re-created
    seqs_b = {s for s, _, _ in q_at_b}
    later = {s for s, _, _ in ctrl._rec_queue[b + 1]}
    assert seqs_b & later or sum(a for _, _, a in q_at_b) > 0
    assert max(s for s, _, _ in ctrl._rec_queue[b + 1]) >= max(seqs_b)
    # a split-and-reset run starts segment 1 with an empty queue
    recorder.instances.clear()
    _yrun(year, arm, t_grid=year.t_grid[b:])
    assert recorder.instances[0]._rec_queue[1] == []


def test_year_computing_only_never_sees_reactor_information(year, recorder,
                                                            monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("computing_only reached a xenon projection")
    monkeypatch.setattr(S, "project_peak_rho", boom)
    monkeypatch.setattr(S, "project_peak_rho_path", boom)
    n = len(year.t_grid)
    r_flat = _yrun(year, "computing_only", R=8000.0, trip=False)
    ramp = np.linspace(8000.0, 2700.0, n)            # arbitrary test array
    r_ramp = _yrun(year, "computing_only", R=ramp, trip=False)
    for ctrl in recorder.instances:
        assert ctrl._rec_project is None
        assert ctrl._project is CS._forbidden_projection
        assert all(o is None for o in ctrl._rec_reactor)
    # R_cap (and therefore iodine/xenon headroom) cannot change a decision
    _same_decisions(r_flat, r_ramp)


def test_year_consolidated_uses_monthly_projection(year, recorder,
                                                   monkeypatch):
    calls = {"n": 0}
    real = S.project_peak_rho_path

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(S, "project_peak_rho_path", counted)
    r_year = _yrun(year, "consolidated")
    ctrl = recorder.instances[0]
    assert ctrl._project is counted and calls["n"] > 0
    assert all(o is not None for o in ctrl._rec_reactor)
    monkeypatch.setattr(S, "project_peak_rho_path", real)
    r_month = _yrun(year, "consolidated", fn=S.run)
    _same(r_year, r_month)


@pytest.mark.parametrize("arm", ["computing_only", "consolidated"])
def test_year_controller_conservation(year, arm):
    r = _yrun(year, arm)
    assert abs(r["residual"]) <= 1.0
    assert r["energy_residual_Wh"] <= 1.0


def test_year_cli_admits_controller_arms(tmp_path):
    """--year --consolidated is no longer rejected; without a cited R_cap
    trajectory nothing is simulated and nothing is written."""
    need = ("demand_curve_year_order1_peak.csv", "pjm_dom_34885183_2025.csv",
            "pjm_genmix_2025.csv")
    if not all(os.path.exists(p) for p in need):
        pytest.skip("year input files not present")
    out = tmp_path / "year_ctrl.csv"
    r = subprocess.run([sys.executable, os.path.join(SRC_DIR, "sim.py"),
                        "--year", "--consolidated",
                        "--computing-only", "--borg", need[0],
                        "--price-file", need[1], "--genmix-file", need[2],
                        "--out", str(out)], cwd=HERE, capture_output=True,
                       text=True, timeout=900)
    txt = r.stdout + r.stderr
    assert "controller arms cannot be combined" not in txt
    if r.returncode != 0:
        pytest.skip(f"year inputs rejected by the strict loaders: "
                    f"{txt.strip().splitlines()[-1]}")
    assert "no cited R_cap trajectory" in txt and not out.exists()
