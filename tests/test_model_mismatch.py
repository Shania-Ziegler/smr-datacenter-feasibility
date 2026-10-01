#!/usr/bin/env python3
"""
test_model_mismatch.py -- regression tests for the opt-in plant/controller
xenon-model mismatch (sim.XeParams, plant_xe / model_xe).

Option A: the controller observes the TRUE current I/X; plant_xe drives only
the simulated reactor (ODE, equilibrium, reactivity -> trip rule, metrics);
model_xe drives only the controller's projection. None = canonical code.
Run from data/:   python3 -m pytest -q test_model_mismatch.py
"""
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
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
CANON_CODE = os.path.join(HERE, "reference_code", "sim.py")
CANON_SIM_SHA = ("297b9a50d7a14ec7d87a5a4ace90f025bf6efb85487526b65756b9ded"
                 "5716273")
TIMING_KEYS = ("timing_mode", "policy_decisions", "policy_total_us",
               "policy_mean_us", "policy_p95_us", "policy_p99_us",
               "policy_max_us", "projection_calls", "projection_total_us",
               "projection_mean_us", "projection_p95_us",
               "projection_p99_us", "projection_max_us")
ALL_ARMS = (("free", None), ("computing_only", None), ("reactive", None),
            ("hysteresis", 2.0), ("hysteresis", 4.0), ("hysteresis", 6.0),
            ("predictive", None), ("consolidated", None))
PROJ_ARMS = ("predictive", "consolidated")
N = S.NOMINAL_XE
LN2 = math.log(2.0)
CASES = {
    "DOE_halflives": dataclasses.replace(
        N, lambda_I=LN2 / (6.57 * 3600.0), lambda_X=LN2 / (9.10 * 3600.0)),
    "lambda_I_x0.95": dataclasses.replace(N, lambda_I=N.lambda_I * 0.95),
    "lambda_I_x1.05": dataclasses.replace(N, lambda_I=N.lambda_I * 1.05),
    "lambda_X_x0.95": dataclasses.replace(N, lambda_X=N.lambda_X * 0.95),
    "lambda_X_x1.05": dataclasses.replace(N, lambda_X=N.lambda_X * 1.05),
    "phi_x0.9": dataclasses.replace(N, phi_full=N.phi_full * 0.9),
    "phi_x1.1": dataclasses.replace(N, phi_full=N.phi_full * 1.1),
}


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


@pytest.fixture(scope="module")
def canon():
    """A frozen copy of the sim.py that produced the reference run, if
    the user keeps one at CANON_CODE (not distributed)."""
    if not os.path.exists(CANON_CODE):
        pytest.skip("frozen reference sim.py not present: " + CANON_CODE)
    with open(CANON_CODE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == CANON_SIM_SHA
    spec = importlib.util.spec_from_file_location("sim_canonical_copy",
                                                  CANON_CODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(c, arm, h=None, mod=S, R=2700.0, **kw):
    base = dict(price=c.price, cheap_thresh=c.thresh, G_max_frac=0.4,
                trip=True, return_series=True,
                eps_buffer=20.0 if arm in PROJ_ARMS else 0.0)
    base.update(kw)
    if arm in CS.CONTROLLER_MODES:
        return mod.run(c.fb_at, c.fm_at, "tiered", c.t_grid, c.u_at,
                       c.P_rated, DT, R, controller_mode=arm, **base)
    if h is not None:
        base["hysteresis_h"] = h
    return mod.run(c.fb_at, c.fm_at, "tiered", c.t_grid, c.u_at, c.P_rated,
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


# ============================== 1. mismatch = 0 is bit-identical ========
@pytest.mark.parametrize("arm,h", ALL_ARMS)
def test_default_bit_identical_to_canonical_code(cell, canon, arm, h):
    """plant_xe = model_xe = None executes exactly the canonical code."""
    _same(_run(cell, arm, h), _run(cell, arm, h, mod=canon))


@pytest.mark.parametrize("arm,h", ALL_ARMS)
def test_explicit_nominal_equals_none(cell, arm, h):
    kw = dict(plant_xe=N)
    if arm in PROJ_ARMS:
        kw["model_xe"] = N
    _same(_run(cell, arm, h), _run(cell, arm, h, **kw))


def test_nominal_is_exactly_the_canonical_constants(canon):
    for f in dataclasses.fields(N):
        assert getattr(N, f.name) == getattr(S, f.name) == \
            getattr(canon, f.name), f.name


def test_parameterized_functions_reproduce_canonical_arithmetic():
    I, X = S.xe_equilibrium(0.83)
    assert S._xe_equilibrium_p(0.83, N) == (I, X)
    assert S._xe_pcm_p(X, N) == S.xe_pcm(X)
    assert S._xe_advance_p(I, X, 0.6, DT, N) == S.xe_advance(I, X, 0.6, DT)
    for p, hz in ((0.3, 10 * 3600.0), (0.0, 36000.0), (0.9, 0.0)):
        assert S._project_peak_rho_p(I, X, p, hz, N) == \
            S.project_peak_rho(I, X, p, hz)
    path = list(np.linspace(0.9, 0.2, 120))
    assert S._project_peak_rho_path_p(I, X, path, 300.0, N) == \
        S.project_peak_rho_path(I, X, path, 300.0)


# ======================== 2./4. controller stays nominal (separation) ===
@pytest.mark.parametrize("arm", PROJ_ARMS)
def test_mismatched_projection_is_nominal_and_sees_true_state(cell, arm,
                                                              monkeypatch):
    """Plant perturbed, model_xe=None: every projection is the canonical
    nominal function, evaluated from the TRUE plant I/X."""
    theta = CASES["phi_x1.1"]
    seen = []
    name = "project_peak_rho" if arm == "predictive" else \
        "project_peak_rho_path"
    real = getattr(S, name)

    def spy(I_, X_, *a):
        v = real(I_, X_, *a)
        seen.append((I_, X_, v))
        return v

    def boom(*_a, **_k):
        raise AssertionError("parameterized projection used under nominal "
                             "controller model")
    monkeypatch.setattr(S, name, spy)
    monkeypatch.setattr(S, "_project_peak_rho_p", boom)
    monkeypatch.setattr(S, "_project_peak_rho_path_p", boom)
    consts = {f.name: getattr(S, f.name) for f in dataclasses.fields(N)}
    rho_eq0 = S.rho_eq
    r = _run(cell, arm, plant_xe=theta)
    assert seen, "no projection happened in the fixture"
    ser = r["series"]
    true_states = set(zip(ser["iodine"].tolist(), ser["xenon"].tolist()))
    assert all((I_, X_) in true_states for I_, X_, _ in seen)
    # controller constants untouched by the plant mismatch
    assert {f.name: getattr(S, f.name) for f in dataclasses.fields(N)} == \
        consts and S.NOMINAL_XE == N and S.rho_eq == rho_eq0


@pytest.mark.parametrize("arm", PROJ_ARMS)
def test_matched_projection_uses_perturbed_model(cell, arm, monkeypatch):
    theta = CASES["lambda_X_x1.05"]
    seen = []
    name = "_project_peak_rho_p" if arm == "predictive" else \
        "_project_peak_rho_path_p"
    real = getattr(S, name)

    def spy(*a):
        seen.append(a[-1])
        return real(*a)

    def boom(*_a, **_k):
        raise AssertionError("nominal projection used under matched model")
    monkeypatch.setattr(S, name, spy)
    monkeypatch.setattr(S, "project_peak_rho", boom)
    monkeypatch.setattr(S, "project_peak_rho_path", boom)
    _run(cell, arm, plant_xe=theta, model_xe=theta)
    assert seen and all(q == theta for q in seen)


def test_projection_from_state_is_nominal_for_every_case():
    I, X = S.xe_equilibrium(1.0)
    nom = S.project_peak_rho(I, X, 0.3, 36000.0)
    for name, q in CASES.items():
        assert S._project_peak_rho_p(I, X, 0.3, 36000.0, q) != nom, name


# ========================== 3. true plant trajectory actually changes ===
@pytest.mark.parametrize("arm", ("free",) + PROJ_ARMS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_true_plant_changes_and_trip_rule_uses_it(cell, arm, case):
    theta = CASES[case]
    r0 = _run(cell, arm)
    r1 = _run(cell, arm, plant_xe=theta)
    s0, s1 = r0["series"], r1["series"]
    assert not np.array_equal(s0["xenon"], s1["xenon"])
    p0 = s1["P_rx"][0]
    assert (s1["iodine"][0], s1["xenon"][0]) == S._xe_equilibrium_p(p0, theta)
    assert np.array_equal(s1["rho"], np.array([S._xe_pcm_p(x, theta)
                                                for x in s1["xenon"]]))
    # headroom / trip rule evaluate the TRUE plant reactivity
    assert np.array_equal(s1["headroom"], 2700.0 - S.sigma_m - s1["rho"])
    hd, tr = s1["headroom"], s1["tripped"]
    for k in range(1, len(hd)):
        if hd[k - 1] < 0.0 and not tr[k - 1]:
            assert tr[k], k


# ================================ matched vs mismatched are distinct ====
@pytest.mark.parametrize("arm", PROJ_ARMS)
def test_matched_and_mismatched_differ(cell, arm):
    theta = CASES["phi_x1.1"]
    mm = _run(cell, arm, plant_xe=theta)
    ma = _run(cell, arm, plant_xe=theta, model_xe=theta)
    assert (not np.array_equal(mm["series"]["P_rx"], ma["series"]["P_rx"])
            or mm["refused_steps"] != ma["refused_steps"])


# ============================== Sigma_f / conversion mismatch rejected ==
@pytest.mark.parametrize("field", S.XE_CONVERSION_FIELDS)
@pytest.mark.parametrize("arm", ("free", "predictive", "consolidated"))
def test_conversion_parameter_mismatch_rejected(cell, field, arm):
    bad = dataclasses.replace(N, **{field: getattr(N, field) * 1.1})
    with pytest.raises(ValueError, match="state-reading artifact"):
        _run(cell, arm, plant_xe=bad)
    if arm in PROJ_ARMS:
        with pytest.raises(ValueError, match="state-reading artifact"):
            _run(cell, arm, plant_xe=N, model_xe=bad)


def test_sigma_f_cancels_analytically():
    q = dataclasses.replace(N, Sigma_f=N.Sigma_f * 1.37)
    for p in (0.2, 0.7, 1.0):
        a = S._xe_pcm_p(S._xe_equilibrium_p(p, q)[1], q)
        b = S.xe_pcm(S.xe_equilibrium(p)[1])
        assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("arm,h", [("free", None), ("reactive", None),
                                   ("hysteresis", 4.0),
                                   ("computing_only", None)])
def test_model_xe_rejected_for_arms_without_projection(cell, arm, h):
    with pytest.raises(ValueError, match="model_xe"):
        _run(cell, arm, h, model_xe=CASES["phi_x1.1"])


# ======================================== 5. conservation under mismatch
@pytest.mark.parametrize("arm", PROJ_ARMS)
@pytest.mark.parametrize("cond", ("mismatched", "matched"))
@pytest.mark.parametrize("case", sorted(CASES))
def test_mismatch_conservation(cell, arm, cond, case):
    theta = CASES[case]
    r = _run(cell, arm, plant_xe=theta,
             model_xe=theta if cond == "matched" else None)
    assert abs(r["residual"]) <= 1.0
    if arm == "consolidated":
        assert r["energy_residual_Wh"] <= 1.0


# ================================ 6. canonical outputs hash-identical ===
@pytest.mark.parametrize("stem", [
    "controller_runs/main_suite",
    "controller_runs/horizon_sensitivity",
    "controller_runs/eps_sensitivity"])
def test_protected_outputs_unchanged(stem):
    _require(stem + "_manifest.json")
    man = json.load(open(stem + "_manifest.json"))
    for k, v in man["outputs"].items():
        with open(v["path"], "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == v["sha256"], \
                (stem, k)
