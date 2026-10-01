#!/usr/bin/env python3
"""
xenon_guard.py -- causal feasibility guard for xenon-limited power moves.

A small, scheduler-independent interface to the same iodine/xenon physics
as sim.py. A caller that holds the current (I, X) state can ask, before
committing a power reduction:

    1. is_dip_safe(state, p_target)        -> is holding p_target projected
                                              to stay under the ceiling?
    2. feasible_floor_checked(state, p)    -> the shallowest projected-safe
                                              setpoint >= p, or an explicit
                                              "infeasible" if even p_max is
                                              projected unsafe
       feasible_floor(state, p)            -> legacy unchecked form (returns
                                              p_max even when p_max is unsafe)
    3. deadtime_if_tripped(state)          -> hours until a restart after a
                                              SCRAM from this state

Scope: it answers only the physics question; what to shed or when to buy
grid power stays with the caller. It is causal (current state in, answer
out) and conservative in the same disclosed way as sim.py's predictive arm:
power is held constant over the horizon (persistence of setpoint). It is
not a clairvoyant oracle and has not been evaluated inside any production
scheduler. In this project the new controller arms use
sim.project_peak_rho_path through a callback; this module is kept as the
standalone interface and as a parity check against sim.py.

Design rules:
  - Same cited constants as sim.py (Choudhury et al. 2025, Table V,
    lambda_I typo corrected), duplicated here so the module has no
    dependencies; test_consistency() and test_oracles.py check that they
    and the constant-hold projection match sim.py exactly.
  - Pure functions, no I/O, no state: (I, X) in, answer out.

Usage:
    from xenon_guard import equilibrium_state, is_dip_safe, \
        feasible_floor_checked, deadtime_if_tripped

    st = equilibrium_state(p=0.90)               # or live (I, X)
    ok, peak = is_dip_safe(st, p_target=0.55, R_cap=2700.0)
    if not ok:
        r = feasible_floor_checked(st, p_want=0.55, R_cap=2700.0)
        if not r.feasible:          # nothing reachable is projected safe
            ...
"""
from dataclasses import dataclass

# ---- cited constants: Choudhury et al. 2025 arXiv:2507.18150v2 Table V ----
# (lambda_I typo corrected). Duplicated from sim.py BY DESIGN; see
# test_consistency() below.
LAMBDA_I = 2.87e-5      # /s
LAMBDA_X = 2.09e-5      # /s
GAMMA_I  = 0.0639
GAMMA_X  = 0.00237
SIGMA_AX = 2.65e-18     # cm^2
NU       = 2.42
SIGMA_F  = 0.39497      # /cm
PHI_FULL = 5.0e13       # n/cm^2/s at 100% power
SIGMA_M  = 100.0        # pcm instrument margin (default; caller may override)


@dataclass(frozen=True)
class XenonState:
    """Iodine-135 and xenon-135 concentrations (atoms/cm^3)."""
    I: float
    X: float


def xe_pcm(X: float) -> float:
    """Xenon reactivity worth (pcm) of concentration X."""
    return SIGMA_AX * X / (NU * SIGMA_F) * 1e5


def equilibrium_state(p: float) -> XenonState:
    """Equilibrium (I, X) at steady power fraction p."""
    I = GAMMA_I * SIGMA_F * PHI_FULL * p / LAMBDA_I
    X = (GAMMA_X * SIGMA_F * PHI_FULL * p + LAMBDA_I * I) / \
        (LAMBDA_X + SIGMA_AX * PHI_FULL * p)
    return XenonState(I=I, X=X)


def _integrate(I, X, p, seconds, dt=150.0):
    """Advance (I, X) by `seconds` at constant power fraction p."""
    f = PHI_FULL * p
    steps = max(1, int(seconds / dt))
    h = seconds / steps
    for _ in range(steps):
        dI = GAMMA_I * SIGMA_F * f - LAMBDA_I * I
        dX = GAMMA_X * SIGMA_F * f + LAMBDA_I * I \
             - LAMBDA_X * X - SIGMA_AX * f * X
        I += dI * h
        X += dX * h
    return I, X


def project_peak_rho(state: XenonState, p_hold: float,
                     horizon_s: float = 10 * 3600.0,
                     dt_proj: float = 300.0) -> float:
    """
    Persistence-of-setpoint projection: hold power at p_hold from the given
    state and return the PEAK xenon reactivity (pcm) over horizon_s.
    Conservative (real recovery burns xenon off sooner) and causal (uses
    only current state). [DISCLOSED assumption, same as sim.py.]
    """
    I, X = state.I, state.X
    peak = xe_pcm(X)
    t = 0.0
    while t < horizon_s:
        I, X = _integrate(I, X, p_hold, dt_proj, dt=dt_proj / 2)
        t += dt_proj
        r = xe_pcm(X)
        if r > peak:
            peak = r
    return peak


def is_dip_safe(state: XenonState, p_target: float, R_cap: float,
                sigma_m: float = SIGMA_M, eps_buffer: float = 20.0,
                horizon_s: float = 10 * 3600.0):
    """
    Q1: may the reactor descend to p_target RIGHT NOW without the ensuing
    xenon transient breaching the ceiling?
    Returns (safe: bool, projected_peak_pcm: float).
    eps_buffer (pcm) is an enforcement margin below R_cap - sigma_m; its
    default absorbed the one-minute discretization grazes of the ceiling
    (183 in a one-month evaluation without it) so that no true crossing
    remained.
    """
    peak = project_peak_rho(state, max(p_target, 1e-3), horizon_s)
    return peak <= (R_cap - sigma_m - eps_buffer), peak


def feasible_floor(state: XenonState, p_want: float, R_cap: float,
                   p_max: float = 1.0, sigma_m: float = SIGMA_M,
                   eps_buffer: float = 20.0,
                   horizon_s: float = 10 * 3600.0, iters: int = 12) -> float:
    """
    Q2: the SHALLOWEST power fraction >= p_want that is xenon-safe from
    this state (== p_want itself when the requested dip is already safe).
    Same conservative bisection as the sim's predictive arm.
    LEGACY / UNCHECKED: if even p_max is unsafe this still returns p_max.
    Use feasible_floor_checked() to get an explicit infeasible result.
    """
    safe, _ = is_dip_safe(state, p_want, R_cap, sigma_m, eps_buffer,
                          horizon_s)
    if safe:
        return p_want
    lo, hi = p_want, p_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        safe, _ = is_dip_safe(state, mid, R_cap, sigma_m, eps_buffer,
                              horizon_s)
        if safe:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class FloorResult:
    """feasible=False means even p_max is projected above the ceiling; then
    floor is None (no safe setpoint exists) and peak_pcm is p_max's peak."""
    feasible: bool
    floor: object
    peak_pcm: float


def feasible_floor_checked(state: XenonState, p_want: float, R_cap: float,
                           p_max: float = 1.0, sigma_m: float = SIGMA_M,
                           eps_buffer: float = 20.0,
                           horizon_s: float = 10 * 3600.0,
                           iters: int = 12) -> FloorResult:
    """Checked form of feasible_floor: identical bisection, but first
    verifies that p_max itself is projected safe and reports infeasible
    instead of returning an unsafe p_max."""
    safe, pk = is_dip_safe(state, p_want, R_cap, sigma_m, eps_buffer,
                           horizon_s)
    if safe:
        return FloorResult(True, p_want, pk)
    safe_top, pk_top = is_dip_safe(state, p_max, R_cap, sigma_m, eps_buffer,
                                   horizon_s)
    if not safe_top:
        return FloorResult(False, None, pk_top)
    f = feasible_floor(state, p_want, R_cap, p_max, sigma_m, eps_buffer,
                       horizon_s, iters)
    ok, pk_f = is_dip_safe(state, f, R_cap, sigma_m, eps_buffer, horizon_s)
    return FloorResult(ok, f if ok else None, pk_f)


def deadtime_if_tripped(state: XenonState, R_cap: float,
                        sigma_m: float = SIGMA_M,
                        trip_margin: float = 50.0,
                        dt: float = 60.0, t_max_h: float = 96.0) -> float:
    """
    Q3: if the reactor SCRAMs from this state, how many hours until it is
    PERMANENTLY restartable -- i.e. past the post-SCRAM xenon peak, with
    headroom >= trip_margin from then on? This is the price of guessing
    wrong -- the number a scheduler should weigh against the value of a
    risky dip. Returns hours (inf if still blocked at t_max_h).

    NOTE on semantics vs sim.py: the sim restarts at FIRST clearance,
    which coincides with permanent clearance when the trip begins in
    violation (the only way sim trips happen: the transient has one peak,
    so first clearance lands on its decaying side). From a HEALTHY state,
    however, first clearance is trivially t=0 while the peak is still
    ahead -- the wrong answer for a planning oracle. Hence: last blocked
    instant, not first clear instant. Identical for real trips, honest
    for hypothetical ones.
    """
    I, X = state.I, state.X
    t = 0.0
    last_blocked = 0.0
    while t < t_max_h * 3600.0:
        if (R_cap - sigma_m - xe_pcm(X)) < trip_margin:
            last_blocked = t
        I, X = _integrate(I, X, 0.0, dt, dt=dt / 5)
        t += dt
    if (R_cap - sigma_m - xe_pcm(X)) < trip_margin:
        return float("inf")
    return last_blocked / 3600.0


def test_consistency():
    """Assert this module's constants match sim.py exactly (run whenever
    sim.py is importable; guards against silent drift)."""
    import sim as S
    pairs = [(LAMBDA_I, S.lambda_I), (LAMBDA_X, S.lambda_X),
             (GAMMA_I, S.gamma_I), (GAMMA_X, S.gamma_X),
             (SIGMA_AX, S.sigma_aX), (NU, S.nu), (SIGMA_F, S.Sigma_f),
             (PHI_FULL, S.phi_full), (SIGMA_M, S.sigma_m)]
    for a, b in pairs:
        assert a == b, f"constant drift vs sim.py: {a} != {b}"
    st = equilibrium_state(1.0)
    assert abs(xe_pcm(st.X) - S.rho_eq) < 1e-9, "rho_eq drift vs sim.py"
    print("xenon_guard constants CONSISTENT with sim.py "
          f"(rho_eq = {xe_pcm(st.X):.1f} pcm)")


if __name__ == "__main__":
    test_consistency()
    # demo: the EOC picture in three calls
    st = equilibrium_state(0.90)
    safe, peak = is_dip_safe(st, 0.55, R_cap=2700.0)
    floor = feasible_floor(st, 0.55, R_cap=2700.0)
    cost = deadtime_if_tripped(st, R_cap=2700.0)
    print(f"EOC (R_cap=2700), running at p=0.90:")
    print(f"  dip to 0.55 safe? {safe} (projected peak {peak:.0f} pcm)")
    print(f"  shallowest safe setpoint: {floor:.3f}")
    print(f"  dead-time if tripped anyway: {cost:.1f} h")