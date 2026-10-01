#!/usr/bin/env python3
"""
consolidated_scheduler.py -- workload-conditioned runtime controller.

"A workload-conditioned runtime controller that coordinates queue slack,
deadline urgency, grid availability and xenon-dependent reactor
feasibility."

This module holds ONLY controller logic: workload splitting, the causal
forecast, least-slack-first deadline scheduling, the spike detector, the
NORMAL / STRESSED / ISLANDED state machine and deterministic stress-event
construction. It contains no reactor constants, no iodine/xenon equations,
no CSV loading and no plotting. The plant (sim.run_controller) owns the
physics and passes in, per step, an Observation built from values at the
CURRENT step only, plus -- for the 'consolidated' arm only -- a projection
callback project_path(I0, X0, p_path, path_dt_s) -> peak pcm.

Arms
  computing_only : causal LSF scheduler with spike, deadline and grid-state
                   handling. Never receives I, X, rho, R_cap, headroom, a
                   ceiling or a projection result, and never calls the
                   projection callback (it is not given one).
  consolidated   : the identical workload/queue/spike/grid/deadline logic
                   plus time-varying-path xenon feasibility.

Decision order at each step (see WorkloadController.step)
   1. observe current demand, fractions, queue, grid state
   2. classify the current demand sample against spike statistics that
      were completed at the PREVIOUS step, then fold the sample in
   3. choose the state: ISLANDED > STRESSED > NORMAL
   4. order all flexible work least-slack-first
   5. choose the grid action allowed in that state
   6. build the requested future reactor-power path from the causal forecast
      and known deadline obligations
   7. consolidated only: project xenon along that path
   8. consolidated only, if the projected peak exceeds the ceiling: cancel the
      discretionary price-driven import, then raise a reactor-power floor
      by deterministic bisection until the path is feasible. If even the
      maximum reachable power is projected unsafe, record
      no_feasible_action (never an accepted safe floor)
   9. surplus from a raised floor or the ramp-down limit displaces grid
      import (all offered work is already served whenever a surplus exists)
  10. the plant enforces ramp limits and the available grid capacity
  11. every deferral, drop, surplus and rigid shortfall is booked explicitly

Lexicographic priorities: (1) reactor feasibility, (2) inflexible demand
and deadline completion, (3) minimum dropped/unserved work, (4) grid cost
and water.

Least-slack-first tie-break (deterministic): items are ordered by
    (remaining_slack_steps, tier rank [mid=0, batch=1], arrival_step, seq)
i.e. equal slack -> mid-tier before batch -> earlier arrival first ->
earlier creation (seq is a per-controller counter; the mid item of a step
is created before its batch item).

Workload mapping (existing Borg-derived fractions; no per-job rigid class
exists in Borg -- rigid compute is the remainder):
    P_batch = fb * P_work, P_mid = fm * P_work,
    P_rigid = (1 - fb - fm) * P_work.
P_FIXED and required cooling are inflexible facility demand (the plant
passes them as base_facility_W).

Queue units follow the legacy simulator: an item's `amount` is a power (W)
held for one dt step, so its energy is amount * dt.
"""
import bisect
import math
from dataclasses import dataclass

# ---------------------------------------------------------------- names --
COMPUTING_ONLY = "computing_only"
CONSOLIDATED = "consolidated"
CONTROLLER_MODES = (COMPUTING_ONLY, CONSOLIDATED)
# Opt-in research arm, deliberately NOT in CONTROLLER_MODES (the canonical
# pair): consolidated with a partial cheap-price import. Where the requested
# price-driven import is projected unsafe, it accepts the LARGEST price
# import whose own projected path is <= the ceiling (bisection on the import
# W) instead of cancelling the whole import. Everything else is identical.
CONSOLIDATED_PARTIAL_GRID = "consolidated_partial_grid"
OPT_IN_MODES = (CONSOLIDATED_PARTIAL_GRID,)
ALL_CONTROLLER_MODES = CONTROLLER_MODES + OPT_IN_MODES
REACTOR_AWARE_MODES = (CONSOLIDATED, CONSOLIDATED_PARTIAL_GRID)

NORMAL, STRESSED, ISLANDED = "NORMAL", "STRESSED", "ISLANDED"
STATES = (NORMAL, STRESSED, ISLANDED)

TIER_RIGID, TIER_MID, TIER_BATCH = "rigid", "mid", "batch"
_TIER_RANK = {TIER_MID: 0, TIER_BATCH: 1}

SRC_NATURAL = "existing_queue_horizon"
SRC_SYNTHETIC = "synthetic_deadline_shock"
DEADLINE_SOURCES = (SRC_NATURAL, SRC_SYNTHETIC)

R_PROJECTED_CEILING = "projected_ceiling"
R_RAMP_LIMIT = "ramp_limit"
R_GRID_UNAVAILABLE = "grid_unavailable"
R_GRID_CAPACITY = "grid_capacity"
R_DEADLINE_PRESSURE = "deadline_pressure"
R_DEMAND_SPIKE = "demand_spike"
R_RIGID_SHORTFALL = "rigid_supply_shortfall"
REASON_CODES = (R_PROJECTED_CEILING, R_RAMP_LIMIT, R_GRID_UNAVAILABLE,
                R_GRID_CAPACITY, R_DEADLINE_PRESSURE, R_DEMAND_SPIKE,
                R_RIGID_SHORTFALL)
RIGID_SHORTFALL_CAUSE = "insufficient_feasible_supply"

SYNTHETIC_DEADLINE_NOTE = ("Deadline shocks are synthetic stress events; the "
                           "Borg trace does not provide job-level deadline "
                           "arrivals.")

EPS_W = 1e-9          # W; amounts at or below this are treated as zero


# ------------------------------------------------------ workload mapping --
def validate_fractions(fb, fm, where=""):
    """Hard-fail unless 0 <= fb, 0 <= fm and fb + fm <= 1 (exact; the
    interpolated Borg fractions are convex combinations of valid samples)."""
    if not (fb >= 0.0 and fm >= 0.0 and fb + fm <= 1.0):
        raise ValueError(f"invalid workload fractions{where}: fb={fb!r}, "
                         f"fm={fm!r}, fb+fm={fb + fm!r} (need fb>=0, fm>=0, "
                         f"fb+fm<=1)")


def split_work(P_work, fb, fm):
    """(P_batch, P_mid, P_rigid) from the existing Borg fractions."""
    return fb * P_work, fm * P_work, (1.0 - fb - fm) * P_work


# ------------------------------------------------------- spike detector --
@dataclass(frozen=True)
class DetectorConfig:
    """Primary preregistered values: k_sigma=3, rated_floor_fraction=0.05,
    identical for every Borg cell and reactor condition."""
    k_sigma: float = 3.0
    rated_floor_fraction: float = 0.05
    tau_s: float = 3600.0        # EWMA time scale (60 min)
    warmup_s: float = 3600.0     # no detection before 60 min of samples


PRIMARY_DETECTOR = DetectorConfig()
SENSITIVITY_K_SIGMA = (2.0, 3.0, 4.0)
SENSITIVITY_FLOOR = (0.025, 0.05, 0.10)


def sensitivity_grid():
    """Every declared (k_sigma, rated_floor_fraction) pair, fixed order."""
    return [(k, f) for k in SENSITIVITY_K_SIGMA for f in SENSITIVITY_FLOOR]


class SpikeDetector:
    """EWMA mean/variance of requested datacenter power.

    alpha = 1 - exp(-dt / tau_s), so the time scale is independent of dt.
    Update (exponentially weighted, West 1979 incremental form):
        d = x - mean;  mean += alpha * d;  var = (1 - alpha) * (var + alpha d^2)
    observe(x) first classifies x against the statistics completed at the
    previous sample, then folds x in (detections therefore never use the
    sample being judged). The mean is initialised to the first sample and
    the variance to zero; no detection is declared until warmup_s of
    samples have been folded in.

    spike  <=>  x - mean_prev > max(k_sigma * std_prev,
                                    rated_floor_fraction * P_rated)
    binding term: 'sigma' when k_sigma*std_prev >= floor, else 'rated_floor'
    (ties are attributed to 'sigma')."""

    def __init__(self, cfg, dt, P_rated_W):
        if not dt > 0 or not cfg.tau_s > 0:
            raise ValueError("dt and tau_s must be > 0")
        self.cfg = cfg
        self.alpha = 1.0 - math.exp(-dt / cfg.tau_s)
        self.warmup_steps = int(math.ceil(cfg.warmup_s / dt))
        self.floor_W = cfg.rated_floor_fraction * P_rated_W
        self.n = 0
        self.mean = None
        self.var = 0.0

    def observe(self, x):
        spike, binding = False, None
        excess, thresh = float("nan"), float("nan")
        if self.n >= self.warmup_steps:
            s_term = self.cfg.k_sigma * math.sqrt(max(self.var, 0.0))
            thresh = max(s_term, self.floor_W)
            excess = x - self.mean
            if excess > thresh:
                spike = True
                binding = "sigma" if s_term >= self.floor_W else "rated_floor"
        if self.mean is None:
            self.mean = float(x)
        else:
            d = x - self.mean
            self.mean += self.alpha * d
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * d * d)
        self.n += 1
        return spike, binding, excess, thresh


# ------------------------------------------------------ controller config --
@dataclass(frozen=True)
class ControllerConfig:
    """Preregistered controller parameters (same for every cell / R_cap).

    horizon_s, path_dt_s : requested-path horizon and segment length; the
        defaults equal the legacy predictive projection (10 h, 300 s).
    urgent_slack_s : queued work with remaining slack <= this is 'deadline
        pressure' (15 min = a quarter of the default 1 h mid-tier horizon).
    stress_hold_s  : STRESSED persists this long after a spike detection
        (60 min, the detector time scale).
    bisect_iters   : floor bisection iterations (legacy predictive: 12).
    p_min_proj     : lower clamp on projected power fractions (legacy
        predictive projects max(p, 1e-3))."""
    detector: DetectorConfig = PRIMARY_DETECTOR
    horizon_s: float = 10 * 3600.0
    path_dt_s: float = 300.0
    urgent_slack_s: float = 900.0
    stress_hold_s: float = 3600.0
    bisect_iters: int = 12
    p_min_proj: float = 1e-3


# ------------------------------------------------------------- queue item --
@dataclass(slots=True)
class QueueItem:
    amount: float            # W for one dt step (legacy convention)
    age_steps: int
    deadline_steps: int
    tier: str
    synthetic_deadline: bool
    deadline_source: str
    arrival_step: int
    seq: int
    due_step: int = 0        # arrival_step + deadline_steps (set on init)

    def __post_init__(self):
        self.due_step = self.arrival_step + self.deadline_steps

    @property
    def slack(self):
        """remaining_slack_steps = deadline_steps - age_steps; the item may
        still be served in a step where this is 0 and is dropped at the end
        of that step (legacy: age incremented, dropped when age > horizon)."""
        return self.deadline_steps - self.age_steps


def lsf_key(it):
    """Least-slack-first key. Because age_steps = k - arrival_step for every
    queued item, remaining slack = due_step - k, so ordering by due_step is
    identical to ordering by remaining slack at every step k -- and it never
    changes over time, which lets the queue stay sorted."""
    return (it.due_step, _TIER_RANK[it.tier], it.arrival_step, it.seq)


# ------------------------------------------------------------ observation --
@dataclass(frozen=True, slots=True)
class ReactorObs:
    """Current reactor state -- given to the consolidated arm ONLY.
    ceiling_pcm is supplied by the plant (R_cap - sigma_m - eps_buffer) so
    no reactor constant lives in this module."""
    I: float
    X: float
    rho_prev_pcm: float
    ceiling_pcm: float


@dataclass(frozen=True, slots=True)
class Observation:
    k: int
    t_s: float
    P_work_W: float          # requested IT work power at this step
    fb: float
    fm: float
    requested_W: float       # exogenous facility request (fixed + PUE*work)
    base_facility_W: float   # inflexible facility demand before IT served
    it_cost: float           # facility W drawn per W of IT served
    P_rated_W: float
    reactor_prev: float      # reactor power fraction at k-1
    reactor_hi: float        # reachable max this step (cap/force/trip/ramp)
    reactor_lo: float        # reachable min this step (ramp-down / trip)
    reactor_cap: float       # min(1, force, trip): no ramp term
    hi_binding: str          # 'cap' | 'force' | 'trip' | 'ramp'
    ramp_step: float         # fraction of rated per step
    grid_cap_W: float        # usable interconnection now
    grid_avail: object       # None if no availability series, else [0, 1]
    grid_connected: bool     # an interconnection exists (G_max_frac > 0)
    price_cheap: bool
    tripped: bool
    emergency_grid: bool
    reactor: object = None   # ReactorObs (consolidated) or None


@dataclass(slots=True)
class Decision:
    k: int
    state: str
    causes: tuple
    spike: bool
    spike_binding: object
    spike_excess_W: float
    spike_threshold_W: float
    grid_W: float
    grid_candidate_W: float  # grid of the requested candidate (pre-feasibility)
    grid_requested_W: float
    grid_price_W: float
    grid_bridge_W: float
    grid_emergency_W: float
    p_requested: float       # reactor fraction the offered work asks for
    p_candidate: float       # requested fraction clipped to reach [lo, hi]
    p_target: float          # accepted reactor fraction for this step
    p_floor: float
    it_served_W: float
    rigid_served_W: float
    rigid_unserved_W: float
    facility_unserved_W: float
    new_served_W: float
    drained_W: float
    drained_b_W: float
    drained_m_W: float
    drained_age_b: float     # sum(take * age) for delay metrics
    drained_age_m: float
    deferred_b_W: float
    deferred_m_W: float
    dropped_b_W: float
    dropped_m_W: float
    dropped_natural_W: float
    dropped_synthetic_W: float
    min_slack: object        # steps, before allocation (None if empty)
    pressure_tier: str
    forecast_W: float
    req_path: object         # (p0, min, mean, max) or None if not built
    acc_path: object
    proj_peak_req: float
    proj_peak_acc: float
    ceiling_pcm: float
    projections: int
    price_import_canceled: bool
    no_feasible_action: bool # nothing reachable projected safe
    pred_miss_W: float       # predicted misses among items due in horizon
    queue_W: float           # after this step's aging/drops
    queue_age_wsum: float    # sum(amount * age) over the queue
    queue_max_age: int
    reasons: tuple
    # price-import diagnostics (W; requested = after the G cap, before
    # feasibility; approved = after feasibility, before surplus displacement)
    price_import_requested_W: float = 0.0
    price_import_approved_W: float = 0.0
    price_import_partial: bool = False   # partial-grid arm: 0 < approved <
    #                                      requested after the search
    partial_audit_missed: bool = False   # audit: a larger audited import
    partial_audit_gap_W: float = 0.0     # was projected feasible (by W)


def _summary(path):
    return (path[0], min(path), sum(path) / len(path), max(path))


def _forbidden_projection(*_a, **_k):
    raise RuntimeError("computing_only must never invoke the xenon "
                       "projection callback")


# -------------------------------------------------------------- controller --
class WorkloadController:
    """Stateful controller for one run. The plant calls, per step k >= 1:
    [apply_deadline_shock(k, ...) at the shock step,] then step(obs).

    The queue is kept sorted by lsf_key (time-invariant), so expiring work,
    work due inside the horizon and deadline drops are always prefixes."""

    def __init__(self, mode, dt, P_rated_W, mw_b, mw_m, cfg=None,
                 project_path=None, known_outages=(), partial_audit_points=0):
        if mode not in ALL_CONTROLLER_MODES:
            raise ValueError(f"unknown controller mode {mode!r}")
        self.mode = mode
        self.reactor_aware = mode in REACTOR_AWARE_MODES
        self.partial_grid = mode == CONSOLIDATED_PARTIAL_GRID
        if partial_audit_points and not self.partial_grid:
            raise ValueError("partial_audit_points applies only to "
                             f"{CONSOLIDATED_PARTIAL_GRID!r}")
        # diagnostic only: at each partial search also project N-1 evenly
        # spaced imports and flag any larger one that is feasible
        self.partial_audit_points = int(partial_audit_points)
        self.cfg = cfg or ControllerConfig()
        self.dt = float(dt)
        self.P_rated = float(P_rated_W)
        self.mw_b, self.mw_m = int(mw_b), int(mw_m)
        if self.reactor_aware:
            if project_path is None:
                raise ValueError("consolidated needs a projection callback")
            self._project = project_path
        else:
            if project_path is not None:
                raise ValueError("computing_only must not be given a "
                                 "projection callback")
            self._project = _forbidden_projection
        # explicitly scheduled (announced) outages only: [(k0, k1), ...)
        self.known_outages = tuple((int(a), int(b)) for a, b in known_outages)
        self.detector = SpikeDetector(self.cfg.detector, dt, P_rated_W)
        self.seg_steps = self.cfg.path_dt_s / self.dt
        self.n_seg = int(self.cfg.horizon_s / self.cfg.path_dt_s)
        if self.n_seg < 2:
            raise ValueError("horizon must span at least two path segments")
        self.h_steps = self.cfg.horizon_s / self.dt
        self.urgent_steps = int(round(self.cfg.urgent_slack_s / self.dt))
        self.hold_steps = int(round(self.cfg.stress_hold_s / self.dt))
        self.queue = []                  # sorted by lsf_key
        self._q_total = 0.0
        self._seq = 0
        self._stress_until = -1
        self.promoted_W = 0.0            # deadline-shock promoted amount

    # ------------------------------------------------------------ events --
    def apply_deadline_shock(self, k, fraction, deadline_steps):
        """Promote `fraction` of every queued natural batch item to a
        remaining slack of `deadline_steps` (only where that is shorter
        than its existing slack). Deterministic: a fixed fraction of each
        item, no sampling. Returns the promoted amount (W-steps)."""
        if not (0.0 < fraction <= 1.0) or deadline_steps < 0:
            raise ValueError("deadline shock needs 0 < fraction <= 1 and "
                             "deadline_steps >= 0")
        promoted, new = 0.0, []
        for it in self.queue:
            if (it.tier == TIER_BATCH and not it.synthetic_deadline
                    and it.slack > deadline_steps and it.amount > EPS_W):
                a = it.amount * fraction
                it.amount -= a
                new.append(QueueItem(a, it.age_steps,
                                     it.age_steps + int(deadline_steps),
                                     TIER_BATCH, True, SRC_SYNTHETIC,
                                     it.arrival_step, self._next_seq()))
                promoted += a
        self.queue = [it for it in self.queue if it.amount > EPS_W]
        for it in new:
            bisect.insort(self.queue, it, key=lsf_key)
        self._q_total = sum(it.amount for it in self.queue)
        self.promoted_W += promoted
        return promoted

    def _next_seq(self):
        self._seq += 1
        return self._seq

    def _merged(self, new):
        """Queue and this step's new items in least-slack-first order."""
        q = self.queue
        i = 0
        for nw in sorted(new, key=lsf_key):
            j = bisect.bisect_left(q, lsf_key(nw), lo=i, key=lsf_key)
            for idx in range(i, j):
                yield q[idx]
            yield nw
            i = j
        for idx in range(i, len(q)):
            yield q[idx]

    # -------------------------------------------------------------- path --
    def _grid_known_zero(self, k_step):
        for a, b in self.known_outages:
            if a <= k_step < b:
                return True
        return False

    def _build_path(self, k, p0, D_fc, pending, grid_W, obs):
        """Requested reactor-power path (fractions), one value per segment.
        Segment 0 holds the current candidate p0. Later segments request
        (forecast facility demand + deadline obligations - grid) / P_rated,
        where grid persists at its current value except inside explicitly
        scheduled outages (known_outages), ramp-limited between segments.
        pending: [(amount_W_steps, steps_until_deadline)] still queued
        after this step and due within the horizon; each is spread evenly
        over the segments before its deadline."""
        n, T = self.n_seg, self.cfg.path_dt_s
        c, dt, P = obs.it_cost, self.dt, self.P_rated
        diff = [0.0] * (n + 1)
        for amt, m in pending:
            ns = min(n - 1, max(1, int(math.ceil(m * dt / T))))
            v = c * amt * dt / (ns * T)
            diff[1] += v
            diff[ns + 1] -= v
        ramp_seg = obs.ramp_step * self.seg_steps
        seg = self.seg_steps
        known = self.known_outages
        path = [p0]
        acc, prev = 0.0, p0
        for j in range(1, n):
            acc += diff[j]
            g = grid_W
            if known and self._grid_known_zero(k + int(j * seg)):
                g = 0.0
            p = (D_fc + acc - g) / P
            p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
            lo_, hi_ = prev - ramp_seg, prev + ramp_seg
            p = lo_ if p < lo_ else (hi_ if p > hi_ else p)
            path.append(p)
            prev = p
        return path

    def _proj(self, robs, path):
        pm = self.cfg.p_min_proj
        return self._project(robs.I, robs.X, [p if p > pm else pm
                                              for p in path],
                             self.cfg.path_dt_s)

    # -------------------------------------------------------------- step --
    def step(self, obs):
        k = obs.k
        if self.mode == COMPUTING_ONLY and obs.reactor is not None:
            raise ValueError("computing_only must not receive reactor state")
        if self.reactor_aware and obs.reactor is None:
            raise ValueError("consolidated needs the current reactor state")
        validate_fractions(obs.fb, obs.fm, f" at step {k}")
        P = self.P_rated
        P_b, P_m, P_r = split_work(obs.P_work_W, obs.fb, obs.fm)
        q = self.queue

        # 2. spike statistics through k-1, then fold in the current sample
        spike, binding, excess, thresh = self.detector.observe(
            obs.requested_W)
        if spike:
            self._stress_until = k + self.hold_steps
        D_fc = self.detector.mean          # EWMA incl. the current sample

        # 3. state
        min_slack = q[0].slack if q else None
        causes = []
        if k < self._stress_until:
            causes.append(R_DEMAND_SPIKE)
        if min_slack is not None and min_slack <= self.urgent_steps:
            causes.append(R_DEADLINE_PRESSURE)
        islanded = obs.grid_avail is not None and obs.grid_avail == 0.0
        state = ISLANDED if islanded else (STRESSED if causes else NORMAL)

        # 4. candidate work, least slack first
        new = []
        if P_m > EPS_W:
            new.append(QueueItem(P_m, 0, self.mw_m, TIER_MID, False,
                                 SRC_NATURAL, k, self._next_seq()))
        if P_b > EPS_W:
            new.append(QueueItem(P_b, 0, self.mw_b, TIER_BATCH, False,
                                 SRC_NATURAL, k, self._next_seq()))
        new_ids = {id(it) for it in new}

        c, base = obs.it_cost, obs.base_facility_W
        expiring = 0.0
        for it in q:                        # prefix: due now
            if it.slack > 0:
                break
            expiring += it.amount
        D_it_all = P_r + self._q_total + P_m + P_b
        need_all = base + c * D_it_all
        need_req = base + c * (P_r + expiring)
        R_hi = obs.reactor_hi * P
        G_cap = obs.grid_cap_W

        # 5. grid action
        g_price = g_bridge = g_emerg = 0.0
        if obs.grid_connected:
            if obs.tripped and obs.emergency_grid:
                g_emerg = need_all
            if state == NORMAL:
                if obs.price_cheap:
                    g_price = need_all
                g_bridge = max(0.0, need_req - R_hi)       # escape only
            elif state == STRESSED:
                g_bridge = max(0.0, need_all - R_hi)       # bridge ramping
            else:                                          # ISLANDED
                g_bridge = max(0.0, need_req - R_hi)
        grid_req = max(g_price, g_bridge, g_emerg)
        grid = min(grid_req, G_cap)
        if obs.tripped and g_emerg > 0.0:
            gr_em, gr_br, gr_pr = grid, 0.0, 0.0
        else:
            gr_em = 0.0
            gr_br = min(g_bridge, grid)
            gr_pr = grid - gr_br

        lo, hi = obs.reactor_lo, obs.reactor_hi

        def allocate(grid_W):
            S = R_hi + grid_W
            fac_short = max(0.0, base - S)
            it_cap = max(0.0, S - base) / c
            r = P_r if P_r < it_cap else it_cap
            it_cap -= r
            takes = []
            if it_cap > EPS_W:
                for it in self._merged(new):
                    t = it.amount if it.amount < it_cap else it_cap
                    it_cap -= t
                    takes.append((it, t))
                    if it_cap <= EPS_W:
                        break
            it_served = r + sum(t for _, t in takes)
            served = (base - fac_short) + c * it_served
            p_req = max(0.0, (served - grid_W) / P)
            return r, takes, it_served, fac_short, p_req

        h_steps = self.h_steps

        def pending_after(takes):
            out = []
            nt = len(takes)
            for i, it in enumerate(self._merged(new)):
                m = it.slack                 # steps still available after k
                if m * self.dt > self.cfg.horizon_s:
                    break                    # due beyond the horizon
                rem = it.amount - (takes[i][1] if i < nt else 0.0)
                if rem > EPS_W and m > 0:
                    out.append((rem, m))
            return out

        # 6. requested path (built for consolidated, or for the log)
        r_srv, takes, it_served, fac_short, p_req = allocate(grid)
        p_requested, grid_candidate = p_req, grid
        p_c = min(max(p_req, lo), hi)
        p_candidate = p_c
        nan = float("nan")
        path_req = acc_path = None
        peak_req = peak_acc = ceiling = nan
        n_proj, canceled, no_feasible = 0, False, False
        best_effort = None
        floor = lo
        reasons = set()
        price_requested = gr_pr
        partial = audit_missed = False
        audit_gap = 0.0

        # 7-8. reactor feasibility (consolidated arms only)
        if self.reactor_aware and not obs.tripped:
            robs = obs.reactor
            ceiling = robs.ceiling_pcm
            path_req = acc_path = self._build_path(
                k, p_c, D_fc, pending_after(takes), grid, obs)
            peak_req = peak_acc = self._proj(robs, path_req)
            n_proj += 1
            if peak_req > ceiling:
                reasons.add(R_PROJECTED_CEILING)
                if gr_pr > 0.0:                    # cancel price-driven dip
                    x_req = gr_pr
                    grid -= gr_pr
                    gr_pr = 0.0
                    canceled = True
                    r_srv, takes, it_served, fac_short, p_req = \
                        allocate(grid)
                    p_c = min(max(p_req, lo), hi)
                    acc_path = self._build_path(k, p_c, D_fc,
                                                pending_after(takes), grid,
                                                obs)
                    peak_acc = self._proj(robs, acc_path)
                    n_proj += 1
                    if self.partial_grid and peak_acc <= ceiling:
                        # largest price import x in (0, x_req) whose own
                        # projected path is <= ceiling. Bracket: x = 0 is
                        # feasible (just projected), x = x_req is not
                        # (peak_req). x_lo only ever moves to a projected-
                        # feasible import, so the accepted import is safe
                        # even if the peak were not monotone in x.
                        g0 = grid

                        def trial(x):
                            a_ = allocate(g0 + x)
                            pc_ = min(max(a_[4], lo), hi)
                            path_ = self._build_path(
                                k, pc_, D_fc, pending_after(a_[1]), g0 + x,
                                obs)
                            return a_, pc_, path_, self._proj(robs, path_)
                        x_lo, x_hi, best = 0.0, x_req, None
                        for _ in range(self.cfg.bisect_iters):
                            x = 0.5 * (x_lo + x_hi)
                            tr = trial(x)
                            n_proj += 1
                            if tr[3] > ceiling:
                                x_hi = x
                            else:
                                x_lo, best = x, tr
                        if best is not None:
                            (r_srv, takes, it_served, fac_short, p_req), \
                                p_c, acc_path, peak_acc = best
                            grid = g0 + x_lo
                            gr_pr = x_lo
                            canceled = False
                            partial = True
                        n_aud = self.partial_audit_points
                        for j in range(1, n_aud):
                            x = x_req * j / n_aud
                            if x > x_hi and trial(x)[3] <= ceiling:
                                audit_missed = True
                                audit_gap = max(audit_gap, x - x_lo)
                if peak_acc > ceiling:
                    base_path = acc_path

                    def floored(f):
                        return [p if p > f else f for p in base_path]
                    top = floored(hi)
                    pk_top = self._proj(robs, top)
                    n_proj += 1
                    if pk_top > ceiling:
                        # NO FEASIBLE ACTION: even the maximum reachable
                        # power is projected unsafe. The plant still needs a
                        # setpoint; the lower-projected-peak candidate is
                        # taken as a best-effort UNSAFE action. It is never
                        # recorded as an accepted safe floor (p_floor = NaN);
                        # the outcome is whatever the plant then records
                        # (crossing / trip / rigid shortfall).
                        no_feasible = True
                        if pk_top <= peak_acc:
                            best_effort, acc_path, peak_acc = hi, top, pk_top
                    else:
                        f_lo, f_hi = p_c, hi
                        for _ in range(self.cfg.bisect_iters):
                            mid = 0.5 * (f_lo + f_hi)
                            n_proj += 1
                            if self._proj(robs, floored(mid)) > ceiling:
                                f_lo = mid
                            else:
                                f_hi = mid
                        cand = floored(f_hi)
                        pk = self._proj(robs, cand)
                        n_proj += 1
                        if pk > ceiling:           # non-monotone: use top
                            floor, acc_path, peak_acc = hi, top, pk_top
                        else:
                            floor, acc_path, peak_acc = f_hi, cand, pk

        p_target = (best_effort if best_effort is not None
                    else min(max(p_c, floor), hi))

        price_approved = gr_pr

        # 9. surplus displaces grid (never fabricates demand)
        surplus = (p_target - p_req) * P
        if surplus > 0.0 and grid > 0.0:
            s = min(gr_pr, surplus); gr_pr -= s; surplus -= s; grid -= s
            s = min(gr_br, surplus); gr_br -= s; surplus -= s; grid -= s
            s = min(gr_em, surplus); gr_em -= s; surplus -= s; grid -= s
            p_req = max(0.0, (base - fac_short + c * it_served - grid) / P)

        rigid_unserved = P_r - r_srv
        if rigid_unserved < EPS_W:
            rigid_unserved = 0.0
        deferred_b = deferred_m = 0.0
        taken_new = {id(it): t for it, t in takes if id(it) in new_ids}
        for it in new:
            rem = it.amount - taken_new.get(id(it), 0.0)
            if rem > EPS_W:
                if it.tier == TIER_BATCH:
                    deferred_b += rem
                else:
                    deferred_m += rem

        # reasons (all known before commit)
        if rigid_unserved > 0.0 or fac_short > EPS_W:
            reasons.add(R_RIGID_SHORTFALL)
        if obs.hi_binding == "ramp" and \
                (need_all - grid) > R_hi * (1 + 1e-12) + EPS_W:
            reasons.add(R_RAMP_LIMIT)
        if not obs.tripped and p_requested < lo - 1e-12:
            reasons.add(R_RAMP_LIMIT)
        if obs.grid_connected and grid_req > EPS_W:
            if islanded:
                reasons.add(R_GRID_UNAVAILABLE)
            elif grid_req > G_cap * (1 + 1e-12) + EPS_W:
                reasons.add(R_GRID_CAPACITY)
        if state == STRESSED and obs.grid_connected and G_cap > 0.0 and (
                obs.price_cheap or g_bridge > EPS_W):
            reasons.update(causes)            # stress modified the path
        if reasons and path_req is None:      # computing_only: for the log
            path_req = acc_path = self._build_path(
                k, p_c, D_fc, pending_after(takes), grid, obs)

        # 11. commit allocation
        drained = drained_b = drained_m = age_b = age_m = new_served = 0.0
        n_q_served = 0
        for it, t in takes:
            it.amount -= t
            if id(it) in new_ids:
                new_served += t
            else:
                n_q_served += 1
                drained += t
                if it.tier == TIER_BATCH:
                    drained_b += t
                    age_b += t * it.age_steps
                else:
                    drained_m += t
                    age_m += t * it.age_steps
        # served queue items are a prefix of the queue; all but possibly
        # the last are exhausted
        cut = 0
        while cut < n_q_served and q[cut].amount <= EPS_W:
            cut += 1
        if cut:
            del q[:cut]
        for it in new:
            if it.amount > EPS_W:
                bisect.insort(q, it, key=lsf_key)

        # pressure tier
        if rigid_unserved > 0.0 or fac_short > EPS_W:
            pressure = TIER_RIGID
        elif deferred_b > 0.0 or deferred_m > 0.0:
            pressure = TIER_BATCH if deferred_b > 0.0 else TIER_MID
        elif q:
            pressure = q[0].tier
        else:
            pressure = "none"

        # aging and deadline drops (legacy order: age += 1, drop age > due);
        # the dropped items are exactly the prefix with due_step <= k
        total = wsum = 0.0
        max_age = 0
        for it in q:
            a_ = it.age_steps + 1
            it.age_steps = a_
            total += it.amount
            wsum += it.amount * a_
            if a_ > max_age:
                max_age = a_
        drop_b = drop_m = drop_nat = drop_syn = 0.0
        nd = 0
        while nd < len(q) and q[nd].age_steps > q[nd].deadline_steps:
            it = q[nd]
            if it.tier == TIER_BATCH:
                drop_b += it.amount
            else:
                drop_m += it.amount
            if it.synthetic_deadline:
                drop_syn += it.amount
            else:
                drop_nat += it.amount
            total -= it.amount
            wsum -= it.amount * it.age_steps
            nd += 1
        if nd:
            del q[:nd]
            max_age = max((it.age_steps for it in q), default=0)
        self._q_total = total if q else 0.0

        # predicted deadline outcome at decision time for items due within
        # the horizon: drain in LSF order at forecast spare capacity
        # (reactor cap + persisted grid - forecast demand); items finishing
        # after their deadline are predicted misses.
        spare = max(0.0, obs.reactor_cap * P + grid - D_fc) / c
        cum = miss = 0.0
        for it in q:
            if it.slack > h_steps:
                break
            cum += it.amount
            if spare <= 0.0 or cum / spare > it.slack + 1:
                miss += it.amount

        return Decision(
            k=k, state=state, causes=tuple(causes), spike=spike,
            spike_binding=binding, spike_excess_W=excess,
            spike_threshold_W=thresh,
            grid_W=grid, grid_candidate_W=grid_candidate,
            grid_requested_W=grid_req, grid_price_W=gr_pr,
            grid_bridge_W=gr_br, grid_emergency_W=gr_em,
            p_requested=p_requested, p_candidate=p_candidate,
            p_target=p_target,
            p_floor=nan if no_feasible else floor,
            it_served_W=it_served, rigid_served_W=r_srv,
            rigid_unserved_W=rigid_unserved, facility_unserved_W=fac_short,
            new_served_W=new_served, drained_W=drained,
            drained_b_W=drained_b, drained_m_W=drained_m,
            drained_age_b=age_b, drained_age_m=age_m,
            deferred_b_W=deferred_b, deferred_m_W=deferred_m,
            dropped_b_W=drop_b, dropped_m_W=drop_m,
            dropped_natural_W=drop_nat, dropped_synthetic_W=drop_syn,
            min_slack=min_slack, pressure_tier=pressure, forecast_W=D_fc,
            req_path=_summary(path_req) if path_req else None,
            acc_path=_summary(acc_path) if acc_path else None,
            proj_peak_req=peak_req, proj_peak_acc=peak_acc,
            ceiling_pcm=ceiling, projections=n_proj,
            price_import_canceled=canceled,
            no_feasible_action=no_feasible,
            pred_miss_W=miss, queue_W=self._q_total, queue_age_wsum=wsum,
            queue_max_age=max_age,
            reasons=tuple(r for r in REASON_CODES if r in reasons),
            price_import_requested_W=price_requested,
            price_import_approved_W=price_approved,
            price_import_partial=partial,
            partial_audit_missed=audit_missed,
            partial_audit_gap_W=audit_gap)


# ------------------------------------------------ decision-log episodes --
class EpisodeLog:
    """Consolidates consecutive steps with an identical reason set into one
    refusal episode; always keeps aggregate per-reason step counts."""

    def __init__(self):
        self.counts = {r: 0 for r in REASON_CODES}
        self.episodes = []
        self._open = None

    def add(self, k, t_h, reasons, row):
        for r in reasons:
            self.counts[r] += 1
        key = reasons
        if self._open is not None and self._open["reasons"] == key and \
                self._open["_last_k"] == k - 1:
            e = self._open
            e["end_h"] = t_h
            e["steps"] += 1
            e["_last_k"] = k
            for f in ("projected_peak_pcm", "requested_grid_W"):
                v = row.get(f)
                if v is not None and v == v and (e["max_" + f] != e["max_" + f]
                                                 or v > e["max_" + f]):
                    e["max_" + f] = v
            return
        self.close()
        e = dict(reasons=key, reason_code=";".join(key), start_h=t_h,
                 end_h=t_h, steps=1, _last_k=k)
        e.update({"first_" + f: v for f, v in row.items()})
        for f in ("projected_peak_pcm", "requested_grid_W"):
            e["max_" + f] = row.get(f, float("nan"))
        self._open = e

    def close(self):
        if self._open is not None:
            e = dict(self._open)
            e.pop("_last_k")
            e.pop("reasons")
            self.episodes.append(e)
            self._open = None


# ------------------------------------------------- stress-event suite ---
@dataclass(frozen=True)
class StressConfig:
    """Deterministic, preregistered stress-event definitions. The same
    start, duration and magnitude are used for every cell, arm and R_cap
    snapshot. No randomness is used (stress_seed is recorded as 'none').

    event_start_h        : 254 h after trace start (day 10 + 14 h); leaves
                           far more than the 6 h settle, the 60 min detector
                           warm-up and the ~66 h measured xenon washout before
                           the event, and > 480 h of recovery after it.
    spike_multiplier     : IT work power x 1.25 (utilisation capped at 1.0,
                           the modelled hardware peak) for spike_duration_h.
    deadline_shock_*     : 50 % of each queued natural batch item promoted
                           to 60 min remaining slack at the shock instant.
    blackout_duration_h  : grid availability 0 for 6 h.
    compound_offset_h    : spike / deadline shock begins 2 h into the
                           blackout."""
    event_start_h: float = 254.0
    spike_multiplier: float = 1.25
    spike_duration_h: float = 2.0
    deadline_shock_fraction: float = 0.5
    deadline_shock_minutes: float = 60.0
    blackout_duration_h: float = 6.0
    compound_offset_h: float = 2.0
    min_pre_event_h: float = 72.0
    min_post_event_h: float = 48.0

    def validate(self):
        if not self.spike_multiplier > 1.0:
            raise ValueError("spike_multiplier must be > 1")
        if not (self.spike_duration_h > 0 and self.blackout_duration_h > 0):
            raise ValueError("event durations must be > 0")
        if not 0.0 < self.deadline_shock_fraction <= 1.0:
            raise ValueError("deadline_shock_fraction must be in (0, 1]")
        if not self.deadline_shock_minutes >= 0.0:
            raise ValueError("deadline_shock_minutes must be >= 0")
        if not 0.0 <= self.compound_offset_h < self.blackout_duration_h:
            raise ValueError("compound_offset_h must lie inside the blackout")


SCENARIOS = ("none", "spike", "deadline_shock", "surprise_blackout",
             "scheduled_outage", "compound_blackout_spike",
             "compound_blackout_deadline")


@dataclass(frozen=True)
class StressScenario:
    name: str
    spike: object = None       # (start_h, duration_h, multiplier)
    deadline: object = None    # (start_h, fraction, minutes)
    blackout: object = None    # (type, start_h, duration_h)

    @property
    def synthetic_deadline(self):
        return self.deadline is not None

    def metadata(self):
        nan = float("nan")
        sp, dl, bo = self.spike, self.deadline, self.blackout
        starts = [x[0] for x in (sp, dl) if x] + ([bo[1]] if bo else [])
        ends = ([sp[0] + sp[1]] if sp else []) + \
               ([bo[1] + bo[2]] if bo else []) + ([dl[0]] if dl else [])
        mag = []
        if sp:
            mag.append(f"spike x{sp[2]:g}")
        if dl:
            mag.append(f"deadline {dl[1]:g} of queued batch -> {dl[2]:g} min")
        if bo:
            mag.append(f"{bo[0]} blackout avail=0")
        return dict(
            scenario=self.name, stress_seed="none (deterministic)",
            event_start_h=min(starts) if starts else nan,
            event_duration_h=(max(ends) - min(starts)) if starts else nan,
            event_magnitude="; ".join(mag) if mag else "none",
            spike_multiplier=sp[2] if sp else nan,
            spike_start_h=sp[0] if sp else nan,
            spike_duration_h=sp[1] if sp else nan,
            blackout_type=bo[0] if bo else "none",
            blackout_start_h=bo[1] if bo else nan,
            blackout_duration_h=bo[2] if bo else nan,
            blackout_known_in_advance=bool(bo and bo[0] == "scheduled"),
            synthetic_deadline=self.synthetic_deadline,
            deadline_source=SRC_SYNTHETIC if dl else SRC_NATURAL,
            deadline_shock_fraction=dl[1] if dl else nan,
            deadline_shock_minutes=dl[2] if dl else nan,
            deadline_shock_start_h=dl[0] if dl else nan,
            cycle_fraction=nan)


def build_scenarios(cfg, names=SCENARIOS):
    cfg.validate()
    s0 = cfg.event_start_h
    sc = s0 + cfg.compound_offset_h
    spike = (s0, cfg.spike_duration_h, cfg.spike_multiplier)
    dl = (s0, cfg.deadline_shock_fraction, cfg.deadline_shock_minutes)
    bo_s = ("surprise", s0, cfg.blackout_duration_h)
    table = {
        "none": StressScenario("none"),
        "spike": StressScenario("spike", spike=spike),
        "deadline_shock": StressScenario("deadline_shock", deadline=dl),
        "surprise_blackout": StressScenario("surprise_blackout",
                                            blackout=bo_s),
        "scheduled_outage": StressScenario(
            "scheduled_outage",
            blackout=("scheduled", s0, cfg.blackout_duration_h)),
        "compound_blackout_spike": StressScenario(
            "compound_blackout_spike", blackout=bo_s,
            spike=(sc, cfg.spike_duration_h, cfg.spike_multiplier)),
        "compound_blackout_deadline": StressScenario(
            "compound_blackout_deadline", blackout=bo_s,
            deadline=(sc, cfg.deadline_shock_fraction,
                      cfg.deadline_shock_minutes)),
    }
    bad = [n for n in names if n not in table]
    if bad:
        raise ValueError(f"unknown stress scenario(s) {bad}; known "
                         f"{list(SCENARIOS)}")
    return [table[n] for n in names]


def _step_of(h, dt):
    return int(round(h * 3600.0 / dt))


@dataclass(frozen=True)
class EventInputs:
    """Per-run exogenous inputs derived from one scenario. Only
    `known_outages` (scheduled outages) is ever shown to a controller in
    advance; a surprise blackout reaches the controller only through the
    current-step availability sample."""
    grid_availability: object       # None or list of floats per step
    known_outages: tuple            # ((k0, k1),) scheduled only
    deadline_shock: object          # (k, fraction, deadline_steps) or None
    spike_window: object            # (k0, k1) or None
    blackout_window: object         # (k0, k1) or None
    spike_multiplier: float
    spike_t_window_s: object        # (s0, s1) relative to t_grid[0]


def build_event_inputs(scn, n_steps, dt, cfg=None):
    """Deterministic per-step inputs for `scn` on a uniform grid of n_steps
    starting at trace time 0. Validates placement (warm-up / recovery)."""
    cfg = cfg or StressConfig()
    horizon_h = n_steps * dt / 3600.0
    md = scn.metadata()
    if scn.name != "none":
        s, dur = md["event_start_h"], md["event_duration_h"]
        if s < cfg.min_pre_event_h:
            raise ValueError(f"{scn.name}: event at {s} h leaves < "
                             f"{cfg.min_pre_event_h} h pre-event warm-up")
        if s + dur > horizon_h - cfg.min_post_event_h:
            raise ValueError(f"{scn.name}: event ends at {s + dur} h, < "
                             f"{cfg.min_post_event_h} h before the trace end "
                             f"({horizon_h:.1f} h)")
    avail, known, bo_w = None, (), None
    if scn.blackout:
        kind, s, d = scn.blackout
        k0, k1 = _step_of(s, dt), _step_of(s + d, dt)
        avail = [1.0] * n_steps
        for k in range(k0, min(k1, n_steps)):
            avail[k] = 0.0
        bo_w = (k0, k1)
        if kind == "scheduled":
            known = ((k0, k1),)
        elif kind != "surprise":
            raise ValueError(f"unknown blackout type {kind!r}")
    shock = None
    if scn.deadline:
        s, f, mins = scn.deadline
        shock = (_step_of(s, dt), f, int(round(mins * 60.0 / dt)))
    sp_w, mult, sp_t = None, 1.0, None
    if scn.spike:
        s, d, mult = scn.spike
        sp_w = (_step_of(s, dt), _step_of(s + d, dt))
        sp_t = (s * 3600.0, (s + d) * 3600.0)
    return EventInputs(avail, known, shock, sp_w, bo_w, mult, sp_t)


def detector_pass(requested_W, dt, P_rated_W, cfg=PRIMARY_DETECTOR):
    """Run the spike detector alone over an exogenous request series fed in
    step order from index 1 (exactly as WorkloadController.step sees it).
    Returns (spike flags, binding terms, excess W) aligned to the input;
    index 0 is never classified. The request series does not depend on any
    controller action, so this equals the in-run detector output."""
    det = SpikeDetector(cfg, dt, P_rated_W)
    n = len(requested_W)
    spike, bind, excess = [False] * n, [None] * n, [float("nan")] * n
    for k in range(1, n):
        spike[k], bind[k], excess[k], _ = det.observe(requested_W[k])
    return spike, bind, excess


def injected_detection_metrics(spk_event, spk_natural, window, hold_steps,
                               dt):
    """Detection metrics for one injected spike (ground truth = the
    injection window). False positives are detections outside
    [start, end + hold) in the injected run that are absent at the same
    step of the paired natural run, i.e. attributable to the injection.
    Natural-trace detections are never labelled true or false."""
    k0, k1 = window
    first = next((k for k in range(k0, min(k1, len(spk_event)))
                  if spk_event[k]), None)
    outside = [k for k in range(1, len(spk_event))
               if not (k0 <= k < k1 + hold_steps)]
    fp = sum(1 for k in outside if spk_event[k] and not spk_natural[k])
    return dict(injected_detected=first is not None,
                injected_detection_delay_min=((first - k0) * dt / 60.0
                                              if first is not None
                                              else float("nan")),
                injected_fp_steps=fp,
                injected_fp_rate=fp / max(len(outside), 1))


def wrap_u_at(u_at, ev, t0):
    """Apply the spike multiplier to utilisation inside the spike window
    (capped at 1.0, the modelled hardware peak). Evaluates u_at only at
    the requested time, so causality of the caller is preserved."""
    if ev.spike_t_window_s is None:
        return u_at
    s0, s1 = ev.spike_t_window_s
    m = ev.spike_multiplier

    def u_spiked(tt):
        u = u_at(tt)
        rel = tt - t0
        if s0 <= rel < s1:
            return min(1.0, u * m)
        return u
    return u_spiked
