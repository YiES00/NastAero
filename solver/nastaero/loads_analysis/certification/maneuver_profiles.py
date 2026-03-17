"""FAR 23 maneuver profiles and discrete gust models for 6-DOF simulation.

Defines control input time histories and gust velocity profiles for:
- §23.331(c): Abrupt elevator pull-up and checked maneuver
- §23.349: Abrupt aileron roll at VA/VC/VD (deflection reduced with speed)
- §23.351: Rudder yaw maneuver
- §23.341: Discrete 1-cosine vertical gust
- Lateral discrete gust for VTP loads

All control/gust functions are implemented as pickle-safe callable classes
(not closures) so they can be serialized for ProcessPoolExecutor.

References
----------
- 14 CFR Part 23: §23.331, §23.333, §23.341, §23.349, §23.351
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .flight_sim import ControlInput
from .aircraft_config import (
    AircraftConfig, WeightCGCondition, SpeedSchedule,
    ControlSurfaceLimits, eas_to_tas, dynamic_pressure_from_eas,
    gust_Ude_at_altitude,
    RHO_0, G_MPS2,
)


# ---------------------------------------------------------------------------
# ManeuverProfile data structure
# ---------------------------------------------------------------------------

@dataclass
class ManeuverProfile:
    """A complete maneuver/gust profile for 6-DOF simulation.

    Contains the control input schedule and optional gust velocity
    function, plus metadata for traceability.
    """
    maneuver_type: str = ""       # elevator_pullup, checked, abrupt_roll,
                                   # yaw, gust_vert, gust_lat
    far_section: str = ""          # e.g. "§23.331(c)"
    t_end: float = 3.0             # Simulation end time (s)
    control_func: Optional[Callable] = None   # (t) -> ControlInput
    gust_func: Optional[Callable] = None      # (t, xe) -> ndarray(3)
    description: str = ""
    delta_e_trim: float = 0.0      # Trim elevator to add as offset


# ---------------------------------------------------------------------------
# Pickle-safe callable classes for control and gust functions
# ---------------------------------------------------------------------------

class _RampControl:
    """Ramp a single control surface from zero to target over t_ramp, then hold.

    Pickle-safe replacement for closure-based control functions.
    """
    __slots__ = ('de_trim', 'de_target', 'da_target', 'dr_target',
                 't_ramp',)

    def __init__(self, de_trim, de_target=None, da_target=0.0, dr_target=0.0,
                 t_ramp=0.2):
        self.de_trim = de_trim
        self.de_target = de_target if de_target is not None else de_trim
        self.da_target = da_target
        self.dr_target = dr_target
        self.t_ramp = t_ramp

    def __call__(self, t):
        if t < 0:
            frac = 0.0
        elif t < self.t_ramp:
            frac = t / self.t_ramp
        else:
            frac = 1.0
        de = self.de_trim + (self.de_target - self.de_trim) * frac
        da = self.da_target * frac
        dr = self.dr_target * frac
        return ControlInput(delta_e=de, delta_a=da, delta_r=dr)


class _RampHoldReleaseControl:
    """Ramp up → hold at peak → ramp back to zero → free response.

    Implements a trapezoidal control pulse for yaw-type maneuvers where
    the critical structural loads occur during the initial transient,
    and holding full deflection indefinitely would saturate the linear
    aero model.

    Timeline: 0→t_ramp_up→(+t_hold)→(+t_ramp_down)→free

    Pickle-safe for ProcessPoolExecutor.
    """
    __slots__ = ('de_trim', 'dr_target', 't_up', 't_hold_end', 't_down_end')

    def __init__(self, de_trim: float, dr_target: float,
                 t_ramp_up: float = 0.15, t_hold: float = 0.1,
                 t_ramp_down: float = 0.15):
        self.de_trim = de_trim
        self.dr_target = dr_target
        self.t_up = t_ramp_up
        self.t_hold_end = t_ramp_up + t_hold
        self.t_down_end = t_ramp_up + t_hold + t_ramp_down

    def __call__(self, t):
        if t < 0:
            frac = 0.0
        elif t < self.t_up:
            # Ramp up
            frac = t / self.t_up
        elif t < self.t_hold_end:
            # Hold at peak
            frac = 1.0
        elif t < self.t_down_end:
            # Ramp back to zero
            frac = 1.0 - (t - self.t_hold_end) / (self.t_down_end - self.t_hold_end)
        else:
            # Free response — no rudder input
            frac = 0.0
        return ControlInput(delta_e=self.de_trim, delta_a=0.0,
                            delta_r=self.dr_target * frac)


class _CheckedControl:
    """Pitch doublet: ramp up → hold → ramp to negative → hold → return.

    Pickle-safe callable for checked maneuver.
    """
    __slots__ = ('de_trim', 'de_pos', 'de_neg', 't_ramp', 't_hold',
                 't1', 't2', 't3', 't4', 't5')

    def __init__(self, de_trim, de_pos, de_neg, t_ramp=0.2, t_hold=0.5):
        self.de_trim = de_trim
        self.de_pos = de_pos
        self.de_neg = de_neg
        self.t_ramp = t_ramp
        self.t_hold = t_hold
        self.t1 = t_ramp
        self.t2 = t_ramp + t_hold
        self.t3 = 2 * t_ramp + t_hold
        self.t4 = 2 * t_ramp + 2 * t_hold
        self.t5 = 3 * t_ramp + 2 * t_hold

    def __call__(self, t):
        if t < 0:
            de = self.de_trim
        elif t < self.t1:
            de = self.de_trim + (self.de_pos - self.de_trim) * (t / self.t_ramp)
        elif t < self.t2:
            de = self.de_pos
        elif t < self.t3:
            frac = (t - self.t2) / self.t_ramp
            de = self.de_pos + (self.de_neg - self.de_pos) * frac
        elif t < self.t4:
            de = self.de_neg
        elif t < self.t5:
            frac = (t - self.t4) / self.t_ramp
            de = self.de_neg + (self.de_trim - self.de_neg) * frac
        else:
            de = self.de_trim
        return ControlInput(delta_e=de, delta_a=0.0, delta_r=0.0)


class _ConstantControl:
    """Constant control (trim hold or gust-only cases)."""
    __slots__ = ('de_trim',)

    def __init__(self, de_trim):
        self.de_trim = de_trim

    def __call__(self, t):
        return ControlInput(delta_e=self.de_trim, delta_a=0.0, delta_r=0.0)


class _VerticalGust:
    """1-cosine vertical gust velocity in body axes."""
    __slots__ = ('Ude_ms', 'gust_length', 'V_tas')

    def __init__(self, Ude_ms, gust_length, V_tas):
        self.Ude_ms = Ude_ms
        self.gust_length = gust_length
        self.V_tas = V_tas

    def __call__(self, t, xe):
        x = self.V_tas * t
        if x < 0 or x > self.gust_length:
            return np.array([0.0, 0.0, 0.0])
        wg = 0.5 * self.Ude_ms * (1.0 - math.cos(
            2.0 * math.pi * x / self.gust_length))
        return np.array([0.0, 0.0, wg])


class _LateralGust:
    """1-cosine lateral gust velocity in body axes."""
    __slots__ = ('Ude_ms', 'gust_length', 'V_tas')

    def __init__(self, Ude_ms, gust_length, V_tas):
        self.Ude_ms = Ude_ms
        self.gust_length = gust_length
        self.V_tas = V_tas

    def __call__(self, t, xe):
        x = self.V_tas * t
        if x < 0 or x > self.gust_length:
            return np.array([0.0, 0.0, 0.0])
        vg = 0.5 * self.Ude_ms * (1.0 - math.cos(
            2.0 * math.pi * x / self.gust_length))
        return np.array([0.0, vg, 0.0])


# ---------------------------------------------------------------------------
# Elevator deflection for target load factor
# ---------------------------------------------------------------------------

def compute_elevator_for_nz(
    nz_target: float,
    V_eas: float,
    weight_N: float,
    S_m2: float,
    CLalpha: float,
    CLdelta_e: float,
    Cmalpha: float,
    Cmdelta_e: float,
    de_physical_limit: float,
) -> float:
    """Compute elevator deflection (rad) for a target load factor.

    Uses the coupled steady-state trim equations:
        CL(α, δe) = nz × W / (q × S)
        Cm(α, δe) = 0  (pitch moment equilibrium)

    Solving gives:
        α = -(Cmδe / Cmα) × δe
        CL = (CLδe − CLα × Cmδe / Cmα) × δe
        δe = CL_target / (CLδe − CLα × Cmδe / Cmα)

    The result is capped at the physical control travel limit.

    Parameters
    ----------
    nz_target : float
        Target load factor (g's).
    V_eas : float
        Equivalent airspeed (m/s).
    weight_N : float
        Aircraft weight (N).
    S_m2 : float
        Reference wing area (m²).
    CLalpha, CLdelta_e, Cmalpha, Cmdelta_e : float
        Longitudinal aero derivatives (per radian).
    de_physical_limit : float
        Physical elevator travel limit (rad, positive).

    Returns
    -------
    float
        Elevator deflection in radians (signed), capped at physical limit.
    """
    qbar = 0.5 * RHO_0 * V_eas ** 2  # EAS → sea-level dynamic pressure
    if qbar < 1.0 or S_m2 < 0.01:
        return math.copysign(de_physical_limit, nz_target - 1.0)

    CL_target = nz_target * weight_N / (qbar * S_m2)

    # Effective elevator lift coefficient (coupled α–δe trim)
    if abs(Cmalpha) < 1e-8:
        CLde_eff = CLdelta_e
    else:
        CLde_eff = CLdelta_e - CLalpha * Cmdelta_e / Cmalpha

    if abs(CLde_eff) < 1e-8:
        return math.copysign(de_physical_limit, nz_target - 1.0)

    de = CL_target / CLde_eff

    # Cap at physical limit
    if abs(de) > de_physical_limit:
        de = math.copysign(de_physical_limit, de)

    return de


def compute_aileron_for_nz(
    nz_limit: float,
    V_eas: float,
    weight_N: float,
    S_m2: float,
    CL_aileron_halfwing: float,
    da_physical_limit: float,
    elastic_factor: float = 15.0,
) -> float:
    """Compute maximum aileron deflection (rad) that keeps the critical
    wing root load factor within the V-n diagram limit.

    During an abrupt roll, one wing sees the symmetric 1g lift plus the
    incremental differential lift from aileron deflection.  The worst-case
    wing root shear is approximately:

        V_root = W/2 + CL_ail_hw × elastic_factor × δa × q × S

    where ``CL_ail_hw`` is the half-wing Z-force coefficient from the VLM
    aileron solution (NOT the rolling moment coefficient Clδa, which is
    span-weighted and underestimates wing root forces dramatically).

    ``elastic_factor`` accounts for the aeroelastic amplification that
    occurs in the flexible-wing SOL 144 solution.  The rigid VLM predicts
    only the aerodynamic loads on a rigid wing; the actual elastic wing
    deforms under aileron loading, causing twist-amplified lift that
    significantly increases wing root forces.  Typical values for GA
    aircraft wings are 10–25×.  Default 15× is a reasonable middle
    estimate calibrated from KC-100 SOL 144 results.

    Setting the wing root load factor to the design limit:

        δa = (nz_limit − 1) × W / (2 × CL_ail_hw × elastic × q × S)

    The result is capped at the physical aileron travel limit.

    Parameters
    ----------
    nz_limit : float
        Positive design limit load factor (g's).
    V_eas : float
        Equivalent airspeed (m/s).
    weight_N : float
        Aircraft weight (N).
    S_m2 : float
        Reference wing area (m²).
    CL_aileron_halfwing : float
        One-sided wing CL from VLM aileron solution (per radian).
    da_physical_limit : float
        Physical aileron travel limit (rad, positive).
    elastic_factor : float
        Aeroelastic amplification factor applied to CL_ail_halfwing.
        Default 15.0.  Set to 1.0 for rigid-body estimate only.

    Returns
    -------
    float
        Maximum aileron deflection in radians (positive), capped at
        physical limit.
    """
    qbar = 0.5 * RHO_0 * V_eas ** 2
    if qbar < 1.0 or S_m2 < 0.01:
        return da_physical_limit

    CL_hw = abs(CL_aileron_halfwing) * elastic_factor
    if CL_hw < 1e-8:
        return da_physical_limit

    delta_nz = nz_limit - 1.0       # available nz increment from 1g trim
    if delta_nz <= 0:
        return 0.0

    da = delta_nz * weight_N / (2.0 * CL_hw * qbar * S_m2)

    # Cap at physical limit
    return min(da, da_physical_limit)


# ---------------------------------------------------------------------------
# Individual maneuver profile generators
# ---------------------------------------------------------------------------

def abrupt_elevator_pullup(
    delta_e_max: float,
    delta_e_trim: float = 0.0,
    t_ramp: float = 0.2,
    t_hold: float = 1.5,
) -> ManeuverProfile:
    """§23.331(c): Abrupt elevator pull-up.

    Ramps elevator from trim to max deflection over t_ramp seconds,
    then holds for t_hold seconds.
    """
    t_end = t_ramp + t_hold + 0.5

    return ManeuverProfile(
        maneuver_type="elevator_pullup",
        far_section="§23.331(c)",
        t_end=t_end,
        control_func=_RampControl(
            de_trim=delta_e_trim, de_target=delta_e_max, t_ramp=t_ramp),
        description=f"Elevator pull-up: δe={math.degrees(delta_e_max):.1f}°, "
                    f"t_ramp={t_ramp:.2f}s",
        delta_e_trim=delta_e_trim,
    )


def checked_maneuver(
    de_pos: float,
    de_neg: float,
    delta_e_trim: float = 0.0,
    t_ramp: float = 0.2,
    t_hold: float = 0.5,
) -> ManeuverProfile:
    """§23.331(c): Checked maneuver (pull-up → push-over doublet)."""
    t5 = 3 * t_ramp + 2 * t_hold
    t_end = t5 + 0.5

    return ManeuverProfile(
        maneuver_type="checked",
        far_section="§23.331(c)",
        t_end=t_end,
        control_func=_CheckedControl(
            de_trim=delta_e_trim, de_pos=de_pos, de_neg=de_neg,
            t_ramp=t_ramp, t_hold=t_hold),
        description=f"Checked: δe={math.degrees(de_pos):.1f}°/"
                    f"{math.degrees(de_neg):.1f}°",
        delta_e_trim=delta_e_trim,
    )


def abrupt_roll(
    delta_a: float,
    delta_e_trim: float = 0.0,
    t_ramp: float = 0.1,
    t_hold: float = 1.0,
) -> ManeuverProfile:
    """§23.349: Abrupt aileron roll input.

    t_hold=1.0 s captures the peak wing-root load during the initial
    rolling acceleration phase (roll-subsidence time constant ~0.3–0.5 s).
    Longer holds cause the aircraft to exceed 90° bank, generating
    secondary yaw–roll coupling loads that are unrealistic for structural
    certification.
    """
    t_end = t_ramp + t_hold + 0.5

    return ManeuverProfile(
        maneuver_type="abrupt_roll",
        far_section="§23.349",
        t_end=t_end,
        control_func=_RampControl(
            de_trim=delta_e_trim, da_target=delta_a, t_ramp=t_ramp),
        description=f"Abrupt roll: δa={math.degrees(delta_a):.1f}°",
        delta_e_trim=delta_e_trim,
    )


def yaw_maneuver(
    delta_r: float,
    delta_e_trim: float = 0.0,
    t_ramp: float = 0.15,
    t_hold: float = 0.1,
    t_release: float = 0.15,
    t_free: float = 0.3,
) -> ManeuverProfile:
    """§23.351: Rudder yaw maneuver — trapezoidal pulse.

    Uses a ramp-hold-release-free profile instead of sustained deflection.
    The critical VTP and fuselage lateral shear loads occur during the
    initial sideslip transient (t ≈ 0.1–0.3 s after application).
    Sustained full-rudder input beyond 0.3 s drives β past the linear
    aero model validity range (~25°), producing non-physical results.

    Default timing (total 0.7 s):
      0–0.15 s  : ramp up to full rudder
      0.15–0.25 s: hold at peak (captures max VTP shear)
      0.25–0.40 s: ramp back to zero (pilot releases rudder)
      0.40–0.70 s: free response (verify directional stability)
    """
    t_end = t_ramp + t_hold + t_release + t_free

    return ManeuverProfile(
        maneuver_type="yaw",
        far_section="§23.351",
        t_end=t_end,
        control_func=_RampHoldReleaseControl(
            de_trim=delta_e_trim, dr_target=delta_r,
            t_ramp_up=t_ramp, t_hold=t_hold, t_ramp_down=t_release),
        description=f"Yaw: δr={math.degrees(delta_r):.1f}°",
        delta_e_trim=delta_e_trim,
    )


def discrete_gust_vertical(
    Ude_fps: float,
    c_bar_m: float,
    V_tas: float,
    delta_e_trim: float = 0.0,
) -> ManeuverProfile:
    """§23.341: Discrete vertical 1-cosine gust.

    w_g(x) = (Ude/2) * (1 - cos(2π·x / (2·H)))
    where H = 12.5 * c̄ is the gust gradient distance.
    """
    Ude_ms = Ude_fps * 0.3048
    H = 12.5 * c_bar_m
    gust_length = 2.0 * H
    t_gust = gust_length / V_tas
    t_end = t_gust + 2.0

    return ManeuverProfile(
        maneuver_type="gust_vert",
        far_section="§23.341",
        t_end=t_end,
        control_func=_ConstantControl(delta_e_trim),
        gust_func=_VerticalGust(Ude_ms, gust_length, V_tas),
        description=f"Vertical gust: Ude={Ude_fps:.0f}fps, "
                    f"H={H:.1f}m, V={V_tas:.1f}m/s",
        delta_e_trim=delta_e_trim,
    )


def discrete_gust_lateral(
    Ude_fps: float,
    c_bar_m: float,
    V_tas: float,
    delta_e_trim: float = 0.0,
) -> ManeuverProfile:
    """Lateral discrete 1-cosine gust for VTP loads.

    Post-gust buffer reduced to 0.5 s to capture the peak sideslip
    response immediately after the gust passage.  The critical VTP
    loads occur during and just after the gust (t ≈ t_gust ± 0.2 s).
    Longer free-response periods allow uncorrected yaw divergence
    to accumulate past the linear aero model validity range (β > 25°).
    """
    Ude_ms = Ude_fps * 0.3048
    H = 12.5 * c_bar_m
    gust_length = 2.0 * H
    t_gust = gust_length / V_tas
    t_end = t_gust + 0.5

    return ManeuverProfile(
        maneuver_type="gust_lat",
        far_section="§23.341",
        t_end=t_end,
        control_func=_ConstantControl(delta_e_trim),
        gust_func=_LateralGust(Ude_ms, gust_length, V_tas),
        description=f"Lateral gust: Ude={Ude_fps:.0f}fps, "
                    f"H={H:.1f}m, V={V_tas:.1f}m/s",
        delta_e_trim=delta_e_trim,
    )


# ---------------------------------------------------------------------------
# Generate all FAR 23 maneuver profiles
# ---------------------------------------------------------------------------

def generate_all_maneuver_profiles(
    config: AircraftConfig,
    wc: WeightCGCondition,
    alt_m: float = 0.0,
    delta_e_trim: float = 0.0,
    derivs=None,
) -> List[Tuple[ManeuverProfile, float, str]]:
    """Generate all FAR 23 maneuver/gust profiles for given flight condition.

    When *derivs* (AeroDerivativeSet) is provided, elevator deflections are
    sized to produce the V-n diagram limit load factor rather than using full
    control travel.  This prevents the 6-DOF simulation from generating load
    factors that exceed the design envelope — the aircraft structure is only
    certified to the V-n limits, so applying larger inputs would be an
    unrealistically severe condition.

    Returns a list of (ManeuverProfile, V_eas, label) tuples.
    """
    sp = config.speeds
    ctrl = config.ctrl_limits
    c_bar_m = config.mean_chord_m
    profiles = []

    de_phys = math.radians(ctrl.elevator_max_deg)   # physical limit

    # Design load factor limits from V-n diagram (§23.337)
    nz_pos = config.nz_max(wc.weight_N)   # e.g. +3.8 g
    nz_neg = config.nz_min(wc.weight_N)   # e.g. -1.52 g

    # Wing area in m² — prefer config value, fall back to derivs
    S_m2 = config.wing_area_m2
    if S_m2 <= 0.01 and derivs is not None:
        S_m2 = derivs.S_ref * 1e-6       # mm² → m²

    maneuver_speeds = {"VA": sp.VA, "VC": sp.VC}
    roll_speeds = {"VA": sp.VA, "VC": sp.VC, "VD": sp.VD}
    yaw_speeds = {"VA": sp.VA, "VC": sp.VC, "VD": sp.VD}

    # --- Helper: elevator deflection for target nz at given speed ----------
    def _de_for_nz(nz_target, V_eas):
        """Return elevator target (rad) that produces nz_target, or
        fall back to physical limit if derivatives are unavailable."""
        if derivs is None:
            return math.copysign(de_phys, nz_target - 1.0)
        return compute_elevator_for_nz(
            nz_target=nz_target,
            V_eas=V_eas,
            weight_N=wc.weight_N,
            S_m2=S_m2,
            CLalpha=derivs.CLalpha,
            CLdelta_e=derivs.CLdelta_e,
            Cmalpha=derivs.Cmalpha,
            Cmdelta_e=derivs.Cmdelta_e,
            de_physical_limit=de_phys,
        )

    # 1. Elevator pull-up / push-over (§23.331(c))
    for sp_label, V_eas in maneuver_speeds.items():
        if V_eas <= 0:
            continue
        de_pos = _de_for_nz(nz_pos, V_eas)   # δe for +nz limit
        de_neg = _de_for_nz(nz_neg, V_eas)   # δe for -nz limit
        profiles.append((
            abrupt_elevator_pullup(de_pos, delta_e_trim),
            V_eas, f"ElevPullUp_{sp_label}_{wc.label}"))
        profiles.append((
            abrupt_elevator_pullup(de_neg, delta_e_trim),
            V_eas, f"ElevPushOver_{sp_label}_{wc.label}"))

    # 2. Checked maneuver (§23.331(c))
    for sp_label, V_eas in maneuver_speeds.items():
        if V_eas <= 0:
            continue
        de_pos = _de_for_nz(nz_pos, V_eas)
        de_neg = _de_for_nz(nz_neg, V_eas)
        profiles.append((
            checked_maneuver(de_pos, de_neg, delta_e_trim),
            V_eas, f"Checked_{sp_label}_{wc.label}"))
        profiles.append((
            checked_maneuver(de_neg, de_pos, delta_e_trim),
            V_eas, f"CheckedRev_{sp_label}_{wc.label}"))

    # 3. Abrupt roll (§23.349) — speed-scheduled per FAR,
    #    additionally limited so wing root nz stays within V-n limit
    for sp_label, V_eas in roll_speeds.items():
        if V_eas <= 0:
            continue
        da_far = ctrl.aileron_at_speed(V_eas, sp.VA, sp.VC, sp.VD)
        if derivs is not None and derivs.CL_aileron_halfwing > 1e-6:
            da_nz = compute_aileron_for_nz(
                nz_limit=nz_pos,
                V_eas=V_eas,
                weight_N=wc.weight_N,
                S_m2=S_m2,
                CL_aileron_halfwing=derivs.CL_aileron_halfwing,
                da_physical_limit=da_far,
            )
            da = min(da_far, da_nz)
        else:
            da = da_far
        profiles.append((
            abrupt_roll(da, delta_e_trim),
            V_eas, f"AbruptRoll+_{sp_label}_{wc.label}"))
        profiles.append((
            abrupt_roll(-da, delta_e_trim),
            V_eas, f"AbruptRoll-_{sp_label}_{wc.label}"))

    # 4. Yaw maneuver (§23.351) — speed-scheduled per FAR
    for sp_label, V_eas in yaw_speeds.items():
        if V_eas <= 0:
            continue
        dr = ctrl.rudder_at_speed(V_eas, sp.VA, sp.VD)
        profiles.append((
            yaw_maneuver(dr, delta_e_trim),
            V_eas, f"Yaw+_{sp_label}_{wc.label}"))
        profiles.append((
            yaw_maneuver(-dr, delta_e_trim),
            V_eas, f"Yaw-_{sp_label}_{wc.label}"))

    # 5. Vertical gusts (§23.341) — altitude-dependent Ude per §23.333(c)
    Ude_VC_alt = gust_Ude_at_altitude(alt_m, config.gust_Ude_VC_fps)
    Ude_VD_alt = gust_Ude_at_altitude(alt_m, config.gust_Ude_VD_fps)

    gust_speeds = {}
    if sp.VB > 0:
        gust_speeds["VB"] = (sp.VB, Ude_VC_alt)
    if sp.VC > 0:
        gust_speeds["VC"] = (sp.VC, Ude_VC_alt)
    if sp.VD > 0:
        gust_speeds["VD"] = (sp.VD, Ude_VD_alt)

    for sp_label, (V_eas, Ude_fps) in gust_speeds.items():
        V_tas = eas_to_tas(V_eas, alt_m)
        profiles.append((
            discrete_gust_vertical(Ude_fps, c_bar_m, V_tas, delta_e_trim),
            V_eas, f"GustVert+_{sp_label}_{wc.label}"))
        profiles.append((
            discrete_gust_vertical(-Ude_fps, c_bar_m, V_tas, delta_e_trim),
            V_eas, f"GustVert-_{sp_label}_{wc.label}"))

    # 6. Lateral gusts — at VC (altitude-adjusted Ude)
    if sp.VC > 0:
        V_tas_vc = eas_to_tas(sp.VC, alt_m)
        Ude_lat = Ude_VC_alt
        profiles.append((
            discrete_gust_lateral(Ude_lat, c_bar_m, V_tas_vc, delta_e_trim),
            sp.VC, f"GustLat+_VC_{wc.label}"))
        profiles.append((
            discrete_gust_lateral(-Ude_lat, c_bar_m, V_tas_vc, delta_e_trim),
            sp.VC, f"GustLat-_VC_{wc.label}"))

    return profiles
