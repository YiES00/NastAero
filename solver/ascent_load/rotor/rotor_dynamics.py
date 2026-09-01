"""Time-varying rotor force models for 6-DOF simulation.

Provides external force callbacks for the 6-DOF integrator that model
rotor forces as a function of time and aircraft state. Used for
dynamic VTOL cases like OEI events, rotor jam, and FCC recovery maneuvers.

Thrust convention (Newton's 3rd law):
  shaft_axis points in the direction the rotor accelerates air (e.g. [0,0,1]
  pushes air downward in NED body axes). The reaction force on the aircraft
  is F = -T × shaft_axis (upward for hover). This gives:
    dw/dt = gz + F_ext_z/m = g - T/m = 0  in hover (correct balance).

Recovery model phases:
  1. Pre-failure: all rotors at nominal thrust T = W/N
  2. Recognition delay: failed rotor off, others unchanged (diverging)
  3. Recovery ramp: linear ramp from uncompensated to closed-loop control
  4. Steady recovery: PD attitude controller + thrust redistribution

The attitude controller uses a pseudo-inverse control allocator:
  [T_total, M_roll, M_pitch] = B @ T_vec  =>  T_vec = B_pinv @ cmd
where B encodes each rotor's contribution to total thrust, roll moment,
and pitch moment from CG. Torque is estimated via momentum-theory
scaling Q ∝ T^1.5.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .bemt_solver import BEMTSolver, RotorLoads
from .rotor_config import RotorDef, VTOLConfig, RotationDir
from .rotor_loads_applicator import rotor_loads_to_nodal_forces


def make_oei_force_func(vtol_config: VTOLConfig,
                         failed_rotor_id: int,
                         failure_time: float,
                         weight_N: float,
                         rho: float,
                         cg_position: Optional[np.ndarray] = None,
                         ) -> callable:
    """Create external force callback for OEI simulation.

    Before failure_time: all hover-capable rotors operative, balanced hover.
    After failure_time: failed rotor thrust drops to zero, remaining
    rotors maintain their pre-failure thrust (no immediate compensation).

    Supports LIFT, TILT, and mixed configurations. For TILT rotors,
    uses effective_shaft_axis to account for tilt angle.

    The moment contribution includes both:
    1. Torque about the shaft axis (reaction torque)
    2. Moment arm contribution: r x F where r is the hub position
       relative to CG. This is the primary source of rolling/pitching
       moment from asymmetric thrust loss.

    Parameters
    ----------
    vtol_config : VTOLConfig
        VTOL configuration.
    failed_rotor_id : int
        ID of the rotor that fails.
    failure_time : float
        Time of failure event (s).
    weight_N : float
        Aircraft weight (N) for thrust computation.
    rho : float
        Air density (kg/m^3).
    cg_position : ndarray (3,) or None
        CG position in model coordinates (mm). If None, uses [0,0,0].

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    hover_rotors = vtol_config.hover_rotors
    n_rotors = len(hover_rotors)
    thrust_per_rotor = weight_N / n_rotors if n_rotors > 0 else 0.0

    # CG position (convert mm -> m)
    mm_to_m = 1e-3
    cg = cg_position * mm_to_m if cg_position is not None else np.zeros(3)

    # Pre-compute nominal rotor forces and hub positions (m) from CG
    nominal_loads: Dict[int, RotorLoads] = {}
    hub_arms: Dict[int, np.ndarray] = {}
    shaft_axes: Dict[int, np.ndarray] = {}
    for rotor in hover_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(
            thrust_per_rotor, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads
        hub_arms[rotor.rotor_id] = rotor.hub_position * mm_to_m - cg
        shaft_axes[rotor.rotor_id] = rotor.effective_shaft_axis

    def oei_force_func(t: float, y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """OEI external force in body axes."""
        F_total = np.zeros(3)
        M_total = np.zeros(3)

        for rotor in hover_rotors:
            if t >= failure_time and rotor.rotor_id == failed_rotor_id:
                continue  # Failed rotor produces no force

            loads = nominal_loads.get(rotor.rotor_id)
            if loads is None:
                continue

            shaft = shaft_axes[rotor.rotor_id]

            # Reaction force on aircraft (opposes air acceleration)
            F_thrust = -loads.thrust * shaft
            F_total += F_thrust

            # Moment from off-CG thrust application: M = r x F
            r_hub = hub_arms[rotor.rotor_id]
            M_total += np.cross(r_hub, F_thrust)

            # Torque about shaft (reaction torque on airframe)
            torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                           else -1.0)
            M_total += -torque_sign * loads.torque * shaft

        return F_total, M_total

    return oei_force_func


def make_rotor_jam_force_func(vtol_config: VTOLConfig,
                               jammed_rotor_id: int,
                               jam_time: float,
                               weight_N: float,
                               rho: float,
                               cg_position: Optional[np.ndarray] = None,
                               ) -> callable:
    """Create external force callback for rotor jam/seizure.

    Before jam_time: all hover-capable rotors operative.
    At jam_time: one rotor suddenly stops — generates a large
    asymmetric drag torque as the windmilling rotor decelerates,
    plus loss of thrust creates rolling/pitching moment from CG offset.

    Supports LIFT, TILT, and mixed configurations.

    Parameters
    ----------
    vtol_config : VTOLConfig
        VTOL configuration.
    jammed_rotor_id : int
        ID of the rotor that jams.
    jam_time : float
        Time of jam event (s).
    weight_N : float
        Aircraft weight (N).
    rho : float
        Air density (kg/m^3).
    cg_position : ndarray (3,) or None
        CG position in model coordinates (mm). If None, uses [0,0,0].

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    hover_rotors = vtol_config.hover_rotors
    n_rotors = len(hover_rotors)
    thrust_per_rotor = weight_N / n_rotors if n_rotors > 0 else 0.0

    # CG position (convert mm -> m)
    mm_to_m = 1e-3
    cg = cg_position * mm_to_m if cg_position is not None else np.zeros(3)

    # Pre-compute nominal loads and hub positions
    nominal_loads: Dict[int, RotorLoads] = {}
    hub_arms: Dict[int, np.ndarray] = {}
    shaft_axes: Dict[int, np.ndarray] = {}
    for rotor in hover_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(
            thrust_per_rotor, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads
        hub_arms[rotor.rotor_id] = rotor.hub_position * mm_to_m - cg
        shaft_axes[rotor.rotor_id] = rotor.effective_shaft_axis

    # Jam torque: sudden brake torque (decelerating rotor)
    jammed_rotor = vtol_config.get_rotor(jammed_rotor_id)
    if jammed_rotor and jammed_rotor_id in nominal_loads:
        jam_torque = nominal_loads[jammed_rotor_id].torque * 3.0  # Impulse
    else:
        jam_torque = 0.0

    def jam_force_func(t: float, y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotor jam external force in body axes."""
        F_total = np.zeros(3)
        M_total = np.zeros(3)

        for rotor in hover_rotors:
            loads = nominal_loads.get(rotor.rotor_id)
            if loads is None:
                continue

            shaft = shaft_axes[rotor.rotor_id]
            r_hub = hub_arms[rotor.rotor_id]

            if t >= jam_time and rotor.rotor_id == jammed_rotor_id:
                # Jammed rotor: no thrust, large brake torque decaying
                dt_since_jam = t - jam_time
                decay = np.exp(-dt_since_jam / 0.2)
                torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                               else -1.0)
                M_total += -torque_sign * jam_torque * decay * shaft
            else:
                # Normal operation: reaction force + moment arm + torque
                F_thrust = -loads.thrust * shaft
                F_total += F_thrust
                M_total += np.cross(r_hub, F_thrust)
                torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                               else -1.0)
                M_total += -torque_sign * loads.torque * shaft

        return F_total, M_total

    return jam_force_func


# ---------------------------------------------------------------------------
# Recovery control helpers
# ---------------------------------------------------------------------------

def _precompute_rotor_data(hover_rotors: List[RotorDef],
                           weight_N: float, rho: float,
                           cg_position: Optional[np.ndarray],
                           ) -> Tuple[Dict, Dict, Dict, Dict, float]:
    """Pre-compute BEMT loads and geometry for all hover rotors.

    Returns (nominal_loads, hub_arms, shaft_axes, rot_signs, T_nominal).
    """
    mm_to_m = 1e-3
    cg = cg_position * mm_to_m if cg_position is not None else np.zeros(3)
    n_rotors = len(hover_rotors)
    T_nominal = weight_N / n_rotors if n_rotors > 0 else 0.0

    nominal_loads: Dict[int, RotorLoads] = {}
    hub_arms: Dict[int, np.ndarray] = {}
    shaft_axes: Dict[int, np.ndarray] = {}
    rot_signs: Dict[int, float] = {}

    for rotor in hover_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(T_nominal, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads
        hub_arms[rotor.rotor_id] = rotor.hub_position * mm_to_m - cg
        shaft_axes[rotor.rotor_id] = rotor.effective_shaft_axis
        rot_signs[rotor.rotor_id] = (
            1.0 if rotor.rotation_dir == RotationDir.CW else -1.0)

    return nominal_loads, hub_arms, shaft_axes, rot_signs, T_nominal


def _build_allocation_matrix(active_rotors: List[RotorDef],
                              hub_arms: Dict[int, np.ndarray],
                              shaft_axes: Dict[int, np.ndarray],
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Build control allocation matrix and its pseudo-inverse.

    Maps individual rotor thrusts to [total_thrust, roll_moment, pitch_moment]:
        cmd = B @ T_vec   =>   T_vec = B_pinv @ cmd

    Uses reaction-force convention: F = -T × shaft, so the moment per
    unit thrust is  m_j = r_j × (-shaft_j) = -(r_j × shaft_j).

    Parameters
    ----------
    active_rotors : list of RotorDef
        Rotors still operative (excluding failed one).
    hub_arms : dict
        {rotor_id: position_vector_from_CG (m)}.
    shaft_axes : dict
        {rotor_id: unit shaft axis (air-acceleration direction)}.

    Returns
    -------
    B : ndarray (3, n_active)
    B_pinv : ndarray (n_active, 3)
        Minimum-norm pseudo-inverse.
    """
    n = len(active_rotors)
    B = np.zeros((3, n))
    for j, rotor in enumerate(active_rotors):
        shaft = shaft_axes[rotor.rotor_id]
        arm = hub_arms[rotor.rotor_id]
        B[0, j] = 1.0  # each rotor contributes T to total upward thrust
        # Moment from reaction force: M = r × (-T·shaft) per unit T
        cross_j = np.cross(arm, shaft)
        B[1, j] = -cross_j[0]   # roll moment (Mx)
        B[2, j] = -cross_j[1]   # pitch moment (My)

    # Pseudo-inverse: B_pinv = B^T (B B^T)^{-1}
    BtB = B @ B.T
    B_pinv = B.T @ np.linalg.inv(BtB)
    return B, B_pinv


def _apply_rotor_forces(active_rotors, T_vec, T_nominal,
                         nominal_loads, hub_arms, shaft_axes, rot_signs):
    """Sum rotor forces/moments for a given thrust vector.

    Uses reaction-force convention: F = -T × shaft (opposes air acceleration).
    Torque estimated via momentum-theory scaling: Q = Q_nom × (T/T_nom)^1.5

    Returns (F_total(3), M_total(3)).
    """
    F_total = np.zeros(3)
    M_total = np.zeros(3)

    for j, rotor in enumerate(active_rotors):
        rid = rotor.rotor_id
        shaft = shaft_axes[rid]
        T_i = T_vec[j]

        # Reaction force on aircraft (opposes air acceleration)
        F_thrust = -T_i * shaft
        F_total += F_thrust

        # Moment from CG offset: M = r × F
        M_total += np.cross(hub_arms[rid], F_thrust)

        # Reaction torque (momentum-theory scaling Q ∝ T^1.5)
        if T_nominal > 0 and nominal_loads[rid].torque > 0:
            T_ratio = T_i / T_nominal
            Q_i = nominal_loads[rid].torque * abs(T_ratio) ** 1.5
        else:
            Q_i = 0.0
        M_total += -rot_signs[rid] * Q_i * shaft

    return F_total, M_total


# ---------------------------------------------------------------------------
# OEI with recovery
# ---------------------------------------------------------------------------

def make_oei_recovery_force_func(
    vtol_config: VTOLConfig,
    failed_rotor_id: int,
    failure_time: float,
    weight_N: float,
    rho: float,
    cg_position: Optional[np.ndarray] = None,
    t_recognition: float = 1.0,
    t_ramp: float = 0.5,
    Ixx: float = 8000.0,
    Iyy: float = 4000.0,
    omega_att: float = 2.0,
    zeta_att: float = 0.7,
) -> callable:
    """Create OEI force callback with FCC recovery control.

    Four-phase simulation:
      Phase 1 (t < failure_time): all rotors at nominal T = W/N.
      Phase 2 (failure_time ≤ t < t_rec_start): failed rotor off,
          remaining rotors at original thrust (uncompensated, aircraft diverges).
      Phase 3 (t_rec_start ≤ t < t_rec_end): linear ramp from
          uncompensated to closed-loop recovery thrust.
      Phase 4 (t ≥ t_rec_end): PD attitude controller + pseudo-inverse
          thrust allocation for moment-balanced hover recovery.

    Parameters
    ----------
    vtol_config : VTOLConfig
    failed_rotor_id : int
    failure_time : float
    weight_N : float
    rho : float
    cg_position : ndarray(3,) or None  (mm)
    t_recognition : float
        FCC/pilot recognition delay (s). Default 1.0 s (FAR 29.903).
    t_ramp : float
        Thrust redistribution ramp time (s). Default 0.5 s.
    Ixx, Iyy : float
        Roll/pitch inertia (kg·m²) for gain computation.
    omega_att : float
        Attitude controller natural frequency (rad/s). Default 2.0.
    zeta_att : float
        Attitude controller damping ratio. Default 0.7.

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    hover_rotors = vtol_config.hover_rotors
    (nominal_loads, hub_arms, shaft_axes,
     rot_signs, T_nominal) = _precompute_rotor_data(
        hover_rotors, weight_N, rho, cg_position)

    # Phase boundaries
    t_rec_start = failure_time + t_recognition
    t_rec_end = t_rec_start + t_ramp

    # Active rotors (excluding failed)
    active_rotors = [r for r in hover_rotors
                     if r.rotor_id != failed_rotor_id]
    n_active = len(active_rotors)

    # Control allocation matrix
    _, B_pinv = _build_allocation_matrix(
        active_rotors, hub_arms, shaft_axes)

    # Thrust limits per rotor
    T_max = T_nominal * 2.0
    T_min = 0.0

    # Uncompensated thrust vector (all remaining at original nominal)
    T_uncomp = np.full(n_active, T_nominal)

    # PD attitude controller gains
    # Kp = I × ω_n², Kd = 2 × ζ × ω_n × I
    Kp_roll = Ixx * omega_att ** 2
    Kd_roll = 2.0 * zeta_att * omega_att * Ixx
    Kp_pitch = Iyy * omega_att ** 2
    Kd_pitch = 2.0 * zeta_att * omega_att * Iyy
    def oei_recovery_func(t: float, y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        F_total = np.zeros(3)
        M_total = np.zeros(3)

        if t < failure_time:
            # Phase 1: all rotors nominal
            for rotor in hover_rotors:
                rid = rotor.rotor_id
                loads = nominal_loads[rid]
                shaft = shaft_axes[rid]
                F_thrust = -loads.thrust * shaft
                F_total += F_thrust
                M_total += np.cross(hub_arms[rid], F_thrust)
                M_total += -rot_signs[rid] * loads.torque * shaft
            return F_total, M_total

        elif t < t_rec_start:
            # Phase 2: failed rotor off, others at nominal (diverging)
            T_vec = T_uncomp
            return _apply_rotor_forces(
                active_rotors, T_vec, T_nominal,
                nominal_loads, hub_arms, shaft_axes, rot_signs)

        else:
            # Phase 3 & 4: recovery controller
            phi, theta = y[6], y[7]
            p, q = y[3], y[4]

            # Desired commands — thrust set to weight/(n-1) to compensate
            # for lost rotor. No heave feedback (avoids positive-feedback
            # saturation under z-down NED convention).
            M_roll_cmd = -Kp_roll * phi - Kd_roll * p
            M_pitch_cmd = -Kp_pitch * theta - Kd_pitch * q
            T_total_cmd = weight_N

            cmd = np.array([T_total_cmd, M_roll_cmd, M_pitch_cmd])
            T_recovery = B_pinv @ cmd
            T_recovery = np.clip(T_recovery, T_min, T_max)

            # Phase 3: ramp from uncompensated to recovery
            if t < t_rec_end:
                alpha = (t - t_rec_start) / t_ramp
                T_vec = T_uncomp * (1.0 - alpha) + T_recovery * alpha
            else:
                T_vec = T_recovery

            return _apply_rotor_forces(
                active_rotors, T_vec, T_nominal,
                nominal_loads, hub_arms, shaft_axes, rot_signs)

    return oei_recovery_func


# ---------------------------------------------------------------------------
# Rotor jam with recovery
# ---------------------------------------------------------------------------

def make_jam_recovery_force_func(
    vtol_config: VTOLConfig,
    jammed_rotor_id: int,
    jam_time: float,
    weight_N: float,
    rho: float,
    cg_position: Optional[np.ndarray] = None,
    t_recognition: float = 1.0,
    t_ramp: float = 0.5,
    Ixx: float = 8000.0,
    Iyy: float = 4000.0,
    omega_att: float = 2.0,
    zeta_att: float = 0.7,
) -> callable:
    """Create rotor-jam force callback with FCC recovery control.

    Same 4-phase structure as OEI recovery, but with an additional
    brake-torque impulse at jam onset (3× nominal torque, τ=0.2 s decay).

    Parameters
    ----------
    vtol_config : VTOLConfig
    jammed_rotor_id : int
    jam_time : float
    weight_N : float
    rho : float
    cg_position : ndarray(3,) or None  (mm)
    t_recognition : float
        FCC recognition delay (s). Default 1.0 s.
    t_ramp : float
        Thrust redistribution ramp time (s). Default 0.5 s.
    Ixx, Iyy : float
        Roll/pitch inertia (kg·m²) for gain computation.
    omega_att : float
        Attitude controller bandwidth (rad/s). Default 2.0.
    zeta_att : float
        Damping ratio. Default 0.7.

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    hover_rotors = vtol_config.hover_rotors
    (nominal_loads, hub_arms, shaft_axes,
     rot_signs, T_nominal) = _precompute_rotor_data(
        hover_rotors, weight_N, rho, cg_position)

    # Jam parameters
    jammed_rotor = vtol_config.get_rotor(jammed_rotor_id)
    jam_torque = 0.0
    if jammed_rotor and jammed_rotor_id in nominal_loads:
        jam_torque = nominal_loads[jammed_rotor_id].torque * 3.0
    jam_tau = 0.2  # brake-torque decay time constant (s)

    # Phase boundaries
    t_rec_start = jam_time + t_recognition
    t_rec_end = t_rec_start + t_ramp

    # Active rotors
    active_rotors = [r for r in hover_rotors
                     if r.rotor_id != jammed_rotor_id]
    n_active = len(active_rotors)

    # Allocation
    _, B_pinv = _build_allocation_matrix(
        active_rotors, hub_arms, shaft_axes)

    T_max = T_nominal * 2.0
    T_min = 0.0
    T_uncomp = np.full(n_active, T_nominal)

    # PD gains
    Kp_roll = Ixx * omega_att ** 2
    Kd_roll = 2.0 * zeta_att * omega_att * Ixx
    Kp_pitch = Iyy * omega_att ** 2
    Kd_pitch = 2.0 * zeta_att * omega_att * Iyy

    def jam_recovery_func(t: float, y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        F_total = np.zeros(3)
        M_total = np.zeros(3)

        # Brake torque on jammed rotor (additive, decaying)
        if t >= jam_time and jammed_rotor is not None:
            dt_jam = t - jam_time
            decay = np.exp(-dt_jam / jam_tau)
            shaft_j = shaft_axes[jammed_rotor_id]
            M_total += -rot_signs[jammed_rotor_id] * jam_torque * decay * shaft_j

        if t < jam_time:
            # Phase 1: all rotors nominal
            for rotor in hover_rotors:
                rid = rotor.rotor_id
                loads = nominal_loads[rid]
                shaft = shaft_axes[rid]
                F_thrust = -loads.thrust * shaft
                F_total += F_thrust
                M_total += np.cross(hub_arms[rid], F_thrust)
                M_total += -rot_signs[rid] * loads.torque * shaft
            return F_total, M_total

        elif t < t_rec_start:
            # Phase 2: jammed rotor off, others at nominal
            T_vec = T_uncomp
            F_act, M_act = _apply_rotor_forces(
                active_rotors, T_vec, T_nominal,
                nominal_loads, hub_arms, shaft_axes, rot_signs)
            F_total += F_act
            M_total += M_act
            return F_total, M_total

        else:
            # Phase 3 & 4: recovery controller
            phi, theta = y[6], y[7]
            p, q = y[3], y[4]

            M_roll_cmd = -Kp_roll * phi - Kd_roll * p
            M_pitch_cmd = -Kp_pitch * theta - Kd_pitch * q
            T_total_cmd = weight_N

            cmd = np.array([T_total_cmd, M_roll_cmd, M_pitch_cmd])
            T_recovery = B_pinv @ cmd
            T_recovery = np.clip(T_recovery, T_min, T_max)

            if t < t_rec_end:
                alpha = (t - t_rec_start) / t_ramp
                T_vec = T_uncomp * (1.0 - alpha) + T_recovery * alpha
            else:
                T_vec = T_recovery

            F_act, M_act = _apply_rotor_forces(
                active_rotors, T_vec, T_nominal,
                nominal_loads, hub_arms, shaft_axes, rot_signs)
            F_total += F_act
            M_total += M_act
            return F_total, M_total

    return jam_recovery_func


# ---------------------------------------------------------------------------
# Post-processing: extract per-rotor thrust schedule from simulation history
# ---------------------------------------------------------------------------

def compute_recovery_thrust_schedule(
    vtol_config: VTOLConfig,
    failed_rotor_id: int,
    failure_time: float,
    weight_N: float,
    rho: float,
    cg_position: Optional[np.ndarray],
    t_arr: np.ndarray,
    states: np.ndarray,
    t_recognition: float = 1.0,
    t_ramp: float = 0.5,
    Ixx: float = 8000.0,
    Iyy: float = 4000.0,
    omega_att: float = 2.0,
    zeta_att: float = 0.7,
) -> Dict[str, np.ndarray]:
    """Re-compute per-rotor thrust schedule from simulation state history.

    Replicates the recovery controller logic to extract the commanded
    thrust for each rotor at every time step. Used for plotting.

    Parameters
    ----------
    vtol_config : VTOLConfig
    failed_rotor_id : int
    failure_time : float
    weight_N, rho : float
    cg_position : ndarray(3,) or None  (mm)
    t_arr : ndarray (N,)
        Time array from simulation.
    states : ndarray (N, 12)
        State history from simulation.
    t_recognition, t_ramp : float
    Ixx, Iyy, omega_att, zeta_att : float

    Returns
    -------
    dict
        {rotor_label: thrust_array(N)} for all hover rotors.
    """
    hover_rotors = vtol_config.hover_rotors
    mm_to_m = 1e-3
    cg = cg_position * mm_to_m if cg_position is not None else np.zeros(3)

    n_all = len(hover_rotors)
    T_nominal = weight_N / n_all if n_all > 0 else 0.0

    # Hub arms and shaft axes
    hub_arms: Dict[int, np.ndarray] = {}
    shaft_axes_local: Dict[int, np.ndarray] = {}
    for rotor in hover_rotors:
        hub_arms[rotor.rotor_id] = rotor.hub_position * mm_to_m - cg
        shaft_axes_local[rotor.rotor_id] = rotor.effective_shaft_axis

    # Active rotors
    active_rotors = [r for r in hover_rotors
                     if r.rotor_id != failed_rotor_id]
    n_active = len(active_rotors)

    # Allocation matrix
    _, B_pinv = _build_allocation_matrix(
        active_rotors, hub_arms, shaft_axes_local)

    T_max = T_nominal * 2.0
    T_uncomp = np.full(n_active, T_nominal)

    # PD gains
    Kp_roll = Ixx * omega_att ** 2
    Kd_roll = 2.0 * zeta_att * omega_att * Ixx
    Kp_pitch = Iyy * omega_att ** 2
    Kd_pitch = 2.0 * zeta_att * omega_att * Iyy

    # Phase boundaries
    t_rec_start = failure_time + t_recognition
    t_rec_end = t_rec_start + t_ramp

    # Output: per-rotor thrust schedule
    N = len(t_arr)
    thrusts: Dict[str, np.ndarray] = {}
    for rotor in hover_rotors:
        thrusts[rotor.label] = np.zeros(N)

    # Active rotor index map
    active_id_to_idx = {r.rotor_id: j for j, r in enumerate(active_rotors)}

    for i in range(N):
        t = t_arr[i]

        if t < failure_time:
            # All rotors at nominal
            for rotor in hover_rotors:
                thrusts[rotor.label][i] = T_nominal

        elif t < t_rec_start:
            # Failed rotor off, others at nominal
            for rotor in hover_rotors:
                if rotor.rotor_id == failed_rotor_id:
                    thrusts[rotor.label][i] = 0.0
                else:
                    thrusts[rotor.label][i] = T_nominal

        else:
            # Recovery controller
            phi = states[i, 6]
            theta = states[i, 7]
            p = states[i, 3]
            q = states[i, 4]

            M_roll_cmd = -Kp_roll * phi - Kd_roll * p
            M_pitch_cmd = -Kp_pitch * theta - Kd_pitch * q
            T_total_cmd = weight_N

            cmd = np.array([T_total_cmd, M_roll_cmd, M_pitch_cmd])
            T_vec = B_pinv @ cmd
            T_vec = np.clip(T_vec, 0.0, T_max)

            # Ramp during phase 3
            if t < t_rec_end:
                alpha = (t - t_rec_start) / t_ramp
                T_vec = T_uncomp * (1.0 - alpha) + T_vec * alpha

            # Store
            for rotor in hover_rotors:
                if rotor.rotor_id == failed_rotor_id:
                    thrusts[rotor.label][i] = 0.0
                else:
                    j = active_id_to_idx[rotor.rotor_id]
                    thrusts[rotor.label][i] = T_vec[j]

    return thrusts
