"""Time-varying rotor force models for 6-DOF simulation.

Provides external force callbacks for the 6-DOF integrator that model
rotor forces as a function of time and aircraft state. Used for
dynamic VTOL cases like OEI events and rotor jam.
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

    Before failure_time: all rotors operative, balanced hover forces.
    After failure_time: failed rotor thrust drops to zero, remaining
    rotors maintain their pre-failure thrust (no immediate compensation).

    The moment contribution includes both:
    1. Torque about the shaft axis (reaction torque)
    2. Moment arm contribution: r × F where r is the hub position
       relative to CG. This is the primary source of rolling moment
       from asymmetric thrust loss.

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
        Air density (kg/m³).
    cg_position : ndarray (3,) or None
        CG position in model coordinates (mm). If None, uses [0,0,0].

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    lift_rotors = vtol_config.lift_rotors
    n_rotors = len(lift_rotors)
    thrust_per_rotor = weight_N / n_rotors if n_rotors > 0 else 0.0

    # CG position (convert mm → m)
    mm_to_m = 1e-3
    cg = cg_position * mm_to_m if cg_position is not None else np.zeros(3)

    # Pre-compute nominal rotor forces and hub positions (m) from CG
    nominal_loads: Dict[int, RotorLoads] = {}
    hub_arms: Dict[int, np.ndarray] = {}  # position relative to CG in meters
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(
            thrust_per_rotor, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads
        # Hub position relative to CG (convert mm → m)
        hub_arms[rotor.rotor_id] = rotor.hub_position * mm_to_m - cg

    def oei_force_func(t: float, y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """OEI external force in body axes."""
        F_total = np.zeros(3)
        M_total = np.zeros(3)

        for rotor in lift_rotors:
            if t >= failure_time and rotor.rotor_id == failed_rotor_id:
                continue  # Failed rotor produces no force

            loads = nominal_loads.get(rotor.rotor_id)
            if loads is None:
                continue

            # Transform to body axes
            shaft = rotor.shaft_axis / np.linalg.norm(rotor.shaft_axis)

            # Thrust force along shaft
            F_thrust = loads.thrust * shaft
            F_total += F_thrust

            # Moment from off-CG thrust application: M = r × F
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

    Before jam_time: all rotors operative.
    At jam_time: one rotor suddenly stops — generates a large
    asymmetric drag torque as the windmilling rotor decelerates,
    plus loss of thrust creates rolling moment from CG offset.

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
        Air density (kg/m³).
    cg_position : ndarray (3,) or None
        CG position in model coordinates (mm). If None, uses [0,0,0].

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    lift_rotors = vtol_config.lift_rotors
    n_rotors = len(lift_rotors)
    thrust_per_rotor = weight_N / n_rotors if n_rotors > 0 else 0.0

    # CG position (convert mm → m)
    mm_to_m = 1e-3
    cg = cg_position * mm_to_m if cg_position is not None else np.zeros(3)

    # Pre-compute nominal loads and hub positions
    nominal_loads: Dict[int, RotorLoads] = {}
    hub_arms: Dict[int, np.ndarray] = {}
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(
            thrust_per_rotor, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads
        hub_arms[rotor.rotor_id] = rotor.hub_position * mm_to_m - cg

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

        for rotor in lift_rotors:
            loads = nominal_loads.get(rotor.rotor_id)
            if loads is None:
                continue

            shaft = rotor.shaft_axis / np.linalg.norm(rotor.shaft_axis)
            r_hub = hub_arms[rotor.rotor_id]

            if t >= jam_time and rotor.rotor_id == jammed_rotor_id:
                # Jammed rotor: no thrust, large brake torque decaying
                dt_since_jam = t - jam_time
                # Exponential decay of brake torque (τ ~ 0.2s)
                decay = np.exp(-dt_since_jam / 0.2)
                torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                               else -1.0)
                M_total += -torque_sign * jam_torque * decay * shaft
                # No thrust from jammed rotor → no r×F contribution
            else:
                # Normal operation: thrust + moment arm + reaction torque
                F_thrust = loads.thrust * shaft
                F_total += F_thrust
                M_total += np.cross(r_hub, F_thrust)
                torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                               else -1.0)
                M_total += -torque_sign * loads.torque * shaft

        return F_total, M_total

    return jam_force_func
