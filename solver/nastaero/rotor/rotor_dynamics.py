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
                         ) -> callable:
    """Create external force callback for OEI simulation.

    Before failure_time: all rotors operative, balanced hover forces.
    After failure_time: failed rotor thrust drops to zero, remaining
    rotors maintain their pre-failure thrust (no immediate compensation).

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

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    lift_rotors = vtol_config.lift_rotors
    n_rotors = len(lift_rotors)
    thrust_per_rotor = weight_N / n_rotors if n_rotors > 0 else 0.0

    # Pre-compute nominal rotor forces for each rotor
    nominal_loads: Dict[int, RotorLoads] = {}
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(
            thrust_per_rotor, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads

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

            # Thrust along shaft
            F_total += loads.thrust * shaft

            # Torque about shaft
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
                               ) -> callable:
    """Create external force callback for rotor jam/seizure.

    Before jam_time: all rotors operative.
    At jam_time: one rotor suddenly stops — generates a large
    asymmetric drag torque as the windmilling rotor decelerates.

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

    Returns
    -------
    callable
        external_force_func(t, y) -> (F_body(3), M_body(3))
    """
    lift_rotors = vtol_config.lift_rotors
    n_rotors = len(lift_rotors)
    thrust_per_rotor = weight_N / n_rotors if n_rotors > 0 else 0.0

    # Pre-compute nominal loads
    nominal_loads: Dict[int, RotorLoads] = {}
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(
            thrust_per_rotor, rotor.rpm_hover, rho)
        nominal_loads[rotor.rotor_id] = loads

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

            if t >= jam_time and rotor.rotor_id == jammed_rotor_id:
                # Jammed rotor: no thrust, large brake torque decaying
                dt_since_jam = t - jam_time
                # Exponential decay of brake torque (τ ~ 0.2s)
                decay = np.exp(-dt_since_jam / 0.2)
                torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                               else -1.0)
                M_total += -torque_sign * jam_torque * decay * shaft
            else:
                # Normal operation
                F_total += loads.thrust * shaft
                torque_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                               else -1.0)
                M_total += -torque_sign * loads.torque * shaft

        return F_total, M_total

    return jam_force_func
