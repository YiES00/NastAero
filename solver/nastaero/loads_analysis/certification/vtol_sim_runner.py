"""VTOL dynamic simulation runner.

Extends the SimRunner for VTOL-specific dynamic cases (OEI, rotor jam)
by injecting rotor force callbacks into the 6-DOF integrator.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from .flight_sim import (
    AircraftParams, AircraftState, ControlInput, SimTimeHistory,
    integrate_6dof, trim_initial_state, compute_nz_from_history,
)
from .sim_runner import CriticalTimePoint
from .vtol_conditions import VTOLCondition, VTOLFlightPhase
from ...rotor.rotor_config import VTOLConfig
from ...rotor.rotor_dynamics import make_oei_force_func, make_rotor_jam_force_func
from ..case_generator import isa_atmosphere

logger = logging.getLogger(__name__)


class VTOLSimRunner:
    """Runner for VTOL dynamic simulation cases.

    Handles OEI events and rotor jam scenarios using the 6-DOF
    integrator with external force callbacks.

    Parameters
    ----------
    params : AircraftParams
        Aircraft parameters (mass, inertia, aero derivatives).
    vtol_config : VTOLConfig
        VTOL rotor configuration.
    """

    def __init__(self, params: AircraftParams, vtol_config: VTOLConfig,
                 cg_position: Optional[np.ndarray] = None):
        self.params = params
        self.vtol_config = vtol_config
        self.cg_position = cg_position  # Model coords (mm)

    def run_oei_case(self, condition: VTOLCondition,
                      weight_N: float,
                      t_sim: float = 3.0,
                      failure_time: float = 1.0,
                      dt: float = 0.005,
                      ) -> Tuple[SimTimeHistory, List[CriticalTimePoint]]:
        """Run OEI dynamic simulation.

        Simulates steady hover → rotor failure → transient response.
        Extracts critical time points (max nz, max rates).

        Parameters
        ----------
        condition : VTOLCondition
            VTOL OEI condition.
        weight_N : float
            Aircraft weight (N).
        t_sim : float
            Total simulation time (s).
        failure_time : float
            Time of failure event (s).
        dt : float
            Integration timestep (s).

        Returns
        -------
        (SimTimeHistory, list of CriticalTimePoint)
        """
        rho, _, _ = isa_atmosphere(condition.altitude_m)
        self.params.rho = rho

        # Initial trim at hover (very low speed)
        V_hover = 5.0  # Use small V for initial trim (avoid div/0)
        initial_state, de_trim = trim_initial_state(
            self.params, V_hover, nz=1.0)

        # Override for hover: mostly stationary
        initial_state.u = 0.1  # Near-zero forward speed
        initial_state.w = 0.0
        initial_state.theta = 0.0

        # Control: hold trim throughout
        ctrl_trim = ControlInput(delta_e=de_trim)
        control_func = lambda t: ctrl_trim

        # External force: OEI callback
        ext_force = make_oei_force_func(
            self.vtol_config,
            failed_rotor_id=condition.failed_rotor_id,
            failure_time=failure_time,
            weight_N=weight_N,
            rho=rho,
            cg_position=self.cg_position,
        )

        # Run simulation
        history = integrate_6dof(
            self.params, initial_state, control_func,
            t_span=(0.0, t_sim), dt=dt,
            external_force_func=ext_force,
        )

        # Recompute load factors
        compute_nz_from_history(self.params, history)

        # Extract critical points
        criticals = self._extract_critical_points(
            history, condition, weight_N)

        logger.info(f"OEI sim: {condition.label}, "
                    f"{len(criticals)} critical points")

        return history, criticals

    def run_rotor_jam_case(self, condition: VTOLCondition,
                            weight_N: float,
                            t_sim: float = 2.0,
                            jam_time: float = 0.5,
                            dt: float = 0.005,
                            ) -> Tuple[SimTimeHistory, List[CriticalTimePoint]]:
        """Run rotor jam/seizure dynamic simulation.

        Parameters
        ----------
        condition : VTOLCondition
            Rotor jam condition.
        weight_N : float
            Aircraft weight (N).
        t_sim : float
            Total simulation time (s).
        jam_time : float
            Time of jam event (s).
        dt : float
            Timestep (s).

        Returns
        -------
        (SimTimeHistory, list of CriticalTimePoint)
        """
        rho, _, _ = isa_atmosphere(condition.altitude_m)
        self.params.rho = rho

        V_hover = 5.0
        initial_state, de_trim = trim_initial_state(
            self.params, V_hover, nz=1.0)
        initial_state.u = 0.1
        initial_state.w = 0.0
        initial_state.theta = 0.0

        ctrl_trim = ControlInput(delta_e=de_trim)
        control_func = lambda t: ctrl_trim

        ext_force = make_rotor_jam_force_func(
            self.vtol_config,
            jammed_rotor_id=condition.failed_rotor_id,
            jam_time=jam_time,
            weight_N=weight_N,
            rho=rho,
            cg_position=self.cg_position,
        )

        history = integrate_6dof(
            self.params, initial_state, control_func,
            t_span=(0.0, t_sim), dt=dt,
            external_force_func=ext_force,
        )

        compute_nz_from_history(self.params, history)

        criticals = self._extract_critical_points(
            history, condition, weight_N)

        logger.info(f"Rotor jam sim: {condition.label}, "
                    f"{len(criticals)} critical points")

        return history, criticals

    def _extract_critical_points(self, history: SimTimeHistory,
                                  condition: VTOLCondition,
                                  weight_N: float,
                                  ) -> List[CriticalTimePoint]:
        """Extract critical time points from simulation history.

        Finds time points with maximum/minimum load factors, angular
        rates, and angular accelerations.

        Parameters
        ----------
        history : SimTimeHistory
            Simulation results.
        condition : VTOLCondition
            Associated condition.
        weight_N : float
            Aircraft weight (N).

        Returns
        -------
        list of CriticalTimePoint
        """
        criticals = []
        N = len(history.t)
        if N < 2:
            return criticals

        # Max/min nz
        idx_nz_max = np.argmax(history.nz)
        idx_nz_min = np.argmin(history.nz)

        for idx, label_suffix in [(idx_nz_max, "nz_max"),
                                   (idx_nz_min, "nz_min")]:
            criticals.append(CriticalTimePoint(
                t=float(history.t[idx]),
                nz=float(history.nz[idx]),
                ny=float(history.ny[idx]) if len(history.ny) > idx else 0.0,
                alpha_deg=float(history.alpha_deg[idx]),
                beta_deg=float(history.beta_deg[idx]),
                p=float(history.p_rate[idx]),
                q=float(history.q_rate[idx]),
                r=float(history.r_rate[idx]),
                p_dot=float(history.p_dot[idx]),
                q_dot=float(history.q_dot[idx]),
                r_dot=float(history.r_dot[idx]),
                reason=f"VTOL_{condition.label}_{label_suffix}",
            ))

        # Max roll rate (asymmetric OEI/jam)
        idx_p_max = np.argmax(np.abs(history.p_rate))
        if abs(history.p_rate[idx_p_max]) > 0.01:
            criticals.append(CriticalTimePoint(
                t=float(history.t[idx_p_max]),
                nz=float(history.nz[idx_p_max]),
                ny=float(history.ny[idx_p_max]) if len(history.ny) > idx_p_max else 0.0,
                alpha_deg=float(history.alpha_deg[idx_p_max]),
                beta_deg=float(history.beta_deg[idx_p_max]),
                p=float(history.p_rate[idx_p_max]),
                q=float(history.q_rate[idx_p_max]),
                r=float(history.r_rate[idx_p_max]),
                p_dot=float(history.p_dot[idx_p_max]),
                q_dot=float(history.q_dot[idx_p_max]),
                r_dot=float(history.r_dot[idx_p_max]),
                reason=f"VTOL_{condition.label}_p_max",
            ))

        # Max yaw rate (rotor torque asymmetry)
        idx_r_max = np.argmax(np.abs(history.r_rate))
        if abs(history.r_rate[idx_r_max]) > 0.01:
            criticals.append(CriticalTimePoint(
                t=float(history.t[idx_r_max]),
                nz=float(history.nz[idx_r_max]),
                ny=float(history.ny[idx_r_max]) if len(history.ny) > idx_r_max else 0.0,
                alpha_deg=float(history.alpha_deg[idx_r_max]),
                beta_deg=float(history.beta_deg[idx_r_max]),
                p=float(history.p_rate[idx_r_max]),
                q=float(history.q_rate[idx_r_max]),
                r=float(history.r_rate[idx_r_max]),
                p_dot=float(history.p_dot[idx_r_max]),
                q_dot=float(history.q_dot[idx_r_max]),
                r_dot=float(history.r_dot[idx_r_max]),
                reason=f"VTOL_{condition.label}_r_max",
            ))

        return criticals
