"""VTOL transient peak loads from 6-DOF OEI/jam simulations.

Bridges rigid-body flight dynamics (6-DOF OEI/jam sims) to structural
loads by computing time-varying forces at each node during the transient:

1. Per-rotor forces reconstructed from thrust schedule (recovery controller)
2. Inertial forces from nz + angular accelerations at all mass nodes
3. Section loads (V, M, T) at wing/boom roots via tip-to-root integration
4. DAF = peak_transient / quasi_static for certification comparison

References
----------
- EASA SC-VTOL-01: OEI and control failure conditions
- FAR 29.561/29.903: Emergency conditions, one engine inoperative
"""
from __future__ import annotations

import logging
import math
import time as time_mod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .flight_sim import (
    AircraftParams, AircraftState, ControlInput, SimTimeHistory,
    integrate_6dof, trim_initial_state, compute_nz_from_history,
)
from .vtol_sim_runner import VTOLSimRunner
from .vtol_conditions import VTOLCondition, VTOLFlightPhase
from .sim_runner import CriticalTimePoint
from .batch_runner import CaseResult
from .aircraft_config import AircraftConfig, WeightCGCondition
from ..case_generator import isa_atmosphere
from ...rotor.rotor_config import VTOLConfig, RotorDef, RotationDir
from ...rotor.bemt_solver import BEMTSolver, RotorLoads
from ...rotor.rotor_dynamics import (
    _precompute_rotor_data, _build_allocation_matrix,
    compute_recovery_thrust_schedule,
)

logger = logging.getLogger(__name__)

_G = 9.80665        # m/s²
_G_MM = 9806.65     # mm/s² (model units: N-mm-s, mass in tonnes)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class TransientLoadSnapshot:
    """Structural loads at a single time point during transient."""
    time: float = 0.0
    nz: float = 1.0
    phi_deg: float = 0.0
    p_dot: float = 0.0
    q_dot: float = 0.0
    r_dot: float = 0.0

    # Section loads at wing roots [N, N·mm]
    wing_root_shear_L: float = 0.0
    wing_root_bending_L: float = 0.0
    wing_root_shear_R: float = 0.0
    wing_root_bending_R: float = 0.0

    # Boom loads (max across all booms)
    max_boom_bending: float = 0.0
    max_boom_shear: float = 0.0


@dataclass
class TransientPeakResult:
    """Complete transient peak loads for one failure event."""
    event_type: str = ""           # "oei" or "jam"
    failed_rotor_id: int = 0
    failed_rotor_label: str = ""
    with_recovery: bool = False
    t_recognition: float = 0.0

    # 6-DOF time histories (full resolution)
    sim_time: np.ndarray = field(default_factory=lambda: np.array([]))
    nz_history: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_history: np.ndarray = field(default_factory=lambda: np.array([]))

    # Structural load time histories (sampled at dt_loads)
    t_loads: np.ndarray = field(default_factory=lambda: np.array([]))
    wing_Mx_L: np.ndarray = field(default_factory=lambda: np.array([]))
    wing_Mx_R: np.ndarray = field(default_factory=lambda: np.array([]))
    wing_Vy_L: np.ndarray = field(default_factory=lambda: np.array([]))
    wing_Vy_R: np.ndarray = field(default_factory=lambda: np.array([]))
    boom_Mx_max: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-rotor thrust time histories {label: ndarray}
    rotor_thrusts: Dict[str, np.ndarray] = field(default_factory=dict)

    # Peak values
    peak_wing_Mx: float = 0.0          # Absolute peak wing root bending [N·mm]
    peak_wing_Mx_time: float = 0.0
    peak_wing_Mx_side: str = ""
    peak_wing_Vy: float = 0.0
    peak_boom_Mx: float = 0.0
    peak_boom_Mx_time: float = 0.0

    # Quasi-static baseline (steady OEI hover at nz=1.0)
    qs_wing_Mx: float = 0.0
    qs_wing_Vy: float = 0.0
    qs_boom_Mx: float = 0.0

    # DAF = peak_transient / quasi_static
    daf_wing_Mx: float = 1.0
    daf_wing_Vy: float = 1.0
    daf_boom_Mx: float = 1.0

    # Critical time points from 6-DOF sim
    critical_points: List[CriticalTimePoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class VTOLTransientLoadsRunner:
    """Automated pipeline: 6-DOF OEI/jam sim → structural peak loads + DAF.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed FEM model.
    vtol_config : VTOLConfig
        VTOL rotor configuration.
    aircraft_config : AircraftConfig
        Aircraft config (weights, speeds, geometry).
    airfoil_config : optional
        Airfoil camber config for DLM correction.
    n_workers : int
        Parallel workers for batch trim.
    """

    def __init__(self, bdf_model, vtol_config: VTOLConfig,
                 aircraft_config: AircraftConfig,
                 airfoil_config=None, n_workers: int = 0):
        self.bdf_model = bdf_model
        self.vtol_config = vtol_config
        self.aircraft_config = aircraft_config
        self.airfoil_config = airfoil_config
        self.n_workers = n_workers

        # Pre-compute mass distribution and geometry
        self._node_masses = self._build_node_mass_map()
        self._cg_mm = self._compute_cg_mm()
        self._total_mass_tonnes = sum(self._node_masses.values())
        self._wing_root_y = self._detect_wing_root_y()
        self._boom_junctions = self._detect_boom_junctions()

        # Pre-compute nominal rotor loads (for torque scaling)
        wc = aircraft_config.weight_cg_conditions[0]
        self._weight_N = wc.weight_N
        rho, _, _ = isa_atmosphere(0.0)
        self._rho = rho
        self._nominal_loads = {}
        hover_rotors = vtol_config.hover_rotors
        n_hover = len(hover_rotors)
        T_nom = self._weight_N / n_hover if n_hover > 0 else 0.0
        self._T_nominal = T_nom
        for rotor in hover_rotors:
            solver = BEMTSolver(rotor.blade, rotor.n_blades)
            loads = solver.solve_for_thrust(T_nom, rotor.rpm_hover, rho)
            self._nominal_loads[rotor.rotor_id] = loads

        # Build AircraftParams for 6-DOF sim
        self._params = self._build_aircraft_params()

        logger.info("VTOLTransientLoadsRunner: %.0f kg, %d hover rotors, "
                    "wing root Y=%.0f mm, %d boom junctions",
                    self._total_mass_tonnes * 1000,
                    n_hover, self._wing_root_y, len(self._boom_junctions))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_oei(self, t_sim: float = 5.0, dt: float = 0.005,
                    t_recognition_list: Optional[List[float]] = None,
                    dt_loads: float = 0.02,
                    ) -> List[TransientPeakResult]:
        """Run OEI transient loads for all rotors × recovery configs.

        Parameters
        ----------
        t_sim : float
            Total simulation time (s).
        dt : float
            Integration timestep (s).
        t_recognition_list : list of float or None
            FCC recognition delays to evaluate (s).
            Default: [0.15, 0.5, 1.0].
        dt_loads : float
            Loads sampling interval (s).

        Returns
        -------
        list of TransientPeakResult
        """
        if t_recognition_list is None:
            t_recognition_list = [0.15, 0.5, 1.0]

        t0 = time_mod.time()
        results = []
        failables = [r for r in self.vtol_config.hover_rotors if r.can_fail]

        for rotor in failables:
            # Without recovery (worst-case, baseline)
            logger.info("OEI no-recovery: %s (ID %d)", rotor.label, rotor.rotor_id)
            result = self._run_oei_event(
                rotor, with_recovery=False, t_sim=t_sim, dt=dt,
                dt_loads=dt_loads)
            results.append(result)

            # With recovery at various t_rec
            for t_rec in t_recognition_list:
                logger.info("OEI recovery t_rec=%.2fs: %s", t_rec, rotor.label)
                result = self._run_oei_event(
                    rotor, with_recovery=True, t_recognition=t_rec,
                    t_sim=t_sim, dt=dt, dt_loads=dt_loads)
                results.append(result)

        elapsed = time_mod.time() - t0
        logger.info("OEI transient loads: %d events, %.1fs", len(results), elapsed)
        return results

    def run_all_jam(self, t_sim: float = 3.0, dt: float = 0.005,
                    dt_loads: float = 0.02,
                    ) -> List[TransientPeakResult]:
        """Run rotor jam transient loads for all rotors."""
        t0 = time_mod.time()
        results = []
        failables = [r for r in self.vtol_config.hover_rotors if r.can_fail]

        for rotor in failables:
            logger.info("Rotor jam: %s (ID %d)", rotor.label, rotor.rotor_id)
            result = self._run_jam_event(
                rotor, t_sim=t_sim, dt=dt, dt_loads=dt_loads)
            results.append(result)

        elapsed = time_mod.time() - t0
        logger.info("Jam transient loads: %d events, %.1fs", len(results), elapsed)
        return results

    # ------------------------------------------------------------------
    # Core event runners
    # ------------------------------------------------------------------

    def _run_oei_event(self, rotor: RotorDef,
                       with_recovery: bool = False,
                       t_recognition: float = 1.0,
                       t_sim: float = 5.0, dt: float = 0.005,
                       failure_time: float = 1.0,
                       dt_loads: float = 0.02,
                       ) -> TransientPeakResult:
        """Run single OEI event and compute structural loads time history."""
        from ...rotor.rotor_dynamics import (
            make_oei_force_func, make_oei_recovery_force_func,
        )

        condition = VTOLCondition(
            label=f"OEI_{rotor.label}",
            phase=VTOLFlightPhase.OEI,
            V_eas=0.0, nz=1.0,
            altitude_m=0.0,
            failed_rotor_id=rotor.rotor_id,
            far_section="SC-VTOL.2140",
        )

        # Run 6-DOF simulation
        sim_runner = VTOLSimRunner(
            self._params, self.vtol_config, cg_position=self._cg_mm)

        if with_recovery:
            # Create recovery force callback directly
            ext_force = make_oei_recovery_force_func(
                self.vtol_config,
                failed_rotor_id=rotor.rotor_id,
                failure_time=failure_time,
                weight_N=self._weight_N,
                rho=self._rho,
                cg_position=self._cg_mm,
                t_recognition=t_recognition,
                t_ramp=0.5,
                Ixx=self._params.Ixx,
                Iyy=self._params.Iyy,
            )

            # Manual sim (VTOLSimRunner doesn't have recovery variant)
            initial_state, de_trim = trim_initial_state(
                self._params, 5.0, nz=1.0)
            initial_state.u = 0.1
            initial_state.w = 0.0
            initial_state.theta = 0.0

            ctrl = ControlInput(delta_e=de_trim)
            history = integrate_6dof(
                self._params, initial_state, lambda t: ctrl,
                t_span=(0.0, t_sim), dt=dt,
                external_force_func=ext_force,
            )
            compute_nz_from_history(self._params, history)
            criticals = sim_runner._extract_critical_points(
                history, condition, self._weight_N)
        else:
            history, criticals = sim_runner.run_oei_case(
                condition, self._weight_N, t_sim=t_sim,
                failure_time=failure_time, dt=dt)

        # Reconstruct per-rotor thrust schedule
        rotor_thrusts = compute_recovery_thrust_schedule(
            self.vtol_config,
            failed_rotor_id=rotor.rotor_id,
            failure_time=failure_time,
            weight_N=self._weight_N,
            rho=self._rho,
            cg_position=self._cg_mm,
            t_arr=history.t,
            states=history.states,
            t_recognition=t_recognition if with_recovery else 1e6,
            t_ramp=0.5 if with_recovery else 0.0,
            Ixx=self._params.Ixx,
            Iyy=self._params.Iyy,
        )

        # Compute structural loads time history
        result = self._compute_loads_time_history(
            history, rotor_thrusts, rotor.rotor_id,
            dt_loads=dt_loads, failure_time=failure_time)

        # Fill metadata
        result.event_type = "oei"
        result.failed_rotor_id = rotor.rotor_id
        result.failed_rotor_label = rotor.label
        result.with_recovery = with_recovery
        result.t_recognition = t_recognition if with_recovery else float('inf')
        result.sim_time = history.t.copy()
        result.nz_history = history.nz.copy()
        result.phi_history = np.degrees(history.states[:, 6])
        result.rotor_thrusts = rotor_thrusts
        result.critical_points = criticals

        # Compute quasi-static baseline and DAF
        self._compute_quasi_static_and_daf(result, rotor.rotor_id)

        logger.info("  %s %s t_rec=%.2fs: peak Mx=%.0f N·mm (DAF=%.2f), "
                    "peak Vy=%.0f N (DAF=%.2f)",
                    result.event_type, rotor.label, result.t_recognition,
                    result.peak_wing_Mx, result.daf_wing_Mx,
                    result.peak_wing_Vy, result.daf_wing_Vy)

        return result

    def _run_jam_event(self, rotor: RotorDef,
                       t_sim: float = 3.0, dt: float = 0.005,
                       jam_time: float = 0.5,
                       dt_loads: float = 0.02,
                       ) -> TransientPeakResult:
        """Run single rotor jam event and compute structural loads."""
        condition = VTOLCondition(
            label=f"Jam_{rotor.label}",
            phase=VTOLFlightPhase.ROTOR_JAM,
            V_eas=0.0, nz=1.0,
            altitude_m=0.0,
            failed_rotor_id=rotor.rotor_id,
            far_section="SC-VTOL.2150",
        )

        sim_runner = VTOLSimRunner(
            self._params, self.vtol_config, cg_position=self._cg_mm)
        history, criticals = sim_runner.run_rotor_jam_case(
            condition, self._weight_N, t_sim=t_sim,
            jam_time=jam_time, dt=dt)

        # For jam, use nominal thrust schedule (no recovery)
        # Failed rotor goes to zero at jam_time
        rotor_thrusts = compute_recovery_thrust_schedule(
            self.vtol_config,
            failed_rotor_id=rotor.rotor_id,
            failure_time=jam_time,
            weight_N=self._weight_N,
            rho=self._rho,
            cg_position=self._cg_mm,
            t_arr=history.t,
            states=history.states,
            t_recognition=1e6,  # No recovery
            t_ramp=0.0,
            Ixx=self._params.Ixx,
            Iyy=self._params.Iyy,
        )

        result = self._compute_loads_time_history(
            history, rotor_thrusts, rotor.rotor_id,
            dt_loads=dt_loads, failure_time=jam_time)

        result.event_type = "jam"
        result.failed_rotor_id = rotor.rotor_id
        result.failed_rotor_label = rotor.label
        result.with_recovery = False
        result.t_recognition = float('inf')
        result.sim_time = history.t.copy()
        result.nz_history = history.nz.copy()
        result.phi_history = np.degrees(history.states[:, 6])
        result.rotor_thrusts = rotor_thrusts
        result.critical_points = criticals

        self._compute_quasi_static_and_daf(result, rotor.rotor_id)

        logger.info("  %s %s: peak Mx=%.0f N·mm (DAF=%.2f), "
                    "peak Vy=%.0f N (DAF=%.2f)",
                    result.event_type, rotor.label,
                    result.peak_wing_Mx, result.daf_wing_Mx,
                    result.peak_wing_Vy, result.daf_wing_Vy)

        return result

    # ------------------------------------------------------------------
    # Structural loads computation
    # ------------------------------------------------------------------

    def _compute_loads_time_history(
        self,
        history: SimTimeHistory,
        rotor_thrusts: Dict[str, np.ndarray],
        failed_rotor_id: int,
        dt_loads: float = 0.02,
        failure_time: float = 1.0,
    ) -> TransientPeakResult:
        """Compute structural section loads at sampled time points.

        At each sample:
        1. Reconstruct per-rotor nodal forces from thrust schedule
        2. Compute inertial forces (nz + angular acceleration)
        3. Integrate section loads at wing root and boom junctions
        """
        N_sim = len(history.t)
        dt_sim = history.t[1] - history.t[0] if N_sim > 1 else 0.005
        step = max(1, int(dt_loads / dt_sim))

        # Sample indices
        indices = list(range(0, N_sim, step))
        if indices[-1] != N_sim - 1:
            indices.append(N_sim - 1)
        N_loads = len(indices)

        # Output arrays
        t_loads = np.zeros(N_loads)
        wing_Mx_L = np.zeros(N_loads)
        wing_Mx_R = np.zeros(N_loads)
        wing_Vy_L = np.zeros(N_loads)
        wing_Vy_R = np.zeros(N_loads)
        boom_Mx_max = np.zeros(N_loads)

        # Map rotor label → RotorDef
        label_to_rotor = {r.label: r for r in self.vtol_config.hover_rotors}

        # Pre-compute rotor thrust arrays indexed by sim step
        # (rotor_thrusts is {label: ndarray(N_sim)})

        for k, idx in enumerate(indices):
            t = history.t[idx]
            state = history.states[idx]
            t_loads[k] = t

            phi = state[6]
            theta = state[7]
            p = state[3]
            q = state[4]
            r = state[5]
            p_dot = history.p_dot[idx]
            q_dot = history.q_dot[idx]
            r_dot = history.r_dot[idx]

            # --- 1. Rotor nodal forces ---
            rotor_nodal = {}
            total_Fz_rotor = 0.0
            for label, thrust_arr in rotor_thrusts.items():
                rotor = label_to_rotor.get(label)
                if rotor is None:
                    continue
                T = thrust_arr[idx]
                if T < 1e-3:
                    continue

                shaft = rotor.effective_shaft_axis
                # Reaction force (opposes air acceleration)
                F_thrust = -T * shaft
                # Reaction torque (momentum-theory scaling)
                nom_loads = self._nominal_loads.get(rotor.rotor_id)
                if nom_loads and self._T_nominal > 0:
                    Q = nom_loads.torque * abs(T / self._T_nominal) ** 1.5
                else:
                    Q = 0.0
                rot_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                            else -1.0)
                M_torque = -rot_sign * Q * shaft

                fvec = np.zeros(6)
                fvec[:3] = F_thrust
                fvec[3:] = M_torque
                nid = rotor.hub_node_id
                if nid in rotor_nodal:
                    rotor_nodal[nid] += fvec
                else:
                    rotor_nodal[nid] = fvec.copy()

                total_Fz_rotor += F_thrust[2]

            # --- 2. Structural nz from total rotor force ---
            # nz = -(total upward force) / W  (body z-axis, up = negative)
            nz_struct = -total_Fz_rotor / self._weight_N if self._weight_N > 0 else 1.0

            # --- 3. Inertial forces at all mass nodes ---
            # F_inertial = -m_node × [nz×g (vertical) + α×r (angular accel)]
            alpha_vec = np.array([p_dot, q_dot, r_dot])  # body angular accel
            omega_vec = np.array([p, q, r])  # body angular velocity
            cg_m = self._cg_mm * 1e-3  # mm → m

            inertial_nodal = {}
            for nid, m_node in self._node_masses.items():
                node = self.bdf_model.nodes.get(nid)
                if node is None:
                    continue

                r_m = node.xyz * 1e-3 - cg_m  # position from CG in meters

                # nz-component (vertical inertial, dominant)
                Fz_inertial = -m_node * 1000.0 * nz_struct * _G  # tonnes→kg

                # Angular acceleration: F = -m × α×r
                F_alpha = -m_node * 1000.0 * np.cross(alpha_vec, r_m)

                # Centripetal: F = -m × ω×(ω×r) (secondary, include for accuracy)
                F_centripetal = -m_node * 1000.0 * np.cross(
                    omega_vec, np.cross(omega_vec, r_m))

                fvec = np.zeros(6)
                fvec[2] = Fz_inertial
                fvec[:3] += F_alpha + F_centripetal
                inertial_nodal[nid] = fvec

            # --- 4. Combine all forces ---
            all_forces = {}
            for nid, fvec in rotor_nodal.items():
                all_forces[nid] = fvec.copy()
            for nid, fvec in inertial_nodal.items():
                if nid in all_forces:
                    all_forces[nid] += fvec
                else:
                    all_forces[nid] = fvec.copy()

            # --- 5. Section loads at wing roots ---
            Vy_L, Mx_L, Vy_R, Mx_R = self._wing_root_section_loads(all_forces)
            wing_Mx_L[k] = Mx_L
            wing_Mx_R[k] = Mx_R
            wing_Vy_L[k] = Vy_L
            wing_Vy_R[k] = Vy_R

            # --- 6. Boom root section loads ---
            boom_Mx = self._boom_root_section_loads(all_forces)
            boom_Mx_max[k] = boom_Mx

        # Find peaks (absolute values)
        result = TransientPeakResult()
        result.t_loads = t_loads
        result.wing_Mx_L = wing_Mx_L
        result.wing_Mx_R = wing_Mx_R
        result.wing_Vy_L = wing_Vy_L
        result.wing_Vy_R = wing_Vy_R
        result.boom_Mx_max = boom_Mx_max

        # Peak wing bending (max absolute across both sides)
        abs_L = np.max(np.abs(wing_Mx_L))
        abs_R = np.max(np.abs(wing_Mx_R))
        if abs_L >= abs_R:
            result.peak_wing_Mx = abs_L
            result.peak_wing_Mx_time = t_loads[np.argmax(np.abs(wing_Mx_L))]
            result.peak_wing_Mx_side = "L"
        else:
            result.peak_wing_Mx = abs_R
            result.peak_wing_Mx_time = t_loads[np.argmax(np.abs(wing_Mx_R))]
            result.peak_wing_Mx_side = "R"

        result.peak_wing_Vy = max(np.max(np.abs(wing_Vy_L)),
                                   np.max(np.abs(wing_Vy_R)))
        result.peak_boom_Mx = np.max(np.abs(boom_Mx_max))
        if len(boom_Mx_max) > 0 and result.peak_boom_Mx > 0:
            result.peak_boom_Mx_time = t_loads[np.argmax(np.abs(boom_Mx_max))]

        return result

    def _wing_root_section_loads(self, nodal_forces: Dict[int, np.ndarray],
                                  ) -> Tuple[float, float, float, float]:
        """Compute shear and bending at left/right wing roots.

        Integrates all forces outboard of the wing root (tip-to-root).

        Returns (Vy_L, Mx_L, Vy_R, Mx_R) in N and N·mm.
        """
        y_root = self._wing_root_y  # mm
        Vy_L = Mx_L = Vy_R = Mx_R = 0.0

        for nid, fvec in nodal_forces.items():
            node = self.bdf_model.nodes.get(nid)
            if node is None:
                continue
            y = node.xyz[1]  # spanwise coordinate (mm)

            if y < -y_root:
                # Left wing outboard
                arm = abs(y) - y_root  # mm from root
                Vy_L += fvec[2]           # vertical shear
                Mx_L += fvec[2] * arm     # bending about root
            elif y > y_root:
                # Right wing outboard
                arm = abs(y) - y_root
                Vy_R += fvec[2]
                Mx_R += fvec[2] * arm

        return Vy_L, Mx_L, Vy_R, Mx_R

    def _boom_root_section_loads(self, nodal_forces: Dict[int, np.ndarray],
                                  ) -> float:
        """Compute max boom root bending moment across all booms.

        Each boom extends from a fuselage junction to a rotor hub.
        Bending = hub force × boom length.
        """
        max_Mx = 0.0

        for junc_y, hub_y, hub_nid in self._boom_junctions:
            # Sum forces at and outboard of the junction
            Mx = 0.0
            for nid, fvec in nodal_forces.items():
                node = self.bdf_model.nodes.get(nid)
                if node is None:
                    continue
                y = node.xyz[1]

                # Determine if this node is on this boom segment
                if hub_y < 0:
                    # Left side: nodes with y < junc_y (outboard)
                    if y < junc_y:
                        arm = abs(y - junc_y)
                        Mx += fvec[2] * arm
                else:
                    # Right side: nodes with y > junc_y (outboard)
                    if y > junc_y:
                        arm = abs(y - junc_y)
                        Mx += fvec[2] * arm

            max_Mx = max(max_Mx, abs(Mx))

        return max_Mx

    # ------------------------------------------------------------------
    # Quasi-static baseline and DAF
    # ------------------------------------------------------------------

    def _compute_quasi_static_and_daf(self, result: TransientPeakResult,
                                       failed_rotor_id: int):
        """Compute quasi-static OEI loads and DAF.

        Quasi-static: steady hover with one rotor off, remaining rotors
        share load to maintain nz=1.0.
        """
        hover_rotors = self.vtol_config.hover_rotors
        n_hover = len(hover_rotors)
        n_active = n_hover - 1
        if n_active <= 0:
            return

        # Each remaining rotor carries W / (N-1)
        T_per_active = self._weight_N / n_active

        # Build quasi-static nodal forces
        qs_forces = {}

        for rotor in hover_rotors:
            if rotor.rotor_id == failed_rotor_id:
                continue

            T = T_per_active
            shaft = rotor.effective_shaft_axis
            F_thrust = -T * shaft
            nom = self._nominal_loads.get(rotor.rotor_id)
            Q = nom.torque * abs(T / self._T_nominal) ** 1.5 if nom else 0.0
            rot_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                        else -1.0)
            M_torque = -rot_sign * Q * shaft

            fvec = np.zeros(6)
            fvec[:3] = F_thrust
            fvec[3:] = M_torque
            nid = rotor.hub_node_id
            if nid in qs_forces:
                qs_forces[nid] += fvec
            else:
                qs_forces[nid] = fvec.copy()

        # Inertial forces at nz=1.0 (no angular accelerations)
        for nid, m_node in self._node_masses.items():
            fvec = np.zeros(6)
            fvec[2] = -m_node * 1000.0 * 1.0 * _G  # nz=1.0
            if nid in qs_forces:
                qs_forces[nid] += fvec
            else:
                qs_forces[nid] = fvec.copy()

        # Section loads
        Vy_L, Mx_L, Vy_R, Mx_R = self._wing_root_section_loads(qs_forces)
        boom_Mx = self._boom_root_section_loads(qs_forces)

        result.qs_wing_Mx = max(abs(Mx_L), abs(Mx_R))
        result.qs_wing_Vy = max(abs(Vy_L), abs(Vy_R))
        result.qs_boom_Mx = boom_Mx

        # DAF
        if result.qs_wing_Mx > 1e-3:
            result.daf_wing_Mx = result.peak_wing_Mx / result.qs_wing_Mx
        if result.qs_wing_Vy > 1e-3:
            result.daf_wing_Vy = result.peak_wing_Vy / result.qs_wing_Vy
        if result.qs_boom_Mx > 1e-3:
            result.daf_boom_Mx = result.peak_boom_Mx / result.qs_boom_Mx

    # ------------------------------------------------------------------
    # Pre-computation helpers
    # ------------------------------------------------------------------

    def _build_node_mass_map(self) -> Dict[int, float]:
        """Build {node_id: mass_in_tonnes} from CONM2 + structural elements."""
        masses: Dict[int, float] = {}

        # CONM2 concentrated masses
        for eid, conm2 in self.bdf_model.conm2s.items():
            nid = conm2.nid
            masses[nid] = masses.get(nid, 0.0) + conm2.mass

        # Structural element lumped masses
        for eid, elem in self.bdf_model.elements.items():
            etype = type(elem).__name__
            if etype == 'CBAR':
                prop = self.bdf_model.properties.get(elem.pid)
                if prop is None:
                    continue
                A = getattr(prop, 'A', 0.0)
                mid = getattr(prop, 'mid', None) or getattr(prop, 'MID', None)
                if mid is None:
                    continue
                mat = self.bdf_model.materials.get(mid)
                if mat is None:
                    continue
                rho = getattr(mat, 'rho', 0.0)
                if rho <= 0 or A <= 0:
                    continue
                n1 = self.bdf_model.nodes.get(elem.nids[0])
                n2 = self.bdf_model.nodes.get(elem.nids[1])
                if n1 is None or n2 is None:
                    continue
                L = np.linalg.norm(n2.xyz - n1.xyz)
                m_elem = rho * A * L  # tonnes (if rho in tonne/mm³)
                for nid in elem.nids[:2]:
                    masses[nid] = masses.get(nid, 0.0) + m_elem / 2.0

            elif etype == 'CQUAD4':
                prop = self.bdf_model.properties.get(elem.pid)
                if prop is None:
                    continue
                t = getattr(prop, 't', 0.0) or getattr(prop, 'T', 0.0)
                mid = getattr(prop, 'mid', None) or getattr(prop, 'MID', None)
                if mid is None:
                    continue
                mat = self.bdf_model.materials.get(mid)
                if mat is None:
                    continue
                rho = getattr(mat, 'rho', 0.0)
                if rho <= 0 or t <= 0:
                    continue
                nodes = [self.bdf_model.nodes.get(nid) for nid in elem.nids[:4]]
                if any(n is None for n in nodes):
                    continue
                d1 = nodes[2].xyz - nodes[0].xyz
                d2 = nodes[3].xyz - nodes[1].xyz
                area = 0.5 * np.linalg.norm(np.cross(d1, d2))
                m_elem = rho * t * area
                for nid in elem.nids[:4]:
                    masses[nid] = masses.get(nid, 0.0) + m_elem / 4.0

            elif etype == 'CTRIA3':
                prop = self.bdf_model.properties.get(elem.pid)
                if prop is None:
                    continue
                t = getattr(prop, 't', 0.0) or getattr(prop, 'T', 0.0)
                mid = getattr(prop, 'mid', None) or getattr(prop, 'MID', None)
                if mid is None:
                    continue
                mat = self.bdf_model.materials.get(mid)
                if mat is None:
                    continue
                rho = getattr(mat, 'rho', 0.0)
                if rho <= 0 or t <= 0:
                    continue
                nodes = [self.bdf_model.nodes.get(nid) for nid in elem.nids[:3]]
                if any(n is None for n in nodes):
                    continue
                d1 = nodes[1].xyz - nodes[0].xyz
                d2 = nodes[2].xyz - nodes[0].xyz
                area = 0.5 * np.linalg.norm(np.cross(d1, d2))
                m_elem = rho * t * area
                for nid in elem.nids[:3]:
                    masses[nid] = masses.get(nid, 0.0) + m_elem / 3.0

        return masses

    def _compute_cg_mm(self) -> np.ndarray:
        """Compute CG position in model coordinates (mm)."""
        total_m = 0.0
        cg = np.zeros(3)
        for nid, m in self._node_masses.items():
            node = self.bdf_model.nodes.get(nid)
            if node is None:
                continue
            cg += m * node.xyz
            total_m += m
        if total_m > 0:
            cg /= total_m
        return cg

    def _detect_wing_root_y(self) -> float:
        """Detect wing root Y-position from model geometry.

        Finds the minimum |Y| of nodes that are clearly on the wing
        (lateral nodes with significant Y displacement from centerline).
        """
        y_values = []
        for nid, node in self.bdf_model.nodes.items():
            y = abs(node.xyz[1])
            if y > 200.0:  # Clearly not centerline
                y_values.append(y)

        if y_values:
            return min(y_values)
        return 500.0  # Default fallback

    def _detect_boom_junctions(self) -> List[Tuple[float, float, int]]:
        """Detect boom-fuselage junctions from rotor hub positions.

        Returns list of (junction_y, hub_y, hub_node_id) tuples.
        Each boom connects a wing/fuselage junction to a rotor hub.
        """
        junctions = []
        for rotor in self.vtol_config.hover_rotors:
            hub_y = rotor.hub_position[1]  # mm
            # Junction is at the wing root Y on the same side
            junc_y = self._wing_root_y if hub_y > 0 else -self._wing_root_y
            junctions.append((junc_y, hub_y, rotor.hub_node_id))
        return junctions

    def _build_aircraft_params(self) -> AircraftParams:
        """Build AircraftParams for 6-DOF simulation."""
        from .aero_derivatives import build_derivative_set, compute_inertia_from_conm2

        wc = self.aircraft_config.weight_cg_conditions[0]
        alt_m = self.aircraft_config.altitudes_m[0] if self.aircraft_config.altitudes_m else 0.0

        # Compute stability derivatives
        from .aircraft_config import eas_to_mach
        mach = eas_to_mach(self.aircraft_config.speeds.VC, alt_m)

        try:
            derivs = build_derivative_set(
                self.bdf_model, self.aircraft_config, wc, mach)
        except Exception as e:
            logger.warning("Could not compute VLM derivatives: %s", e)
            from .aero_derivatives import AeroDerivativeSet
            derivs = AeroDerivativeSet()

        # Compute inertia
        cg_xyz = np.array([wc.cg_x, 0.0, 0.0])
        inertia = compute_inertia_from_conm2(self.bdf_model, cg_xyz)

        mm_to_m = 1e-3
        rho, _, _ = isa_atmosphere(alt_m)

        params = AircraftParams(
            mass=inertia["mass_kg"],
            S=derivs.S_ref * mm_to_m ** 2 if derivs.S_ref > 0 else 20.0,
            b=derivs.b_ref * mm_to_m if derivs.b_ref > 0 else 12.0,
            c_bar=derivs.c_bar * mm_to_m if derivs.c_bar > 0 else 1.5,
            Ixx=inertia["Ixx"],
            Iyy=inertia["Iyy"],
            Izz=inertia["Izz"],
            Ixz=inertia["Ixz"],
            derivs=derivs,
            rho=rho,
        )
        return params


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summarize_transient_results(results: List[TransientPeakResult]) -> str:
    """Format transient results as a summary table."""
    lines = []
    lines.append("=" * 90)
    lines.append("VTOL Transient Peak Loads Summary")
    lines.append("=" * 90)
    lines.append(f"{'Event':<25s} {'Rotor':<15s} {'t_rec[s]':>8s} "
                 f"{'Peak Mx':>12s} {'QS Mx':>12s} {'DAF_Mx':>8s} "
                 f"{'Peak Vy':>10s} {'DAF_Vy':>8s}")
    lines.append("-" * 90)

    for r in results:
        t_rec_str = f"{r.t_recognition:.2f}" if r.t_recognition < 1e5 else "None"
        event_str = f"{r.event_type}"
        if r.with_recovery:
            event_str += " (recovery)"
        lines.append(
            f"{event_str:<25s} {r.failed_rotor_label:<15s} {t_rec_str:>8s} "
            f"{r.peak_wing_Mx:>12.0f} {r.qs_wing_Mx:>12.0f} "
            f"{r.daf_wing_Mx:>8.2f} "
            f"{r.peak_wing_Vy:>10.0f} {r.daf_wing_Vy:>8.2f}")

    lines.append("=" * 90)

    # Worst-case summary
    if results:
        worst = max(results, key=lambda r: r.daf_wing_Mx)
        lines.append(f"Worst DAF_Mx: {worst.daf_wing_Mx:.2f} "
                     f"({worst.event_type} {worst.failed_rotor_label}, "
                     f"t_rec={worst.t_recognition:.2f}s, "
                     f"peak at t={worst.peak_wing_Mx_time:.3f}s)")

        worst_vy = max(results, key=lambda r: r.daf_wing_Vy)
        lines.append(f"Worst DAF_Vy: {worst_vy.daf_wing_Vy:.2f} "
                     f"({worst_vy.event_type} {worst_vy.failed_rotor_label})")

    return "\n".join(lines)
