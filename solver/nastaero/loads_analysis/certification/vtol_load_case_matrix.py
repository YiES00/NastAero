"""VTOL load case matrix generator.

Generates VTOL-specific load cases (hover, OEI, transition, VTOL landing,
rotor jam) and computes rotor forces for each condition using BEMT.

Case ID numbering:
- 1-9999: Conventional flight cases (from LoadCaseMatrix)
- 10000-19999: Dynamic simulation cases
- 20000+: VTOL-specific cases
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from ..case_generator import TrimCondition, isa_atmosphere
from .aircraft_config import AircraftConfig, WeightCGCondition, dynamic_pressure_from_eas, eas_to_mach
from .load_case_matrix import CertLoadCase
from .vtol_conditions import (
    VTOLCondition, VTOLFlightPhase,
    generate_hover_conditions,
    generate_oei_conditions,
    generate_transition_conditions,
    generate_vtol_landing_conditions,
    generate_rotor_jam_conditions,
)
from ...rotor.rotor_config import VTOLConfig, RotorDef, RotorType
from ...rotor.bemt_solver import BEMTSolver, RotorLoads
from ...rotor.forward_flight import ForwardFlightBEMT
from ...rotor.rotor_loads_applicator import rotor_loads_to_nodal_forces, all_rotor_forces


# VTOL case ID ranges
VTOL_CASE_ID_START = 20000
HOVER_ID_START = 20000
OEI_HOVER_ID_START = 20100
TRANSITION_ID_START = 20200
OEI_TRANSITION_ID_START = 20300
VTOL_LANDING_ID_START = 20400
ROTOR_JAM_ID_START = 20500


class VTOLLoadCaseMatrix:
    """Generator for VTOL-specific load cases.

    Takes a VTOLConfig and AircraftConfig, generates all VTOL conditions,
    computes rotor forces via BEMT, and produces CertLoadCase objects
    ready for structural analysis.

    Parameters
    ----------
    vtol_config : VTOLConfig
        VTOL rotor configuration.
    aircraft_config : AircraftConfig
        Base aircraft configuration.
    """

    def __init__(self, vtol_config: VTOLConfig,
                 aircraft_config: AircraftConfig):
        self.vtol_config = vtol_config
        self.aircraft_config = aircraft_config
        self.cases: List[CertLoadCase] = []
        self._bemt_cache: Dict[int, BEMTSolver] = {}
        self._ff_cache: Dict[int, ForwardFlightBEMT] = {}
        self._next_id = VTOL_CASE_ID_START

    def _get_bemt_solver(self, rotor: RotorDef) -> BEMTSolver:
        """Get or create BEMT solver for a rotor (cached)."""
        if rotor.rotor_id not in self._bemt_cache:
            self._bemt_cache[rotor.rotor_id] = BEMTSolver(
                rotor.blade, rotor.n_blades)
        return self._bemt_cache[rotor.rotor_id]

    def _get_ff_solver(self, rotor: RotorDef) -> ForwardFlightBEMT:
        """Get or create forward-flight BEMT solver."""
        if rotor.rotor_id not in self._ff_cache:
            self._ff_cache[rotor.rotor_id] = ForwardFlightBEMT(
                rotor.blade, rotor.n_blades)
        return self._ff_cache[rotor.rotor_id]

    def _next_case_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    def _compute_rotor_forces_hover(self, condition: VTOLCondition,
                                     wc: WeightCGCondition,
                                     ) -> Dict[int, np.ndarray]:
        """Compute rotor forces for hover/OEI conditions.

        Parameters
        ----------
        condition : VTOLCondition
            VTOL flight condition.
        wc : WeightCGCondition
            Weight condition.

        Returns
        -------
        dict
            {node_id: force_vector(6)} for all active rotors.
        """
        rho, _, _ = isa_atmosphere(condition.altitude_m)
        total_weight_N = wc.weight_N * condition.nz

        # Identify active lift rotors
        lift_rotors = self.vtol_config.lift_rotors
        active_rotors = [r for r in lift_rotors
                         if r.rotor_id != condition.failed_rotor_id]
        n_active = len(active_rotors)
        if n_active == 0:
            return {}

        # Each active rotor shares the load equally
        thrust_per_rotor = total_weight_N * condition.thrust_fraction / n_active

        loads_map: Dict[int, RotorLoads] = {}
        for rotor in active_rotors:
            solver = self._get_bemt_solver(rotor)
            rpm = rotor.rpm_hover * condition.rotor_rpm_factor

            # Solve for required thrust
            loads = solver.solve_for_thrust(
                thrust_per_rotor, rpm, rho, V_inf=0.0)
            loads_map[rotor.rotor_id] = loads

        return all_rotor_forces(active_rotors, loads_map)

    def _compute_rotor_forces_transition(self, condition: VTOLCondition,
                                          wc: WeightCGCondition,
                                          ) -> Dict[int, np.ndarray]:
        """Compute rotor forces for transition conditions.

        In transition, lift rotors provide partial thrust (decreasing
        with speed) and cruise rotor provides forward thrust.
        """
        rho, _, _ = isa_atmosphere(condition.altitude_m)
        total_weight_N = wc.weight_N * condition.nz

        loads_map: Dict[int, RotorLoads] = {}
        all_active = []

        # Lift rotors: partial thrust
        lift_rotors = [r for r in self.vtol_config.lift_rotors
                       if r.rotor_id != condition.failed_rotor_id]
        n_lift = len(lift_rotors)

        if n_lift > 0:
            thrust_per_lift = (total_weight_N * condition.thrust_fraction
                               / n_lift)
            for rotor in lift_rotors:
                rpm = rotor.rpm_hover * condition.rotor_rpm_factor
                # Use forward-flight BEMT for non-zero V
                if condition.V_eas > 1.0:
                    solver = self._get_ff_solver(rotor)
                    loads = solver.solve_for_thrust(
                        thrust_per_lift, rpm, condition.V_eas,
                        alpha_shaft=np.pi / 2,  # Vertical shaft
                        rho=rho)
                else:
                    solver = self._get_bemt_solver(rotor)
                    loads = solver.solve_for_thrust(
                        thrust_per_lift, rpm, rho)
                loads_map[rotor.rotor_id] = loads
                all_active.append(rotor)

        # Cruise rotor: forward thrust (if V > 0)
        for rotor in self.vtol_config.cruise_rotors:
            if condition.V_eas > 1.0 and rotor.rpm_cruise > 0:
                # Simple thrust estimate for pusher
                solver = self._get_bemt_solver(rotor)
                # Cruise thrust ~ drag at this speed (simplified)
                # Use a fraction of weight as drag estimate
                drag_est = total_weight_N * 0.05  # L/D ~ 20
                loads = solver.solve_for_thrust(
                    drag_est, rotor.rpm_cruise, rho,
                    V_inf=condition.V_eas)
                loads_map[rotor.rotor_id] = loads
                all_active.append(rotor)

        return all_rotor_forces(all_active, loads_map)

    def _condition_to_cert_case(self, condition: VTOLCondition,
                                 wc: WeightCGCondition,
                                 rotor_forces: Dict[int, np.ndarray],
                                 ) -> CertLoadCase:
        """Convert VTOLCondition + rotor forces to CertLoadCase."""
        # For hover (q=0), we cannot use standard trim
        if condition.V_eas < 1.0:
            # Static analysis — no aeroelastic trim
            tc = TrimCondition(
                case_id=self._next_case_id(),
                mach=0.0, q=0.0, nz=condition.nz,
                label=condition.label,
                altitude_m=condition.altitude_m,
            )
            solve_type = "static_rotor"
        else:
            mach = eas_to_mach(condition.V_eas, condition.altitude_m)
            q = dynamic_pressure_from_eas(condition.V_eas)
            tc = TrimCondition(
                case_id=self._next_case_id(),
                mach=mach, q=q, nz=condition.nz,
                fixed_vars={
                    "ROLL": 0.0, "YAW": 0.0,
                    "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                },
                free_vars=["ANGLEA", "ELEV"],
                label=condition.label,
                altitude_m=condition.altitude_m,
            )
            solve_type = "trim"

        # Map phase to category
        phase_to_category = {
            VTOLFlightPhase.HOVER: "vtol_hover",
            VTOLFlightPhase.OEI: "vtol_oei",
            VTOLFlightPhase.TRANSITION: "vtol_transition",
            VTOLFlightPhase.VTOL_LANDING: "vtol_landing",
            VTOLFlightPhase.ROTOR_JAM: "vtol_rotor_jam",
        }

        return CertLoadCase(
            trim_condition=tc,
            category=phase_to_category.get(condition.phase, "vtol"),
            far_section=condition.far_section,
            weight_cg=wc,
            altitude_m=condition.altitude_m,
            config_label=f"VTOL {condition.phase.value}",
            solve_type=solve_type,
            rotor_forces=rotor_forces if rotor_forces else None,
        )

    def generate_all(self) -> List[CertLoadCase]:
        """Generate all VTOL load cases with pre-computed rotor forces.

        Returns
        -------
        list of CertLoadCase
            All VTOL load cases ready for structural analysis.
        """
        self.cases = []
        self._next_id = VTOL_CASE_ID_START

        altitudes = self.aircraft_config.altitudes_m
        lift_rotor_ids = [r.rotor_id for r in self.vtol_config.lift_rotors
                          if r.can_fail]
        all_rotor_ids = [r.rotor_id for r in self.vtol_config.rotors
                         if r.can_fail]

        # Generate conditions
        conditions: List[VTOLCondition] = []
        conditions.extend(generate_hover_conditions(altitudes))
        conditions.extend(generate_oei_conditions(
            self.vtol_config.n_lift_rotors, lift_rotor_ids, altitudes))
        conditions.extend(generate_transition_conditions(
            self.vtol_config.v_mca, self.vtol_config.v_transition_end,
            altitudes))
        conditions.extend(generate_vtol_landing_conditions(altitudes))
        conditions.extend(generate_rotor_jam_conditions(
            all_rotor_ids, altitudes))

        # Convert each condition to CertLoadCase with rotor forces
        for wc in self.aircraft_config.weight_cg_conditions:
            for cond in conditions:
                # Compute rotor forces
                if cond.phase in (VTOLFlightPhase.HOVER,
                                  VTOLFlightPhase.OEI,
                                  VTOLFlightPhase.VTOL_LANDING):
                    rotor_forces = self._compute_rotor_forces_hover(cond, wc)
                elif cond.phase == VTOLFlightPhase.TRANSITION:
                    rotor_forces = self._compute_rotor_forces_transition(
                        cond, wc)
                elif cond.phase == VTOLFlightPhase.ROTOR_JAM:
                    # Jam: use last-known forces, then set failed rotor to 0
                    rotor_forces = self._compute_rotor_forces_hover(cond, wc)
                else:
                    rotor_forces = {}

                case = self._condition_to_cert_case(cond, wc, rotor_forces)
                self.cases.append(case)

        return self.cases

    def summary(self) -> Dict[str, int]:
        """Return case count by VTOL category."""
        counts: Dict[str, int] = {}
        for c in self.cases:
            counts[c.category] = counts.get(c.category, 0) + 1
        return counts
