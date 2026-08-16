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
from ...rotor.rotor_config import (VTOLConfig, RotorDef, RotorType,
                                   RotationDir)
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

        # Identify active hover-capable rotors (LIFT + TILT)
        hover_rotors = self.vtol_config.hover_rotors
        active_rotors = [r for r in hover_rotors
                         if r.rotor_id != condition.failed_rotor_id]
        n_active = len(active_rotors)
        if n_active == 0:
            return {}

        # Each active rotor shares the load equally.
        # thrust_fraction already includes the nz factor where applicable
        # (hover conditions set tf=nz), so do NOT multiply by nz again.
        thrust_per_rotor = wc.weight_N * condition.thrust_fraction / n_active

        loads_map: Dict[int, RotorLoads] = {}
        for rotor in active_rotors:
            solver = self._get_bemt_solver(rotor)
            rpm = rotor.rpm_hover * condition.rotor_rpm_factor

            # Solve for required thrust
            loads = solver.solve_for_thrust(
                thrust_per_rotor, rpm, rho, V_inf=0.0)
            loads_map[rotor.rotor_id] = loads

        self._last_loads_map = loads_map
        return all_rotor_forces(active_rotors, loads_map)

    def _compute_rotor_forces_transition(self, condition: VTOLCondition,
                                          wc: WeightCGCondition,
                                          ) -> Dict[int, np.ndarray]:
        """Compute rotor forces for transition conditions.

        In transition, lift rotors provide partial thrust (decreasing
        with speed) and cruise rotor provides forward thrust.
        """
        rho, _, _ = isa_atmosphere(condition.altitude_m)

        loads_map: Dict[int, RotorLoads] = {}
        all_active = []

        # Hover-capable rotors: partial thrust in transition
        hover_rotors = [r for r in self.vtol_config.hover_rotors
                        if r.rotor_id != condition.failed_rotor_id]
        n_hover = len(hover_rotors)

        if n_hover > 0:
            # thrust_fraction is set to (rotor lift share) * nz by the
            # transition condition generator — no extra nz multiply here.
            thrust_per_lift = (wc.weight_N * condition.thrust_fraction
                               / n_hover)
            for rotor in hover_rotors:
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
                solver = self._get_bemt_solver(rotor)
                # Speed-dependent drag: D = CD0*S * q
                # Estimate CD0*S from cruise L/D~20:
                #   D_cruise = W / 20 at V_cruise
                #   CD0*S = D_cruise / q_cruise
                q_eas = 0.5 * rho * condition.V_eas ** 2
                # Use W/(L/D) as cruise drag reference, scale by q ratio
                wt_ref = wc.weight_N
                drag_est = wt_ref * 0.05 * (q_eas / (0.5 * rho * 35.0 ** 2))
                drag_est = max(drag_est, wt_ref * 0.01)  # Minimum drag floor
                loads = solver.solve_for_thrust(
                    drag_est, rotor.rpm_cruise, rho,
                    V_inf=condition.V_eas)
                loads_map[rotor.rotor_id] = loads
                all_active.append(rotor)

        self._last_loads_map = loads_map
        return all_rotor_forces(all_active, loads_map)

    def _compute_rotor_forces_tilt(self, condition: VTOLCondition,
                                   wc: WeightCGCondition,
                                   ) -> Dict[int, np.ndarray]:
        """틸트 변환 조건의 로터 하중 — 전열 틸트(σ) + 후열 수직.

        배분은 vtol_conditions.tilt_allocation의 Fx/Fz 2식 평형을
        따르고, 전열 추력은 축 a(σ) = [sinσ, 0, cosσ](전방 틸트,
        추력 +x)로 회전 조립한다. BEMT는 축 경사각
        alpha_shaft = 90° − σ의 전진비행 해로 달성 추력·토크를
        구한다(포화 시 달성값 사용 — 파이프라인 관례)."""
        import math

        from .vtol_conditions import tilt_allocation

        rho, _, _ = isa_atmosphere(condition.altitude_m)
        tilt_rotors = [r for r in self.vtol_config.rotors
                       if r.rotor_type == RotorType.TILT
                       and r.rotor_id != condition.failed_rotor_id]
        aft_rotors = [r for r in self.vtol_config.rotors
                      if r.rotor_type == RotorType.LIFT
                      and r.rotor_id != condition.failed_rotor_id]
        if not tilt_rotors:
            return {}
        sigma = condition.tilt_deg
        F, A, _L, _ok = tilt_allocation(
            condition.V_eas, sigma, condition.nz, wc.weight_N,
            getattr(self.aircraft_config, "wing_area_m2", 0.0),
            1.0, rho, n_tilt=len(tilt_rotors) or 1,
            n_aft=len(aft_rotors) or 1)
        s_r = math.radians(sigma)
        axis_f = np.array([math.sin(s_r), 0.0, math.cos(s_r)])
        axis_a = np.array([0.0, 0.0, 1.0])

        # ── 틸트 고착(M6): 고착 로터는 stuck 각, 건강 로터는 스케줄
        #    각. 추력은 전 틸트 로터 공통 지령 T_cmd로, Fx 평형
        #    T_cmd·(Σ_h sinσ_c + sinσ_s) = D 에서 결정한다(제어법칙
        #    없는 결정 규칙). 비대칭 롤·요 잔차는 관성 릴리프 폐합.
        stuck_id = (condition.failed_rotor_id
                    if condition.stuck_tilt_deg is not None else None)
        if stuck_id is not None:
            from .vtol_conditions import tilt_drag_estimate
            tilt_all = [r for r in self.vtol_config.rotors
                        if r.rotor_type == RotorType.TILT]
            healthy = [r for r in tilt_all if r.rotor_id != stuck_id]
            stuck = [r for r in tilt_all if r.rotor_id == stuck_id]
            s_s = math.radians(condition.stuck_tilt_deg)
            axis_s = np.array([math.sin(s_s), 0.0, math.cos(s_s)])
            D = tilt_drag_estimate(condition.V_eas, wc.weight_N)
            denom = (len(healthy) * math.sin(s_r) + math.sin(s_s))
            T_cmd = D / max(denom, 1e-6)
            T_cap = 1.5 * wc.weight_N / max(
                len(self.vtol_config.rotors), 1)
            T_cmd = min(T_cmd, T_cap)
            # 수직 잔여는 날개(능력 한도) + 후열
            Fz_tilt = T_cmd * (len(healthy) * math.cos(s_r)
                               + math.cos(s_s))
            q = 0.5 * rho * condition.V_eas ** 2
            L_cap = 1.0 * q * getattr(self.aircraft_config,
                                      "wing_area_m2", 0.0)
            L_wing = min(L_cap, max(0.0, condition.nz * wc.weight_N
                                    - Fz_tilt))
            A = max(0.0, condition.nz * wc.weight_N - Fz_tilt - L_wing)
            forces: Dict[int, np.ndarray] = {}

            def _apply_s(rotors, T_each, axis, alpha_shaft):
                for r in rotors:
                    if T_each <= 1e-9:
                        continue
                    rpm = r.rpm_hover * condition.rotor_rpm_factor
                    if condition.V_eas > 1.0:
                        solver = self._get_ff_solver(r)
                        loads = solver.solve_for_thrust(
                            T_each, rpm, condition.V_eas,
                            alpha_shaft=alpha_shaft, rho=rho)
                    else:
                        solver = self._get_bemt_solver(r)
                        loads = solver.solve_for_thrust(T_each, rpm,
                                                        rho)
                    rot_sign = (1.0 if r.rotation_dir
                                == RotationDir.CW else -1.0)
                    fvec = np.zeros(6)
                    fvec[:3] = loads.thrust * axis
                    fvec[3:] = -rot_sign * loads.torque * axis
                    nid = r.hub_node_id
                    forces[nid] = forces.get(nid, np.zeros(6)) + fvec

            _apply_s(healthy, T_cmd, axis_f, math.pi / 2 - s_r)
            _apply_s(stuck, T_cmd, axis_s, math.pi / 2 - s_s)
            if aft_rotors and A > 0:
                _apply_s(aft_rotors, A / len(aft_rotors), axis_a,
                         math.pi / 2)
            return forces

        forces: Dict[int, np.ndarray] = {}

        def _apply(rotors, T_each, axis, alpha_shaft):
            for r in rotors:
                if T_each <= 1e-9:
                    continue
                rpm = r.rpm_hover * condition.rotor_rpm_factor
                if condition.V_eas > 1.0:
                    solver = self._get_ff_solver(r)
                    loads = solver.solve_for_thrust(
                        T_each, rpm, condition.V_eas,
                        alpha_shaft=alpha_shaft, rho=rho)
                else:
                    solver = self._get_bemt_solver(r)
                    loads = solver.solve_for_thrust(T_each, rpm, rho)
                rot_sign = (1.0 if r.rotation_dir == RotationDir.CW
                            else -1.0)
                fvec = np.zeros(6)
                fvec[:3] = loads.thrust * axis
                fvec[3:] = -rot_sign * loads.torque * axis
                nid = r.hub_node_id
                forces[nid] = forces.get(nid, np.zeros(6)) + fvec

        _apply(tilt_rotors, F / len(tilt_rotors), axis_f,
               math.pi / 2 - s_r)
        if aft_rotors:
            _apply(aft_rotors, A / len(aft_rotors), axis_a,
                   math.pi / 2)
        return forces

    @staticmethod
    def _saturation_summary(loads_map) -> tuple:
        """(지령 실현 가능 여부, 최대 추력 부족률) — 로터 하중 맵 기준."""
        worst = 0.0
        for ld in (loads_map or {}).values():
            worst = max(worst, getattr(ld, "thrust_shortfall_frac", 0.0))
        return (worst <= 0.0), worst

    def _condition_to_cert_case(self, condition: VTOLCondition,
                                 wc: WeightCGCondition,
                                 rotor_forces: Dict[int, np.ndarray],
                                 saturation: tuple = (True, 0.0),
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
            VTOLFlightPhase.TILT_TRANSITION: "vtol_tilt_transition",
            VTOLFlightPhase.TILT_STUCK: "vtol_tilt_stuck",
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
            rotor_command_feasible=saturation[0],
            rotor_thrust_shortfall=saturation[1],
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
        hover_rotor_ids = [r.rotor_id for r in self.vtol_config.hover_rotors
                           if r.can_fail]
        all_rotor_ids = [r.rotor_id for r in self.vtol_config.rotors
                         if r.can_fail]

        # Generate conditions
        conditions: List[VTOLCondition] = []
        conditions.extend(generate_hover_conditions(altitudes))
        conditions.extend(generate_oei_conditions(
            self.vtol_config.n_hover_rotors, hover_rotor_ids, altitudes))
        # Pass wing parameters for capability-based thrust fraction
        cfg = self.aircraft_config
        # 틸트 구성이면 L+C 천이 스케줄 대신 틸트 변환 회랑을 쓴다
        has_tilt = any(r.rotor_type == RotorType.TILT
                       for r in self.vtol_config.rotors)
        if has_tilt:
            from .vtol_conditions import (
                generate_tilt_transition_conditions,
            )
            n_tilt = sum(1 for r in self.vtol_config.rotors
                         if r.rotor_type == RotorType.TILT)
            n_aft = sum(1 for r in self.vtol_config.rotors
                        if r.rotor_type == RotorType.LIFT)
            conditions.extend(generate_tilt_transition_conditions(
                self.vtol_config.v_mca,
                self.vtol_config.v_transition_end, altitudes,
                wing_area_m2=getattr(cfg, 'wing_area_m2', 0.0),
                CL_transition=1.0,
                weight_N=(cfg.weight_cg_conditions[0].weight_N
                          if cfg.weight_cg_conditions else 0.0),
                n_tilt=n_tilt, n_aft=n_aft,
                CLalpha_gust=getattr(cfg, 'CLalpha', 0.0),
                mean_chord_gust=getattr(cfg, 'mean_chord_m', 1.0),
            ))
            # 틸트 고착(M6) — 틸트로터 고유 고장 모드
            from .vtol_conditions import (
                generate_tilt_stuck_conditions,
            )
            tilt_ids = [r.rotor_id for r in self.vtol_config.rotors
                        if r.rotor_type == RotorType.TILT
                        and r.can_fail]
            conditions.extend(generate_tilt_stuck_conditions(
                self.vtol_config.v_mca,
                self.vtol_config.v_transition_end, altitudes,
                tilt_ids,
                wing_area_m2=getattr(cfg, 'wing_area_m2', 0.0),
                CL_transition=1.0,
                weight_N=(cfg.weight_cg_conditions[0].weight_N
                          if cfg.weight_cg_conditions else 0.0),
                n_tilt=n_tilt, n_aft=n_aft,
            ))
        if not has_tilt:
            conditions.extend(generate_transition_conditions(
                self.vtol_config.v_mca,
                self.vtol_config.v_transition_end, altitudes,
                wing_area_m2=getattr(cfg, 'wing_area_m2', 0.0),
                CL_transition=1.0,
                weight_N=(cfg.weight_cg_conditions[0].weight_N
                          if cfg.weight_cg_conditions else 0.0),
            ))
            # 천이 준정적 수직 돌풍 — 호버 정적 돌풍(±0.3)과 고정익
            # Pratt(VB/VC/VD) 사이 회랑의 공백을 잇는 케이스
            from .vtol_conditions import (
                generate_transition_gust_conditions,
            )
            conditions.extend(generate_transition_gust_conditions(
                self.vtol_config.v_mca,
                self.vtol_config.v_transition_end, altitudes,
                wing_area_m2=getattr(cfg, 'wing_area_m2', 0.0),
                CL_transition=1.0,
                weight_N=(cfg.weight_cg_conditions[0].weight_N
                          if cfg.weight_cg_conditions else 0.0),
                CLalpha=getattr(cfg, 'CLalpha', 0.0),
                mean_chord_m=getattr(cfg, 'mean_chord_m', 1.0),
            ))
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
                elif cond.phase in (VTOLFlightPhase.TILT_TRANSITION,
                                    VTOLFlightPhase.TILT_STUCK):
                    rotor_forces = self._compute_rotor_forces_tilt(
                        cond, wc)
                else:
                    rotor_forces = {}

                sat = self._saturation_summary(
                    getattr(self, "_last_loads_map", None))
                case = self._condition_to_cert_case(cond, wc, rotor_forces,
                                                    saturation=sat)
                self.cases.append(case)

        return self.cases

    def summary(self) -> Dict[str, int]:
        """Return case count by VTOL category."""
        counts: Dict[str, int] = {}
        for c in self.cases:
            counts[c.category] = counts.get(c.category, 0) + 1
        return counts
