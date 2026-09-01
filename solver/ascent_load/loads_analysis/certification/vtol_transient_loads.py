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

    # Baseline hover (all rotors active, nz=1.0)
    baseline_wing_Mx: float = 0.0
    baseline_wing_Vy: float = 0.0

    # DAF = peak_transient / max(baseline, qs_oei)
    daf_wing_Mx: float = 1.0
    daf_wing_Vy: float = 1.0
    daf_boom_Mx: float = 1.0

    # Critical time points from 6-DOF sim
    critical_points: List[CriticalTimePoint] = field(default_factory=list)

    # 피크 시점의 절점하중 {nid: ndarray(6)} — 케이스화해 VMT/포락선에
    # 참여시키기 위한 스냅샷 (호버 돌풍 과도 경로에서 채움)
    peak_nodal_forces: Optional[Dict[int, np.ndarray]] = None


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
                 airfoil_config=None, n_workers: int = 0,
                 inflow_tau_coeff: float = None):
        # inflow_tau_coeff: Pitt-Peters 시정수 계수 재정의(민감도 스윕용,
        # r3 MC4). None이면 PittPetersInflow 기본값 4/(3pi).
        self.inflow_tau_coeff = inflow_tau_coeff
        self.bdf_model = bdf_model
        self.vtol_config = vtol_config
        self.aircraft_config = aircraft_config
        self.airfoil_config = airfoil_config
        self.n_workers = n_workers

        # Auto-detect unit system from model coordinate scale
        # N-mm-s models: extents ~1000-10000 mm; N-m-kg-s models: extents ~1-20 m
        max_extent = max(
            max(abs(n.xyz[i]) for n in bdf_model.nodes.values())
            for i in range(3)
        )
        if max_extent > 100.0:
            # Millimeter model (GACOMP style): N-mm-s, mass in tonnes
            self._length_scale = 1.0      # coordinates in mm
            self._mass_to_kg = 1000.0     # tonnes → kg
            self._length_to_m = 1e-3      # mm → m
            self._g_model = 9806.65       # mm/s²
            self._wing_root_threshold = 200.0  # mm
            logger.info("Detected N-mm-s unit system (extent=%.0f mm)", max_extent)
        else:
            # Meter model (UAM style): N-m-kg-s, mass in kg
            self._length_scale = 1.0      # coordinates in m
            self._mass_to_kg = 1.0        # already kg
            self._length_to_m = 1.0       # already m
            self._g_model = 9.80665       # m/s²
            self._wing_root_threshold = 0.2  # m (200mm)
            logger.info("Detected N-m-kg-s unit system (extent=%.1f m)", max_extent)

        # Pre-compute mass distribution and geometry
        self._node_masses = self._build_node_mass_map()
        self._cg = self._compute_cg()            # In model coords
        self._cg_mm = self._cg * (self._length_to_m / 1e-3)  # Always in mm (for rotor_dynamics)
        self._total_mass_kg = sum(m * self._mass_to_kg
                                   for m in self._node_masses.values())
        self._wing_root_y = self._detect_wing_root_y()
        self._boom_junctions = self._detect_boom_junctions()

        # Pre-compute nominal rotor loads (for torque scaling)
        # Use heaviest weight condition (MTOW) for consistent force balance
        wc = max(aircraft_config.weight_cg_conditions,
                 key=lambda w: w.weight_N)
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
                    "wing root Y=%.3f, %d boom junctions",
                    self._total_mass_kg,
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

    def run_all_hover_gust(self, V_de_list: Tuple[float, ...] = (7.62,),
                           T_g: float = 0.5, V_wind: float = 10.0,
                           t_sim: float = 3.0, dt: float = 0.005,
                           dt_loads: float = 0.02,
                           ) -> List[TransientPeakResult]:
        """호버 1-cos 돌풍 과도 하중 (상하 양방향, 강도별).

        논문 2의 통합 MultiRotorAggregate 경로: 로터별 Pitt-Peters
        유입류 ODE + 기체 상하 동역학 + 로터별 돌풍 도달 시차
        (Δt_i = x_i / V_wind). 피크 시점의 절점하중 스냅샷을 함께
        저장해 배치 러너가 인증 케이스로 합류시킨다. DAF 기준은
        전 로터가 시차·유입류 지연 없이 피크 돌풍을 동시에 받는
        무시차 준정적 상한(quasi-static no-lag bound)이다.
        """
        t0 = time_mod.time()
        results = []
        for V_de in V_de_list:
            for sign, tag in ((-1.0, "up"), (+1.0, "down")):
                result = self._run_hover_gust_event(
                    sign * V_de, T_g, V_wind,
                    t_sim=t_sim, dt=dt, dt_loads=dt_loads)
                result.failed_rotor_label = (
                    f"Vde={V_de:.2f}m/s {tag} Tg={T_g:.2f}s")
                results.append(result)
        elapsed = time_mod.time() - t0
        logger.info("Hover-gust transient loads: %d events, %.1fs",
                    len(results), elapsed)
        return results

    # ------------------------------------------------------------------
    # Core event runners
    # ------------------------------------------------------------------

    def _run_oei_event(self, rotor: RotorDef,
                       with_recovery: bool = False,
                       t_recognition: float = 1.0,
                       t_sim: float = 5.0, dt: float = 0.005,
                       failure_time: float = 0.1,
                       dt_loads: float = 0.02,
                       ) -> TransientPeakResult:
        """Run single OEI event and compute structural loads time history.

        failure_time is kept small (0.1s) to minimise the uncontrolled
        pre-failure hover phase.  A baseline FCC attitude-hold PD damper
        is always active to prevent unrealistic divergence.
        """
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
            base_force = make_oei_recovery_force_func(
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
        else:
            base_force = make_oei_force_func(
                self.vtol_config,
                failed_rotor_id=rotor.rotor_id,
                failure_time=failure_time,
                weight_N=self._weight_N,
                rho=self._rho,
                cg_position=self._cg_mm,
            )

        # Always wrap with FCC baseline attitude-hold PD to prevent
        # unrealistic hover divergence.  For recovery cases the internal
        # PD drives the same setpoint (phi=theta=0), so the two loops
        # are additive and remain stable.
        ext_force = self._wrap_with_attitude_hold(base_force)

        # Run sim
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
                       jam_time: float = 0.1,
                       dt_loads: float = 0.02,
                       ) -> TransientPeakResult:
        """Run single rotor jam event and compute structural loads."""
        from ...rotor.rotor_dynamics import make_rotor_jam_force_func

        condition = VTOLCondition(
            label=f"Jam_{rotor.label}",
            phase=VTOLFlightPhase.ROTOR_JAM,
            V_eas=0.0, nz=1.0,
            altitude_m=0.0,
            failed_rotor_id=rotor.rotor_id,
            far_section="SC-VTOL.2150",
        )

        # Build jam force function with attitude stabilisation
        base_force = make_rotor_jam_force_func(
            self.vtol_config,
            jammed_rotor_id=rotor.rotor_id,
            jam_time=jam_time,
            weight_N=self._weight_N,
            rho=self._rho,
            cg_position=self._cg_mm,
        )
        ext_force = self._wrap_with_attitude_hold(base_force)

        # Run sim
        initial_state, de_trim = trim_initial_state(
            self._params, 5.0, nz=1.0)
        initial_state.u = 0.1
        initial_state.w = 0.0
        initial_state.theta = 0.0

        ctrl = ControlInput(delta_e=de_trim)
        sim_runner = VTOLSimRunner(
            self._params, self.vtol_config, cg_position=self._cg_mm)

        history = integrate_6dof(
            self._params, initial_state, lambda t: ctrl,
            t_span=(0.0, t_sim), dt=dt,
            external_force_func=ext_force,
        )
        compute_nz_from_history(self._params, history)
        criticals = sim_runner._extract_critical_points(
            history, condition, self._weight_N)

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

    def _run_hover_gust_event(self, w_amp: float, T_g: float,
                              V_wind: float, t_sim: float = 3.0,
                              dt: float = 0.005, dt_loads: float = 0.02,
                              ) -> TransientPeakResult:
        """단일 1-cos 호버 돌풍 이벤트 — 집계 유입류 ODE + 하중 이력.

        w_amp 부호 규약은 Pitt-Peters 유입류와 동일하게 양(+)=하강
        돌풍. 무시차 준정적 상한(모든 로터가 피크 돌풍을 동시에,
        유입류 완전 발달 상태로 받는 경우)을 DAF 기준으로 쓴다.
        """
        from scipy.integrate import solve_ivp

        from ...rotor.dynamic_inflow import (
            MultiRotorAggregate, PittPetersInflow,
        )

        lift_rotors = list(self.vtol_config.hover_rotors)
        n = len(lift_rotors)
        if n == 0:
            raise ValueError("hover_rotors is empty")
        T_hover = self._weight_N / n

        _tk = ({'tau_coeff': self.inflow_tau_coeff}
               if self.inflow_tau_coeff else {})
        pps = [PittPetersInflow(rotor_radius=r.blade.radius, rho=self._rho,
                                T_steady=T_hover, **_tk)
               for r in lift_rotors]
        # hub_position은 모델 단위와 무관하게 설정 규약상 mm다
        # (rotor_config.RotorDef, rotor_dynamics의 1e-3 하드코딩과 동일).
        # 모델 단위 환산계수를 쓰면 미터 덱에서 1000배 어긋난다.
        x_m = np.array([float(r.hub_position[0]) * 1e-3
                        for r in lift_rotors])
        x_min = float(x_m.min())
        agg = MultiRotorAggregate(rotors=pps, rotor_x_positions=x_m,
                                  body_mass=self._total_mass_kg)

        # 로터별 도달 시차는 derivative 내부에서 x_i/V_wind로 처리하므로
        # 여기서는 지연 없는 기준 1-cos 프로파일만 준다
        def gust_func(t: float, x: float) -> float:
            if 0.0 <= t <= T_g:
                return 0.5 * w_amp * (1.0 - math.cos(2.0 * math.pi
                                                     * t / T_g))
            return 0.0

        _ = x_min   # (미사용 — 절대 x는 공통 오프셋만 더함)
        t_eval = np.arange(0.0, t_sim, dt)
        sol = solve_ivp(
            lambda t, y: agg.derivative(y, t, gust_func, V_wind=V_wind),
            (0.0, t_sim), agg.initial_state(),
            t_eval=t_eval, max_step=dt)
        nus = sol.y[:n, :]                      # (n, N_t)

        # 로터별 추력 이력 (음수 추력은 물리적으로 0으로 클립)
        thrust = np.empty_like(nus)
        for i, pp in enumerate(pps):
            thrust[i] = np.maximum(
                T_hover + 2.0 * self._rho * pp.A * pp.nu_steady
                * (nus[i] - pp.nu_steady), 0.0)

        # ---- 하중 이력 (dt_loads 샘플) ----
        step = max(1, int(dt_loads / dt))
        indices = list(range(0, len(sol.t), step))
        N_loads = len(indices)
        t_loads = np.zeros(N_loads)
        wing_Mx_L = np.zeros(N_loads)
        wing_Mx_R = np.zeros(N_loads)
        wing_Vy_L = np.zeros(N_loads)
        wing_Vy_R = np.zeros(N_loads)
        boom_Mx = np.zeros(N_loads)
        nz_hist = np.zeros(N_loads)

        def _nodal_forces(T_per_rotor: np.ndarray):
            rotor_nodal: Dict[int, np.ndarray] = {}
            total_Fz = 0.0
            for r, T in zip(lift_rotors, T_per_rotor):
                shaft = r.effective_shaft_axis
                F_thrust = T * shaft
                nom = self._nominal_loads.get(r.rotor_id)
                if nom and self._T_nominal > 0:
                    Q = nom.torque * abs(T / self._T_nominal) ** 1.5
                else:
                    Q = 0.0
                rot_sign = (1.0 if r.rotation_dir == RotationDir.CW
                            else -1.0)
                fvec = np.zeros(6)
                fvec[:3] = F_thrust
                fvec[3:] = -rot_sign * Q * shaft
                nid = r.hub_node_id
                if nid in rotor_nodal:
                    rotor_nodal[nid] += fvec
                else:
                    rotor_nodal[nid] = fvec
                total_Fz += F_thrust[2]
            nz_struct = (total_Fz / self._weight_N
                         if self._weight_N > 0 else 1.0)
            forces = {nid: f.copy() for nid, f in rotor_nodal.items()}
            for nid, m_node in self._node_masses.items():
                if self.bdf_model.nodes.get(nid) is None:
                    continue
                fvec = np.zeros(6)
                fvec[2] = -m_node * self._mass_to_kg * nz_struct * _G
                if nid in forces:
                    forces[nid] += fvec
                else:
                    forces[nid] = fvec
            return forces, nz_struct

        peak_forces = None
        peak_absMx = -1.0
        for k, idx in enumerate(indices):
            t_loads[k] = sol.t[idx]
            forces, nz_struct = _nodal_forces(thrust[:, idx])
            nz_hist[k] = nz_struct
            Vy_L, Mx_L, Vy_R, Mx_R = self._wing_root_section_loads(forces)
            wing_Mx_L[k], wing_Mx_R[k] = Mx_L, Mx_R
            wing_Vy_L[k], wing_Vy_R[k] = Vy_L, Vy_R
            boom_Mx[k] = self._boom_root_section_loads(forces)
            m = max(abs(Mx_L), abs(Mx_R))
            if m > peak_absMx:
                peak_absMx = m
                peak_forces = forces

        # ---- 무시차 준정적 상한 (DAF 기준) ----
        T_qs = np.array([max(T_hover - 2.0 * self._rho * pp.A
                             * pp.nu_steady * w_amp, 0.0) for pp in pps])
        qs_forces, _ = _nodal_forces(T_qs)
        qVy_L, qMx_L, qVy_R, qMx_R = self._wing_root_section_loads(qs_forces)
        qs_boom = self._boom_root_section_loads(qs_forces)

        result = TransientPeakResult()
        result.event_type = "gust"
        result.with_recovery = False
        result.t_recognition = float("inf")
        result.sim_time = sol.t.copy()
        result.nz_history = np.interp(sol.t, t_loads, nz_hist)
        result.t_loads = t_loads
        result.wing_Mx_L, result.wing_Mx_R = wing_Mx_L, wing_Mx_R
        result.wing_Vy_L, result.wing_Vy_R = wing_Vy_L, wing_Vy_R
        result.boom_Mx_max = boom_Mx
        result.rotor_thrusts = {r.label: thrust[i].copy()
                                for i, r in enumerate(lift_rotors)}
        result.peak_nodal_forces = peak_forces

        abs_L = float(np.nanmax(np.abs(wing_Mx_L)))
        abs_R = float(np.nanmax(np.abs(wing_Mx_R)))
        if abs_L >= abs_R:
            result.peak_wing_Mx = abs_L
            result.peak_wing_Mx_time = t_loads[int(np.nanargmax(
                np.abs(wing_Mx_L)))]
            result.peak_wing_Mx_side = "L"
        else:
            result.peak_wing_Mx = abs_R
            result.peak_wing_Mx_time = t_loads[int(np.nanargmax(
                np.abs(wing_Mx_R)))]
            result.peak_wing_Mx_side = "R"
        result.peak_wing_Vy = max(float(np.nanmax(np.abs(wing_Vy_L))),
                                  float(np.nanmax(np.abs(wing_Vy_R))))
        result.peak_boom_Mx = float(np.nanmax(np.abs(boom_Mx)))
        if result.peak_boom_Mx > 0:
            result.peak_boom_Mx_time = t_loads[int(np.nanargmax(
                np.abs(boom_Mx)))]

        result.qs_wing_Mx = max(abs(qMx_L), abs(qMx_R))
        result.qs_wing_Vy = max(abs(qVy_L), abs(qVy_R))
        result.qs_boom_Mx = abs(qs_boom)
        result.daf_wing_Mx = (result.peak_wing_Mx / result.qs_wing_Mx
                              if result.qs_wing_Mx > 1e-9 else 1.0)
        result.daf_wing_Vy = (result.peak_wing_Vy / result.qs_wing_Vy
                              if result.qs_wing_Vy > 1e-9 else 1.0)
        result.daf_boom_Mx = (result.peak_boom_Mx / result.qs_boom_Mx
                              if result.qs_boom_Mx > 1e-9 else 1.0)

        logger.info("  gust w=%.2f m/s Tg=%.2fs: peak Mx=%.0f N·mm "
                    "(no-lag ratio %.2f), nz %.3f..%.3f",
                    w_amp, T_g, result.peak_wing_Mx, result.daf_wing_Mx,
                    float(nz_hist.min()), float(nz_hist.max()))
        return result

    # ------------------------------------------------------------------
    # 천이 과도 돌풍 (전진비 유입류) — 회랑 준정적 돌풍의 과도 대응
    # ------------------------------------------------------------------
    def run_all_transition_gust(self, V_de_list: Tuple[float, ...] = (7.62,),
                                T_g: float = 0.5, V_wind: float = 10.0,
                                n_speeds: int = 3, t_sim: float = 3.0,
                                dt: float = 0.005, dt_loads: float = 0.02,
                                ) -> List["TransientPeakResult"]:
        """천이 회랑 1-cos 돌풍 과도 하중 (상하 양방향, 대표 속도점).

        논문 2 집계 모델의 전진비 일반화: 로터별 Pitt-Peters 유입류가
        동결 질량유량 U = sqrt(V^2 + nu^2)로 스케일되고(V=0에서 호버와
        정확히 중첩), 날개는 준정적 돌풍 응답 ΔL = ½ρVS·CLα·(−w−ż)로
        기체 상하 동역학에 참여한다. 돌풍 도달 시차는 전방 조우 폐쇄
        속도 (V_wind + V)로 계산돼 전진 속도가 클수록 시차 감쇠가
        줄어든다. 하중 스냅샷은 재트림 스크린과 같은 트림 아핀
        조립(C0 + nz_eff·C1 + 로터 + 관성 + relief)을 쓰므로 날개
        공력 굽힘이 올바르게 포함된다.
        """
        from .retrim_events import solve_transition_trim_affine
        from .vtol_conditions import generate_transition_conditions

        t0 = time_mod.time()
        wc = self.aircraft_config.weight_cg_conditions[0]
        conds = [c for c in generate_transition_conditions(
            self.vtol_config.v_mca, self.vtol_config.v_transition_end,
            [0.0],
            wing_area_m2=getattr(self.aircraft_config, "wing_area_m2",
                                 0.0),
            CL_transition=1.0, weight_N=wc.weight_N)
            if c.thrust_fraction > 0.05 and abs(c.nz - 1.0) < 1e-9]
        if not conds:
            return []
        # 대표 속도점: 회랑을 고르게 덮는 최대 n_speeds개
        idx = np.unique(np.linspace(0, len(conds) - 1,
                                    min(n_speeds, len(conds))
                                    ).round().astype(int))
        conds = [conds[i] for i in idx]
        pairs = solve_transition_trim_affine(
            self.bdf_model, self.aircraft_config, wc,
            [c.V_eas for c in conds])

        results = []
        for cond in conds:
            key = round(cond.V_eas, 3)
            if key not in pairs:
                continue
            C0f, C1f = pairs[key]
            for V_de in V_de_list:
                for sign, tag in ((-1.0, "up"), (+1.0, "down")):
                    r = self._run_transition_gust_event(
                        cond, C0f, C1f, sign * V_de, T_g, V_wind,
                        t_sim=t_sim, dt=dt, dt_loads=dt_loads)
                    r.failed_rotor_label = (
                        f"TR V={cond.V_eas:.1f}m/s Vde={V_de:.2f}m/s "
                        f"{tag} Tg={T_g:.2f}s")
                    results.append(r)
        logger.info("Transition-gust transient loads: %d events, %.1fs",
                    len(results), time_mod.time() - t0)
        return results

    def _run_transition_gust_event(self, cond, C0f, C1f, w_amp: float,
                                   T_g: float, V_wind: float,
                                   t_sim: float = 3.0, dt: float = 0.005,
                                   dt_loads: float = 0.02,
                                   ) -> "TransientPeakResult":
        """단일 천이 1-cos 돌풍 이벤트 — 전진비 유입류 + 날개 준정적."""
        from scipy.integrate import solve_ivp

        from ...rotor.dynamic_inflow import (
            MultiRotorAggregate, PittPetersInflow,
        )
        from ..trim_loads import (
            apply_inertia_relief, compute_nodal_inertial_forces,
        )
        from .vtol_load_case_matrix import VTOLLoadCaseMatrix

        lift_rotors = list(self.vtol_config.hover_rotors)
        n = len(lift_rotors)
        W = self._weight_N
        V = float(cond.V_eas)
        S = getattr(self.aircraft_config, "wing_area_m2", 0.0)
        CLa = getattr(self.aircraft_config, "CLalpha", 0.0)
        rho = self._rho

        # 로터/푸셔 명목 6분력 (파이프라인 천이 경로와 동일).
        # BEMT가 콜렉티브 한계로 포화할 수 있으므로, 명목 추력은
        # 스케줄 목표(tf·W/n)가 아니라 달성된 허브 수직력으로 잡고
        # 잔여는 파이프라인 관례대로 날개(nz_eff)가 부담한다.
        matrix = VTOLLoadCaseMatrix(self.vtol_config,
                                    self.aircraft_config)
        wc = self.aircraft_config.weight_cg_conditions[0]
        forces_nom = matrix._compute_rotor_forces_transition(cond, wc)
        hub_ids = {r.hub_node_id for r in lift_rotors}
        rotor_vec = {r.hub_node_id:
                     np.array(forces_nom[r.hub_node_id], float)
                     for r in lift_rotors}
        pusher = {nid: np.array(v, float)
                  for nid, v in forces_nom.items() if nid not in hub_ids}
        T_ach = np.array([max(rotor_vec[r.hub_node_id][2], 1e-6)
                          for r in lift_rotors])   # 달성 수직 추력 [N]
        L0 = W - float(T_ach.sum())                # 날개 명목 양력 [N]

        _tk = ({'tau_coeff': self.inflow_tau_coeff}
               if self.inflow_tau_coeff else {})
        pps = [PittPetersInflow(rotor_radius=r.blade.radius, rho=rho,
                                T_steady=float(T_i), V_forward=V, **_tk)
               for r, T_i in zip(lift_rotors, T_ach)]
        # hub_position은 설정 규약상 항상 mm (위 주석 참조)
        x_m = np.array([float(r.hub_position[0]) * 1e-3
                        for r in lift_rotors])
        agg = MultiRotorAggregate(rotors=pps, rotor_x_positions=x_m,
                                  body_mass=self._total_mass_kg)
        V_rel = V_wind + V                        # 전방 조우 폐쇄 속도
        x_wing_m = float(self._cg_mm[0]) * 1e-3   # _cg_mm은 이미 mm

        def gust(t: float, x: float) -> float:
            if 0.0 <= t <= T_g:
                return 0.5 * w_amp * (1.0 - math.cos(2.0 * math.pi
                                                     * t / T_g))
            return 0.0

        def wing_dL_up(t: float, z_dot: float) -> float:
            """날개 준정적 돌풍 양력 증분 (+상방) [N]."""
            t_w = t - x_wing_m / V_rel
            w_wing = gust(t_w, x_wing_m) if t_w >= 0.0 else 0.0
            return 0.5 * rho * V * S * CLa * (-w_wing - z_dot)

        def rhs(t, y):
            dydt = agg.derivative(y, t, gust, V_wind=V_rel)
            z_dot = y[agg._idx_z_dot]
            dydt[agg._idx_z_dot] += (wing_dL_up(t, z_dot)
                                     / self._total_mass_kg)
            return dydt

        t_eval = np.arange(0.0, t_sim, dt)
        sol = solve_ivp(rhs, (0.0, t_sim), agg.initial_state(),
                        t_eval=t_eval, max_step=dt)
        nus = sol.y[:n, :]
        z_dot_hist = sol.y[agg._idx_z_dot, :]
        thrust = np.empty_like(nus)
        for i, pp in enumerate(pps):
            thrust[i] = np.maximum(
                T_ach[i] + pp.thrust_slope * (nus[i] - pp.nu_steady),
                0.0)

        # ---- 하중 이력: 트림 아핀 조립 (재트림 스크린과 동일 경로) ----
        def _assemble(T_arr: np.ndarray, dL_up: float):
            nz_eff = (L0 + dL_up) / W
            forces: Dict[int, np.ndarray] = {}

            def _acc(fdict, scale=1.0):
                for nid, v in fdict.items():
                    forces[nid] = (forces.get(nid, np.zeros(6))
                                   + scale * np.asarray(v, float))

            _acc(C0f)
            _acc(C1f, nz_eff)
            _acc(pusher)
            total_Tz = 0.0
            for i, (r, T) in enumerate(zip(lift_rotors, T_arr)):
                scale = T / T_ach[i]
                _acc({r.hub_node_id: rotor_vec[r.hub_node_id]}, scale)
                total_Tz += scale * rotor_vec[r.hub_node_id][2]
            nz_struct = (total_Tz + L0 + dL_up) / W if W > 0 else 1.0
            inertial = compute_nodal_inertial_forces(
                self.bdf_model, nz_struct - nz_eff, self._g_model)
            _acc(inertial)
            # 관성 해제는 모델 좌표계·단위에서 수행한다 (_cg_mm은 mm
            # 고정이라 미터 덱에서 팔이 1000배가 된다)
            apply_inertia_relief(self.bdf_model, {}, forces,
                                 cg=self._cg, g=self._g_model)
            return forces, nz_struct

        step = max(1, int(dt_loads / dt))
        indices = list(range(0, len(sol.t), step))
        N_loads = len(indices)
        t_loads = np.zeros(N_loads)
        wing_Mx_L = np.zeros(N_loads)
        wing_Mx_R = np.zeros(N_loads)
        wing_Vy_L = np.zeros(N_loads)
        wing_Vy_R = np.zeros(N_loads)
        boom_Mx = np.zeros(N_loads)
        nz_hist = np.zeros(N_loads)
        peak_forces = None
        peak_absMx = -1.0
        for k, idx in enumerate(indices):
            t_k = sol.t[idx]
            t_loads[k] = t_k
            dL = wing_dL_up(t_k, z_dot_hist[idx])
            forces, nz_struct = _assemble(thrust[:, idx], dL)
            nz_hist[k] = nz_struct
            Vy_L, Mx_L, Vy_R, Mx_R = self._wing_root_section_loads(forces)
            wing_Mx_L[k], wing_Mx_R[k] = Mx_L, Mx_R
            wing_Vy_L[k], wing_Vy_R[k] = Vy_L, Vy_R
            boom_Mx[k] = self._boom_root_section_loads(forces)
            m = max(abs(Mx_L), abs(Mx_R))
            if m > peak_absMx:
                peak_absMx = m
                peak_forces = forces

        # ---- 무시차 준정적 상한 (전 로터·날개가 피크 돌풍 동시 수신) ----
        T_qs = np.array([max(T_i - pp.thrust_slope * w_amp, 0.0)
                         for T_i, pp in zip(T_ach, pps)])
        dL_qs = 0.5 * rho * V * S * CLa * (-w_amp)
        qs_forces, _ = _assemble(T_qs, dL_qs)
        qVy_L, qMx_L, qVy_R, qMx_R = self._wing_root_section_loads(qs_forces)
        qs_boom = self._boom_root_section_loads(qs_forces)

        result = TransientPeakResult()
        result.event_type = "gust"
        result.with_recovery = False
        result.t_recognition = float("inf")
        result.sim_time = sol.t.copy()
        result.nz_history = np.interp(sol.t, t_loads, nz_hist)
        result.t_loads = t_loads
        result.wing_Mx_L, result.wing_Mx_R = wing_Mx_L, wing_Mx_R
        result.wing_Vy_L, result.wing_Vy_R = wing_Vy_L, wing_Vy_R
        result.boom_Mx_max = boom_Mx
        result.rotor_thrusts = {r.label: thrust[i].copy()
                                for i, r in enumerate(lift_rotors)}
        result.peak_nodal_forces = peak_forces

        abs_L = float(np.nanmax(np.abs(wing_Mx_L)))
        abs_R = float(np.nanmax(np.abs(wing_Mx_R)))
        if abs_L >= abs_R:
            result.peak_wing_Mx = abs_L
            result.peak_wing_Mx_time = t_loads[int(np.nanargmax(
                np.abs(wing_Mx_L)))]
            result.peak_wing_Mx_side = "L"
        else:
            result.peak_wing_Mx = abs_R
            result.peak_wing_Mx_time = t_loads[int(np.nanargmax(
                np.abs(wing_Mx_R)))]
            result.peak_wing_Mx_side = "R"
        result.peak_wing_Vy = max(float(np.nanmax(np.abs(wing_Vy_L))),
                                  float(np.nanmax(np.abs(wing_Vy_R))))
        result.peak_boom_Mx = float(np.nanmax(np.abs(boom_Mx)))
        if result.peak_boom_Mx > 0:
            result.peak_boom_Mx_time = t_loads[int(np.nanargmax(
                np.abs(boom_Mx)))]
        result.qs_wing_Mx = max(abs(qMx_L), abs(qMx_R))
        result.qs_wing_Vy = max(abs(qVy_L), abs(qVy_R))
        result.qs_boom_Mx = abs(qs_boom)
        result.daf_wing_Mx = (result.peak_wing_Mx / result.qs_wing_Mx
                              if result.qs_wing_Mx > 1e-9 else 1.0)
        result.daf_wing_Vy = (result.peak_wing_Vy / result.qs_wing_Vy
                              if result.qs_wing_Vy > 1e-9 else 1.0)
        result.daf_boom_Mx = (result.peak_boom_Mx / result.qs_boom_Mx
                              if result.qs_boom_Mx > 1e-9 else 1.0)
        logger.info("  TR gust V=%.1f w=%.2f: peak Mx=%.0f N·mm "
                    "(no-lag ratio %.2f), nz %.3f..%.3f",
                    V, w_amp, result.peak_wing_Mx, result.daf_wing_Mx,
                    float(nz_hist.min()), float(nz_hist.max()))
        return result

    # ------------------------------------------------------------------
    # 틸트 변환 기동 스윕 (B단계) — 준정적 스케줄 + 자이로스코픽 항
    # ------------------------------------------------------------------
    def run_tilt_conversion_sweep(self, V_eas: Optional[float] = None,
                                  sigma_rate_dps: float = 10.0,
                                  dt: float = 0.25,
                                  blades_mass_kg: float = 1.44,
                                  components=None) -> dict:
        """일정 속도 변환 기동의 하중 스윕 — σ(t) 램프 + 자이로 항.

        변환은 느린 기동(σ̇ ~ 10°/s, 수 초 척도; 유입류 시간상수
        ~0.05 s ≪ 변환 시간)이므로 골격은 준정적 회랑 배분이고,
        시간 의존 항은 틸트 각속도가 로터 각운동량에 만드는
        자이로스코픽 허브 모멘트 −dH/dt = −s·I_p·Ω·σ̇·[cosσ,0,−sinσ]
        (s = 회전 방향 부호)이다. 하중 조립은 재트림/천이 돌풍과
        같은 트림 아핀 경로(C0 + nz_eff·C1 + 로터 + 관성 + relief)를
        쓴다. 반환 dict: t, sigma_deg, VMT 피크(회랑 중심 정적 케이스
        대비 비), gyro_share(자이로 항이 피크 비틀림에 기여한 비율).
        """
        import math as _m

        from ..trim_loads import (
            apply_inertia_relief, compute_nodal_inertial_forces,
        )
        from .batch_runner import BatchResult, CaseResult
        from .retrim_events import solve_transition_trim_affine
        from .vmt_bridge import compute_vmt_for_batch
        from .vtol_conditions import (
            VTOLCondition, VTOLFlightPhase, tilt_allocation,
        )
        from .vtol_load_case_matrix import VTOLLoadCaseMatrix
        from ...rotor.rotor_config import RotationDir, RotorType

        wc = self.aircraft_config.weight_cg_conditions[0]
        W = wc.weight_N
        S = getattr(self.aircraft_config, "wing_area_m2", 0.0)
        matrix = VTOLLoadCaseMatrix(self.vtol_config,
                                    self.aircraft_config)
        tilt_rotors = [r for r in self.vtol_config.rotors
                       if r.rotor_type == RotorType.TILT]
        n_tilt = len(tilt_rotors)
        n_aft = sum(1 for r in self.vtol_config.rotors
                    if r.rotor_type == RotorType.LIFT)

        # 회랑 가용 σ 대역
        import numpy as _np
        if V_eas is None:
            V_eas = 0.5 * (self.vtol_config.v_mca
                           + self.vtol_config.v_transition_end)
        from .vtol_conditions import TILT_GRID_DEG
        grid = _np.arange(TILT_GRID_DEG, 90.0 + 1e-9,
                          TILT_GRID_DEG)
        feas = [sg for sg in grid
                if tilt_allocation(V_eas, float(sg), 1.0, W, S, 1.0,
                                   self._rho, n_tilt, n_aft)[3]]
        if not feas:
            raise ValueError(f"no feasible corridor at V={V_eas}")
        s0, s1 = feas[0] + 1.0, feas[-1] - 1.0
        t_conv = (s1 - s0) / sigma_rate_dps
        t_arr = _np.arange(0.0, t_conv + 1e-9, dt)
        sig_arr = s0 + sigma_rate_dps * t_arr
        sd_rad = _m.radians(sigma_rate_dps)

        pair = solve_transition_trim_affine(
            self.bdf_model, self.aircraft_config, wc, [V_eas])
        C0f, C1f = pair[round(V_eas, 3)]

        def _assemble(sigma_deg: float, with_gyro: bool):
            cond = VTOLCondition(
                label="sweep", phase=VTOLFlightPhase.TILT_TRANSITION,
                V_eas=V_eas, nz=1.0, tilt_deg=float(sigma_deg))
            forces = matrix._compute_rotor_forces_tilt(cond, wc)
            if with_gyro:
                s_r = _m.radians(sigma_deg)
                dHdir = _np.array([_m.cos(s_r), 0.0, -_m.sin(s_r)])
                for r in tilt_rotors:
                    I_p = blades_mass_kg * r.blade.radius ** 2 / 3.0
                    omega = r.rpm_hover * 2.0 * _m.pi / 60.0
                    sgn = (1.0 if r.rotation_dir == RotationDir.CW
                           else -1.0)
                    M = -sgn * I_p * omega * sd_rad * dHdir * 1e3
                    f = forces.setdefault(r.hub_node_id,
                                          _np.zeros(6))
                    f[3:] += M                     # N·mm
            _F, _A, L_wing, _ok = tilt_allocation(
                V_eas, sigma_deg, 1.0, W, S, 1.0, self._rho,
                n_tilt, n_aft)
            nz_eff = L_wing / W
            out = {}
            for nid, v in C0f.items():
                out[nid] = _np.array(v, float).copy()
            for nid, v in C1f.items():
                out[nid] = out.get(nid, _np.zeros(6)) \
                    + nz_eff * _np.asarray(v, float)
            for nid, v in forces.items():
                out[nid] = out.get(nid, _np.zeros(6)) + v
            Fz_rotor = sum(v[2] for v in forces.values())
            nz_struct = (Fz_rotor + L_wing) / W
            inertial = compute_nodal_inertial_forces(
                self.bdf_model, nz_struct - nz_eff, self._g_model)
            for nid, v in inertial.items():
                out[nid] = out.get(nid, _np.zeros(6)) + v
            # 관성 해제는 모델 좌표계·단위에서 (위 주석 참조)
            apply_inertia_relief(self.bdf_model, {}, out,
                                 cg=self._cg, g=self._g_model)
            return out

        # 스윕 + 회랑 중심 정적 케이스 + 자이로 오프 대조
        batch = BatchResult()
        s_c = 0.5 * (s0 + s1)
        specs = ([("static", s_c, False), ("nogyro_c", s_c, True)]
                 + [(f"t{k}", float(sg), True)
                    for k, sg in enumerate(sig_arr)])
        for k, (tag, sg, gy) in enumerate(specs):
            batch.case_results.append(CaseResult(
                case_id=970000 + k, category="tilt_sweep",
                converged=True,
                nodal_forces=_assemble(sg, gy and tag != "static"),
                nz=1.0, label=f"{tag} s={sg:.1f}"))
            batch.completed_ids.add(970000 + k)
        vmt = compute_vmt_for_batch(self.bdf_model, batch,
                                    components=components,
                                    fuselage_cg_x=float(
                                        self._cg_mm[0]))

        def _wing_extreme(cid, qty):
            best = 0.0
            for comp in ("Right Wing", "Left Wing"):
                if comp in vmt[cid]:
                    arr = _np.asarray(vmt[cid][comp][qty], float)
                    m = arr[int(_np.argmax(_np.abs(arr)))]
                    if abs(m) > abs(best):
                        best = float(m)
            return best

        tor = [_wing_extreme(970000 + 2 + k, "torsion")
               for k in range(len(sig_arr))]
        ben = [_wing_extreme(970000 + 2 + k, "bending")
               for k in range(len(sig_arr))]
        tor_static = _wing_extreme(970000, "torsion")
        k_pk = int(_np.argmax(_np.abs(tor)))
        # 자이로 항 기여 — 허브 모멘트는 x·z축(롤·요)이고 CW/CCW
        # 쌍이 루트 단면에서는 상쇄되므로, 외곽 허브 안쪽 스테이션
        # 포함 전 스테이션에서 on/off 차의 최댓값으로 잰다.
        gyro_delta = 0.0
        for comp in ("Right Wing", "Left Wing"):
            for qty in ("torsion", "bending", "shear"):
                if comp not in vmt[970000]:
                    continue
                a_on = _np.asarray(vmt[970001][comp][qty], float)
                a_off = _np.asarray(vmt[970000][comp][qty], float)
                gyro_delta = max(gyro_delta,
                                 float(_np.abs(a_on - a_off).max()))
        gyro_share = gyro_delta / max(abs(tor[k_pk]), 1e-9)
        r0 = tilt_rotors[0]
        gyro_hub_Nmm = (blades_mass_kg * r0.blade.radius ** 2 / 3.0
                        * r0.rpm_hover * 2.0 * _m.pi / 60.0
                        * sd_rad * 1e3)
        return {
            "gyro_hub_moment_Nmm": float(gyro_hub_Nmm),
            "V_eas": V_eas, "sigma_rate_dps": sigma_rate_dps,
            "t": t_arr, "sigma_deg": sig_arr,
            "wing_torsion": _np.array(tor),
            "wing_bending": _np.array(ben),
            "peak_torsion": float(tor[k_pk]),
            "peak_sigma": float(sig_arr[k_pk]),
            "static_center_torsion": float(tor_static),
            "peak_over_static": float(abs(tor[k_pk])
                                      / max(abs(tor_static), 1e-9)),
            "gyro_share": float(gyro_share),
        }

    def _wrap_with_attitude_hold(self, base_force_func):
        """Wrap an external force function with FCC attitude-hold PD damper.

        In hover, the aircraft has no aerodynamic damping.  Real eVTOLs
        use always-active flight controllers to stabilise attitude.
        This wrapper adds proportional + derivative torques to simulate
        the FCC baseline attitude hold, preventing unrealistic divergence
        of the 6-DOF simulation before/after OEI events.

        PD gains are set to give ωn ≈ 3 rad/s, ζ ≈ 0.7 (typical FCC).
        """
        Ixx = self._params.Ixx
        Iyy = self._params.Iyy
        Izz = self._params.Izz

        omega_n = 6.0   # rad/s  — attitude natural frequency
        zeta = 1.0      # critically damped

        # PD gains: Kp = I × ωn², Kd = 2 ζ ωn I
        Kp_roll = Ixx * omega_n ** 2
        Kd_roll = 2.0 * zeta * omega_n * Ixx
        Kp_pitch = Iyy * omega_n ** 2
        Kd_pitch = 2.0 * zeta * omega_n * Iyy
        Kd_yaw = 2.0 * zeta * omega_n * Izz  # yaw rate damper only

        def stabilised_force(t, state):
            F_body, M_body = base_force_func(t, state)
            phi = state[6]
            theta = state[7]
            p = state[3]
            q = state[4]
            r = state[5]

            # Attitude-hold: command phi=0, theta=0
            M_out = np.array(M_body, dtype=float)
            M_out[0] += -Kp_roll * phi - Kd_roll * p
            M_out[1] += -Kp_pitch * theta - Kd_pitch * q
            M_out[2] += -Kd_yaw * r
            return F_body, M_out

        return stabilised_force

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

        # Sample indices — limit to failure_time + 1.0s to avoid
        # post-divergence artifacts in the uncontrolled phase
        t_max = failure_time + 1.0
        idx_max = min(N_sim - 1, int(t_max / dt_sim))
        indices = list(range(0, idx_max + 1, step))
        if indices[-1] != idx_max:
            indices.append(idx_max)
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

            # Skip diverged timesteps (NaN/inf from unstable sims)
            if not np.all(np.isfinite(state)):
                break  # Once diverged, all subsequent steps are useless

            phi = state[6]
            theta = state[7]
            p = state[3]
            q = state[4]
            r = state[5]
            p_dot = history.p_dot[idx]
            q_dot = history.q_dot[idx]
            r_dot = history.r_dot[idx]

            # Check for physical divergence (states still finite but unphysical)
            if (abs(phi) > np.pi/2 or abs(theta) > np.pi/2
                    or abs(p) > 20.0 or abs(q) > 20.0 or abs(r) > 20.0):
                break  # Aircraft tumbling — stop here

            # Also check angular accelerations for divergence
            if not (np.isfinite(p_dot) and np.isfinite(q_dot) and np.isfinite(r_dot)):
                break

            # --- 1. Rotor nodal forces ---
            rotor_nodal = {}
            total_Fz_rotor = 0.0
            for label, thrust_arr in rotor_thrusts.items():
                rotor = label_to_rotor.get(label)
                if rotor is None:
                    continue
                T = thrust_arr[idx]
                if not np.isfinite(T) or T < 1e-3:
                    continue

                shaft = rotor.effective_shaft_axis
                # Reaction force (opposes air acceleration)
                F_thrust = T * shaft  # Thrust acts upward on airframe
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
            nz_struct = total_Fz_rotor / self._weight_N if self._weight_N > 0 else 1.0

            # --- 3. Inertial forces at all mass nodes ---
            # F_inertial = -m_node × [nz×g (vertical) + α×r (angular accel)]
            alpha_vec = np.array([p_dot, q_dot, r_dot])  # body angular accel [rad/s²]
            omega_vec = np.array([p, q, r])  # body angular velocity [rad/s]
            cg_m = self._cg * self._length_to_m  # CG in meters

            inertial_nodal = {}
            for nid, m_node in self._node_masses.items():
                node = self.bdf_model.nodes.get(nid)
                if node is None:
                    continue

                r_m = node.xyz * self._length_to_m - cg_m  # position from CG in meters
                m_kg = m_node * self._mass_to_kg

                # nz-component (vertical inertial, dominant)
                Fz_inertial = -m_kg * nz_struct * _G  # [N]

                # Angular acceleration: F = -m × α×r  [N]
                F_alpha = -m_kg * np.cross(alpha_vec, r_m)

                # Centripetal: F = -m × ω×(ω×r)  [N]
                F_centripetal = -m_kg * np.cross(
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

        # Peak wing bending (max absolute across both sides, ignoring NaN)
        abs_L = float(np.nanmax(np.abs(wing_Mx_L))) if np.any(wing_Mx_L != 0) else 0.0
        abs_R = float(np.nanmax(np.abs(wing_Mx_R))) if np.any(wing_Mx_R != 0) else 0.0
        if abs_L >= abs_R and abs_L > 0:
            result.peak_wing_Mx = abs_L
            result.peak_wing_Mx_time = t_loads[np.nanargmax(np.abs(wing_Mx_L))]
            result.peak_wing_Mx_side = "L"
        elif abs_R > 0:
            result.peak_wing_Mx = abs_R
            result.peak_wing_Mx_time = t_loads[np.nanargmax(np.abs(wing_Mx_R))]
            result.peak_wing_Mx_side = "R"

        result.peak_wing_Vy = max(
            float(np.nanmax(np.abs(wing_Vy_L))) if np.any(wing_Vy_L != 0) else 0.0,
            float(np.nanmax(np.abs(wing_Vy_R))) if np.any(wing_Vy_R != 0) else 0.0)
        abs_boom = np.abs(boom_Mx_max)
        result.peak_boom_Mx = float(np.nanmax(abs_boom)) if np.any(abs_boom > 0) else 0.0
        if result.peak_boom_Mx > 0:
            result.peak_boom_Mx_time = t_loads[np.nanargmax(abs_boom)]

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
            F_thrust = T * shaft  # Thrust acts upward on airframe
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
            fvec[2] = -m_node * self._mass_to_kg * 1.0 * _G  # nz=1.0 [N]
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

        # Compute baseline hover loads (all rotors active, nz=1.0)
        bl_forces = {}
        T_nom = self._weight_N / len(hover_rotors)
        for rotor in hover_rotors:
            shaft = rotor.effective_shaft_axis
            fvec = np.zeros(6)
            fvec[:3] = T_nom * shaft
            nid = rotor.hub_node_id
            bl_forces[nid] = fvec.copy()
        for nid, m_node in self._node_masses.items():
            fvec = np.zeros(6)
            fvec[2] = -m_node * self._mass_to_kg * 1.0 * _G
            if nid in bl_forces:
                bl_forces[nid] += fvec
            else:
                bl_forces[nid] = fvec.copy()
        _, bl_Mx_L, _, bl_Mx_R = self._wing_root_section_loads(bl_forces)
        bl_Vy_L, _, bl_Vy_R, _ = self._wing_root_section_loads(bl_forces)
        result.baseline_wing_Mx = max(abs(bl_Mx_L), abs(bl_Mx_R))
        result.baseline_wing_Vy = max(abs(bl_Vy_L), abs(bl_Vy_R))

        # DAF — use baseline hover as denominator (more physical than QS OEI
        # which can be smaller than baseline due to thrust redistribution)
        denom_Mx = max(result.baseline_wing_Mx, result.qs_wing_Mx)
        denom_Vy = max(result.baseline_wing_Vy, result.qs_wing_Vy)
        if denom_Mx > 1e-3:
            result.daf_wing_Mx = result.peak_wing_Mx / denom_Mx
        if denom_Vy > 1e-3:
            result.daf_wing_Vy = result.peak_wing_Vy / denom_Vy
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
            nid = conm2.node_id
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
                n1 = self.bdf_model.nodes.get(elem.node_ids[0])
                n2 = self.bdf_model.nodes.get(elem.node_ids[1])
                if n1 is None or n2 is None:
                    continue
                L = np.linalg.norm(n2.xyz - n1.xyz)
                m_elem = rho * A * L  # tonnes (if rho in tonne/mm³)
                for nid in elem.node_ids[:2]:
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
                nodes = [self.bdf_model.nodes.get(nid) for nid in elem.node_ids[:4]]
                if any(n is None for n in nodes):
                    continue
                d1 = nodes[2].xyz - nodes[0].xyz
                d2 = nodes[3].xyz - nodes[1].xyz
                area = 0.5 * np.linalg.norm(np.cross(d1, d2))
                m_elem = rho * t * area
                for nid in elem.node_ids[:4]:
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
                nodes = [self.bdf_model.nodes.get(nid) for nid in elem.node_ids[:3]]
                if any(n is None for n in nodes):
                    continue
                d1 = nodes[1].xyz - nodes[0].xyz
                d2 = nodes[2].xyz - nodes[0].xyz
                area = 0.5 * np.linalg.norm(np.cross(d1, d2))
                m_elem = rho * t * area
                for nid in elem.node_ids[:3]:
                    masses[nid] = masses.get(nid, 0.0) + m_elem / 3.0

        return masses

    def _compute_cg(self) -> np.ndarray:
        """Compute CG position in model coordinates."""
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
        threshold = self._wing_root_threshold
        y_values = []
        for nid, node in self.bdf_model.nodes.items():
            y = abs(node.xyz[1])
            if y > threshold:
                y_values.append(y)

        if y_values:
            return min(y_values)
        # Fallback: 500mm or 0.5m depending on unit system
        return 500.0 if self._length_to_m < 0.01 else 0.5

    def _detect_boom_junctions(self) -> List[Tuple[float, float, int]]:
        """Detect boom-fuselage junctions from rotor hub positions.

        Returns list of (junction_y, hub_y, hub_node_id) tuples.
        All Y values in model coordinates.
        """
        junctions = []
        for rotor in self.vtol_config.hover_rotors:
            # Hub position is in mm (config); convert to model coords
            hub_y_model = rotor.hub_position[1] * 1e-3 / self._length_to_m
            # Junction is at the wing root Y on the same side
            junc_y = self._wing_root_y if hub_y_model > 0 else -self._wing_root_y
            junctions.append((junc_y, hub_y_model, rotor.hub_node_id))
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

        # Compute inertia — pass correct unit conversions for this model
        # cg_x is always in mm (config convention); convert to model coords
        cg_xyz = np.array([wc.cg_x * 1e-3 / self._length_to_m, 0.0, 0.0])
        inertia = compute_inertia_from_conm2(
            self.bdf_model, cg_xyz,
            mass_to_kg=self._mass_to_kg,
            length_to_m=self._length_to_m)

        # Convert reference dimensions from model coords to SI (m, m²)
        L = self._length_to_m  # model length unit → m
        rho, _, _ = isa_atmosphere(alt_m)

        params = AircraftParams(
            mass=inertia["mass_kg"],
            S=derivs.S_ref * L ** 2 if derivs.S_ref > 0 else 20.0,
            b=derivs.b_ref * L if derivs.b_ref > 0 else 12.0,
            c_bar=derivs.c_bar * L if derivs.c_bar > 0 else 1.5,
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
