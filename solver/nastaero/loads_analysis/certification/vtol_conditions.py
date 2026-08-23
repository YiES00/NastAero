"""VTOL flight conditions and load factor definitions.

Defines the VTOL-specific flight envelope including hover, transition,
OEI (One Engine Inoperative), and VTOL landing conditions per
EASA SC-VTOL and FAA Part 23 MOC for powered lift.

References
----------
- EASA SC-VTOL-01: Special Condition for VTOL Aircraft
- FAA Part 23 Means of Compliance for Powered Lift
- K-UAM Airworthiness Standards (한국형 도심항공교통)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class VTOLFlightPhase(Enum):
    """VTOL flight phase categories."""
    HOVER = "hover"
    TRANSITION = "transition"
    CRUISE = "cruise"
    VTOL_LANDING = "vtol_landing"
    OEI = "oei"
    ROTOR_JAM = "rotor_jam"
    TILT_TRANSITION = "tilt_transition"
    TILT_STUCK = "tilt_stuck"


@dataclass
class VTOLCondition:
    """A single VTOL flight condition.

    Attributes
    ----------
    label : str
        Descriptive label.
    phase : VTOLFlightPhase
        Flight phase category.
    V_eas : float
        Equivalent airspeed (m/s). 0 for hover.
    nz : float
        Load factor at CG. Includes rotor thrust component.
    altitude_m : float
        Altitude (m).
    thrust_fraction : float
        Fraction of total weight supported by rotors (1.0 = full hover).
    failed_rotor_id : int or None
        ID of failed rotor for OEI/jam conditions. None if all operative.
    rotor_rpm_factor : float
        RPM multiplier relative to design RPM (1.0 = nominal).
    sink_rate_mps : float
        Vertical sink rate for VTOL landing (m/s, positive = down).
    far_section : str
        Applicable regulation section reference.
    """
    label: str = ""
    phase: VTOLFlightPhase = VTOLFlightPhase.HOVER
    V_eas: float = 0.0
    nz: float = 1.0
    altitude_m: float = 0.0
    thrust_fraction: float = 1.0
    failed_rotor_id: Optional[int] = None
    rotor_rpm_factor: float = 1.0
    sink_rate_mps: float = 0.0
    tilt_deg: float = 0.0        # 틸트각 (0 = 수직/호버, 90 = 수평/순항)
    stuck_tilt_deg: Optional[float] = None  # 고착 나셀의 틸트각 (M6)
    far_section: str = "SC-VTOL"


def generate_hover_conditions(altitudes_m: List[float],
                               ) -> List[VTOLCondition]:
    """Generate hover load conditions.

    Includes:
    - Hover at 1.0g (steady hover)
    - Hover at 1.15g (maneuvering margin per SC-VTOL)
    - Hover with vertical gust (effective nz = 1.0 ± Δnz_gust)

    Parameters
    ----------
    altitudes_m : list of float
        Analysis altitudes.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    for alt in altitudes_m:
        # Steady hover
        conditions.append(VTOLCondition(
            label=f"Hover 1.0g h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.0,
            altitude_m=alt,
            thrust_fraction=1.0,
            far_section="SC-VTOL.2215",
        ))
        # Maneuvering hover (15% thrust margin)
        conditions.append(VTOLCondition(
            label=f"Hover 1.15g h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.15,
            altitude_m=alt,
            thrust_fraction=1.15,
            far_section="SC-VTOL.2215",
        ))
        # Vertical gust in hover (positive)
        conditions.append(VTOLCondition(
            label=f"Hover gust+ h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.3,
            altitude_m=alt,
            thrust_fraction=1.3,
            far_section="SC-VTOL.2215",
        ))
        # Vertical gust in hover (negative)
        conditions.append(VTOLCondition(
            label=f"Hover gust- h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=0.7,
            altitude_m=alt,
            thrust_fraction=0.7,
            far_section="SC-VTOL.2215",
        ))
    return conditions


def generate_oei_conditions(n_lift_rotors: int,
                             rotor_ids: List[int],
                             altitudes_m: List[float],
                             ) -> List[VTOLCondition]:
    """Generate One Engine Inoperative (OEI) conditions.

    Each lift rotor is failed in turn. Remaining rotors increase
    thrust to maintain hover (if possible).

    Parameters
    ----------
    n_lift_rotors : int
        Total number of lift rotors.
    rotor_ids : list of int
        IDs of rotors that can fail.
    altitudes_m : list of float
        Analysis altitudes.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    # OEI thrust factor: remaining rotors share full weight
    # rpm_factor > 1.0 to compensate (limited by motor capability)
    rpm_factor = min(1.15, (n_lift_rotors / (n_lift_rotors - 1)) ** 0.5)

    for alt in altitudes_m:
        for rid in rotor_ids:
            # OEI hover
            conditions.append(VTOLCondition(
                label=f"OEI Hover R{rid} fail h={alt:.0f}m",
                phase=VTOLFlightPhase.OEI,
                V_eas=0.0, nz=1.0,
                altitude_m=alt,
                thrust_fraction=1.0,
                failed_rotor_id=rid,
                rotor_rpm_factor=rpm_factor,
                far_section="SC-VTOL.2140",
            ))
    return conditions


def generate_transition_conditions(v_mca: float,
                                    v_transition_end: float,
                                    altitudes_m: List[float],
                                    n_speed_steps: int = 8,
                                    wing_area_m2: float = 0.0,
                                    CL_transition: float = 1.0,
                                    weight_N: float = 0.0,
                                    ) -> List[VTOLCondition]:
    """Generate transition corridor conditions.

    Thrust fraction is based on wing lift capability at each speed:
    the wing carries what it can (up to CL_transition), and rotors
    supplement the remainder. This avoids the unrealistic linear
    model that would reduce rotor thrust at speeds where the wing
    cannot yet provide sufficient lift.

    Parameters
    ----------
    v_mca : float
        Minimum controllable airspeed in VTOL mode (m/s EAS).
    v_transition_end : float
        Speed at which transition is complete (m/s EAS).
    altitudes_m : list of float
        Analysis altitudes.
    n_speed_steps : int
        Number of speed points in transition corridor.
    wing_area_m2 : float
        Wing reference area (m²). If 0, falls back to linear model.
    CL_transition : float
        Maximum CL used during transition (conservative, below CLmax
        to maintain stall margin). Default 1.0.
    weight_N : float
        Aircraft weight (N) for wing lift fraction calculation.
        If 0, falls back to linear model.

    Returns
    -------
    list of VTOLCondition
    """
    import numpy as np

    conditions = []
    # Extend speed range: start from low speed (just above hover)
    # to v_transition_end, with finer spacing
    v_start = max(1.0, v_mca * 0.5)  # Start below V_MCA
    speeds = np.linspace(v_start, v_transition_end, n_speed_steps)

    use_wing_capability = (wing_area_m2 > 0 and weight_N > 0)

    for alt in altitudes_m:
        from ..case_generator import isa_atmosphere
        rho, _, _ = isa_atmosphere(alt)

        for V in speeds:
            if use_wing_capability:
                # Wing-capability-based thrust fraction
                q = 0.5 * rho * V ** 2
                L_wing_max = CL_transition * q * wing_area_m2
                if L_wing_max >= weight_N:
                    tf = 0.0  # Wing carries full weight
                else:
                    tf = 1.0 - L_wing_max / weight_N
            else:
                # Fallback: linear model
                if v_transition_end > 0:
                    tf = max(0.0, 1.0 - V / v_transition_end)
                else:
                    tf = 1.0

            # nz = 1.0 (level flight) and nz_max for maneuvering
            for nz in [1.0, 1.5]:
                conditions.append(VTOLCondition(
                    label=(f"Transition V={V:.1f}m/s nz={nz:.1f} "
                           f"h={alt:.0f}m"),
                    phase=VTOLFlightPhase.TRANSITION,
                    V_eas=V, nz=nz,
                    altitude_m=alt,
                    thrust_fraction=tf * nz,
                    far_section="SC-VTOL.2215",
                ))
    return conditions


def transition_rotor_fraction(V: float, rho: float,
                              wing_area_m2: float,
                              CL_transition: float,
                              weight_N: float,
                              v_transition_end: float) -> float:
    """천이 속도 V에서의 로터 양력 분담 tf (nz=1 기준).

    generate_transition_conditions의 날개 능력 기반 모델과 동일:
    날개가 CL_transition까지 담당하고 로터가 잔여를 부담한다."""
    if wing_area_m2 > 0 and weight_N > 0:
        q = 0.5 * rho * V ** 2
        L_wing_max = CL_transition * q * wing_area_m2
        if L_wing_max >= weight_N:
            return 0.0
        return 1.0 - L_wing_max / weight_N
    if v_transition_end > 0:
        return max(0.0, 1.0 - V / v_transition_end)
    return 1.0


def transition_gust_delta_nz(V: float, share_rotor: float,
                             weight_N: float, wing_area_m2: float,
                             CLalpha: float, mean_chord_m: float,
                             gust_ude_fps: float = 66.0,
                             hover_gust_dnz: float = 0.3,
                             altitude_m: float = 0.0) -> float:
    """천이 회랑의 준정적 수직 돌풍 하중배수 증분.

    날개 성분은 Pratt(§23.341, 러프에어 Ude 기본 66 fps)로, 로터
    성분은 호버 돌풍 관례(±hover_gust_dnz)를 로터 분담 비율로
    스케일해 더한다:

        dn(V) = dn_Pratt(V; Ude) + share_rotor(V) * hover_gust_dnz

    V→0에서 호버 돌풍(±0.3)으로, 날개 전담(share→0)에서 순수
    Pratt로 연속적으로 이어진다."""
    from .vn_diagram import pratt_gust_delta_nz

    dn_wing = 0.0
    if V > 0 and wing_area_m2 > 0 and weight_N > 0 and CLalpha > 0:
        dn_wing = pratt_gust_delta_nz(
            V, gust_ude_fps, weight_N / wing_area_m2, CLalpha,
            mean_chord_m, weight_N, wing_area_m2,
            altitude_m=altitude_m)
    return dn_wing + share_rotor * hover_gust_dnz


def generate_transition_gust_conditions(
        v_mca: float, v_transition_end: float,
        altitudes_m: List[float], n_speed_steps: int = 8,
        wing_area_m2: float = 0.0, CL_transition: float = 1.0,
        weight_N: float = 0.0, CLalpha: float = 0.0,
        mean_chord_m: float = 1.0,
        gust_ude_fps: float = 66.0,
        hover_gust_dnz: float = 0.3) -> List[VTOLCondition]:
    """천이 회랑 준정적 수직 돌풍 케이스 (상승/하강 각 1개/속도점).

    nz_g = 1 ± dn(V)에서 로터는 스케줄 분담을 유지하고
    (tf_g = max(0, share·nz_g) — 하강 돌풍에서 추력은 0 아래로
    내려갈 수 없음), 날개가 잔여 nz_eff를 트림한다(음수 허용).
    호버 정적 돌풍(nz=1±0.3)과 고정익 Pratt 돌풍(VB/VC/VD) 사이의
    공백 구간을 잇는다.
    """
    import numpy as np

    from ..case_generator import isa_atmosphere

    v_start = max(1.0, v_mca * 0.5)
    speeds = np.linspace(v_start, v_transition_end, n_speed_steps)
    conditions = []
    for alt in altitudes_m:
        rho, _, _ = isa_atmosphere(alt)
        for V in speeds:
            share = transition_rotor_fraction(
                V, rho, wing_area_m2, CL_transition, weight_N,
                v_transition_end)
            dn = transition_gust_delta_nz(
                float(V), share, weight_N, wing_area_m2, CLalpha,
                mean_chord_m, gust_ude_fps=gust_ude_fps,
                hover_gust_dnz=hover_gust_dnz, altitude_m=alt)
            for sgn, tag in ((+1.0, "+"), (-1.0, "-")):
                nz_g = 1.0 + sgn * dn
                tf_g = max(0.0, share * nz_g)
                conditions.append(VTOLCondition(
                    label=(f"Transition gust{tag} V={V:.1f}m/s "
                           f"nz={nz_g:.2f} h={alt:.0f}m"),
                    phase=VTOLFlightPhase.TRANSITION,
                    V_eas=float(V), nz=nz_g,
                    altitude_m=alt,
                    thrust_fraction=tf_g,
                    far_section="SC-VTOL.2215/23.341",
                ))
    return conditions


# 틸트 회랑 가용성 판정 격자 (deg). 2.0에서는 저속 회랑이 단일
# 격자점으로 붕괴하고 하단 경계가 격자 시작점에 걸려 잘렸다.
# 0.5에서 회랑 폭이 0.25 격자 대비 0.5 이내로 수렴한다.
TILT_GRID_DEG = 0.5


def tilt_drag_estimate(V: float, weight_N: float) -> float:
    """천이/틸트 경로 공용 항력 추정 — 순항 L/D 기반 스케일.

    _compute_rotor_forces_transition의 푸셔 항력 관례와 동일:
    D = 0.05·W·(q/q_35) 에 하한 0.01·W."""
    q_ratio = (V / 35.0) ** 2
    return max(weight_N * 0.05 * q_ratio, weight_N * 0.01)


def tilt_allocation(V: float, sigma_deg: float, nz: float,
                    weight_N: float, wing_area_m2: float,
                    CL_transition: float, rho: float,
                    n_tilt: int = 4, n_aft: int = 4,
                    thrust_max_frac: float = 1.5):
    """제어법칙 없는 결정적 틸트 배분 — Fx/Fz 2식 평형.

    틸트각 sigma(0=수직, 90=수평, 추력 +x 전방)에서:
      Fx:  F_front·sin(sigma) = D(V)          (전열 총추력 F 결정)
      Fz:  L_wing + F·cos(sigma) + A_aft = nz·W
    날개는 능력 한계(CL·q·S)까지 담당하고 후열이 잔여를 부담한다.
    반환: (F_front_total, A_aft_total, L_wing, feasible) —
    feasible은 로터당 추력 한계(1.5·T_hover)와 잔여 부호 조건.
    """
    import math as _m

    W = weight_N
    T_hover = W / (n_tilt + n_aft)
    T_cap_front = n_tilt * thrust_max_frac * T_hover
    T_cap_aft = n_aft * thrust_max_frac * T_hover
    D = tilt_drag_estimate(V, W)
    s = _m.sin(_m.radians(max(sigma_deg, 1e-3)))
    c = _m.cos(_m.radians(sigma_deg))
    F = D / s
    q = 0.5 * rho * V ** 2
    L_cap = CL_transition * q * wing_area_m2
    L_wing = min(L_cap, max(0.0, nz * W - F * c))
    A = max(0.0, nz * W - F * c - L_wing)
    feasible = (F <= T_cap_front + 1e-9) and (A <= T_cap_aft + 1e-9) \
        and (nz * W - F * c - L_wing >= -0.05 * W)
    return F, A, L_wing, feasible


def generate_tilt_transition_conditions(
        v_mca: float, v_transition_end: float,
        altitudes_m: List[float], n_speed_steps: int = 8,
        wing_area_m2: float = 0.0, CL_transition: float = 1.0,
        weight_N: float = 0.0, n_tilt: int = 4, n_aft: int = 4,
        thrust_max_frac: float = 1.5,
        CLalpha_gust: float = 0.0,
        mean_chord_gust: float = 1.0) -> List[VTOLCondition]:
    """틸트 변환 회랑 케이스 — 물리 유도 회랑의 중심·양 경계.

    각 속도에서 틸트각 격자를 배분 가용성으로 선별해 변환 회랑
    [sigma_lo(V), sigma_hi(V)]를 구성한다. 저속·과대 틸트 쪽 경계는
    후열 추력 한계(수직 지지 부족), 고속·과소 틸트 쪽 경계는 전열
    추력 한계(F = D/sin(sigma) 상한)가 만든다 — 실기 변환 회랑의 두
    경계와 같은 구조. 케이스는 회랑 중심과 양 경계에서 nz {1.0,
    1.5}로 생성된다(제어법칙 없이 규정 스케줄로 정의되는 준정적
    상태; Fx 잔차는 파이프라인 관례대로 관성 릴리프가 폐합).
    """
    import numpy as np

    from ..case_generator import isa_atmosphere

    conditions = []
    v_start = max(1.0, v_mca * 0.5)
    speeds = np.linspace(v_start, v_transition_end, n_speed_steps)
    sigma_grid = np.arange(TILT_GRID_DEG, 90.0 + 1e-9,
                           TILT_GRID_DEG)
    for alt in altitudes_m:
        rho, _, _ = isa_atmosphere(alt)
        for V in speeds:
            feas = [sg for sg in sigma_grid
                    if tilt_allocation(float(V), float(sg), 1.0,
                                       weight_N, wing_area_m2,
                                       CL_transition, rho, n_tilt,
                                       n_aft, thrust_max_frac)[3]]
            if not feas:
                continue
            s_lo, s_hi = feas[0], feas[-1]
            s_c = 0.5 * (s_lo + s_hi)
            picks = sorted({round(s_lo, 1), round(s_c, 1),
                            round(s_hi, 1)})
            for sg in picks:
                for nz in (1.0, 1.5):
                    F, A, L, ok = tilt_allocation(
                        float(V), float(sg), nz, weight_N,
                        wing_area_m2, CL_transition, rho, n_tilt,
                        n_aft, thrust_max_frac)
                    if not ok:
                        continue
                    conditions.append(VTOLCondition(
                        label=(f"Tilt conv V={V:.1f}m/s "
                               f"s={sg:.1f}deg nz={nz:.1f} "
                               f"h={alt:.0f}m"),
                        phase=VTOLFlightPhase.TILT_TRANSITION,
                        V_eas=float(V), nz=nz, altitude_m=alt,
                        thrust_fraction=(F + A) / weight_N
                        if weight_N > 0 else 0.0,
                        tilt_deg=float(sg),
                        far_section="SC-VTOL.2215/2160",
                    ))
            # 회랑 준정적 수직 돌풍 — L+C의 천이 돌풍(①)과 동일한
            # 혼합 모델을 틸트 회랑 중심 스케줄에 적용
            s_c2 = 0.5 * (s_lo + s_hi)
            F0, A0, _L0, _ok0 = tilt_allocation(
                float(V), float(s_c2), 1.0, weight_N, wing_area_m2,
                CL_transition, rho, n_tilt, n_aft, thrust_max_frac)
            share_rot = ((F0 * np.cos(np.radians(s_c2)) + A0)
                         / weight_N if weight_N > 0 else 1.0)
            dn = transition_gust_delta_nz(
                float(V), share_rot, weight_N, wing_area_m2,
                CLalpha_gust, mean_chord_gust)
            for sgn, tag in ((+1.0, "+"), (-1.0, "-")):
                nz_g = 1.0 + sgn * dn
                F, A, _L, ok = tilt_allocation(
                    float(V), float(s_c2), max(nz_g, 0.05),
                    weight_N, wing_area_m2, CL_transition, rho,
                    n_tilt, n_aft, thrust_max_frac)
                conditions.append(VTOLCondition(
                    label=(f"Tilt conv gust{tag} V={V:.1f}m/s "
                           f"s={s_c2:.1f}deg nz={nz_g:.2f} "
                           f"h={alt:.0f}m"),
                    phase=VTOLFlightPhase.TILT_TRANSITION,
                    V_eas=float(V), nz=nz_g, altitude_m=alt,
                    thrust_fraction=(F + A) / weight_N
                    if weight_N > 0 else 0.0,
                    tilt_deg=float(s_c2),
                    far_section="SC-VTOL.2215/23.341",
                ))
    return conditions


def generate_tilt_stuck_conditions(
        v_mca: float, v_transition_end: float,
        altitudes_m: List[float], tilt_rotor_ids: List[int],
        wing_area_m2: float = 0.0, CL_transition: float = 1.0,
        weight_N: float = 0.0, n_tilt: int = 4, n_aft: int = 4,
        stuck_angles: tuple = (0.0, 90.0),
        n_speed_steps: int = 4) -> List[VTOLCondition]:
    """틸트 액추에이터 고착(M6) 조건 — 틸트로터 고유 고장 모드.

    변환 중 한 나셀이 stuck 각도에 얼어붙고(추력은 지령 가능 —
    액추에이터 고장이지 모터 고장이 아님) 나머지는 회랑 스케줄
    각도를 유지한다. 대표 고착각은 0°(미변환: 남들이 순항 전환하는
    동안 수직 고정)와 90°(조기 전환 고정)이며, 로터별로 좌우 비대칭
    Fx·Fz가 생겨 롤·요 모멘트가 발생한다(관성 릴리프 폐합 관례).
    """
    import numpy as np

    from ..case_generator import isa_atmosphere

    conditions = []
    v_start = max(1.0, v_mca * 0.5)
    speeds = np.linspace(v_start, v_transition_end, n_speed_steps)
    sigma_grid = np.arange(TILT_GRID_DEG, 90.0 + 1e-9,
                           TILT_GRID_DEG)
    for alt in altitudes_m:
        rho, _, _ = isa_atmosphere(alt)
        for V in speeds:
            feas = [sg for sg in sigma_grid
                    if tilt_allocation(float(V), float(sg), 1.0,
                                       weight_N, wing_area_m2,
                                       CL_transition, rho, n_tilt,
                                       n_aft)[3]]
            if not feas:
                continue
            s_c = 0.5 * (feas[0] + feas[-1])
            for rid in tilt_rotor_ids:
                for s_s in stuck_angles:
                    if abs(s_s - s_c) < 5.0:
                        continue          # 스케줄과 사실상 동일
                    conditions.append(VTOLCondition(
                        label=(f"Tilt stuck R{rid} s={s_s:.0f}deg "
                               f"(sched {s_c:.1f}deg) V={V:.1f}m/s "
                               f"h={alt:.0f}m"),
                        phase=VTOLFlightPhase.TILT_STUCK,
                        V_eas=float(V), nz=1.0, altitude_m=alt,
                        tilt_deg=float(s_c),
                        stuck_tilt_deg=float(s_s),
                        failed_rotor_id=rid,
                        far_section="SC-VTOL.2150",
                    ))
    return conditions


def generate_vtol_landing_conditions(altitudes_m: List[float],
                                      ) -> List[VTOLCondition]:
    """Generate VTOL landing conditions.

    Vertical descent landings at specified sink rates per SC-VTOL.

    Parameters
    ----------
    altitudes_m : list of float
        Only ground level (alt=0) is relevant for landing.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    # VTOL landing sink rates and load factors
    # SC-VTOL requires 2.0g for normal VTOL landing
    sink_rates = [
        (1.5, 2.0, "Normal VTOL landing"),
        (3.0, 2.5, "Emergency VTOL landing"),
    ]

    for sink_rate_mps, nz, desc in sink_rates:
        conditions.append(VTOLCondition(
            label=f"{desc} Vs={sink_rate_mps:.1f}m/s",
            phase=VTOLFlightPhase.VTOL_LANDING,
            V_eas=0.0, nz=nz,
            altitude_m=0.0,
            thrust_fraction=0.5,  # Partial thrust during touchdown
            sink_rate_mps=sink_rate_mps,
            far_section="SC-VTOL.2480",
        ))
    return conditions


def generate_rotor_jam_conditions(rotor_ids: List[int],
                                   altitudes_m: List[float],
                                   ) -> List[VTOLCondition]:
    """Generate rotor jam/seizure conditions.

    Sudden stop of a single rotor — produces asymmetric torque and
    yaw moment. Treated as a dynamic case.

    Parameters
    ----------
    rotor_ids : list of int
        Rotor IDs that can jam.
    altitudes_m : list of float
        Analysis altitudes.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    for alt in altitudes_m:
        for rid in rotor_ids:
            conditions.append(VTOLCondition(
                label=f"Rotor Jam R{rid} h={alt:.0f}m",
                phase=VTOLFlightPhase.ROTOR_JAM,
                V_eas=0.0, nz=1.0,
                altitude_m=alt,
                thrust_fraction=1.0,
                failed_rotor_id=rid,
                far_section="SC-VTOL.2150",
            ))
    return conditions
