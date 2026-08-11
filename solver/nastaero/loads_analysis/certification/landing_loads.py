"""Quasi-static landing load computation per FAA Part 23 §23.471-§23.511.

Generates nodal force distributions for landing conditions by combining
gear reaction forces with inertial loads. The resulting forces can be
fed directly into the VMT pipeline for envelope processing.

Landing conditions covered:
- §23.479: Level landing (3-point)
- §23.481: Tail-down landing
- §23.483: One-wheel landing
- §23.485: Side load
- §23.487: Rebound landing
- §23.491-§23.497: Ground handling (taxi, braking, turning)

References
----------
- 14 CFR §23.471-§23.511: Ground load conditions
- 14 CFR §23.473: Landing load factors
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aircraft_config import (
    AircraftConfig, WeightCGCondition, LandingGearConfig,
    G_MPS2, FPS_TO_MPS,
)


# ---------------------------------------------------------------------------
# Landing condition types
# ---------------------------------------------------------------------------

class LandingConditionType(Enum):
    """Landing condition types per Part 23."""
    LEVEL_LANDING = "level_landing"              # §23.479
    TAIL_DOWN = "tail_down"                      # §23.481
    ONE_WHEEL = "one_wheel"                      # §23.483
    SIDE_LOAD = "side_load"                      # §23.485
    REBOUND = "rebound"                          # §23.487
    TAXI = "taxi"                                # §23.491
    BRAKED_ROLL = "braked_roll"                  # §23.493
    TURNING = "turning"                          # §23.497
    NOSE_WHEEL_YAW = "nose_wheel_yaw"           # §23.499


class GroundConditionType(Enum):
    """Ground handling condition types."""
    TAXI = "taxi"
    BRAKED_ROLL = "braked_roll"
    TURNING = "turning"
    NOSE_WHEEL_YAW = "nose_wheel_yaw"


# ---------------------------------------------------------------------------
# Landing condition dataclass
# ---------------------------------------------------------------------------

@dataclass
class LandingCondition:
    """A single landing or ground load condition.

    Attributes
    ----------
    case_id : int
        Unique case identifier.
    condition_type : LandingConditionType
        Type of landing/ground condition.
    nz_cg : float
        Load factor at CG (g's).
    weight_cg : WeightCGCondition
        Weight/CG condition for this case.
    label : str
        Human-readable description.
    far_section : str
        FAR section reference.
    attitude_deg : float
        Aircraft pitch attitude at touchdown (degrees).
    lateral_factor : float
        Lateral load factor (fraction of vertical) for side loads.
    drag_factor : float
        Drag load factor (fraction of vertical) for braking.
    main_gear_vertical_frac : float
        Fraction of total vertical load on main gear (default 1.0).
    nose_gear_vertical_frac : float
        Fraction of total vertical load on nose gear (default 0.0).
    one_wheel_side : str
        For one-wheel landing: "left" or "right".
    """
    case_id: int = 0
    condition_type: LandingConditionType = LandingConditionType.LEVEL_LANDING
    nz_cg: float = 2.67
    weight_cg: Optional[WeightCGCondition] = None
    label: str = ""
    far_section: str = "§23.479"
    attitude_deg: float = 0.0
    lateral_factor: float = 0.0
    drag_factor: float = 0.0
    main_gear_vertical_frac: float = 1.0
    nose_gear_vertical_frac: float = 0.0
    one_wheel_side: str = ""
    # §23.473(e): 착륙 충격 동안 양력이 지지하는 중량비 (지면반력은
    # (nz − lift_factor)·W가 된다). 지상 조건(§23.485~499)은 0.
    lift_factor: float = 0.0
    # §23.485: 메인 좌우에 서로 다른 측력 (W 비율; 부호 = +y 방향)
    side_frac_per_main: Optional[Tuple[float, ...]] = None
    # §23.493: 노즈휠은 브레이크가 없어 드래그 미적용 등, 노즈 전용
    # 드래그 계수 (None이면 drag_factor를 따름)
    nose_drag_factor: Optional[float] = None
    # §23.499 노즈휠 보조 조건: "aft"(0.8V 드래그) / "fwd"(0.4V 전방)
    # / "side"(0.7V 측력) — 수직 2.25×정적하중과 조합
    nose_supp_dir: str = ""


# ---------------------------------------------------------------------------
# Gear reaction force computation
# ---------------------------------------------------------------------------

def compute_gear_reactions(condition: LandingCondition,
                            gear: LandingGearConfig,
                            ) -> Dict[int, np.ndarray]:
    """Compute gear reaction forces for a landing condition.

    Returns nodal forces at each gear attachment point.

    Parameters
    ----------
    condition : LandingCondition
        The landing condition.
    gear : LandingGearConfig
        Landing gear configuration.

    Returns
    -------
    dict of {node_id: ndarray(6)}
        Gear reaction forces [Fx, Fy, Fz, Mx, My, Mz] at each node.
    """
    W = condition.weight_cg.weight_N if condition.weight_cg else 0.0
    nz = condition.nz_cg

    # 지면반력 하중배수 = nz − 양력비 (§23.473(e): 착륙 충격 중 양력이
    # lift_factor·W를 지지). 지상 조건은 lift_factor=0 → 기존과 동일.
    F_vertical_total = (nz - condition.lift_factor) * W

    forces: Dict[int, np.ndarray] = {}

    # §23.499 노즈휠 보조 조건 — 수직 2.25×정적하중 + 방향별 성분
    if condition.nose_supp_dir:
        v_static = W * condition.nose_gear_vertical_frac
        v = 2.25 * v_static
        n_nose = max(len(gear.nose_gear_node_ids), 1)
        for nid in gear.nose_gear_node_ids:
            f = np.zeros(6)
            f[2] = v / n_nose
            if condition.nose_supp_dir == "aft":
                f[0] = -0.8 * v / n_nose          # 드래그 (후방)
            elif condition.nose_supp_dir == "fwd":
                f[0] = +0.4 * v / n_nose          # 전방
            elif condition.nose_supp_dir == "side":
                f[1] = 0.7 * v / n_nose           # 측력
            forces[nid] = f
        return forces

    # Main gear vertical loads
    F_main_total = F_vertical_total * condition.main_gear_vertical_frac
    n_main = len(gear.main_gear_node_ids)
    if n_main > 0:
        if condition.condition_type == LandingConditionType.ONE_WHEEL:
            # §23.483: 수평착륙에서 그 쪽이 받았을 반력을 편측에 그대로
            for i, nid in enumerate(gear.main_gear_node_ids):
                f = np.zeros(6)
                on_this = ((condition.one_wheel_side == "left" and i == 0)
                           or (condition.one_wheel_side == "right"
                               and i == n_main - 1))
                if on_this:
                    f[2] = F_main_total
                    if condition.drag_factor > 0:
                        f[0] = -F_main_total * condition.drag_factor
                forces[nid] = f
        else:
            F_main_per = F_main_total / n_main
            for i, nid in enumerate(gear.main_gear_node_ids):
                f = np.zeros(6)
                f[2] = F_main_per
                # §23.485: 좌우 비대칭 측력 (W 비율로 직접 지정)
                if condition.side_frac_per_main is not None:
                    if i < len(condition.side_frac_per_main):
                        f[1] = condition.side_frac_per_main[i] * W
                elif condition.lateral_factor > 0:
                    f[1] = F_main_per * condition.lateral_factor
                if condition.drag_factor > 0:
                    f[0] = -F_main_per * condition.drag_factor
                forces[nid] = f

    # Nose gear vertical loads
    F_nose_total = F_vertical_total * condition.nose_gear_vertical_frac
    n_nose = len(gear.nose_gear_node_ids)
    if n_nose > 0 and F_nose_total > 0:
        nose_drag = (condition.nose_drag_factor
                     if condition.nose_drag_factor is not None
                     else condition.drag_factor)
        F_nose_per = F_nose_total / n_nose
        for nid in gear.nose_gear_node_ids:
            f = np.zeros(6)
            f[2] = F_nose_per
            if nose_drag > 0:
                f[0] = -F_nose_per * nose_drag
            forces[nid] = f

    return forces


def compute_landing_inertial_forces(condition: LandingCondition,
                                      node_masses: Dict[int, float],
                                      g: float = 9810.0,
                                      ) -> Dict[int, np.ndarray]:
    """Compute inertial forces on all structural nodes.

    F_inertial = -mass × nz × g (downward, opposing gear reaction).

    Parameters
    ----------
    condition : LandingCondition
        Landing condition with nz_cg.
    node_masses : dict of {node_id: mass}
        Lumped mass at each structural node.
    g : float
        Gravitational acceleration in model units (default 9810 mm/s²).

    Returns
    -------
    dict of {node_id: ndarray(6)}
        Inertial forces at each node.
    """
    nz = condition.nz_cg
    forces: Dict[int, np.ndarray] = {}

    for nid, mass in node_masses.items():
        if mass <= 0:
            continue
        f = np.zeros(6)
        f[2] = -mass * nz * g  # Fz downward (inertial)
        forces[nid] = f

    return forces


def compute_landing_lift_forces(condition: LandingCondition,
                                node_masses: Dict[int, float],
                                g: float = 9810.0,
                                ) -> Dict[int, np.ndarray]:
    """§23.473(e) 착륙 충격 중 양력 (lift_factor·W, 상향) 분포.

    질량 비례 분포(간이) — 합력이 CG를 지나므로 지면반력
    (nz−L)·W, 관성 −nz·W와 함께 ΣFz=0의 자기평형을 이룬다.
    """
    L = condition.lift_factor
    if L <= 0:
        return {}
    forces: Dict[int, np.ndarray] = {}
    for nid, mass in node_masses.items():
        if mass <= 0:
            continue
        f = np.zeros(6)
        f[2] = mass * L * g          # 상향
        forces[nid] = f
    return forces


def spinup_drag_factor(weight_N: float) -> float:
    """스핀업 드래그 계수 K — FAR 23 Appendix C 23.1.

    W ≤ 3,000 lb → 0.25, W ≥ 6,000 lb → 0.33, 사이 선형 보간
    (FAR 23 LOADS Manual, LANDLOADS.BAS). §23.479(b)의 최소 25%
    드래그 요구를 겸한다.
    """
    w_lb = weight_N / 4.44822
    if w_lb <= 3000.0:
        return 0.25
    if w_lb >= 6000.0:
        return 0.33
    return 0.25 + (0.33 - 0.25) * (w_lb - 3000.0) / 3000.0


def combine_forces(*force_dicts: Dict[int, np.ndarray]
                    ) -> Dict[int, np.ndarray]:
    """Combine multiple nodal force distributions.

    Parameters
    ----------
    *force_dicts : dict of {node_id: ndarray(6)}
        Any number of force distributions.

    Returns
    -------
    dict of {node_id: ndarray(6)}
        Combined forces.
    """
    combined: Dict[int, np.ndarray] = {}
    for fd in force_dicts:
        for nid, f in fd.items():
            if nid in combined:
                combined[nid] = combined[nid] + f
            else:
                combined[nid] = f.copy()
    return combined


# ---------------------------------------------------------------------------
# Landing condition generators per Part 23 sections
# ---------------------------------------------------------------------------

def generate_level_landing(case_id_start: int,
                            config: AircraftConfig,
                            weight_cg: WeightCGCondition,
                            ) -> List[LandingCondition]:
    """Generate level landing conditions per §23.479.

    Level landing with all wheels contacting simultaneously.
    Vertical load distributed between main and nose gear.

    Parameters
    ----------
    case_id_start : int
        Starting case ID.
    config : AircraftConfig
        Aircraft configuration.
    weight_cg : WeightCGCondition
        Weight/CG condition.

    Returns
    -------
    list of LandingCondition
    """
    gear = config.landing_gear
    nz, _, _ = gear.landing_load_factors(weight_cg.weight_N,
                                         config.wing_area_m2)
    K = spinup_drag_factor(weight_cg.weight_N)

    # 3점 자세: CG 위치의 정적 분배 / 노즈클리어 자세: 메인만
    main_frac, nose_frac = _compute_gear_load_distribution(
        weight_cg.cg_x, gear.main_gear_x, gear.nose_gear_x)

    cases = []
    attitudes = [("3pt", main_frac, nose_frac),
                 ("nose-clear", 1.0, 0.0)]
    cid = case_id_start
    for att, mf, nf in attitudes:
        for drag, tag in ((0.0, ""), (K, f" drag K={K:.2f}")):
            cases.append(LandingCondition(
                case_id=cid,
                condition_type=LandingConditionType.LEVEL_LANDING,
                nz_cg=nz,
                weight_cg=weight_cg,
                label=(f"Level landing {att}{tag} nz={nz:.2f} "
                       f"{weight_cg.label}"),
                far_section="§23.479",
                lift_factor=2 / 3,
                drag_factor=drag,
                main_gear_vertical_frac=mf,
                nose_gear_vertical_frac=nf,
            ))
            cid += 1
    return cases


def generate_tail_down_landing(case_id_start: int,
                                 config: AircraftConfig,
                                 weight_cg: WeightCGCondition,
                                 ) -> List[LandingCondition]:
    """Generate tail-down landing per §23.481.

    Main gear only, high pitch attitude, no nose gear contact.
    """
    gear = config.landing_gear
    nz, _, _ = gear.landing_load_factors(weight_cg.weight_N,
                                         config.wing_area_m2)

    # §23.481(b): 지면반력은 수직 (드래그 없음), 메인만
    cases = [LandingCondition(
        case_id=case_id_start,
        condition_type=LandingConditionType.TAIL_DOWN,
        nz_cg=nz,
        weight_cg=weight_cg,
        label=f"Tail-down landing nz={nz:.2f} {weight_cg.label}",
        far_section="§23.481",
        attitude_deg=10.0,  # Typical tail-down attitude
        lift_factor=2 / 3,
        main_gear_vertical_frac=1.0,
        nose_gear_vertical_frac=0.0,
    )]
    return cases


def generate_one_wheel_landing(case_id_start: int,
                                 config: AircraftConfig,
                                 weight_cg: WeightCGCondition,
                                 ) -> List[LandingCondition]:
    """Generate one-wheel landing per §23.483.

    수평착륙 자세에서 편측 메인 기어가 §23.479의 그 쪽 반력
    (전체의 절반)을 단독으로 받는다.
    """
    gear = config.landing_gear
    nz, _, _ = gear.landing_load_factors(weight_cg.weight_N,
                                         config.wing_area_m2)

    cases = []
    for i, side in enumerate(["left", "right"]):
        cases.append(LandingCondition(
            case_id=case_id_start + i,
            condition_type=LandingConditionType.ONE_WHEEL,
            nz_cg=nz,
            weight_cg=weight_cg,
            label=f"One-wheel {side} nz={nz:.2f} {weight_cg.label}",
            far_section="§23.483",
            lift_factor=2 / 3,
            main_gear_vertical_frac=0.5,   # §23.479의 편측 반력과 동일
            nose_gear_vertical_frac=0.0,
            one_wheel_side=side,
        ))
    return cases


def generate_side_load(case_id_start: int,
                        config: AircraftConfig,
                        weight_cg: WeightCGCondition,
                        ) -> List[LandingCondition]:
    """Generate side load conditions per §23.485.

    수평 자세, 메인만 접지, 수직 하중배수 1.33 (좌우 균등),
    측력은 한쪽 안쪽으로 0.5W + 반대쪽 바깥으로 0.33W
    (limit side inertia factor 0.83). 좌/우 미끄럼 두 방향 생성.
    """
    cases = []
    for i, sgn in enumerate((+1.0, -1.0)):
        cases.append(LandingCondition(
            case_id=case_id_start + i,
            condition_type=LandingConditionType.SIDE_LOAD,
            nz_cg=1.33,
            weight_cg=weight_cg,
            label=(f"Side load {'+y' if sgn > 0 else '-y'} "
                   f"{weight_cg.label}"),
            far_section="§23.485",
            side_frac_per_main=(sgn * 0.5, sgn * 0.33),
            main_gear_vertical_frac=1.0,
            nose_gear_vertical_frac=0.0,
        ))
    return cases


def generate_rebound(case_id_start: int,
                      config: AircraftConfig,
                      weight_cg: WeightCGCondition,
                      ) -> List[LandingCondition]:
    """Generate rebound condition per §23.487.

    Spring-back after landing: nz = -1.0 (upward acceleration at gear).
    Typically nz_rebound = 20:1 to 10:1 gear spring, ~1.5g upward simplified.
    """
    cases = [LandingCondition(
        case_id=case_id_start,
        condition_type=LandingConditionType.REBOUND,
        nz_cg=1.5,  # Conservative rebound factor
        weight_cg=weight_cg,
        label=f"Rebound nz=1.5 {weight_cg.label}",
        far_section="§23.487",
        main_gear_vertical_frac=1.0,
        nose_gear_vertical_frac=0.0,
    )]
    return cases


def generate_ground_handling(case_id_start: int,
                               config: AircraftConfig,
                               weight_cg: WeightCGCondition,
                               ) -> List[LandingCondition]:
    """Generate ground handling conditions per §23.491-§23.497.

    Includes taxi, braked roll, and turning.
    """
    gear = config.landing_gear
    main_frac, nose_frac = _compute_gear_load_distribution(
        weight_cg.cg_x, gear.main_gear_x, gear.nose_gear_x)

    cases = []

    # §23.491 Taxi — nz = 1.0 + taxi bump factor
    cases.append(LandingCondition(
        case_id=case_id_start,
        condition_type=LandingConditionType.TAXI,
        nz_cg=1.5,  # 1g + 0.5g taxi bump per §23.491
        weight_cg=weight_cg,
        label=f"Taxi nz=1.5 {weight_cg.label}",
        far_section="§23.491",
        main_gear_vertical_frac=main_frac,
        nose_gear_vertical_frac=nose_frac,
    ))

    # §23.493 Braked roll — 수직 하중배수 1.33, 브레이크 휠(메인)에만
    # 마찰계수 0.8의 드래그 (노즈휠은 브레이크 없음)
    cases.append(LandingCondition(
        case_id=case_id_start + 1,
        condition_type=LandingConditionType.BRAKED_ROLL,
        nz_cg=1.33,
        weight_cg=weight_cg,
        label=f"Braked roll nz=1.33 {weight_cg.label}",
        far_section="§23.493",
        drag_factor=0.8,
        nose_drag_factor=0.0,
        main_gear_vertical_frac=main_frac,
        nose_gear_vertical_frac=nose_frac,
    ))

    # §23.497 Turning — nz = 1.0 + 0.5g lateral
    cases.append(LandingCondition(
        case_id=case_id_start + 2,
        condition_type=LandingConditionType.TURNING,
        nz_cg=1.0,
        weight_cg=weight_cg,
        label=f"Turning {weight_cg.label}",
        far_section="§23.497",
        lateral_factor=0.5,
        main_gear_vertical_frac=main_frac,
        nose_gear_vertical_frac=nose_frac,
    ))

    # §23.499 노즈휠 보조 조건 — 수직 2.25×정적하중과 조합해
    # aft(0.8V 드래그)/fwd(0.4V 전방)/side(0.7V 측력) 세 케이스
    if nose_frac > 0:
        for j, sup in enumerate(("aft", "fwd", "side")):
            cases.append(LandingCondition(
                case_id=case_id_start + 3 + j,
                condition_type=LandingConditionType.NOSE_WHEEL_YAW,
                nz_cg=1.0,
                weight_cg=weight_cg,
                label=f"Nose-wheel {sup} 2.25xstatic {weight_cg.label}",
                far_section="§23.499",
                nose_supp_dir=sup,
                main_gear_vertical_frac=0.0,
                nose_gear_vertical_frac=nose_frac,
            ))

    return cases


def generate_all_landing_conditions(config: AircraftConfig,
                                      case_id_start: int = 5000,
                                      ) -> List[LandingCondition]:
    """Generate all landing and ground handling conditions.

    Parameters
    ----------
    config : AircraftConfig
        Aircraft configuration.
    case_id_start : int
        Starting case ID for landing cases.

    Returns
    -------
    list of LandingCondition
        All landing and ground handling cases.
    """
    cases: List[LandingCondition] = []
    cid = case_id_start

    for wc in config.weight_cg_conditions:
        # §23.479 Level landing
        new = generate_level_landing(cid, config, wc)
        cases.extend(new)
        cid += len(new)

        # §23.481 Tail-down
        new = generate_tail_down_landing(cid, config, wc)
        cases.extend(new)
        cid += len(new)

        # §23.483 One-wheel
        new = generate_one_wheel_landing(cid, config, wc)
        cases.extend(new)
        cid += len(new)

        # §23.485 Side load
        new = generate_side_load(cid, config, wc)
        cases.extend(new)
        cid += len(new)

        # §23.487 Rebound
        new = generate_rebound(cid, config, wc)
        cases.extend(new)
        cid += len(new)

        # §23.491-499 Ground handling
        new = generate_ground_handling(cid, config, wc)
        cases.extend(new)
        cid += len(new)

    return cases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_gear_load_distribution(cg_x: float,
                                      main_x: float,
                                      nose_x: float,
                                      ) -> Tuple[float, float]:
    """Compute static gear load distribution from CG position.

    Parameters
    ----------
    cg_x : float
        CG x-position.
    main_x : float
        Main gear x-position.
    nose_x : float
        Nose gear x-position.

    Returns
    -------
    (main_fraction, nose_fraction) : tuple of float
        Fraction of total vertical load on each gear set.
    """
    wheelbase = abs(main_x - nose_x)
    if wheelbase < 1e-10:
        return 1.0, 0.0

    # Main gear takes load proportional to CG distance from nose gear
    # and vice versa (lever arm principle)
    dist_to_nose = abs(cg_x - nose_x)
    dist_to_main = abs(cg_x - main_x)

    main_frac = dist_to_nose / wheelbase
    nose_frac = dist_to_main / wheelbase

    # Clamp to [0, 1]
    main_frac = max(0.0, min(1.0, main_frac))
    nose_frac = max(0.0, min(1.0, nose_frac))

    # Normalize
    total = main_frac + nose_frac
    if total > 0:
        main_frac /= total
        nose_frac /= total

    return main_frac, nose_frac
