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

    # Total vertical reaction = nz × W (upward)
    F_vertical_total = nz * W

    forces: Dict[int, np.ndarray] = {}

    # Main gear vertical loads
    F_main_total = F_vertical_total * condition.main_gear_vertical_frac
    n_main = len(gear.main_gear_node_ids)
    if n_main > 0:
        # One-wheel landing: all load on one side
        if condition.condition_type == LandingConditionType.ONE_WHEEL:
            # Apply 0.75 factor per §23.483
            F_main_per = F_main_total * 0.75  # All on one wheel
            for i, nid in enumerate(gear.main_gear_node_ids):
                f = np.zeros(6)
                if condition.one_wheel_side == "left" and i == 0:
                    f[2] = F_main_per  # Fz upward
                elif condition.one_wheel_side == "right" and i == n_main - 1:
                    f[2] = F_main_per  # Fz upward
                # Lateral load for side landing
                if condition.lateral_factor > 0:
                    f[1] = F_main_per * condition.lateral_factor
                # Drag load
                if condition.drag_factor > 0:
                    f[0] = -F_main_per * condition.drag_factor  # rearward
                forces[nid] = f
        else:
            # Normal distribution across main gear
            F_main_per = F_main_total / n_main
            for nid in gear.main_gear_node_ids:
                f = np.zeros(6)
                f[2] = F_main_per  # Fz upward
                # Lateral load (side load condition)
                if condition.lateral_factor > 0:
                    f[1] = F_main_per * condition.lateral_factor
                # Drag load (braking)
                if condition.drag_factor > 0:
                    f[0] = -F_main_per * condition.drag_factor
                forces[nid] = f

    # Nose gear vertical loads
    F_nose_total = F_vertical_total * condition.nose_gear_vertical_frac
    n_nose = len(gear.nose_gear_node_ids)
    if n_nose > 0 and F_nose_total > 0:
        F_nose_per = F_nose_total / n_nose
        for nid in gear.nose_gear_node_ids:
            f = np.zeros(6)
            f[2] = F_nose_per
            if condition.drag_factor > 0:
                f[0] = -F_nose_per * condition.drag_factor
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
    nz = gear.compute_nz_landing(weight_cg.weight_N)

    # Compute fore/aft distribution from CG position
    # Main gear takes most of the load
    main_frac, nose_frac = _compute_gear_load_distribution(
        weight_cg.cg_x, gear.main_gear_x, gear.nose_gear_x)

    cases = []
    cases.append(LandingCondition(
        case_id=case_id_start,
        condition_type=LandingConditionType.LEVEL_LANDING,
        nz_cg=nz,
        weight_cg=weight_cg,
        label=f"Level landing nz={nz:.2f} {weight_cg.label}",
        far_section="§23.479",
        attitude_deg=0.0,
        main_gear_vertical_frac=main_frac,
        nose_gear_vertical_frac=nose_frac,
    ))

    # With drag (spin-up condition) — 0.25 drag factor
    cases.append(LandingCondition(
        case_id=case_id_start + 1,
        condition_type=LandingConditionType.LEVEL_LANDING,
        nz_cg=nz,
        weight_cg=weight_cg,
        label=f"Level landing drag nz={nz:.2f} {weight_cg.label}",
        far_section="§23.479",
        attitude_deg=0.0,
        drag_factor=0.25,
        main_gear_vertical_frac=main_frac,
        nose_gear_vertical_frac=nose_frac,
    ))

    return cases


def generate_tail_down_landing(case_id_start: int,
                                 config: AircraftConfig,
                                 weight_cg: WeightCGCondition,
                                 ) -> List[LandingCondition]:
    """Generate tail-down landing per §23.481.

    Main gear only, high pitch attitude, no nose gear contact.
    """
    gear = config.landing_gear
    nz = gear.compute_nz_landing(weight_cg.weight_N)

    cases = [LandingCondition(
        case_id=case_id_start,
        condition_type=LandingConditionType.TAIL_DOWN,
        nz_cg=nz,
        weight_cg=weight_cg,
        label=f"Tail-down landing nz={nz:.2f} {weight_cg.label}",
        far_section="§23.481",
        attitude_deg=10.0,  # Typical tail-down attitude
        main_gear_vertical_frac=1.0,
        nose_gear_vertical_frac=0.0,
    )]
    return cases


def generate_one_wheel_landing(case_id_start: int,
                                 config: AircraftConfig,
                                 weight_cg: WeightCGCondition,
                                 ) -> List[LandingCondition]:
    """Generate one-wheel landing per §23.483.

    Load on one main gear only, with 0.75 factor applied.
    Both left and right sides.
    """
    gear = config.landing_gear
    nz = gear.compute_nz_landing(weight_cg.weight_N)

    cases = []
    for i, side in enumerate(["left", "right"]):
        cases.append(LandingCondition(
            case_id=case_id_start + i,
            condition_type=LandingConditionType.ONE_WHEEL,
            nz_cg=nz,
            weight_cg=weight_cg,
            label=f"One-wheel {side} nz={nz:.2f} {weight_cg.label}",
            far_section="§23.483",
            main_gear_vertical_frac=1.0,
            nose_gear_vertical_frac=0.0,
            one_wheel_side=side,
        ))
    return cases


def generate_side_load(case_id_start: int,
                        config: AircraftConfig,
                        weight_cg: WeightCGCondition,
                        ) -> List[LandingCondition]:
    """Generate side load condition per §23.485.

    Lateral tire reaction as fraction of vertical, with reduced vertical.
    """
    gear = config.landing_gear
    nz = gear.compute_nz_landing(weight_cg.weight_N)
    # §23.485: vertical = nz/2, lateral = 0.8 × vertical per side
    nz_side = nz * 0.5

    cases = []
    cases.append(LandingCondition(
        case_id=case_id_start,
        condition_type=LandingConditionType.SIDE_LOAD,
        nz_cg=nz_side,
        weight_cg=weight_cg,
        label=f"Side load nz={nz_side:.2f} {weight_cg.label}",
        far_section="§23.485",
        lateral_factor=0.8,
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

    # §23.493 Braked roll — nz = 1.0 + 0.4g drag
    cases.append(LandingCondition(
        case_id=case_id_start + 1,
        condition_type=LandingConditionType.BRAKED_ROLL,
        nz_cg=1.0,
        weight_cg=weight_cg,
        label=f"Braked roll {weight_cg.label}",
        far_section="§23.493",
        drag_factor=0.8,  # Braking coefficient
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

    # §23.499 Nose-wheel yaw — lateral load on nose gear
    if nose_frac > 0:
        cases.append(LandingCondition(
            case_id=case_id_start + 3,
            condition_type=LandingConditionType.NOSE_WHEEL_YAW,
            nz_cg=1.0,
            weight_cg=weight_cg,
            label=f"Nose-wheel yaw {weight_cg.label}",
            far_section="§23.499",
            lateral_factor=0.8,
            main_gear_vertical_frac=0.0,
            nose_gear_vertical_frac=1.0,
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
