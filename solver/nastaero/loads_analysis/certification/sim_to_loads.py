"""Convert 6-DOF critical time points to SOL 144 load cases.

Each CriticalTimePoint from the flight dynamics simulation provides a
complete aerodynamic state (α, β, δe, δa, δr, nz). These are converted
to CertLoadCase objects with fully constrained TrimConditions (zero free
variables), so SOL 144 computes only the elastic structural deformation
under the prescribed aerodynamic loads.

SOL 144 compatibility
---------------------
When n_trim_free = 0:
  - n_constraints = min(0, 2) = 0  (no trim equations)
  - System: (K - q·Q_aa)·u = F_grav + F_aero_fixed
  - Pure static aeroelastic deformation under fixed loads

This is exactly what we want: the 6-DOF simulation determines the
rigid-body flight condition, and SOL 144 gives the elastic structural
response at that condition.

References
----------
- Phase 4d of the FAA Part 23 certification framework
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..case_generator import TrimCondition
from .aircraft_config import (
    AircraftConfig, WeightCGCondition,
    eas_to_mach, eas_to_tas, dynamic_pressure_from_eas,
)
from .load_case_matrix import CertLoadCase
from .sim_runner import CriticalTimePoint, SimRunResult

logger = logging.getLogger(__name__)


def deduplicate_critical_points(
    all_points: List[CriticalTimePoint],
    max_cases: int = 50,
    min_per_type: int = 2,
) -> List[CriticalTimePoint]:
    """Reduce critical time points by removing flight-state duplicates.

    When the 6-DOF simulation produces more critical time points than
    the downstream stress team can practically analyse, this function
    selects a diverse subset that still covers the full design space.

    Strategy
    --------
    1. Group points by ``maneuver_type`` (don't mix fundamentally
       different load paths).
    2. Allocate a budget to each type proportional to its original count,
       with a guaranteed minimum per type.
    3. Within each type, select diverse representatives using farthest-
       point sampling on a normalised flight-state vector
       ``[nz, α, β, V_eas, p, q, r, δe, δa, δr]``, seeded with the
       most extreme point (highest weighted-extremity score).

    Parameters
    ----------
    all_points : list of CriticalTimePoint
        All valid critical points from all simulations.
    max_cases : int
        Maximum number of points to retain.
    min_per_type : int
        Minimum points to keep per maneuver type.

    Returns
    -------
    list of CriticalTimePoint
        Reduced set of critical points.
    """
    if len(all_points) <= max_cases:
        return list(all_points)

    # ---- 1. Group by maneuver type ----
    groups: Dict[str, List[CriticalTimePoint]] = {}
    for pt in all_points:
        groups.setdefault(pt.maneuver_type, []).append(pt)

    n_types = len(groups)

    # ---- 2. Budget allocation ----
    # Guarantee min_per_type each, then distribute the remainder
    # proportionally to original group sizes.
    guaranteed = min(min_per_type, max_cases // max(n_types, 1))
    budget: Dict[str, int] = {mt: guaranteed for mt in groups}
    remaining = max_cases - sum(budget.values())

    if remaining > 0:
        total_pts = sum(len(pts) for pts in groups.values())
        # Proportional share (fractional)
        for mt in sorted(groups, key=lambda x: -len(groups[x])):
            extra = int(remaining * len(groups[mt]) / max(total_pts, 1))
            budget[mt] += extra
        # Distribute leftover 1-by-1 to largest groups
        leftover = max_cases - sum(budget.values())
        for mt in sorted(groups, key=lambda x: -len(groups[x])):
            if leftover <= 0:
                break
            budget[mt] += 1
            leftover -= 1

    # ---- 3. Select representatives per type ----
    selected: List[CriticalTimePoint] = []
    for mt, pts in sorted(groups.items()):
        n_keep = min(budget.get(mt, min_per_type), len(pts))
        if len(pts) <= n_keep:
            selected.extend(pts)
        else:
            chosen = _select_diverse_points(pts, n_keep)
            selected.extend(chosen)

    logger.info("Deduplication: %d → %d critical points (budget %d)",
                len(all_points), len(selected), max_cases)
    for mt in sorted(groups):
        before = len(groups[mt])
        after = sum(1 for p in selected if p.maneuver_type == mt)
        logger.info("  %s: %d → %d", mt, before, after)

    return selected


def _select_diverse_points(
    points: List[CriticalTimePoint],
    n_keep: int,
) -> List[CriticalTimePoint]:
    """Select diverse representatives using farthest-point sampling.

    Builds a normalised 10-D flight-state vector per point, then
    greedily selects the most extreme point first and iteratively adds
    the point that is farthest (in Euclidean distance) from all already-
    selected points, with a small bias toward extremity to break ties.

    Parameters
    ----------
    points : list of CriticalTimePoint
        Points within a single maneuver type.
    n_keep : int
        Number of points to select.

    Returns
    -------
    list of CriticalTimePoint
        The selected subset, ordered by original index.
    """
    n = len(points)
    if n <= n_keep:
        return list(points)

    # Build state matrix: [nz, alpha, beta, V_eas, p, q, r, de, da, dr]
    state = np.array([
        [pt.nz, pt.alpha_deg, pt.beta_deg, pt.V_eas,
         pt.p, pt.q, pt.r, pt.delta_e, pt.delta_a, pt.delta_r]
        for pt in points
    ], dtype=np.float64)

    # Normalise each dimension to [0, 1]
    mins = state.min(axis=0)
    maxs = state.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1.0
    norm = (state - mins) / ranges

    # Extremity score — higher weight for load-factor and angular rates
    # because these drive structural loads most directly.
    weights = np.array([3.0, 1.5, 1.5, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    scores = np.sum(np.abs(norm - 0.5) * weights, axis=1)

    # ---- Farthest-point sampling ----
    selected_idx: List[int] = [int(np.argmax(scores))]

    for _ in range(n_keep - 1):
        # Min distance from each candidate to any already-selected point
        min_dists = np.full(n, np.inf)
        for si in selected_idx:
            dists = np.linalg.norm(norm - norm[si], axis=1)
            min_dists = np.minimum(min_dists, dists)

        # Exclude already-selected
        for si in selected_idx:
            min_dists[si] = -1.0

        # Combine distance with a small extremity bias for tie-breaking
        max_score = np.max(scores) + 1e-10
        combined = min_dists * (1.0 + 0.3 * scores / max_score)
        next_idx = int(np.argmax(combined))
        selected_idx.append(next_idx)

    # Return in original order for reproducibility
    return [points[i] for i in sorted(selected_idx)]


def critical_points_to_load_cases(
    sim_results: List[SimRunResult],
    config: AircraftConfig,
    case_id_start: int = 10000,
    max_dynamic_cases: int = 50,
) -> List[CertLoadCase]:
    """Convert 6-DOF critical time points to CertLoadCase objects.

    Each CriticalTimePoint is converted to a fully constrained
    TrimCondition with all control surfaces and angles fixed.
    The resulting CertLoadCase can be solved by the existing
    BatchRunner alongside static cases.

    Before conversion, critical points are deduplicated by flight state
    similarity to reduce the total case count. Points within each
    maneuver type are clustered by a normalised state vector
    (nz, α, β, V, p, q, r, δe, δa, δr) and the most diverse,
    highest-priority representatives are retained.

    Parameters
    ----------
    sim_results : list of SimRunResult
        Results from SimRunner.run_all().
    config : AircraftConfig
        Aircraft configuration (for W/CG lookup).
    case_id_start : int
        Starting case ID for dynamic cases (default 10000 to
        separate from static cases which start at 1).
    max_dynamic_cases : int
        Maximum number of dynamic load cases after deduplication.
        Set to 0 to disable deduplication.

    Returns
    -------
    list of CertLoadCase
        Dynamic load cases ready for batch trim execution.
    """
    cases = []
    case_id = case_id_start

    # Build W/CG lookup by label
    wc_map: Dict[str, WeightCGCondition] = {
        wc.label: wc for wc in config.weight_cg_conditions
    }

    # Collect all valid critical time points
    all_points: List[CriticalTimePoint] = []
    for result in sim_results:
        for ctp in result.critical_points:
            if (math.isnan(ctp.nz) or math.isnan(ctp.alpha_deg)
                    or math.isnan(ctp.beta_deg) or math.isnan(ctp.delta_e)):
                logger.warning("  Skipping NaN critical point: %s (t=%.3fs)",
                               ctp.reason, ctp.t)
                continue
            all_points.append(ctp)

    # Deduplicate if needed
    if max_dynamic_cases > 0 and len(all_points) > max_dynamic_cases:
        all_points = deduplicate_critical_points(
            all_points, max_cases=max_dynamic_cases,
        )

    for ctp in all_points:
        # Look up weight/CG condition
        wc = wc_map.get(ctp.weight_label)
        if wc is None:
            # Use first weight condition as fallback
            wc = config.weight_cg_conditions[0] if config.weight_cg_conditions else None
            if wc is None:
                continue

        # Flight condition
        alt_m = config.altitudes_m[0] if config.altitudes_m else 0.0
        V_eas = ctp.V_eas
        mach = eas_to_mach(V_eas, alt_m)
        q = dynamic_pressure_from_eas(V_eas)
        V_tas = eas_to_tas(V_eas, alt_m)

        # Convert alpha/beta from degrees to radians for ANGLEA/SIDES
        alpha_rad = math.radians(ctp.alpha_deg)
        beta_rad = math.radians(ctp.beta_deg)

        # Build fully constrained trim condition
        # All variables fixed → 0 free variables → pure elastic solution
        # Include ALL AESTAT/AESURF trim variables to fully constrain
        # the trim solution (0 free vars → pure elastic response).
        # CRITICAL: URDD2/URDD4/URDD6 must also be specified as fixed,
        # otherwise they become free variables with zero normalwash,
        # creating a degenerate constraint system that produces
        # incorrect force magnitudes.
        fixed_vars = {
            "ANGLEA": alpha_rad,
            "SIDES": beta_rad,
            "ROLL": 0.0, "YAW": 0.0,
            "ELEV": ctp.delta_e,
            "ARON": ctp.delta_a,
            "RUD": ctp.delta_r,
            "URDD3": ctp.nz,   # Normal load factor
            "URDD2": 0.0,      # Lateral acceleration (zero for symmetric)
            "URDD4": 0.0,      # Roll acceleration
            "URDD6": 0.0,      # Yaw acceleration
        }

        # Build descriptive label
        reason_short = ctp.reason.replace(" ", "")
        category = f"dynamic_{ctp.maneuver_type}"
        label = (f"Dyn_{ctp.maneuver_type}_{reason_short}_"
                 f"V{V_eas:.0f}_{ctp.weight_label}")

        tc = TrimCondition(
            case_id=case_id,
            mach=mach,
            q=q,
            nz=ctp.nz,
            fixed_vars=fixed_vars,
            free_vars=[],     # All constrained — key for 0-free-var path
            label=label,
            altitude_m=alt_m,
            velocity=V_tas,
        )

        # Flight state snapshot for report
        flight_state = {
            "V_eas_m_s": V_eas,
            "V_tas_m_s": V_tas,
            "altitude_m": alt_m,
            "mach": mach,
            "alpha_deg": ctp.alpha_deg,
            "beta_deg": ctp.beta_deg,
            "nz": ctp.nz,
            "ny": ctp.ny,
            "p_rad_s": ctp.p,
            "q_rad_s": ctp.q,
            "r_rad_s": ctp.r,
            "p_dot_rad_s2": ctp.p_dot,
            "q_dot_rad_s2": ctp.q_dot,
            "r_dot_rad_s2": ctp.r_dot,
            "delta_e_deg": math.degrees(ctp.delta_e),
            "delta_a_deg": math.degrees(ctp.delta_a),
            "delta_r_deg": math.degrees(ctp.delta_r),
            "reason": ctp.reason,
            "maneuver_type": ctp.maneuver_type,
            "t": ctp.t,
        }

        clc = CertLoadCase(
            trim_condition=tc,
            category=category,
            far_section=ctp.far_section,
            weight_cg=wc,
            altitude_m=alt_m,
            config_label="Clean",
            solve_type="trim",
            flight_state=flight_state,
        )

        cases.append(clc)
        case_id += 1

    # Log summary by category
    cat_counts: Dict[str, int] = {}
    for c in cases:
        cat_counts[c.category] = cat_counts.get(c.category, 0) + 1

    logger.info("Dynamic load cases: %d total (IDs %d-%d)",
                len(cases), case_id_start,
                case_id_start + len(cases) - 1 if cases else case_id_start)
    for cat, cnt in sorted(cat_counts.items()):
        logger.info("  %s: %d", cat, cnt)

    return cases


def summarize_critical_points(
    sim_results: List[SimRunResult],
) -> Dict[str, int]:
    """Summarize critical point counts by maneuver type.

    Returns
    -------
    dict
        Mapping maneuver_type → count.
    """
    counts: Dict[str, int] = {}
    for result in sim_results:
        mt = result.maneuver_type or "unknown"
        counts[mt] = counts.get(mt, 0) + len(result.critical_points)
    return counts
