"""Parallel 6-DOF simulation orchestration and critical point extraction.

Orchestrates the execution of all maneuver/gust simulations:
1. Computes VLM derivatives and inertia tensor (once per W/CG condition)
2. Generates all FAR 23 maneuver profiles
3. Runs simulations in parallel via ProcessPoolExecutor
4. Extracts critical time points from each time history

The critical points are then converted to CertLoadCase objects by
sim_to_loads.py for detailed SOL 144 analysis.

References
----------
- Phase 4b-4d of the FAA Part 23 certification framework
"""
from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .aero_derivatives import (
    AeroDerivativeSet, build_derivative_set, compute_inertia_from_conm2,
)
from .aircraft_config import (
    AircraftConfig, WeightCGCondition,
    eas_to_tas, eas_to_mach, dynamic_pressure_from_eas,
)
from .flight_sim import (
    AircraftParams, AircraftState, ControlInput,
    SimTimeHistory, integrate_6dof, trim_initial_state,
    compute_nz_from_history,
)
from .maneuver_profiles import ManeuverProfile, generate_all_maneuver_profiles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Critical time point data structure
# ---------------------------------------------------------------------------

@dataclass
class CriticalTimePoint:
    """A critical time point extracted from simulation time history.

    Represents an instant where a response quantity (nz, angular rate, etc.)
    reaches an extreme value. The complete aerodynamic state at this point
    is captured for conversion to a SOL 144 load case.
    """
    t: float = 0.0           # Time (s)
    nz: float = 0.0          # Normal load factor (g's)
    ny: float = 0.0          # Lateral load factor (g's)
    alpha_deg: float = 0.0   # Angle of attack (degrees)
    beta_deg: float = 0.0    # Sideslip angle (degrees)
    p: float = 0.0           # Roll rate (rad/s)
    q: float = 0.0           # Pitch rate (rad/s)
    r: float = 0.0           # Yaw rate (rad/s)
    p_dot: float = 0.0       # Roll acceleration (rad/s²)
    q_dot: float = 0.0       # Pitch acceleration (rad/s²)
    r_dot: float = 0.0       # Yaw acceleration (rad/s²)
    V_total: float = 0.0     # Total airspeed (m/s)
    delta_e: float = 0.0     # Elevator (rad)
    delta_a: float = 0.0     # Aileron (rad)
    delta_r: float = 0.0     # Rudder (rad)
    reason: str = ""         # Why this point is critical
    maneuver_type: str = ""  # Maneuver category
    V_eas: float = 0.0       # Equivalent airspeed (m/s)
    weight_label: str = ""   # Weight/CG condition label
    far_section: str = ""    # FAR section reference


@dataclass
class SimRunResult:
    """Result of a single 6-DOF maneuver/gust simulation."""
    maneuver_type: str = ""
    V_eas: float = 0.0
    weight_label: str = ""
    label: str = ""
    far_section: str = ""
    critical_points: List[CriticalTimePoint] = field(default_factory=list)
    # Time history is NOT stored to save memory (only critical points kept)


# ---------------------------------------------------------------------------
# Critical point extraction
# ---------------------------------------------------------------------------

def extract_critical_points(
    history: SimTimeHistory,
    maneuver_type: str,
    V_eas: float,
    weight_label: str,
    far_section: str,
    label: str,
    max_points: int = 5,
    min_dt: float = 0.05,
    max_abs_nz: float = 8.0,
) -> List[CriticalTimePoint]:
    """Extract critical time points from a simulation time history.

    Identifies extrema of key response quantities:
    - max nz, min nz (normal load factor)
    - max |ny| (lateral load factor)
    - max |q| (pitch rate)
    - max |p| (roll rate)
    - max |r| (yaw rate)
    - max |β| (sideslip angle)

    Removes duplicates within min_dt seconds, returns up to max_points.

    Parameters
    ----------
    history : SimTimeHistory
        Complete time history from 6-DOF simulation.
    maneuver_type : str
        Maneuver type identifier.
    V_eas : float
        Equivalent airspeed (m/s).
    weight_label : str
        Weight/CG condition label.
    far_section : str
        FAR section reference.
    label : str
        Descriptive label.
    max_points : int
        Maximum number of critical points to extract per simulation.
    min_dt : float
        Minimum time separation between critical points (s).

    Returns
    -------
    list of CriticalTimePoint
        Extracted critical time points.
    """
    N = len(history.t)
    if N < 3:
        return []

    # Check for NaN/Inf in time history — simulation may have diverged
    if np.any(np.isnan(history.nz)) or np.any(np.isinf(history.nz)):
        logger.warning("  NaN/Inf in nz time history — simulation diverged, skipping")
        return []

    # Skip initial transient (first 0.05s) to avoid startup artifacts
    i_start = max(1, int(0.05 / (history.t[1] - history.t[0])))

    candidates = []

    def _add_candidate(idx, reason, score):
        """Add a candidate critical point with a relevance score."""
        if idx < i_start or idx >= N:
            return
        candidates.append((idx, reason, abs(score)))

    # --- Normal load factor extrema ---
    nz = history.nz[i_start:]
    if len(nz) > 0:
        i_max_nz = np.argmax(nz) + i_start
        i_min_nz = np.argmin(nz) + i_start
        _add_candidate(i_max_nz, "max nz", history.nz[i_max_nz])
        _add_candidate(i_min_nz, "min nz", -history.nz[i_min_nz])

    # --- Lateral load factor extrema ---
    ny = history.ny[i_start:]
    if len(ny) > 0 and np.max(np.abs(ny)) > 0.01:
        i_max_ny = np.argmax(np.abs(ny)) + i_start
        _add_candidate(i_max_ny, "max |ny|", history.ny[i_max_ny])

    # --- Angular rate extrema ---
    for rate_arr, rate_name in [
        (history.q_rate, "q"),
        (history.p_rate, "p"),
        (history.r_rate, "r"),
    ]:
        arr = rate_arr[i_start:]
        if len(arr) > 0 and np.max(np.abs(arr)) > 0.01:
            i_max = np.argmax(np.abs(arr)) + i_start
            _add_candidate(i_max, f"max |{rate_name}|", rate_arr[i_max])

    # --- Sideslip extrema ---
    beta = history.beta_deg[i_start:]
    if len(beta) > 0 and np.max(np.abs(beta)) > 0.1:
        i_max_beta = np.argmax(np.abs(beta)) + i_start
        _add_candidate(i_max_beta, "max |β|", history.beta_deg[i_max_beta])

    # --- Alpha extrema ---
    alpha = history.alpha_deg[i_start:]
    if len(alpha) > 0:
        i_max_alpha = np.argmax(alpha) + i_start
        i_min_alpha = np.argmin(alpha) + i_start
        _add_candidate(i_max_alpha, "max α", history.alpha_deg[i_max_alpha])
        _add_candidate(i_min_alpha, "min α", -history.alpha_deg[i_min_alpha])

    # Sort by relevance score (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Remove duplicates within min_dt
    selected_times = []
    result = []

    for idx, reason, score in candidates:
        if len(result) >= max_points:
            break
        t = history.t[idx]
        # Check for duplicates
        if any(abs(t - st) < min_dt for st in selected_times):
            continue
        selected_times.append(t)

        # Extract state at this time point
        s = history.states[idx]
        V = math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)

        # Skip critical points with unrealistically high load factors
        # (beyond what the linear aero model can physically represent,
        #  even with alpha stall clamping)
        nz_val = history.nz[idx]
        if abs(nz_val) > max_abs_nz:
            logger.debug("  Skipping critical point (|nz|=%.1f > %.1f): %s",
                         abs(nz_val), max_abs_nz, reason)
            continue

        # Angular accelerations (may not exist in legacy histories)
        _pdot = history.p_dot[idx] if hasattr(history, 'p_dot') and len(history.p_dot) > idx else 0.0
        _qdot = history.q_dot[idx] if hasattr(history, 'q_dot') and len(history.q_dot) > idx else 0.0
        _rdot = history.r_dot[idx] if hasattr(history, 'r_dot') and len(history.r_dot) > idx else 0.0

        ctp = CriticalTimePoint(
            t=t,
            nz=nz_val,
            ny=history.ny[idx],
            alpha_deg=history.alpha_deg[idx],
            beta_deg=history.beta_deg[idx],
            p=history.p_rate[idx],
            q=history.q_rate[idx],
            r=history.r_rate[idx],
            p_dot=float(_pdot),
            q_dot=float(_qdot),
            r_dot=float(_rdot),
            V_total=V,
            delta_e=history.controls[idx, 0],
            delta_a=history.controls[idx, 1],
            delta_r=history.controls[idx, 2],
            reason=reason,
            maneuver_type=maneuver_type,
            V_eas=V_eas,
            weight_label=weight_label,
            far_section=far_section,
        )
        result.append(ctp)

    return result


# ---------------------------------------------------------------------------
# Single simulation worker (pickle-safe for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_single_sim(args: Tuple) -> SimRunResult:
    """Run a single 6-DOF simulation (worker function for parallel execution).

    Parameters
    ----------
    args : tuple
        (params, profile, V_eas, alt_m, weight_label, label, max_abs_nz)

    Returns
    -------
    SimRunResult
        Result with extracted critical points (time history discarded).
    """
    # Support both old (6-element) and new (7-element) arg format
    if len(args) == 7:
        params, profile, V_eas, alt_m, weight_label, label, max_abs_nz = args
    else:
        params, profile, V_eas, alt_m, weight_label, label = args
        max_abs_nz = 8.0

    from .aircraft_config import eas_to_tas
    from .flight_sim import (
        AircraftParams, integrate_6dof, trim_initial_state,
        compute_nz_from_history,
    )

    V_tas = eas_to_tas(V_eas, alt_m)
    if V_tas < 1.0:
        return SimRunResult(
            maneuver_type=profile.maneuver_type,
            V_eas=V_eas,
            weight_label=weight_label,
            label=label,
            far_section=profile.far_section,
        )

    # Compute trim initial condition
    initial_state, de_trim = trim_initial_state(params, V_tas)

    # Update the control function's trim elevator for this speed
    # The control functions store de_trim internally — update it
    ctrl_func = profile.control_func
    if hasattr(ctrl_func, 'de_trim'):
        ctrl_func.de_trim = de_trim

    # Integrate
    history = integrate_6dof(
        params=params,
        initial_state=initial_state,
        control_func=ctrl_func,
        t_span=(0.0, profile.t_end),
        dt=0.005,
        gust_func=profile.gust_func,
    )

    # Recompute load factors from aerodynamic model
    compute_nz_from_history(params, history)

    # Extract critical points
    crits = extract_critical_points(
        history=history,
        maneuver_type=profile.maneuver_type,
        V_eas=V_eas,
        weight_label=weight_label,
        far_section=profile.far_section,
        label=label,
        max_abs_nz=max_abs_nz,
    )

    return SimRunResult(
        maneuver_type=profile.maneuver_type,
        V_eas=V_eas,
        weight_label=weight_label,
        label=label,
        far_section=profile.far_section,
        critical_points=crits,
    )


# ---------------------------------------------------------------------------
# SimRunner — orchestrates all simulations
# ---------------------------------------------------------------------------

class SimRunner:
    """Orchestrates 6-DOF flight dynamics simulations.

    Workflow:
    1. Compute VLM stability derivatives (once per W/CG condition)
    2. Compute inertia tensor from CONM2 distribution
    3. Generate all FAR 23 maneuver profiles
    4. Run simulations in parallel
    5. Collect critical time points

    Parameters
    ----------
    config : AircraftConfig
        Aircraft configuration.
    bdf_model : BDFModel
        Parsed BDF model.
    n_workers : int
        Number of parallel workers. 0 = auto (os.cpu_count()).
    """

    def __init__(self, config: AircraftConfig, bdf_model, n_workers: int = 0):
        self.config = config
        self.bdf_model = bdf_model
        self.n_workers = n_workers if n_workers > 0 else os.cpu_count() or 1

    def run_all(self) -> Tuple[List[SimRunResult], Dict[str, Any]]:
        """Run all 6-DOF simulations and extract critical points.

        Returns
        -------
        (results, info)
            results: List of SimRunResult with critical points.
            info: Summary dict with derivative set, inertia, counts, timing.
        """
        t_start = time.time()
        all_results = []
        total_crits = 0
        total_sims = 0

        info = {}

        for wc in self.config.weight_cg_conditions:
            logger.info("  W/CG condition: %s (%.0f N)", wc.label, wc.weight_N)

            # --- 1. Compute derivatives ---
            # Use VC Mach as representative
            alt_m = self.config.altitudes_m[0] if self.config.altitudes_m else 0.0
            mach = eas_to_mach(self.config.speeds.VC, alt_m)

            derivs = build_derivative_set(
                self.bdf_model, self.config, wc, mach,
            )
            info["derivs"] = derivs

            # --- 2. Compute inertia ---
            cg_xyz = np.array([wc.cg_x, 0.0, 0.0])
            inertia = compute_inertia_from_conm2(self.bdf_model, cg_xyz)
            info["inertia"] = inertia

            mass_kg = inertia["mass_kg"]
            if mass_kg < 1.0:
                logger.warning("  Mass too low (%.2f kg) — skipping", mass_kg)
                continue

            # Build AircraftParams (SI units)
            mm_to_m = 1e-3
            params = AircraftParams(
                mass=mass_kg,
                S=derivs.S_ref * mm_to_m**2,    # mm² → m²
                b=derivs.b_ref * mm_to_m,        # mm → m
                c_bar=derivs.c_bar * mm_to_m,    # mm → m
                Ixx=inertia["Ixx"],
                Iyy=inertia["Iyy"],
                Izz=inertia["Izz"],
                Ixz=inertia["Ixz"],
                derivs=derivs,
            )

            # Compute trim elevator for reference 1g condition at VC
            V_tas_vc = eas_to_tas(self.config.speeds.VC, alt_m)
            _, de_trim = trim_initial_state(params, V_tas_vc)

            # Compute V-n design load factor limits (§23.337)
            nz_pos_limit = self.config.nz_max(wc.weight_N)
            nz_neg_limit = self.config.nz_min(wc.weight_N)
            # Allow 10% dynamic overshoot tolerance for transient peaks
            max_abs_nz = nz_pos_limit * 1.1
            logger.info("  V-n limits: nz_pos=+%.2f, nz_neg=%.2f, "
                        "max_abs_nz=%.2f (1.1×)",
                        nz_pos_limit, nz_neg_limit, max_abs_nz)

            # --- 3. Generate maneuver profiles ---
            # Pass derivs so elevator deflections are sized to the V-n
            # diagram limit load factor instead of full control travel.
            profiles = generate_all_maneuver_profiles(
                self.config, wc, alt_m,
                delta_e_trim=de_trim,
                derivs=derivs,
            )
            logger.info("  %d maneuver profiles generated", len(profiles))

            # --- 4. Build simulation tasks ---
            rho, _, _ = _isa_atmosphere(alt_m)
            params.rho = rho

            tasks = []
            for prof, V_eas, label in profiles:
                tasks.append((params, prof, V_eas, alt_m, wc.label, label,
                              max_abs_nz))

            total_sims += len(tasks)

            # --- 5. Run simulations (parallel or sequential) ---
            results = self._execute_sims(tasks)
            all_results.extend(results)

            # Count critical points
            n_crits = sum(len(r.critical_points) for r in results)
            total_crits += n_crits
            logger.info("  → %d critical time points from %d simulations",
                        n_crits, len(results))

        elapsed = time.time() - t_start
        info["n_sims"] = total_sims
        info["n_critical_points"] = total_crits
        info["elapsed_s"] = elapsed

        logger.info("6-DOF simulation complete: %d sims, %d critical points (%.2fs)",
                    total_sims, total_crits, elapsed)

        return all_results, info

    def _execute_sims(self, tasks: List[Tuple]) -> List[SimRunResult]:
        """Execute simulation tasks, using parallel or sequential mode."""
        if len(tasks) == 0:
            return []

        # For small task counts or single worker, run sequentially
        if len(tasks) <= 4 or self.n_workers <= 1:
            results = []
            for task in tasks:
                try:
                    result = _run_single_sim(task)
                    results.append(result)
                except Exception as e:
                    logger.warning("  Sim failed (%s): %s", task[5], e)
            return results

        # Parallel execution
        results = [None] * len(tasks)
        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {}
                for i, task in enumerate(tasks):
                    fut = executor.submit(_run_single_sim, task)
                    futures[fut] = i

                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        logger.warning("  Sim %d failed: %s", idx, e)
                        results[idx] = SimRunResult(
                            label=tasks[idx][5],
                        )
        except Exception as e:
            logger.warning("  Parallel execution failed, falling back to sequential: %s", e)
            results = []
            for task in tasks:
                try:
                    result = _run_single_sim(task)
                    results.append(result)
                except Exception as ex:
                    logger.warning("  Sim failed (%s): %s", task[5], ex)

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Helper: ISA atmosphere (avoid circular import)
# ---------------------------------------------------------------------------

def _isa_atmosphere(altitude_m: float) -> Tuple[float, float, float]:
    """ISA atmosphere model (standalone to avoid import issues)."""
    g0 = 9.80665
    R = 287.05287
    gamma = 1.4
    T0 = 288.15
    P0 = 101325.0
    rho0 = 1.225
    L = 0.0065

    if altitude_m < 11000:
        T = T0 - L * altitude_m
        P = P0 * (T / T0) ** (g0 / (R * L))
    else:
        T11 = T0 - L * 11000.0
        P11 = P0 * (T11 / T0) ** (g0 / (R * L))
        T = T11
        P = P11 * math.exp(-g0 * (altitude_m - 11000.0) / (R * T11))

    rho = P / (R * T)
    a = math.sqrt(gamma * R * T)
    return rho, T, a
