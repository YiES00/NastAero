"""Parallel batch trim execution with checkpointing.

Orchestrates the execution of all certification load cases:
- Groups flight cases by weight/CG condition (shared TrimSharedData)
- Executes trim solutions in batches with progress logging
- Saves/resumes from .naero checkpoint files
- Handles landing cases via quasi-static analysis path

The BatchRunner is the central execution engine that connects:
  LoadCaseMatrix → SOL 144 trim solver → nodal loads → VMT pipeline

References
----------
- Phase 4 of the FAA Part 23 certification framework
"""
from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import math

from ..case_generator import (
    TrimCondition, trim_condition_to_trim_card, trim_conditions_to_model,
)
from .aircraft_config import (
    AircraftConfig, WeightCGCondition, CONM2Adjuster,
)
from .load_case_matrix import CertLoadCase, LoadCaseMatrix
from .landing_loads import (
    LandingCondition, compute_gear_reactions,
    compute_landing_inertial_forces, combine_forces,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch result container
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result of a single load case execution.

    Attributes
    ----------
    case_id : int
        Case identifier.
    category : str
        Case category (symmetric, gust, rolling, yaw, etc.).
    far_section : str
        FAR section reference.
    converged : bool
        Whether the trim converged (always True for landing static).
    nodal_forces : dict or None
        {node_id: ndarray(6)} nodal force distribution.
    trim_vars : dict or None
        Solved trim variables (ANGLEA, ELEV values).
    weight_label : str
        Weight/CG condition label.
    altitude_m : float
        Flight altitude.
    nz : float
        Load factor.
    mach : float
        Mach number.
    label : str
        Human-readable case description.
    """
    case_id: int = 0
    category: str = ""
    far_section: str = ""
    converged: bool = False
    nodal_forces: Optional[Dict[int, np.ndarray]] = None
    trim_vars: Optional[Dict[str, float]] = None
    weight_label: str = ""
    altitude_m: float = 0.0
    nz: float = 0.0
    mach: float = 0.0
    label: str = ""
    flight_state: Optional[Dict] = None


def _build_flight_state(case: CertLoadCase,
                        trim_vars: Optional[Dict[str, float]] = None,
                        ) -> Dict:
    """Build flight state dict for a load case.

    For dynamic cases (from 6-DOF sim), uses the pre-populated
    ``CertLoadCase.flight_state``.  For static cases, extracts state
    from trim variables and TrimCondition metadata.

    Parameters
    ----------
    case : CertLoadCase
        The load case.
    trim_vars : dict, optional
        Solved/fixed trim variables from SOL 144.

    Returns
    -------
    dict
        Flight state snapshot (speeds, angles, rates, controls).
    """
    # Dynamic cases already have full flight state
    if case.flight_state is not None:
        return dict(case.flight_state)

    # Static cases — reconstruct from trim variables
    tc = case.trim_condition
    tv = trim_vars or (tc.fixed_vars if tc else {})

    alpha_rad = tv.get("ANGLEA", 0.0)
    beta_rad = tv.get("SIDES", 0.0)
    nz = tv.get("URDD3", tc.nz if tc else 1.0)

    return {
        "V_eas_m_s": tc.velocity if tc else 0.0,
        "V_tas_m_s": tc.velocity if tc else 0.0,
        "altitude_m": case.altitude_m,
        "mach": tc.mach if tc else 0.0,
        "alpha_deg": math.degrees(alpha_rad),
        "beta_deg": math.degrees(beta_rad),
        "nz": nz,
        "ny": 0.0,
        "p_rad_s": 0.0,
        "q_rad_s": 0.0,
        "r_rad_s": 0.0,
        "p_dot_rad_s2": 0.0,
        "q_dot_rad_s2": 0.0,
        "r_dot_rad_s2": 0.0,
        "delta_e_deg": math.degrees(tv.get("ELEV", 0.0)),
        "delta_a_deg": math.degrees(tv.get("ARON", 0.0)),
        "delta_r_deg": math.degrees(tv.get("RUD", 0.0)),
        "reason": "",
        "maneuver_type": case.category,
    }


@dataclass
class BatchResult:
    """Container for all batch execution results.

    Attributes
    ----------
    case_results : list of CaseResult
        Individual case results.
    completed_ids : set of int
        Case IDs that have been completed.
    config : AircraftConfig
        Aircraft configuration used.
    wall_time_s : float
        Total wall-clock time for execution.
    """
    case_results: List[CaseResult] = field(default_factory=list)
    completed_ids: set = field(default_factory=set)
    config: Optional[AircraftConfig] = None
    wall_time_s: float = 0.0

    @property
    def n_converged(self) -> int:
        """Number of converged cases."""
        return sum(1 for r in self.case_results if r.converged)

    @property
    def n_total(self) -> int:
        """Total number of cases."""
        return len(self.case_results)

    def get_result(self, case_id: int) -> Optional[CaseResult]:
        """Get result by case ID."""
        for r in self.case_results:
            if r.case_id == case_id:
                return r
        return None

    def results_by_category(self, category: str) -> List[CaseResult]:
        """Get results filtered by category."""
        return [r for r in self.case_results if r.category == category]

    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        cats = {}
        for r in self.case_results:
            if r.category not in cats:
                cats[r.category] = {"total": 0, "converged": 0}
            cats[r.category]["total"] += 1
            if r.converged:
                cats[r.category]["converged"] += 1
        return {
            "total": self.n_total,
            "converged": self.n_converged,
            "wall_time_s": self.wall_time_s,
            "by_category": cats,
        }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class BatchRunner:
    """Parallel batch execution engine for certification load cases.

    Groups cases by weight/CG condition to reuse TrimSharedData,
    executes in configurable batch sizes, and supports checkpointing.

    Parameters
    ----------
    matrix : LoadCaseMatrix
        The load case matrix to execute.
    bdf_model : BDFModel
        The base BDF model (will be modified for each W/CG condition).
    n_workers : int
        Number of parallel workers (0 = sequential).
    batch_size : int
        Number of subcases per batch.
    checkpoint_dir : str
        Directory for checkpoint files. None to disable.

    Example
    -------
    >>> runner = BatchRunner(matrix, model, n_workers=4)
    >>> results = runner.run()
    >>> print(results.summary())
    """

    def __init__(self, matrix: LoadCaseMatrix,
                 bdf_model=None,
                 n_workers: int = 0,
                 batch_size: int = 50,
                 checkpoint_dir: Optional[str] = None,
                 airfoil_config=None):
        self.matrix = matrix
        self.bdf_model = bdf_model
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.airfoil_config = airfoil_config
        self._batch_result = BatchResult(config=matrix.config)
        self._q_scale = self._detect_q_scale()

    def _detect_q_scale(self) -> float:
        """Detect model unit system and return q conversion factor.

        The certification pipeline computes dynamic pressure q in SI units
        (Pa = N/m²) from ISA atmosphere and EAS speeds in m/s.

        If the FE model uses millimetres (N-mm-sec), q must be converted
        to N/mm² (MPa) by multiplying by 1e-6.

        Detection heuristic: AEROS.REFC > 100 → mm-based model.
        """
        if self.bdf_model is None:
            return 1.0

        refc = 0.0
        if self.bdf_model.aeros:
            refc = self.bdf_model.aeros.refc
        elif self.bdf_model.aero:
            refc = self.bdf_model.aero.refc

        if refc > 100:  # mm-based model
            logger.info("Model unit detection: mm-based (refc=%.1f), "
                        "q_scale=1e-6 (Pa → N/mm²)", refc)
            return 1e-6
        return 1.0

    # ---------------------------------------------------------------
    # Model state save / restore (for W/CG group isolation)
    # ---------------------------------------------------------------

    def _save_model_state(self) -> dict:
        """Snapshot model trims, subcases, and CONM2 masses.

        Used to isolate W/CG groups: each group may modify CONM2 masses
        and inject TRIM cards/subcases. This captures the pre-modification
        state so it can be restored after the group is processed.
        """
        state = {
            'trims': dict(self.bdf_model.trims),
            'subcases': list(self.bdf_model.subcases),
            'conm2_masses': {},
        }
        for eid, m in self.bdf_model.masses.items():
            if hasattr(m, 'type') and m.type == 'CONM2':
                state['conm2_masses'][eid] = m.mass
        return state

    def _restore_model_state(self, state: dict) -> None:
        """Restore model trims, subcases, and CONM2 masses from snapshot."""
        self.bdf_model.trims = state['trims']
        self.bdf_model.subcases = state['subcases']
        for eid, mass_val in state['conm2_masses'].items():
            if eid in self.bdf_model.masses:
                self.bdf_model.masses[eid].mass = mass_val

    def run(self, resume: bool = False) -> BatchResult:
        """Execute all load cases.

        Parameters
        ----------
        resume : bool
            If True, resume from checkpoint.

        Returns
        -------
        BatchResult
            All case results.
        """
        t0 = time.time()

        # Load checkpoint if resuming
        if resume and self.checkpoint_dir:
            self._load_checkpoint()

        # Process flight cases (grouped by weight/CG)
        self._run_flight_cases()

        # Process landing cases
        self._run_landing_cases()

        # Final checkpoint after all cases
        if self.checkpoint_dir:
            self._save_checkpoint()

        self._batch_result.wall_time_s = time.time() - t0

        logger.info(
            "Batch complete: %d/%d converged in %.1fs",
            self._batch_result.n_converged,
            self._batch_result.n_total,
            self._batch_result.wall_time_s,
        )

        return self._batch_result

    # ---------------------------------------------------------------
    # Flight cases (SOL 144 trim)
    # ---------------------------------------------------------------

    def _run_flight_cases(self) -> None:
        """Execute flight load cases grouped by weight/CG.

        For each W/CG group:
        1. Save model state (trims, subcases, CONM2 masses)
        2. Apply CONM2 adjustments for this W/CG condition
        3. Solve all cases in batches via SOL 144
        4. Restore model state for the next W/CG group
        """
        # Group cases by weight/CG label
        groups: Dict[str, List[CertLoadCase]] = {}
        for c in self.matrix.flight_cases:
            wc_label = c.weight_cg.label if c.weight_cg else "default"
            if wc_label not in groups:
                groups[wc_label] = []
            groups[wc_label].append(c)

        total_flight = len(self.matrix.flight_cases)
        processed = 0

        for wc_label, cases in groups.items():
            logger.info(
                "Processing W/CG group '%s' (%d cases)",
                wc_label, len(cases))

            # Skip already completed cases
            remaining = [c for c in cases
                          if c.case_id not in self._batch_result.completed_ids]

            if not remaining:
                processed += len(cases)
                continue

            # Save model state before this W/CG group
            saved_state = None
            if self.bdf_model:
                saved_state = self._save_model_state()

            try:
                # Apply CONM2 adjustments for this W/CG
                if self.bdf_model and cases[0].weight_cg:
                    wc = cases[0].weight_cg
                    if wc.conm2_adjustments:
                        CONM2Adjuster.apply_adjustments(
                            self.bdf_model, wc.conm2_adjustments)

                # Process in batches
                for batch_start in range(0, len(remaining), self.batch_size):
                    batch = remaining[batch_start:
                                       batch_start + self.batch_size]

                    results = self._solve_trim_batch(batch)
                    self._batch_result.case_results.extend(results)
                    for r in results:
                        self._batch_result.completed_ids.add(r.case_id)

                    processed += len(batch)
                    logger.info(
                        "Progress: %d/%d (%.1f%%) | Category: %s",
                        processed, total_flight,
                        100.0 * processed / max(total_flight, 1),
                        batch[0].category if batch else "?",
                    )

                    # Checkpoint after each batch
                    if self.checkpoint_dir:
                        self._save_checkpoint()

            finally:
                # Restore model state for next W/CG group
                if saved_state is not None:
                    self._restore_model_state(saved_state)

    def _solve_trim_batch(self, cases: List[CertLoadCase]
                            ) -> List[CaseResult]:
        """Solve a batch of trim cases via SOL 144.

        Workflow:
        1. Convert TrimConditions → TRIM cards + subcases (injected into model)
        2. Filter model.subcases to only the new ones (avoid re-solving
           original BDF subcases)
        3. Call solve_trim() which builds TrimSharedData once and solves
           all subcases (sequential or parallel)
        4. Map SubcaseResult back to CaseResult via subcase_id → case index

        Parameters
        ----------
        cases : list of CertLoadCase
            Cases to solve.

        Returns
        -------
        list of CaseResult
        """
        results = []

        if self.bdf_model is None:
            # No model: placeholder results
            for c in cases:
                tc = c.trim_condition
                results.append(CaseResult(
                    case_id=tc.case_id,
                    category=c.category,
                    far_section=c.far_section,
                    converged=False,
                    weight_label=(c.weight_cg.label
                                   if c.weight_cg else ""),
                    altitude_m=c.altitude_m,
                    nz=tc.nz,
                    mach=tc.mach,
                    label=tc.label,
                    flight_state=_build_flight_state(c),
                ))
            return results

        # Build TrimCondition list
        tcs = [c.trim_condition for c in cases]

        # Scale q from SI (Pa) to model units if needed.
        # The certification pipeline computes q in Pa (N/m²) from
        # ISA atmosphere and EAS speeds.  An N-mm-sec model needs
        # q in N/mm² (MPa), which is 1e-6 × Pa.
        original_qs = None
        if self._q_scale != 1.0:
            original_qs = [tc.q for tc in tcs]
            for tc in tcs:
                tc.q *= self._q_scale
            logger.info("  q scaled by %.1e for model units "
                        "(e.g. %.4f Pa → %.4e model)",
                        self._q_scale,
                        original_qs[0] if original_qs else 0,
                        tcs[0].q if tcs else 0)

        try:
            # 1) Inject TRIM cards + subcases into model
            created = trim_conditions_to_model(self.bdf_model, tcs)
            # created = [(trim_card, subcase_id), ...]

            # Build subcase_id → case index mapping
            sc_id_to_idx = {sc_id: i for i, (_, sc_id) in enumerate(created)}

            # 2) Keep only newly created subcases (avoid re-solving
            #    original BDF subcases from previous groups)
            new_sc_ids = {sc_id for _, sc_id in created}
            original_subcases = self.bdf_model.subcases
            self.bdf_model.subcases = [
                sc for sc in self.bdf_model.subcases
                if sc.id in new_sc_ids
            ]

            try:
                # 3) Solve via SOL 144
                from ...solvers.sol144 import solve_trim
                result_data = solve_trim(
                    self.bdf_model,
                    n_workers=self.n_workers,
                    airfoil_config=self.airfoil_config,
                )
            finally:
                # Restore subcases list (outer _run_flight_cases
                # does full restore, but we need this for batch loop)
                self.bdf_model.subcases = original_subcases

            # 4) Map SubcaseResults → CaseResults
            solved_case_ids = set()
            for sc_result in result_data.subcases:
                idx = sc_id_to_idx.get(sc_result.subcase_id)
                if idx is None:
                    continue  # Pre-existing subcase, skip
                if idx >= len(cases):
                    continue

                c = cases[idx]
                tc = c.trim_condition

                # Convergence: nodal forces computed successfully
                converged = (sc_result.nodal_combined_forces is not None
                             and len(sc_result.nodal_combined_forces) > 0)

                results.append(CaseResult(
                    case_id=tc.case_id,
                    category=c.category,
                    far_section=c.far_section,
                    converged=converged,
                    nodal_forces=sc_result.nodal_combined_forces,
                    trim_vars=sc_result.trim_variables,
                    weight_label=(c.weight_cg.label
                                   if c.weight_cg else ""),
                    altitude_m=c.altitude_m,
                    nz=tc.nz,
                    mach=tc.mach,
                    label=tc.label,
                    flight_state=_build_flight_state(
                        c, sc_result.trim_variables),
                ))
                solved_case_ids.add(tc.case_id)

            # 5) Handle cases that didn't get a SubcaseResult
            for c in cases:
                if c.case_id not in solved_case_ids:
                    tc = c.trim_condition
                    results.append(CaseResult(
                        case_id=tc.case_id,
                        category=c.category,
                        far_section=c.far_section,
                        converged=False,
                        weight_label=(c.weight_cg.label
                                       if c.weight_cg else ""),
                        altitude_m=c.altitude_m,
                        nz=tc.nz,
                        mach=tc.mach,
                        label=tc.label,
                        flight_state=_build_flight_state(c),
                    ))

        except Exception as e:
            logger.error("Solver failed for batch of %d cases: %s",
                         len(cases), e)
            for c in cases:
                tc = c.trim_condition
                results.append(CaseResult(
                    case_id=tc.case_id,
                    category=c.category,
                    far_section=c.far_section,
                    converged=False,
                    weight_label=(c.weight_cg.label
                                   if c.weight_cg else ""),
                    altitude_m=c.altitude_m,
                    nz=tc.nz,
                    mach=tc.mach,
                    label=tc.label,
                    flight_state=_build_flight_state(c),
                ))

        finally:
            # Restore original q values so TrimConditions aren't mutated
            if original_qs is not None:
                for i, tc in enumerate(tcs):
                    tc.q = original_qs[i]

        return results

    # ---------------------------------------------------------------
    # Landing cases (quasi-static)
    # ---------------------------------------------------------------

    def _detect_gravity(self) -> float:
        """Detect gravitational acceleration from BDF model unit system."""
        if self.bdf_model is None:
            return 9810.0  # Default N-mm-sec
        refc = 0.0
        if self.bdf_model.aeros:
            refc = self.bdf_model.aeros.refc
        elif self.bdf_model.aero:
            refc = self.bdf_model.aero.refc
        return 9810.0 if refc > 100 else 9.81

    def _run_landing_cases(self) -> None:
        """Execute landing/ground load cases."""
        for cond in self.matrix.landing_cases:
            if cond.case_id in self._batch_result.completed_ids:
                continue

            result = self._solve_landing_case(cond)
            self._batch_result.case_results.append(result)
            self._batch_result.completed_ids.add(result.case_id)

    def _solve_landing_case(self, cond: LandingCondition) -> CaseResult:
        """Solve a single landing case.

        Computes gear reaction forces + inertial forces and combines.

        Parameters
        ----------
        cond : LandingCondition
            The landing condition.

        Returns
        -------
        CaseResult
        """
        gear = self.matrix.config.landing_gear

        # Gear reaction forces
        gear_forces = compute_gear_reactions(cond, gear)

        # Inertial forces (if model available for node masses)
        nodal_forces = gear_forces  # At minimum, we have gear reactions

        # If model is available, compute full inertial distribution
        if self.bdf_model is not None:
            # Extract node masses from CONM2s
            node_masses: Dict[int, float] = {}
            if hasattr(self.bdf_model, 'conm2s'):
                for eid, conm2 in self.bdf_model.conm2s.items():
                    nid = conm2.node_id
                    if nid not in node_masses:
                        node_masses[nid] = 0.0
                    node_masses[nid] += conm2.mass

            # Detect gravity from model unit system
            g = self._detect_gravity()
            inertial = compute_landing_inertial_forces(
                cond, node_masses, g=g)
            nodal_forces = combine_forces(gear_forces, inertial)

        return CaseResult(
            case_id=cond.case_id,
            category="landing",
            far_section=cond.far_section,
            converged=True,
            nodal_forces=nodal_forces,
            weight_label=(cond.weight_cg.label
                           if cond.weight_cg else ""),
            nz=cond.nz_cg,
            label=cond.label,
        )

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        cp_path = os.path.join(self.checkpoint_dir, "batch_checkpoint.json")

        # Save metadata (not nodal forces — those go to .naero)
        data = {
            "completed_ids": list(self._batch_result.completed_ids),
            "n_results": len(self._batch_result.case_results),
            "results_metadata": [],
        }

        for r in self._batch_result.case_results:
            data["results_metadata"].append({
                "case_id": r.case_id,
                "category": r.category,
                "far_section": r.far_section,
                "converged": r.converged,
                "weight_label": r.weight_label,
                "altitude_m": r.altitude_m,
                "nz": r.nz,
                "mach": r.mach,
                "label": r.label,
            })

        with open(cp_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug("Checkpoint saved: %d cases", len(data["completed_ids"]))

    def _load_checkpoint(self) -> None:
        """Resume from checkpoint file."""
        if not self.checkpoint_dir:
            return

        cp_path = os.path.join(self.checkpoint_dir, "batch_checkpoint.json")
        if not os.path.exists(cp_path):
            return

        with open(cp_path, 'r') as f:
            data = json.load(f)

        self._batch_result.completed_ids = set(data.get("completed_ids", []))

        for meta in data.get("results_metadata", []):
            self._batch_result.case_results.append(CaseResult(
                case_id=meta["case_id"],
                category=meta["category"],
                far_section=meta["far_section"],
                converged=meta["converged"],
                weight_label=meta["weight_label"],
                altitude_m=meta["altitude_m"],
                nz=meta["nz"],
                mach=meta["mach"],
                label=meta["label"],
            ))

        logger.info(
            "Resumed from checkpoint: %d cases completed",
            len(self._batch_result.completed_ids))
