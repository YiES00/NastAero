"""Export critical design loads as Nastran FORCE/MOMENT BDF cards.

After envelope processing identifies design-driving load cases, this module
exports the corresponding nodal force distributions as BDF-format FORCE* and
MOMENT* cards.  Stress analysts apply these external loads to the full-vehicle
FE model to compute internal loads for component sizing.

Outputs
-------
- Individual BDF files per critical (or all converged) load case
- Master BDF with INCLUDE + SUBCASE definitions
- Summary CSV of exported cases

References
----------
- Phase 5+ of the FAA Part 23 certification framework
"""
from __future__ import annotations

import csv
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .batch_runner import BatchResult, CaseResult
from .envelope import CriticalCase, EnvelopeProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_critical_forces(
    batch_result: BatchResult,
    proc: EnvelopeProcessor,
    model,
    output_dir: str,
    include_all: bool = False,
) -> Dict[str, Any]:
    """Export critical design loads as Nastran FORCE/MOMENT BDF cards.

    Parameters
    ----------
    batch_result : BatchResult
        Completed batch results containing nodal forces.
    proc : EnvelopeProcessor
        Envelope processor with identified critical cases.
    model : BDFModel
        The FE model (used only for metadata).
    output_dir : str
        Directory to write BDF files into.
    include_all : bool
        If True, export all converged cases (not just critical).

    Returns
    -------
    dict
        Summary with keys: n_cases, n_force_cards, n_moment_cards,
        master_bdf, summary_csv, case_files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect critical cases and build reason mapping
    all_critical = proc.get_critical_cases()
    critical_ids: Set[int] = {cc.case_id for cc in all_critical}
    critical_reasons = _build_critical_reasons(all_critical)

    # 2. Determine which cases to export
    if include_all:
        export_ids = sorted({
            cr.case_id for cr in batch_result.case_results
            if cr.converged and cr.nodal_forces
        })
    else:
        export_ids = sorted(critical_ids)

    # 3. Write individual BDF files
    case_files: Dict[int, str] = {}
    total_force_cards = 0
    total_moment_cards = 0

    for cid in export_ids:
        cr = batch_result.get_result(cid)
        if not cr or not cr.nodal_forces:
            continue

        safe_label = _safe_filename(cr.label or f"case_{cid}")
        safe_cat = _safe_filename(cr.category or "unknown")
        fname = f"case_{cid:04d}_{safe_cat}_{safe_label}.bdf"
        fpath = os.path.join(output_dir, fname)

        header = _build_header(cr, critical_reasons.get(cid, []))
        n_force, n_moment = _write_force_bdf(
            cr.nodal_forces, fpath, load_sid=cid, header=header,
        )
        case_files[cid] = fname
        total_force_cards += n_force
        total_moment_cards += n_moment

    # 4. Master BDF
    master_path = os.path.join(output_dir, "critical_loads_master.bdf")
    _write_master_bdf(case_files, batch_result, output_dir, master_path)

    # 5. Summary CSV
    csv_path = os.path.join(output_dir, "critical_loads_summary.csv")
    _write_summary_csv(
        case_files, batch_result, critical_reasons, csv_path,
    )

    result = {
        "n_cases": len(case_files),
        "n_force_cards": total_force_cards,
        "n_moment_cards": total_moment_cards,
        "master_bdf": master_path,
        "summary_csv": csv_path,
        "case_files": case_files,
    }

    logger.info(
        "Exported %d cases (%d FORCE, %d MOMENT cards) to %s",
        result["n_cases"], total_force_cards, total_moment_cards, output_dir,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_critical_reasons(
    critical_cases: List[CriticalCase],
) -> Dict[int, List[Tuple[str, str, str, float, float]]]:
    """Map case_id -> list of (component, quantity, extreme, station, value)."""
    reasons: Dict[int, list] = {}
    for cc in critical_cases:
        reasons.setdefault(cc.case_id, []).append(
            (cc.component, cc.quantity, cc.extreme, cc.station, cc.value)
        )
    return reasons


def _safe_filename(s: str) -> str:
    """Sanitise a string for safe use in file names."""
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9_\-+.]", "", s)
    # Trim to reasonable length
    return s[:60]


_QTY_LABELS = {"V": "Shear", "M": "Bending", "T": "Torsion"}
_QTY_UNITS = {"V": "N", "M": "N\u00b7mm", "T": "N\u00b7mm"}


def _build_header(
    cr: CaseResult,
    reasons: List[Tuple[str, str, str, float, float]],
) -> str:
    """Build a BDF comment header with case metadata."""
    lines = [
        "$ =====================================================",
        "$ CERTIFICATION CRITICAL DESIGN LOAD",
        f"$ Case ID: {cr.case_id}",
    ]
    if cr.label:
        lines.append(f"$ Label: {cr.label}")
    lines.append(f"$ Category: {cr.category} | FAR: {cr.far_section}")
    lines.append(
        f"$ nz = {cr.nz:+.2f}, Mach = {cr.mach:.3f}, "
        f"Alt = {cr.altitude_m:.0f} m"
    )
    if cr.weight_label:
        lines.append(f"$ Weight: {cr.weight_label}")
    lines.append("$")

    if reasons:
        lines.append("$ CRITICAL FOR:")
        for comp, qty, ext, sta, val in reasons:
            qlbl = _QTY_LABELS.get(qty, qty)
            qunit = _QTY_UNITS.get(qty, "")
            lines.append(
                f"$   {comp}: {ext} {qlbl} at sta {sta:.1f} mm "
                f"({val:,.0f} {qunit})"
            )
    else:
        lines.append("$ (exported as non-critical converged case)")

    lines.append("$ =====================================================")
    lines.append("$")
    return "\n".join(lines) + "\n"


def _write_force_bdf(
    nodal_forces: Dict[int, np.ndarray],
    filepath: str,
    load_sid: int,
    header: str,
) -> Tuple[int, int]:
    """Write FORCE*/MOMENT* BDF cards with metadata header.

    Reuses the Nastran fixed-16 format from trim_loads.write_force_cards().

    Returns (n_force_cards, n_moment_cards).
    """
    n_force = 0
    n_moment = 0
    cid = 0  # basic coordinate system

    with open(filepath, "w") as f:
        f.write(header)

        for nid in sorted(nodal_forces.keys()):
            fv = nodal_forces[nid]

            # Translational forces
            f_mag = np.linalg.norm(fv[:3])
            if f_mag > 1e-20:
                n1, n2, n3 = fv[:3] / f_mag
                f.write(
                    "FORCE*  %16d%16d%16d%16.8E\n"
                    % (load_sid, nid, cid, f_mag)
                )
                f.write(
                    "*       %16.8E%16.8E%16.8E\n" % (n1, n2, n3)
                )
                n_force += 1

            # Rotational moments
            m_mag = np.linalg.norm(fv[3:6])
            if m_mag > 1e-20:
                mn1, mn2, mn3 = fv[3:6] / m_mag
                f.write(
                    "MOMENT* %16d%16d%16d%16.8E\n"
                    % (load_sid, nid, cid, m_mag)
                )
                f.write(
                    "*       %16.8E%16.8E%16.8E\n" % (mn1, mn2, mn3)
                )
                n_moment += 1

    logger.info(
        "  BDF written: %s (%d FORCE, %d MOMENT cards)",
        os.path.basename(filepath), n_force, n_moment,
    )
    return n_force, n_moment


def _write_master_bdf(
    case_files: Dict[int, str],
    batch_result: BatchResult,
    output_dir: str,
    master_path: str,
) -> None:
    """Write master BDF with INCLUDE references and SUBCASE definitions."""
    with open(master_path, "w") as f:
        f.write("$ =============================================================\n")
        f.write("$ MASTER BDF — Critical Design Loads\n")
        f.write("$ Generated by NastAero Certification Loads Analysis\n")
        f.write(f"$ Total cases: {len(case_files)}\n")
        f.write("$ =============================================================\n")
        f.write("$\n")
        f.write("$ USAGE:\n")
        f.write("$   1. Include this file in your detailed FE model\n")
        f.write("$   2. Each SUBCASE applies one critical load condition\n")
        f.write("$   3. SID in FORCE/MOMENT cards matches SUBCASE LOAD\n")
        f.write("$\n")

        # INCLUDE statements in Bulk Data section
        f.write("$ --- INCLUDE statements (Bulk Data) ---\n")
        f.write("$\n")
        for cid in sorted(case_files.keys()):
            fname = case_files[cid]
            f.write(f"INCLUDE '{fname}'\n")

        f.write("$\n")
        f.write("$ --- SUBCASE definitions (Case Control) ---\n")
        f.write("$ Copy the following into your Case Control section:\n")
        f.write("$\n")

        for cid in sorted(case_files.keys()):
            cr = batch_result.get_result(cid)
            label = cr.label if cr and cr.label else f"Case {cid}"
            cat = cr.category if cr else ""
            f.write(f"$ SUBCASE {cid}\n")
            f.write(f"$   SUBTITLE = {label} [{cat}]\n")
            f.write(f"$   LOAD = {cid}\n")
            f.write("$\n")

    logger.info("Master BDF written: %s", master_path)


def _write_summary_csv(
    case_files: Dict[int, str],
    batch_result: BatchResult,
    critical_reasons: Dict[int, list],
    csv_path: str,
) -> None:
    """Write summary CSV of exported cases."""
    fieldnames = [
        "case_id", "label", "category", "far_section",
        "nz", "mach", "altitude_m", "weight",
        "n_critical_for", "critical_for", "bdf_file",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cid in sorted(case_files.keys()):
            cr = batch_result.get_result(cid)
            reasons = critical_reasons.get(cid, [])

            # Format critical reasons
            reason_strs = []
            for comp, qty, ext, sta, val in reasons:
                qlbl = _QTY_LABELS.get(qty, qty)
                reason_strs.append(
                    f"{comp} {ext} {qlbl} @{sta:.0f}mm"
                )

            writer.writerow({
                "case_id": cid,
                "label": cr.label if cr else "",
                "category": cr.category if cr else "",
                "far_section": cr.far_section if cr else "",
                "nz": f"{cr.nz:.2f}" if cr else "",
                "mach": f"{cr.mach:.3f}" if cr else "",
                "altitude_m": f"{cr.altitude_m:.0f}" if cr else "",
                "weight": cr.weight_label if cr else "",
                "n_critical_for": len(reasons),
                "critical_for": "; ".join(reason_strs),
                "bdf_file": case_files[cid],
            })

    logger.info("Summary CSV written: %s", csv_path)
