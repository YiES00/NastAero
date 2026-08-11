"""Certification report generation for FAA Part 23.

Produces structured summary tables and compliance matrices from
envelope processing results.

Report contents:
- Critical loads table: design-driving loads per component/station
- Regulatory compliance matrix: FAR §23.321-§23.511 coverage
- Lateral maneuver summary: roll/yaw case results
- Landing loads summary: ground condition results
- Load case statistics: category counts, convergence rates

References
----------
- Phase 6 of the FAA Part 23 certification framework
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .batch_runner import BatchResult, CaseResult
from .envelope import (
    EnvelopeProcessor, CriticalCase, ComponentEnvelope,
)
from .load_case_matrix import LoadCaseMatrix


# ---------------------------------------------------------------------------
# Report tables
# ---------------------------------------------------------------------------

@dataclass
class CriticalLoadsRow:
    """One row in the critical loads table.

    Attributes
    ----------
    component : str
        Structural component.
    station : float
        Span station.
    quantity : str
        V, M, or T.
    max_value : float
        Maximum value.
    max_case_id : int
        Case producing max.
    max_category : str
        Category of max case.
    max_nz : float
        nz of max case.
    min_value : float
        Minimum value.
    min_case_id : int
        Case producing min.
    min_category : str
        Category of min case.
    min_nz : float
        nz of min case.
    """
    component: str = ""
    station: float = 0.0
    quantity: str = ""
    max_value: float = 0.0
    max_case_id: int = 0
    max_category: str = ""
    max_nz: float = 0.0
    min_value: float = 0.0
    min_case_id: int = 0
    min_category: str = ""
    min_nz: float = 0.0


@dataclass
class ComplianceEntry:
    """FAR section compliance entry.

    Attributes
    ----------
    section : str
        FAR section (e.g., "§23.337").
    title : str
        Section title.
    n_cases : int
        Number of cases covering this section.
    status : str
        "covered", "partial", or "not_covered".
    notes : str
        Additional notes.
    """
    section: str = ""
    title: str = ""
    n_cases: int = 0
    status: str = "not_covered"
    notes: str = ""


# FAR Part 23 section reference table
FAR_SECTIONS = {
    "§23.321": "Flight loads — General",
    "§23.331(c)": "Checked maneuver conditions",
    "§23.333": "Flight envelope",
    "§23.337": "Limit maneuvering load factors",
    "§23.341": "Gust load factors",
    "§23.345": "High lift devices",
    "§23.349": "Rolling conditions",
    "§23.351": "Yaw maneuver conditions",
    "§23.471": "Ground loads — General",
    "§23.473": "Landing load factors",
    "§23.479": "Level landing conditions",
    "§23.481": "Tail-down landing",
    "§23.483": "One-wheel landing",
    "§23.485": "Side load conditions",
    "§23.487": "Rebound landing",
    "§23.491": "Taxi conditions",
    "§23.493": "Braked roll conditions",
    "§23.497": "Turning conditions",
    "§23.499": "Nose-wheel yaw",
}


# ---------------------------------------------------------------------------
# Certification report
# ---------------------------------------------------------------------------

class CertificationReport:
    """Report generator for Part 23 certification loads analysis.

    Parameters
    ----------
    matrix : LoadCaseMatrix
        The load case matrix.
    batch_result : BatchResult
        Execution results.
    envelope_proc : EnvelopeProcessor or None
        Envelope processing results.

    Example
    -------
    >>> report = CertificationReport(matrix, batch_result, envelope_proc)
    >>> table = report.critical_loads_table()
    >>> compliance = report.regulatory_compliance_matrix()
    >>> report.to_csv("cert_report.csv")
    """

    def __init__(self, matrix: LoadCaseMatrix,
                 batch_result: BatchResult,
                 envelope_proc: Optional[EnvelopeProcessor] = None):
        self.matrix = matrix
        self.batch_result = batch_result
        self.envelope_proc = envelope_proc

    # ---------------------------------------------------------------
    # Critical loads table
    # ---------------------------------------------------------------

    def critical_loads_table(self, components: List[str] = None
                              ) -> List[CriticalLoadsRow]:
        """Generate critical loads table.

        For each component and station, reports max/min V, M, T
        with the controlling case information.

        Parameters
        ----------
        components : list of str, optional
            Components to include. None = all.

        Returns
        -------
        list of CriticalLoadsRow
        """
        if self.envelope_proc is None:
            return []

        rows = []
        critical = self.envelope_proc.get_critical_cases()

        # Group by (component, station)
        from collections import defaultdict
        groups = defaultdict(list)
        for cc in critical:
            if components and cc.component not in components:
                continue
            groups[(cc.component, cc.station)].append(cc)

        for (comp, sta), cases in sorted(groups.items()):
            for qty in ["V", "M", "T"]:
                max_cc = next(
                    (c for c in cases
                     if c.quantity == qty and c.extreme == "max"),
                    None)
                min_cc = next(
                    (c for c in cases
                     if c.quantity == qty and c.extreme == "min"),
                    None)

                rows.append(CriticalLoadsRow(
                    component=comp,
                    station=sta,
                    quantity=qty,
                    max_value=max_cc.value if max_cc else 0.0,
                    max_case_id=max_cc.case_id if max_cc else 0,
                    max_category=max_cc.category if max_cc else "",
                    max_nz=max_cc.nz if max_cc else 0.0,
                    min_value=min_cc.value if min_cc else 0.0,
                    min_case_id=min_cc.case_id if min_cc else 0,
                    min_category=min_cc.category if min_cc else "",
                    min_nz=min_cc.nz if min_cc else 0.0,
                ))

        return rows

    # ---------------------------------------------------------------
    # Regulatory compliance matrix
    # ---------------------------------------------------------------

    def regulatory_compliance_matrix(self) -> List[ComplianceEntry]:
        """Generate FAR section compliance matrix.

        Checks which FAR sections are covered by the load case matrix.

        Returns
        -------
        list of ComplianceEntry
        """
        # Count cases per FAR section
        section_counts: Dict[str, int] = {}

        for c in self.matrix.flight_cases:
            sec = c.far_section
            if sec:
                section_counts[sec] = section_counts.get(sec, 0) + 1

        for c in self.matrix.landing_cases:
            sec = c.far_section
            if sec:
                section_counts[sec] = section_counts.get(sec, 0) + 1

        entries = []
        for section, title in FAR_SECTIONS.items():
            count = section_counts.get(section, 0)
            if count > 0:
                status = "covered"
            else:
                status = "not_covered"

            entries.append(ComplianceEntry(
                section=section,
                title=title,
                n_cases=count,
                status=status,
            ))

        return entries

    # ---------------------------------------------------------------
    # Category summaries
    # ---------------------------------------------------------------

    def lateral_maneuver_summary(self) -> Dict[str, Any]:
        """Summary of rolling and yaw maneuver cases.

        Returns
        -------
        dict with rolling and yaw case statistics.
        """
        rolling = self.batch_result.results_by_category("rolling")
        yaw = self.batch_result.results_by_category("yaw")

        return {
            "rolling": {
                "n_cases": len(rolling),
                "n_converged": sum(1 for r in rolling if r.converged),
                "nz_range": (
                    min((r.nz for r in rolling), default=0),
                    max((r.nz for r in rolling), default=0),
                ),
            },
            "yaw": {
                "n_cases": len(yaw),
                "n_converged": sum(1 for r in yaw if r.converged),
                "nz_range": (
                    min((r.nz for r in yaw), default=0),
                    max((r.nz for r in yaw), default=0),
                ),
            },
        }

    def landing_loads_summary(self) -> Dict[str, Any]:
        """Summary of landing and ground handling cases.

        Returns
        -------
        dict with landing case statistics.
        """
        landing = self.batch_result.results_by_category("landing")

        return {
            "n_cases": len(landing),
            "n_converged": sum(1 for r in landing if r.converged),
            "nz_range": (
                min((r.nz for r in landing), default=0),
                max((r.nz for r in landing), default=0),
            ),
            "far_sections": sorted(set(
                r.far_section for r in landing if r.far_section)),
        }

    # ---------------------------------------------------------------
    # Full summary
    # ---------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Complete report summary.

        Returns
        -------
        dict with full statistics.
        """
        compliance = self.regulatory_compliance_matrix()
        n_covered = sum(1 for e in compliance if e.status == "covered")

        return {
            "total_cases": self.matrix.total_cases,
            "flight_cases": len(self.matrix.flight_cases),
            "landing_cases": len(self.matrix.landing_cases),
            "converged": self.batch_result.n_converged,
            "convergence_rate": (
                self.batch_result.n_converged / max(self.batch_result.n_total, 1)
            ),
            "far_sections_covered": n_covered,
            "far_sections_total": len(FAR_SECTIONS),
            "compliance_rate": n_covered / max(len(FAR_SECTIONS), 1),
            "category_breakdown": self.batch_result.summary()["by_category"],
            "lateral_maneuvers": self.lateral_maneuver_summary(),
            "landing_loads": self.landing_loads_summary(),
        }

    # ---------------------------------------------------------------
    # CSV export
    # ---------------------------------------------------------------

    def to_csv(self, filepath: str) -> None:
        """Export critical loads table to CSV.

        Parameters
        ----------
        filepath : str
            Output CSV file path.
        """
        rows = self.critical_loads_table()

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Component', 'Station', 'Quantity',
                'Max Value', 'Max Case ID', 'Max Category', 'Max nz',
                'Min Value', 'Min Case ID', 'Min Category', 'Min nz',
            ])
            for row in rows:
                writer.writerow([
                    row.component, f"{row.station:.1f}", row.quantity,
                    f"{row.max_value:.4g}", row.max_case_id,
                    row.max_category, f"{row.max_nz:.3f}",
                    f"{row.min_value:.4g}", row.min_case_id,
                    row.min_category, f"{row.min_nz:.3f}",
                ])

    def compliance_to_csv(self, filepath: str) -> None:
        """Export compliance matrix to CSV.

        Parameters
        ----------
        filepath : str
            Output CSV file path.
        """
        entries = self.regulatory_compliance_matrix()

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'FAR Section', 'Title', 'N Cases', 'Status', 'Notes',
            ])
            for e in entries:
                writer.writerow([
                    e.section, e.title, e.n_cases, e.status, e.notes,
                ])
