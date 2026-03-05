"""VMT envelope processing and critical case identification.

Processes VMT results from all certification load cases to:
- Compute min/max envelopes per component and station
- Identify critical (design-driving) load cases
- Generate potato plot data (V-M, M-T scatter)
- Provide summary statistics for report generation

The EnvelopeProcessor is the post-processing engine that connects:
  BatchResult → VMT curves → envelopes → critical cases

References
----------
- Phase 5 of the FAA Part 23 certification framework
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .batch_runner import BatchResult, CaseResult


# ---------------------------------------------------------------------------
# Critical case record
# ---------------------------------------------------------------------------

@dataclass
class CriticalCase:
    """Record of a critical (envelope-defining) load case.

    Attributes
    ----------
    station : float
        Span station position.
    component : str
        Structural component name (e.g., "Wing", "HTP").
    quantity : str
        Load quantity: "V" (shear), "M" (bending), "T" (torsion).
    extreme : str
        "max" or "min".
    value : float
        The extreme value.
    case_id : int
        ID of the controlling load case.
    category : str
        Load case category (symmetric, gust, rolling, etc.).
    far_section : str
        FAR section reference.
    nz : float
        Load factor of the controlling case.
    label : str
        Case label.
    """
    station: float = 0.0
    component: str = ""
    quantity: str = ""
    extreme: str = ""
    value: float = 0.0
    case_id: int = 0
    category: str = ""
    far_section: str = ""
    nz: float = 0.0
    label: str = ""


# ---------------------------------------------------------------------------
# Station envelope data
# ---------------------------------------------------------------------------

@dataclass
class StationEnvelope:
    """Envelope data at a single span station.

    Attributes
    ----------
    station : float
        Span station position.
    V_max : float
        Maximum shear force.
    V_min : float
        Minimum shear force.
    M_max : float
        Maximum bending moment.
    M_min : float
        Minimum bending moment.
    T_max : float
        Maximum torsion.
    T_min : float
        Minimum torsion.
    V_max_case_id : int
        Case ID producing max shear.
    V_min_case_id : int
        Case ID producing min shear.
    M_max_case_id : int
        Case ID producing max bending.
    M_min_case_id : int
        Case ID producing min bending.
    T_max_case_id : int
        Case ID producing max torsion.
    T_min_case_id : int
        Case ID producing min torsion.
    """
    station: float = 0.0
    V_max: float = -math.inf
    V_min: float = math.inf
    M_max: float = -math.inf
    M_min: float = math.inf
    T_max: float = -math.inf
    T_min: float = math.inf
    V_max_case_id: int = 0
    V_min_case_id: int = 0
    M_max_case_id: int = 0
    M_min_case_id: int = 0
    T_max_case_id: int = 0
    T_min_case_id: int = 0


# ---------------------------------------------------------------------------
# Component envelope
# ---------------------------------------------------------------------------

@dataclass
class ComponentEnvelope:
    """Envelope for an entire structural component.

    Attributes
    ----------
    component : str
        Component name.
    stations : list of float
        Span station positions.
    envelopes : list of StationEnvelope
        Envelope data per station.
    n_cases : int
        Number of cases in envelope.
    """
    component: str = ""
    stations: List[float] = field(default_factory=list)
    envelopes: List[StationEnvelope] = field(default_factory=list)
    n_cases: int = 0

    @property
    def V_max_array(self) -> np.ndarray:
        """Max shear envelope array."""
        return np.array([e.V_max for e in self.envelopes])

    @property
    def V_min_array(self) -> np.ndarray:
        """Min shear envelope array."""
        return np.array([e.V_min for e in self.envelopes])

    @property
    def M_max_array(self) -> np.ndarray:
        """Max bending moment envelope array."""
        return np.array([e.M_max for e in self.envelopes])

    @property
    def M_min_array(self) -> np.ndarray:
        """Min bending moment envelope array."""
        return np.array([e.M_min for e in self.envelopes])

    @property
    def T_max_array(self) -> np.ndarray:
        """Max torsion envelope array."""
        return np.array([e.T_max for e in self.envelopes])

    @property
    def T_min_array(self) -> np.ndarray:
        """Min torsion envelope array."""
        return np.array([e.T_min for e in self.envelopes])


# ---------------------------------------------------------------------------
# Potato plot data
# ---------------------------------------------------------------------------

@dataclass
class PotatoData:
    """Data for potato plot (V-M or M-T scatter) at a span station.

    Attributes
    ----------
    station : float
        Span station position.
    component : str
        Component name.
    x_values : list of float
        X-axis values (e.g., shear V or bending M).
    y_values : list of float
        Y-axis values (e.g., bending M or torsion T).
    case_ids : list of int
        Corresponding case IDs.
    categories : list of str
        Category of each case.
    x_label : str
        X-axis label (e.g., "Shear V (N)").
    y_label : str
        Y-axis label (e.g., "Bending M (N-mm)").
    hull_x : ndarray or None
        Convex hull x-coordinates.
    hull_y : ndarray or None
        Convex hull y-coordinates.
    """
    station: float = 0.0
    component: str = ""
    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    case_ids: List[int] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    hull_x: Optional[np.ndarray] = None
    hull_y: Optional[np.ndarray] = None

    @property
    def n_points(self) -> int:
        return len(self.x_values)


# ---------------------------------------------------------------------------
# Envelope processor
# ---------------------------------------------------------------------------

class EnvelopeProcessor:
    """Post-processor for computing VMT envelopes and critical cases.

    Processes VMT data from all load cases to build envelopes,
    identify critical cases, and generate potato plot data.

    Parameters
    ----------
    batch_result : BatchResult
        Results from BatchRunner.
    vmt_data : dict, optional
        Precomputed VMT data: {case_id: {component: VMTCurve}}.
        If None, VMT must be computed externally.

    Example
    -------
    >>> proc = EnvelopeProcessor(batch_result, vmt_data)
    >>> proc.compute_envelopes()
    >>> critical = proc.get_critical_cases("Wing")
    >>> potato = proc.compute_potato("Wing", station=5000.0)
    """

    def __init__(self, batch_result: BatchResult,
                 vmt_data: Optional[Dict[int, Dict[str, Any]]] = None):
        self.batch_result = batch_result
        self.vmt_data = vmt_data or {}
        self._component_envelopes: Dict[str, ComponentEnvelope] = {}
        self._critical_cases: List[CriticalCase] = []

    # ---------------------------------------------------------------
    # VMT data ingestion
    # ---------------------------------------------------------------

    def add_vmt_curve(self, case_id: int, component: str,
                       stations: np.ndarray,
                       shear: np.ndarray,
                       bending: np.ndarray,
                       torsion: np.ndarray) -> None:
        """Add VMT curve data for a specific case and component.

        Parameters
        ----------
        case_id : int
            Load case ID.
        component : str
            Component name.
        stations, shear, bending, torsion : ndarray
            VMT arrays at span stations.
        """
        if case_id not in self.vmt_data:
            self.vmt_data[case_id] = {}

        self.vmt_data[case_id][component] = {
            "stations": stations,
            "shear": shear,
            "bending": bending,
            "torsion": torsion,
        }

    # ---------------------------------------------------------------
    # Envelope computation
    # ---------------------------------------------------------------

    def compute_envelopes(self) -> Dict[str, ComponentEnvelope]:
        """Compute VMT envelopes for all components.

        Iterates through all cases and updates min/max at each station.

        Returns
        -------
        dict of {component: ComponentEnvelope}
        """
        self._component_envelopes = {}

        for case_id, comp_data in self.vmt_data.items():
            # Look up case metadata
            case_result = self.batch_result.get_result(case_id)

            for comp_name, vmt in comp_data.items():
                stations = vmt["stations"]
                shear = vmt["shear"]
                bending = vmt["bending"]
                torsion = vmt["torsion"]

                if comp_name not in self._component_envelopes:
                    self._component_envelopes[comp_name] = ComponentEnvelope(
                        component=comp_name,
                        stations=list(stations),
                        envelopes=[StationEnvelope(station=s)
                                    for s in stations],
                    )

                env = self._component_envelopes[comp_name]
                env.n_cases += 1

                for i, sta in enumerate(stations):
                    se = env.envelopes[i]

                    if shear[i] > se.V_max:
                        se.V_max = shear[i]
                        se.V_max_case_id = case_id
                    if shear[i] < se.V_min:
                        se.V_min = shear[i]
                        se.V_min_case_id = case_id

                    if bending[i] > se.M_max:
                        se.M_max = bending[i]
                        se.M_max_case_id = case_id
                    if bending[i] < se.M_min:
                        se.M_min = bending[i]
                        se.M_min_case_id = case_id

                    if torsion[i] > se.T_max:
                        se.T_max = torsion[i]
                        se.T_max_case_id = case_id
                    if torsion[i] < se.T_min:
                        se.T_min = torsion[i]
                        se.T_min_case_id = case_id

        return self._component_envelopes

    def get_envelope(self, component: str) -> Optional[ComponentEnvelope]:
        """Get envelope for a specific component."""
        return self._component_envelopes.get(component)

    # ---------------------------------------------------------------
    # Critical case identification
    # ---------------------------------------------------------------

    def identify_critical_cases(self) -> List[CriticalCase]:
        """Identify all critical (envelope-defining) cases.

        For each station and quantity (V, M, T), records the case
        that produces the max and min value.

        Returns
        -------
        list of CriticalCase
        """
        self._critical_cases = []

        for comp_name, env in self._component_envelopes.items():
            for se in env.envelopes:
                for qty, extreme, value, cid in [
                    ("V", "max", se.V_max, se.V_max_case_id),
                    ("V", "min", se.V_min, se.V_min_case_id),
                    ("M", "max", se.M_max, se.M_max_case_id),
                    ("M", "min", se.M_min, se.M_min_case_id),
                    ("T", "max", se.T_max, se.T_max_case_id),
                    ("T", "min", se.T_min, se.T_min_case_id),
                ]:
                    if abs(value) == math.inf:
                        continue

                    # Look up case metadata
                    cr = self.batch_result.get_result(cid)
                    category = cr.category if cr else ""
                    far_section = cr.far_section if cr else ""
                    nz = cr.nz if cr else 0.0
                    label = cr.label if cr else ""

                    self._critical_cases.append(CriticalCase(
                        station=se.station,
                        component=comp_name,
                        quantity=qty,
                        extreme=extreme,
                        value=value,
                        case_id=cid,
                        category=category,
                        far_section=far_section,
                        nz=nz,
                        label=label,
                    ))

        return self._critical_cases

    def get_critical_cases(self, component: str = None,
                             quantity: str = None,
                             ) -> List[CriticalCase]:
        """Get critical cases with optional filtering.

        Parameters
        ----------
        component : str, optional
            Filter by component name.
        quantity : str, optional
            Filter by quantity ("V", "M", "T").

        Returns
        -------
        list of CriticalCase
        """
        result = self._critical_cases
        if component:
            result = [c for c in result if c.component == component]
        if quantity:
            result = [c for c in result if c.quantity == quantity]
        return result

    def critical_case_frequency(self) -> Dict[int, int]:
        """Count how many times each case appears as critical.

        Returns
        -------
        dict of {case_id: count}
            Frequency of each case appearing as critical.
        """
        freq: Dict[int, int] = {}
        for cc in self._critical_cases:
            freq[cc.case_id] = freq.get(cc.case_id, 0) + 1
        return freq

    def critical_category_distribution(self) -> Dict[str, int]:
        """Count critical cases by category.

        Returns
        -------
        dict of {category: count}
        """
        dist: Dict[str, int] = {}
        for cc in self._critical_cases:
            dist[cc.category] = dist.get(cc.category, 0) + 1
        return dist

    # ---------------------------------------------------------------
    # Potato plot data
    # ---------------------------------------------------------------

    def compute_potato(self, component: str, station: float,
                         x_quantity: str = "V", y_quantity: str = "M",
                         ) -> PotatoData:
        """Compute potato plot data at a specific span station.

        Parameters
        ----------
        component : str
            Component name.
        station : float
            Target span station (nearest station used).
        x_quantity : str
            X-axis quantity: "V", "M", or "T".
        y_quantity : str
            Y-axis quantity: "V", "M", or "T".

        Returns
        -------
        PotatoData
        """
        qty_map = {"V": "shear", "M": "bending", "T": "torsion"}
        labels = {"V": "Shear V (N)", "M": "Bending M (N-mm)",
                   "T": "Torsion T (N-mm)"}

        potato = PotatoData(
            station=station,
            component=component,
            x_label=labels.get(x_quantity, x_quantity),
            y_label=labels.get(y_quantity, y_quantity),
        )

        for case_id, comp_data in self.vmt_data.items():
            if component not in comp_data:
                continue

            vmt = comp_data[component]
            stations = vmt["stations"]

            # Find nearest station index
            idx = int(np.argmin(np.abs(np.array(stations) - station)))

            x_arr = vmt[qty_map[x_quantity]]
            y_arr = vmt[qty_map[y_quantity]]

            potato.x_values.append(float(x_arr[idx]))
            potato.y_values.append(float(y_arr[idx]))
            potato.case_ids.append(case_id)

            # Look up category
            cr = self.batch_result.get_result(case_id)
            potato.categories.append(cr.category if cr else "")

        # Compute convex hull if enough points
        if len(potato.x_values) >= 3:
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack([potato.x_values, potato.y_values])
                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                # Close the hull
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                potato.hull_x = hull_pts[:, 0]
                potato.hull_y = hull_pts[:, 1]
            except (ImportError, Exception):
                pass  # scipy not available or degenerate hull

        return potato

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Summary of envelope processing results.

        Returns
        -------
        dict with keys: components, n_critical, category_dist
        """
        return {
            "components": list(self._component_envelopes.keys()),
            "n_critical": len(self._critical_cases),
            "category_distribution": self.critical_category_distribution(),
            "case_frequency_top10": sorted(
                self.critical_case_frequency().items(),
                key=lambda x: x[1], reverse=True)[:10],
        }
