"""VMT envelope processing and critical design-load selection.

Processes VMT results from all certification load cases to:
- Compute min/max envelopes per component and station
- Identify critical cases from axis extremes AND from the convex hulls of
  the V-M / V-T / M-T interaction (potato) plots at every load station
- Reduce hundreds/thousands of cases to a compact set of design load
  conditions (the deliverable for detailed stress analysis)
- Generate potato plot data and summary statistics for reporting

The EnvelopeProcessor is the post-processing engine that connects:
  BatchResult → VMT curves → envelopes → interaction hulls → critical cases

``select_critical_design_loads`` is the end-to-end driver: it runs the whole
chain and, given an output directory, exports one Nastran FORCE/MOMENT BDF
per design case (plus a master INCLUDE deck) for the full-vehicle mesh.

References
----------
- Phase 5 of the FAA Part 23 certification framework
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .batch_runner import BatchResult, CaseResult

logger = logging.getLogger(__name__)

# 구성품 국부 6분력의 물리량 이름 (r4 MC3). 전역 3성분 "V"/"M"/"T"와
# 구분되는 별도 이름이라 임계 케이스 기록에서 서로 섞이지 않는다.
LOCAL_QUANTITIES = ("N", "Vy", "Vz", "Mx", "My", "Mz")


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
        Load quantity. Global-axis: "V" (shear), "M" (bending),
        "T" (torsion). Component-local 6-component (r4 MC3):
        "N" (axial), "Vy"/"Vz" (chord/normal shear), "Mx" (torsion
        about the member axis), "My"/"Mz" (bending).
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
# Design load condition (compact, post-reduction)
# ---------------------------------------------------------------------------

@dataclass
class DesignLoadCase:
    """One selected design load condition after envelope reduction.

    A single load case that governs one or more component/station/quantity
    extremes. The compact set of these is the deliverable handed to detailed
    stress analysis.

    Attributes
    ----------
    case_id : int
        Load case ID.
    category, far_section, nz, label : ...
        Case metadata (copied from the controlling case).
    n_govern : int
        Number of (component, station, quantity) extremes this case governs.
    governs : list of (component, station, quantity, extreme)
        Every extremum this case defines.
    components : list of str
        Distinct components this case drives.
    """
    case_id: int = 0
    category: str = ""
    far_section: str = ""
    nz: float = 0.0
    label: str = ""
    n_govern: int = 0
    governs: List[Tuple[str, float, str, str]] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    # r3 MC2: physical realizability of the rotor thrust command. False
    # means the BEMT saturated at the collective limit and the achieved
    # thrust fell short of the command -- the case is a propulsion-limit
    # state, not a realized flight condition.
    rotor_command_feasible: bool = True
    rotor_thrust_shortfall: float = 0.0

    @property
    def by_envelope(self) -> List[Tuple[str, float, str, str]]:
        """Extrema governed by the axis-aligned V/M/T envelope."""
        return [g for g in self.governs if g[3] in ("max", "min")]

    @property
    def by_interaction(self) -> List[Tuple[str, float, str, str]]:
        """Extrema governed by an interaction (2-D or 3-D) hull."""
        return [g for g in self.governs if g[3] in ("hull", "hull3d")]

    @property
    def basis(self) -> str:
        """Selection basis: ``envelope``, ``interaction``, ``both``, or
        ``propulsion-limit`` (saturated command exceeding the feasible
        envelope, r3 MC2)."""
        e, h = bool(self.by_envelope), bool(self.by_interaction)
        if e and h:
            return "both"
        if e:
            return "envelope"
        if h:
            return "interaction"
        if any(g[3] == "propulsion-limit" for g in self.governs):
            return "propulsion-limit"
        return ""

    def why(self) -> str:
        """One-line human-readable reason this case is a design load.

        Groups the governed extrema by component and, within each, lists the
        axis-envelope extremes (``V+`` for max shear, ``M-`` for min bending,
        ...) and the interaction planes (``V-M``, ...) it sits on the hull of.
        """
        from collections import defaultdict
        per: Dict[str, Dict[str, set]] = defaultdict(
            lambda: {"env": set(), "hull": set(), "plim": set()})
        for comp, _sta, qty, ext in self.governs:
            if ext in ("max", "min"):
                per[comp]["env"].add(f"{qty}{'+' if ext == 'max' else '-'}")
            elif ext in ("hull", "hull3d"):
                per[comp]["hull"].add(qty)
            elif ext == "propulsion-limit":
                per[comp]["plim"].add(qty)
        parts = []
        for comp in sorted(per):
            segs = []
            if per[comp]["env"]:
                segs.append("env " + ",".join(sorted(per[comp]["env"])))
            if per[comp]["hull"]:
                segs.append("potato " + ",".join(sorted(per[comp]["hull"])))
            if per[comp]["plim"]:
                segs.append("propulsion-limit "
                            + ",".join(sorted(per[comp]["plim"])))
            parts.append(f"{comp}: {'; '.join(segs)}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Convex-hull helper for interaction-diagram critical selection
# ---------------------------------------------------------------------------

def _hull_vertex_indices(points: np.ndarray) -> List[int]:
    """Indices of the convex-hull vertices of a 2-D point set.

    Robust to degeneracy: the four axis extremes are always included (they
    lie on the hull and provide a fallback when SciPy is unavailable or the
    points are collinear/coincident). Axes are scaled to a common range
    before hulling so that quantities of very different magnitude (shear in
    N, moment in N-mm) yield a physically meaningful hull; axis-independent
    scaling is affine and does not change which points are vertices.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= 2:
        return list(range(n))

    idx: set = set()
    for col in range(pts.shape[1]):
        idx.add(int(np.argmax(pts[:, col])))
        idx.add(int(np.argmin(pts[:, col])))

    span = np.ptp(pts, axis=0)
    span[span == 0] = 1.0
    scaled = (pts - pts.min(axis=0)) / span
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(scaled)
        idx.update(int(v) for v in hull.vertices)
    except Exception:
        pass  # collinear / coincident / SciPy missing -> axis extremes only
    return sorted(idx)


def _iter_components(vmt_data: Dict[int, Dict[str, Any]]):
    """Yield (component_name, None) for each distinct component in vmt_data."""
    seen: set = set()
    for comp_data in vmt_data.values():
        for name in comp_data:
            if name not in seen:
                seen.add(name)
                yield name, None


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
    # 구성품 국부 6분력 극값 (r4 MC3). local[qty] = [min, max],
    # local_case[qty] = [min_case_id, max_case_id]. 국부 배열이 없는
    # 구성품에서는 비어 있으므로 전역 3성분 경로와 독립이다.
    local: Dict[str, List[float]] = field(default_factory=dict)
    local_case: Dict[str, List[int]] = field(default_factory=dict)

    def update_local(self, qty: str, value: float, case_id: int) -> None:
        cur = self.local.get(qty)
        if cur is None:
            self.local[qty] = [value, value]
            self.local_case[qty] = [case_id, case_id]
            return
        if value < cur[0]:
            cur[0] = value
            self.local_case[qty][0] = case_id
        if value > cur[1]:
            cur[1] = value
            self.local_case[qty][1] = case_id


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

                # 국부 6분력 극값 (r4 MC3). 국부 스테이션은 부재축
                # 투영이라 전역 스테이션과 개수는 같지만 좌표가 다르므로
                # 인덱스로 대응시킨다(두 축 모두 같은 n_stations 등간격).
                if vmt.get("local_stations") is not None:
                    for qty in LOCAL_QUANTITIES:
                        arr = vmt.get(qty)
                        if arr is None:
                            continue
                        for i in range(min(len(arr), len(env.envelopes))):
                            env.envelopes[i].update_local(
                                qty, float(arr[i]), case_id)

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

    def identify_local_critical_cases(self) -> List[CriticalCase]:
        """구성품 국부 6분력의 축별 극값을 임계 케이스로 추가한다 (r4 MC3).

        전역 3성분 선정은 경사·후퇴 구성품에서 축력·시위 전단·면내
        굽힘을 볼 수 없다. 국부 물리량은 "N"/"Vy"/"Vz"/"Mx"/"My"/"Mz"
        라는 별도 이름으로 기록되므로 기존 V/M/T 기록과 섞이지 않고,
        설계 세트 선정에서 함께 순위에 반영된다.
        """
        added: List[CriticalCase] = []
        for comp_name, env in self._component_envelopes.items():
            for se in env.envelopes:
                for qty, pair in se.local.items():
                    ids = se.local_case.get(qty, [0, 0])
                    for k, extreme in ((0, "min"), (1, "max")):
                        value, cid = pair[k], ids[k]
                        if abs(value) == math.inf:
                            continue
                        cr = self.batch_result.get_result(cid)
                        cc = CriticalCase(
                            station=se.station,
                            component=comp_name,
                            quantity=qty,
                            extreme=extreme,
                            value=value,
                            case_id=cid,
                            category=cr.category if cr else "",
                            far_section=cr.far_section if cr else "",
                            nz=cr.nz if cr else 0.0,
                            label=cr.label if cr else "",
                        )
                        self._critical_cases.append(cc)
                        added.append(cc)
        logger.info("Local 6-component critical records: %d", len(added))
        return added

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

    # ---------------------------------------------------------------
    # Interaction-diagram (potato-plot) critical case selection
    # ---------------------------------------------------------------

    def add_interaction_critical_cases(
        self,
        planes: Tuple[Tuple[str, str], ...] = (("V", "M"), ("V", "T"), ("M", "T")),
    ) -> List[CriticalCase]:
        """Select critical cases from the interaction (potato-plot) hulls.

        Axis-aligned min/max (see :meth:`identify_critical_cases`) catches
        a case only when it maximises a single quantity. The design-driving
        case under *combined* loading is generally an outer point of the
        two-dimensional V-M / V-T / M-T scatter that is extreme in neither
        axis alone. This method walks every station of every component and,
        for each interaction plane, marks the load cases sitting on the
        convex hull (the outermost points of the potato plot) as critical.

        The resulting :class:`CriticalCase` records use ``quantity`` set to
        the plane label (e.g. ``"V-M"``) and ``extreme = "hull"``. They are
        appended to the internal critical-case list, so they flow through
        :meth:`get_critical_cases`, the FORCE-card export, and
        :meth:`select_design_cases`.

        Parameters
        ----------
        planes : tuple of (str, str)
            Interaction planes to hull, as (x_quantity, y_quantity) pairs.

        Returns
        -------
        list of CriticalCase
            The interaction-hull critical cases added by this call.
        """
        qty_map = {"V": "shear", "M": "bending", "T": "torsion"}
        added: List[CriticalCase] = []

        for comp_name, comp_data in _iter_components(self.vmt_data):
            # case_ids and per-case arrays available for this component
            case_ids = [cid for cid, cd in self.vmt_data.items()
                        if comp_name in cd]
            if len(case_ids) < 2:
                continue
            stations = self.vmt_data[case_ids[0]][comp_name]["stations"]

            for i, sta in enumerate(stations):
                for xq, yq in planes:
                    xs, ys, cids = [], [], []
                    for cid in case_ids:
                        vmt = self.vmt_data[cid][comp_name]
                        xs.append(float(vmt[qty_map[xq]][i]))
                        ys.append(float(vmt[qty_map[yq]][i]))
                        cids.append(cid)

                    pts = np.column_stack([xs, ys])
                    vert = _hull_vertex_indices(pts)
                    centroid = pts.mean(axis=0)
                    span = np.ptp(pts, axis=0)
                    span[span == 0] = 1.0

                    for vi in vert:
                        cid = cids[vi]
                        cr = self.batch_result.get_result(cid)
                        # radial extremity in normalised plane (informational)
                        d = float(np.hypot(*((pts[vi] - centroid) / span)))
                        cc = CriticalCase(
                            station=float(sta),
                            component=comp_name,
                            quantity=f"{xq}-{yq}",
                            extreme="hull",
                            value=d,
                            case_id=cid,
                            category=cr.category if cr else "",
                            far_section=cr.far_section if cr else "",
                            nz=cr.nz if cr else 0.0,
                            label=cr.label if cr else "",
                        )
                        self._critical_cases.append(cc)
                        added.append(cc)

        return added

    def add_interaction_critical_cases_3d(self) -> List[CriticalCase]:
        """Select critical cases from the 3-D (V, M, T) convex hull.

        The three 2-D potato hulls cover failure modes that combine two
        of the three section quantities; a case can still be extreme
        only in a mixed V+M+T direction while sitting inside all three
        projections. Hulling the full (V, M, T) point cloud at every
        station catches those triple-combination criticals -- complete
        coverage for any linear failure function of the three
        quantities. Every 2-D hull vertex is also a 3-D hull vertex
        (projection preserves convex combinations), so this is a
        superset of the plane selection; the price is a larger vertex
        count. Records carry ``quantity="V-M-T"``, ``extreme="hull3d"``.
        """
        added: List[CriticalCase] = []
        for comp_name, comp_data in _iter_components(self.vmt_data):
            case_ids = [cid for cid, cd in self.vmt_data.items()
                        if comp_name in cd]
            if len(case_ids) < 3:
                continue
            stations = self.vmt_data[case_ids[0]][comp_name]["stations"]

            for i, sta in enumerate(stations):
                pts = np.array([
                    [float(self.vmt_data[cid][comp_name]["shear"][i]),
                     float(self.vmt_data[cid][comp_name]["bending"][i]),
                     float(self.vmt_data[cid][comp_name]["torsion"][i])]
                    for cid in case_ids])
                vert = _hull_vertex_indices(pts)
                centroid = pts.mean(axis=0)
                span = np.ptp(pts, axis=0)
                span[span == 0] = 1.0

                for vi in vert:
                    cid = case_ids[vi]
                    cr = self.batch_result.get_result(cid)
                    d = float(np.linalg.norm((pts[vi] - centroid) / span))
                    cc = CriticalCase(
                        station=float(sta),
                        component=comp_name,
                        quantity="V-M-T",
                        extreme="hull3d",
                        value=d,
                        case_id=cid,
                        category=cr.category if cr else "",
                        far_section=cr.far_section if cr else "",
                        nz=cr.nz if cr else 0.0,
                        label=cr.label if cr else "",
                    )
                    self._critical_cases.append(cc)
                    added.append(cc)
        return added

    # ---------------------------------------------------------------
    # Compact design-load set (hundreds/thousands -> tens)
    # ---------------------------------------------------------------

    def select_design_cases(self) -> List["DesignLoadCase"]:
        """Reduce critical cases to a compact set of design load conditions.

        The critical-case list (axis extremes plus interaction hulls, over
        every component and station) names the same handful of cases many
        times over. This collapses that list to one :class:`DesignLoadCase`
        per distinct load case, recording every (component, station,
        quantity) extremum it governs, and ranks them by how much of the
        structure each one drives (most-governing first).

        This is the end of the loads analysis: a batch of hundreds or
        thousands of load cases reduces to the few tens of design load
        conditions a stress analyst must actually run against the detailed
        model.

        Returns
        -------
        list of DesignLoadCase
            Sorted by number of governed extremes, descending.
        """
        by_case: Dict[int, DesignLoadCase] = {}
        for cc in self._critical_cases:
            if abs(cc.value) == math.inf:
                continue
            dc = by_case.get(cc.case_id)
            if dc is None:
                dc = DesignLoadCase(
                    case_id=cc.case_id, category=cc.category,
                    far_section=cc.far_section, nz=cc.nz, label=cc.label,
                )
                by_case[cc.case_id] = dc
            dc.governs.append((cc.component, cc.station, cc.quantity, cc.extreme))
            if cc.component not in dc.components:
                dc.components.append(cc.component)

        for dc in by_case.values():
            dc.n_govern = len(dc.governs)

        return sorted(by_case.values(),
                      key=lambda d: (d.n_govern, len(d.components)),
                      reverse=True)


# ---------------------------------------------------------------------------
# Design-load summary table ("what was selected and why")
# ---------------------------------------------------------------------------

def design_load_table(design_cases: List["DesignLoadCase"],
                       top: Optional[int] = None) -> List[Dict[str, Any]]:
    """Tabulate the selected design loads with their selection reason.

    Each row states which case it is, how much structure it governs, the
    components it drives, whether it was selected by the axis-aligned V/M/T
    envelope, by an interaction (potato-plot) hull, or both, and a one-line
    reason string.

    Parameters
    ----------
    design_cases : list of DesignLoadCase
        Output of :meth:`EnvelopeProcessor.select_design_cases` (already
        ranked most-governing first).
    top : int, optional
        Keep only the first ``top`` rows.

    Returns
    -------
    list of dict
        Rows with keys: rank, case_id, label, category, far_section, nz,
        n_govern, components, basis, envelope_quantities,
        interaction_planes, why.
    """
    rows: List[Dict[str, Any]] = []
    for rank, d in enumerate(design_cases, 1):
        if top is not None and rank > top:
            break
        rows.append({
            "rank": rank,
            "case_id": d.case_id,
            "label": d.label,
            "category": d.category,
            "far_section": d.far_section,
            "nz": d.nz,
            "n_govern": d.n_govern,
            "components": sorted(set(d.components)),
            "basis": d.basis,
            "envelope_quantities": sorted({g[2] for g in d.by_envelope}),
            "interaction_planes": sorted({g[2] for g in d.by_interaction}),
            "rotor_command_feasible": d.rotor_command_feasible,
            "rotor_thrust_shortfall": d.rotor_thrust_shortfall,
            "why": d.why(),
        })
    return rows


def write_design_load_summary_csv(design_cases: List["DesignLoadCase"],
                                  csv_path: str,
                                  top: Optional[int] = None) -> str:
    """Write the design-load summary table (see :func:`design_load_table`).

    This is the final loads-analysis deliverable list: every selected design
    load condition with the reason it was selected.
    """
    import csv as _csv
    rows = design_load_table(design_cases, top=top)
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rank", "case_id", "label", "category", "far_section",
                    "nz", "n_govern", "components", "basis",
                    "envelope_quantities", "interaction_planes",
                    "rotor_command_feasible", "rotor_thrust_shortfall",
                    "why"])
        for r in rows:
            w.writerow([
                r["rank"], r["case_id"], r["label"], r["category"],
                r["far_section"], f"{r['nz']:+.2f}", r["n_govern"],
                "; ".join(r["components"]), r["basis"],
                ",".join(r["envelope_quantities"]),
                ",".join(r["interaction_planes"]),
                "Y" if r["rotor_command_feasible"] else "N",
                f"{r['rotor_thrust_shortfall']:.3f}",
                r["why"],
            ])
    logger.info("Design-load summary written: %s (%d conditions)",
                csv_path, len(rows))
    return csv_path


# ---------------------------------------------------------------------------
# Propulsion-limit (saturated rotor command) screening -- r3 MC2
# ---------------------------------------------------------------------------

def _screen_infeasible_against_envelope(
    proc: "EnvelopeProcessor",
    vmt_data: Dict[int, Dict[str, Any]],
    infeasible_ids: List[int],
) -> List[Dict[str, Any]]:
    """Compare saturated-command cases against the feasible envelope.

    Returns one record per (case, component, station, quantity) where the
    saturated case falls OUTSIDE the feasible-case envelope band. These
    are the loads the feasible-only selection would silently miss.
    """
    records: List[Dict[str, Any]] = []
    for cid in infeasible_ids:
        for comp_name, vmt in vmt_data.get(cid, {}).items():
            env = proc.get_envelope(comp_name)
            if env is None:
                continue
            stations = vmt["stations"]
            for qty, arr in (("V", vmt["shear"]), ("M", vmt["bending"]),
                             ("T", vmt["torsion"])):
                for i, sta in enumerate(stations):
                    if i >= len(env.envelopes):
                        break
                    se = env.envelopes[i]
                    lo = getattr(se, f"{qty}_min")
                    hi = getattr(se, f"{qty}_max")
                    v = float(arr[i])
                    if v > hi or v < lo:
                        bound = hi if v > hi else lo
                        records.append({
                            "case_id": cid, "component": comp_name,
                            "station": float(sta), "quantity": qty,
                            "value": v, "feasible_bound": float(bound),
                            "extreme": "max" if v > hi else "min",
                        })
    return records


def _append_propulsion_limit_cases(
    batch_result: BatchResult,
    design_cases: List["DesignLoadCase"],
    exceedances: List[Dict[str, Any]],
    shortfall: Dict[int, float],
) -> int:
    """Append flagged design cases for envelope-exceeding saturated cases."""
    by_case: Dict[int, List[Dict[str, Any]]] = {}
    for e in exceedances:
        by_case.setdefault(e["case_id"], []).append(e)
    existing = {dc.case_id for dc in design_cases}
    appended = 0
    for cid, recs in sorted(by_case.items()):
        if cid in existing:
            continue
        cr = batch_result.get_result(cid)
        dc = DesignLoadCase(
            case_id=cid,
            category=getattr(cr, "category", "") if cr else "",
            far_section=getattr(cr, "far_section", "") if cr else "",
            nz=getattr(cr, "nz", 0.0) if cr else 0.0,
            label=getattr(cr, "label", "") if cr else "",
            rotor_command_feasible=False,
            rotor_thrust_shortfall=shortfall.get(cid, 0.0),
        )
        for e in recs:
            dc.governs.append((e["component"], e["station"], e["quantity"],
                               "propulsion-limit"))
            if e["component"] not in dc.components:
                dc.components.append(e["component"])
        dc.n_govern = len(dc.governs)
        design_cases.append(dc)
        appended += 1
    return appended


# ---------------------------------------------------------------------------
# End-to-end driver: batch results -> design load conditions -> FORCE cards
# ---------------------------------------------------------------------------

def select_critical_design_loads(
    model: Any,
    batch_result: BatchResult,
    output_dir: Optional[str] = None,
    n_stations: int = 50,
    fuselage_cg_x: Optional[float] = None,
    planes: Tuple[Tuple[str, str], ...] = (("V", "M"), ("V", "T"), ("M", "T")),
    include_axis: bool = True,
    include_3d: bool = True,
    infeasible_policy: str = "separate",
    components: Any = None,
    vmt_data: Optional[Dict[int, Dict[str, Any]]] = None,
    include_local6: bool = True,
) -> Dict[str, Any]:
    """Run the full critical-design-load selection, the last loads step.

    Given a completed load-case batch (hundreds to thousands of cases with
    recovered nodal forces), this:

    1. integrates shear-moment-torsion (V-M-T) along every structural
       component -- wing, HTP, VTP, and fuselage -- at ``n_stations`` load
       stations each;
    2. builds the V/M/T envelopes and marks the axis-aligned min/max cases
       (optional, ``include_axis``);
    3. draws the V-M, V-T, and M-T interaction (potato) plots at every
       station and marks the load cases on their convex hulls -- the design
       cases under combined loading;
    4. reduces the union of those critical cases to a compact set of design
       load conditions (typically a few tens), ranked by how much structure
       each governs; and
    5. writes, for every selected case, a Nastran-format FORCE/MOMENT BDF
       (plus a master INCLUDE deck and a summary CSV) ready to apply to the
       full-vehicle FE mesh for detailed stress analysis.

    Parameters
    ----------
    model : BDFModel
        The structural model (node positions, component identification).
    batch_result : BatchResult
        Completed batch with per-case ``nodal_forces``.
    output_dir : str, optional
        If given, FORCE/MOMENT BDF cards are exported here.
    n_stations : int
        Load stations per component for V-M-T integration.
    fuselage_cg_x : float, optional
        Aircraft CG X (mm) for fuselage V-M-T integration about the CG.
    planes : tuple of (str, str)
        Interaction planes to hull. Default: V-M, V-T, M-T.
    include_axis : bool
        Also mark axis-aligned envelope extremes as critical.
    include_3d : bool
        Also hull the full (V, M, T) point cloud at every station
        (default). The planar hulls only cover failure functions of two
        quantities at a time; a case interior to all three projections can
        still be extreme along a mixed V+M+T direction. On distributed-
        propulsion re-trim matrices that residual reaches several percent
        of the local load range, so the three-dimensional pass is the
        default and should only be turned off to reproduce planar-selection
        results. Cost is negligible (hulls over all stations take seconds);
        the price is a slightly larger design set.
    include_local6 : bool
        Also mark the axis extremes of the component-local six-component
        section loads (N, Vy, Vz, Mx, My, Mz) as critical (default,
        r4 MC3). The global-axis V/M/T records are unchanged; setting
        this False reproduces the global-only selection of earlier
        releases, which is how the two design sets are compared.
    infeasible_policy : str
        How cases whose rotor thrust command was NOT achieved (BEMT
        collective saturation, ``CaseResult.rotor_command_feasible``
        False) enter the selection (r3 MC2). ``'separate'`` (default):
        the envelope and design set are built from feasible cases only;
        saturated cases are then screened against that envelope and any
        that exceed it anywhere are appended as flagged propulsion-limit
        design cases, with the full accounting returned under
        ``result['propulsion_limit']``. ``'include'``: legacy behavior,
        everything pooled (reproduces archived selections). ``'exclude'``:
        saturated cases dropped entirely.

    Returns
    -------
    dict
        Keys: ``n_cases_in`` (batch size), ``n_critical`` (critical records),
        ``n_design_cases`` (compact set size), ``compression`` (ratio),
        ``design_cases`` (list of DesignLoadCase), ``components``, and, when
        ``output_dir`` is given, ``export`` (the FORCE-card export summary).
    """
    from .vmt_bridge import compute_vmt_for_batch

    if infeasible_policy not in ("separate", "include", "exclude"):
        raise ValueError(
            f"infeasible_policy must be 'separate', 'include' or "
            f"'exclude', got {infeasible_policy!r}")

    if vmt_data is None:
        vmt_data = compute_vmt_for_batch(
            model, batch_result, components=components,
            n_stations=n_stations, fuselage_cg_x=fuselage_cg_x,
        )

    feas_flag = {
        cr.case_id: bool(getattr(cr, "rotor_command_feasible", True))
        for cr in batch_result.case_results
    }
    shortfall = {
        cr.case_id: float(getattr(cr, "rotor_thrust_shortfall", 0.0))
        for cr in batch_result.case_results
    }
    infeasible_ids = sorted(
        cid for cid in vmt_data if not feas_flag.get(cid, True))

    if infeasible_policy == "include":
        pooled = vmt_data
    else:
        pooled = {cid: d for cid, d in vmt_data.items()
                  if feas_flag.get(cid, True)}

    proc = EnvelopeProcessor(batch_result, pooled)
    proc.compute_envelopes()
    if include_axis:
        proc.identify_critical_cases()
    if planes:
        proc.add_interaction_critical_cases(planes=planes)
    if include_3d:
        proc.add_interaction_critical_cases_3d()
    if include_local6:
        proc.identify_local_critical_cases()
    design_cases = proc.select_design_cases()

    # Stamp physical realizability on every selected case
    for dc in design_cases:
        dc.rotor_command_feasible = feas_flag.get(dc.case_id, True)
        dc.rotor_thrust_shortfall = shortfall.get(dc.case_id, 0.0)

    propulsion_limit: Dict[str, Any] = {
        "policy": infeasible_policy,
        "n_infeasible": len(infeasible_ids),
        "infeasible_case_ids": infeasible_ids,
        "exceedances": [],
        "n_appended_design_cases": 0,
    }
    if infeasible_policy == "separate" and infeasible_ids:
        exceed = _screen_infeasible_against_envelope(
            proc, vmt_data, infeasible_ids)
        propulsion_limit["exceedances"] = exceed
        appended = _append_propulsion_limit_cases(
            batch_result, design_cases, exceed, shortfall)
        propulsion_limit["n_appended_design_cases"] = appended
        if exceed:
            logger.warning(
                "Propulsion-limit screening: %d saturated case(s) exceed "
                "the feasible envelope at %d station extremes; appended "
                "%d flagged design case(s)",
                len({e['case_id'] for e in exceed}), len(exceed), appended)
        else:
            logger.info(
                "Propulsion-limit screening: %d saturated case(s), none "
                "exceed the feasible envelope", len(infeasible_ids))

    n_in = sum(1 for cr in batch_result.case_results
               if cr.converged and cr.nodal_forces)
    n_design = len(design_cases)

    result: Dict[str, Any] = {
        "n_cases_in": n_in,
        "n_critical": len(proc.get_critical_cases()),
        "n_design_cases": n_design,
        "compression": (n_in / n_design) if n_design else 0.0,
        "design_cases": design_cases,
        "design_table": design_load_table(design_cases),
        "components": list(proc._component_envelopes.keys()),
        "processor": proc,
        "propulsion_limit": propulsion_limit,
    }

    if output_dir is not None:
        from .force_export import export_critical_forces
        plim_ids = sorted({e["case_id"]
                           for e in propulsion_limit["exceedances"]})
        result["export"] = export_critical_forces(
            batch_result, proc, model, output_dir,
            extra_case_ids=plim_ids or None,
        )
        import os as _os
        result["design_summary_csv"] = write_design_load_summary_csv(
            design_cases, _os.path.join(output_dir, "design_loads_summary.csv"),
        )

    logger.info(
        "Critical design loads: %d cases -> %d design conditions "
        "(%.1f:1) over %d components",
        n_in, n_design, result["compression"], len(result["components"]),
    )
    return result
