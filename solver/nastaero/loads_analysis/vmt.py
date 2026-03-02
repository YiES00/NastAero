"""VMT (Shear/Bending Moment/Torsion) internal loads integration.

Computes sectional shear force (V), bending moment (M), and torsion (T)
distributions along the span of structural components by integrating
nodal forces from tip to root.

Usage:
    from nastaero.loads_analysis.component_id import identify_components
    from nastaero.loads_analysis.vmt import compute_vmt_all

    components = identify_components(model)
    result = compute_vmt_all(model, sc.nodal_combined_forces, components)
    for curve in result.curves:
        print(f"{curve.component_name}: V_root={curve.shear[0]:.0f} N")
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .component_id import ComponentDef, ComponentSet


_AXIS_LABELS = {0: 'X (mm)', 1: 'Y (mm)', 2: 'Z (mm)'}


@dataclass
class VMTCurve:
    """VMT results for a single component and load condition."""
    component_name: str
    stations: np.ndarray            # (n_stations,) span positions
    shear: np.ndarray               # (n_stations,) V - shear force [N]
    bending_moment: np.ndarray      # (n_stations,) M - bending moment [N-mm]
    torsion: np.ndarray             # (n_stations,) T - torsion [N-mm]
    span_axis: int = 1
    station_label: str = 'Y (mm)'
    load_type: str = 'combined'
    subcase_id: int = 0


@dataclass
class VMTResult:
    """Collection of VMT curves."""
    curves: List[VMTCurve] = field(default_factory=list)

    def get_curves(
        self,
        component_name: str = None,
        load_type: str = None,
        subcase_id: int = None,
    ) -> List[VMTCurve]:
        """Filter curves by component, load type, and/or subcase."""
        result = self.curves
        if component_name is not None:
            name_lower = component_name.lower()
            result = [c for c in result if name_lower in c.component_name.lower()]
        if load_type is not None:
            result = [c for c in result if c.load_type == load_type]
        if subcase_id is not None:
            result = [c for c in result if c.subcase_id == subcase_id]
        return result

    @property
    def component_names(self) -> List[str]:
        return sorted(set(c.component_name for c in self.curves))


def compute_vmt(
    model: Any,
    nodal_forces: Dict[int, np.ndarray],
    component: ComponentDef,
    n_stations: int = 50,
    elastic_axis_frac: float = 0.40,
    load_type: str = 'combined',
    subcase_id: int = 0,
) -> VMTCurve:
    """Compute VMT at span stations for one component.

    Parameters
    ----------
    model : BDFModel or VizModel
    nodal_forces : Dict[int, ndarray(6)]
        Nodal forces [Fx, Fy, Fz, Mx, My, Mz] in BASIC frame.
    component : ComponentDef
    n_stations : int
        Number of evenly-spaced span stations.
    elastic_axis_frac : float
        Chord fraction for elastic axis (torsion reference), 0.0=LE, 1.0=TE.
    load_type : str
    subcase_id : int

    Returns
    -------
    VMTCurve
    """
    span_ax = component.span_axis
    shear_ax = component.shear_axis
    bend_ax = component.bending_axis
    torsion_ax = component.torsion_axis
    sign = component.integration_sign

    # Collect node data for this component
    valid_nids = []
    for nid in component.node_ids:
        if nid in model.nodes and nid in nodal_forces:
            valid_nids.append(nid)

    if not valid_nids:
        return _empty_curve(component, load_type, subcase_id)

    k = len(valid_nids)
    all_xyz = np.array([model.nodes[nid].xyz_global for nid in valid_nids],
                        dtype=np.float64)   # (K, 3)
    all_f6 = np.array([nodal_forces[nid] for nid in valid_nids],
                       dtype=np.float64)    # (K, 6)
    all_span = all_xyz[:, span_ax]          # (K,)

    span_min, span_max = all_span.min(), all_span.max()
    if span_max - span_min < 1e-6:
        return _empty_curve(component, load_type, subcase_id)

    # Adjust n_stations if too few nodes
    n_stations = min(n_stations, max(k // 2, 10))

    # Create stations from root to tip
    if sign > 0:
        # outboard = +direction → stations from min (root) to max (tip)
        stations = np.linspace(span_min, span_max, n_stations)
    else:
        # outboard = -direction → stations from max (root) to min (tip)
        stations = np.linspace(span_max, span_min, n_stations)

    # Precompute elastic axis X-position at each station (for torsion reference)
    # Use chord-fraction method: at each span station, find x_min/x_max,
    # place reference at x_min + fraction * (x_max - x_min)
    # For non-X chord axis, determine the "chordwise" axis
    if span_ax == 1:    # wing/HTP: span=Y, chord=X
        chord_ax = 0
    elif span_ax == 2:  # VTP: span=Z, chord=X
        chord_ax = 0
    else:               # fuselage: span=X, chord=Z (vertical)
        chord_ax = 2

    # Compute reference X at each station by binning nearby nodes
    ref_chord = _compute_elastic_axis(all_xyz, all_span, stations,
                                       chord_ax, elastic_axis_frac)

    # Integration: at each station, sum forces OUTBOARD of the cut
    V = np.zeros(n_stations)
    M = np.zeros(n_stations)
    T = np.zeros(n_stations)

    for i, s_cut in enumerate(stations):
        # Select outboard nodes
        if sign > 0:
            mask = all_span >= s_cut
        else:
            mask = all_span <= s_cut

        if not np.any(mask):
            continue

        F_out = all_f6[mask, :3]        # (m, 3) forces
        M_out = all_f6[mask, 3:6]       # (m, 3) moments
        xyz_out = all_xyz[mask]         # (m, 3) positions

        # Cut point: at current station, on the elastic axis
        cut_point = np.zeros(3)
        cut_point[span_ax] = s_cut
        cut_point[chord_ax] = ref_chord[i]
        # The third axis: use mean of outboard nodes
        third_ax = 3 - span_ax - chord_ax
        cut_point[third_ax] = xyz_out[:, third_ax].mean()

        # Moment arm from cut point to each node
        r = xyz_out - cut_point         # (m, 3)

        # Sum forces
        sum_F = F_out.sum(axis=0)       # (3,)

        # Sum moments: cross(r, F) + direct nodal moments
        sum_M = np.cross(r, F_out).sum(axis=0) + M_out.sum(axis=0)

        V[i] = sum_F[shear_ax]
        M[i] = sum_M[bend_ax]
        T[i] = sum_M[torsion_ax]

    return VMTCurve(
        component_name=component.name,
        stations=stations,
        shear=V,
        bending_moment=M,
        torsion=T,
        span_axis=span_ax,
        station_label=_AXIS_LABELS.get(span_ax, 'Station'),
        load_type=load_type,
        subcase_id=subcase_id,
    )


def compute_vmt_all(
    model: Any,
    nodal_forces: Dict[int, np.ndarray],
    components: ComponentSet,
    n_stations: int = 50,
    load_type: str = 'combined',
    subcase_id: int = 0,
) -> VMTResult:
    """Compute VMT for all components.

    Parameters
    ----------
    model : BDFModel or VizModel
    nodal_forces : Dict[int, ndarray(6)]
    components : ComponentSet
    n_stations : int
    load_type : str
    subcase_id : int

    Returns
    -------
    VMTResult
    """
    result = VMTResult()
    for comp in components.components:
        curve = compute_vmt(model, nodal_forces, comp,
                            n_stations=n_stations,
                            load_type=load_type,
                            subcase_id=subcase_id)
        result.curves.append(curve)
    return result


def _compute_elastic_axis(
    all_xyz: np.ndarray,
    all_span: np.ndarray,
    stations: np.ndarray,
    chord_ax: int,
    frac: float,
) -> np.ndarray:
    """Estimate elastic axis chord position at each station.

    Uses a sliding-window approach: for each station, find nearby nodes
    and compute x_le + frac * chord from their chord-axis extent.
    """
    n_stations = len(stations)
    ref = np.zeros(n_stations)
    span_range = all_span.max() - all_span.min()
    half_bin = max(span_range / (n_stations * 0.8), 1.0)

    chord_vals = all_xyz[:, chord_ax]

    for i, s in enumerate(stations):
        nearby = np.abs(all_span - s) < half_bin
        if np.any(nearby):
            c_min = chord_vals[nearby].min()
            c_max = chord_vals[nearby].max()
            ref[i] = c_min + frac * (c_max - c_min)
        else:
            # Fallback: use global mean
            ref[i] = chord_vals.mean()

    return ref


def _empty_curve(component: ComponentDef, load_type: str, subcase_id: int) -> VMTCurve:
    """Return an empty VMTCurve for a component with no valid data."""
    return VMTCurve(
        component_name=component.name,
        stations=np.array([0.0]),
        shear=np.array([0.0]),
        bending_moment=np.array([0.0]),
        torsion=np.array([0.0]),
        span_axis=component.span_axis,
        station_label=_AXIS_LABELS.get(component.span_axis, 'Station'),
        load_type=load_type,
        subcase_id=subcase_id,
    )
