"""VMT (Shear/Bending Moment/Torsion) internal loads integration.

Computes sectional shear force (V), bending moment (M), and torsion (T)
distributions along the span of structural components by integrating
nodal forces from tip to root.

Two recovery paths coexist:

1. GLOBAL-AXIS path (legacy, unchanged): V/M/T are single global-axis
   projections (component.shear_axis / bending_axis / torsion_axis),
   stations parameterized on the global span axis, cut point third
   coordinate = mean of ALL outboard nodes. All archived envelope and
   paper numbers come from this path; it is preserved bit-for-bit.

2. LOCAL 6-COMPONENT path (2026-09, peer-review r3 MC1): a per-component
   orthonormal frame (e1 = span direction from the dominant principal
   axis of the node cloud, oriented outboard; e3 = up-hint axis
   orthogonalized against e1; e2 = e3 x e1, right-handed e1 x e2 = e3)
   recovers the full section 6-vector
       N  = F . e1   (axial)        Mx = M . e1   (torsion about span)
       Vy = F . e2   (chord shear)  My = M . e2   (bending about e2)
       Vz = F . e3   (normal shear) Mz = M . e3   (in-plane bending)
   Stations parameterize the PROJECTION s = x . e1, the cut is the
   half-space s >= s_cut (plane normal to the member axis, which for a
   canted surface such as a 40-deg V-tail differs from a global-axis
   cut), and the cut point lies on the elastic-axis estimate of a local
   sliding window (chord: window min + frac * range on the global chord
   axis; third coordinate: window mean), projected onto the cut plane.
   For planar components aligned with global axes the two paths agree
   exactly (see tests/test_vmt_local6.py).

Usage:
    from ascent_load.loads_analysis.component_id import identify_components
    from ascent_load.loads_analysis.vmt import compute_vmt_all

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
    """VMT results for a single component and load condition.

    The first four arrays are the legacy global-axis projections used by
    the envelope machinery and all archived results. The ``local_*``
    fields carry the component-local 6-component recovery (None when not
    computed, e.g. for curves assembled by affine curve algebra rather
    than from nodal forces).
    """
    component_name: str
    stations: np.ndarray            # (n_stations,) span positions
    shear: np.ndarray               # (n_stations,) V - shear force [N]
    bending_moment: np.ndarray      # (n_stations,) M - bending moment [N-mm]
    torsion: np.ndarray             # (n_stations,) T - torsion [N-mm]
    span_axis: int = 1
    station_label: str = 'Y (mm)'
    load_type: str = 'combined'
    subcase_id: int = 0
    # --- component-local 6-component recovery (r3 MC1) ---
    local_stations: Optional[np.ndarray] = None   # (n,) s = x . e1 [mm]
    local_N: Optional[np.ndarray] = None          # (n,) axial F.e1 [N]
    local_Vy: Optional[np.ndarray] = None         # (n,) chord shear F.e2 [N]
    local_Vz: Optional[np.ndarray] = None         # (n,) normal shear F.e3 [N]
    local_Mx: Optional[np.ndarray] = None         # (n,) torsion M.e1 [N-mm]
    local_My: Optional[np.ndarray] = None         # (n,) bending M.e2 [N-mm]
    local_Mz: Optional[np.ndarray] = None         # (n,) bending M.e3 [N-mm]
    local_frame: Optional[np.ndarray] = None      # (3,3) rows = e1, e2, e3
    local_cut_points: Optional[np.ndarray] = None  # (n, 3) global coords


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

    loc = _local6_recovery(all_xyz, all_f6, component, n_stations,
                           elastic_axis_frac)
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
        local_stations=None if loc is None else loc['stations'],
        local_N=None if loc is None else loc['N'],
        local_Vy=None if loc is None else loc['Vy'],
        local_Vz=None if loc is None else loc['Vz'],
        local_Mx=None if loc is None else loc['Mx'],
        local_My=None if loc is None else loc['My'],
        local_Mz=None if loc is None else loc['Mz'],
        local_frame=None if loc is None else loc['frame'],
        local_cut_points=None if loc is None else loc['cut_points'],
    )


def compute_vmt_fuselage_cg(
    model: Any,
    nodal_forces: Dict[int, np.ndarray],
    component: ComponentDef,
    cg_x: float,
    n_stations: int = 50,
    elastic_axis_frac: float = 0.40,
    load_type: str = 'combined',
    subcase_id: int = 0,
) -> VMTCurve:
    """Compute fuselage VMT by integrating from CG forward and aft.

    Instead of integrating all forces from one end, this splits the
    fuselage at the CG position and integrates:
    - Forward of CG: sum forces/moments from nose toward CG
    - Aft of CG: sum forces/moments from tail toward CG

    This produces a VMT distribution that peaks at the CG and goes to
    zero at both ends, which is the standard fuselage loads presentation.

    Parameters
    ----------
    model : BDFModel or VizModel
    nodal_forces : Dict[int, ndarray(6)]
    component : ComponentDef
        Fuselage component definition (span_axis=0).
    cg_x : float
        Aircraft CG position along X axis (mm).
    n_stations : int
    elastic_axis_frac : float
    load_type : str
    subcase_id : int

    Returns
    -------
    VMTCurve
        Combined forward+aft VMT with stations from nose to tail.
    """
    span_ax = component.span_axis
    shear_ax = component.shear_axis
    bend_ax = component.bending_axis
    torsion_ax = component.torsion_axis

    # Collect node data
    valid_nids = []
    for nid in component.node_ids:
        if nid in model.nodes and nid in nodal_forces:
            valid_nids.append(nid)

    if not valid_nids:
        return _empty_curve(component, load_type, subcase_id)

    k = len(valid_nids)
    all_xyz = np.array([model.nodes[nid].xyz_global for nid in valid_nids],
                        dtype=np.float64)
    all_f6 = np.array([nodal_forces[nid] for nid in valid_nids],
                       dtype=np.float64)
    all_span = all_xyz[:, span_ax]

    span_min, span_max = all_span.min(), all_span.max()
    if span_max - span_min < 1e-6:
        return _empty_curve(component, load_type, subcase_id)

    n_stations = min(n_stations, max(k // 2, 10))

    # Create stations from nose to tail (ascending X)
    stations = np.linspace(span_min, span_max, n_stations)

    # Chord axis for elastic axis computation
    if span_ax == 1:
        chord_ax = 0
    elif span_ax == 2:
        chord_ax = 0
    else:
        chord_ax = 2

    ref_chord = _compute_elastic_axis(all_xyz, all_span, stations,
                                       chord_ax, elastic_axis_frac)

    V = np.zeros(n_stations)
    M = np.zeros(n_stations)
    T = np.zeros(n_stations)

    for i, s_cut in enumerate(stations):
        if s_cut <= cg_x:
            # Forward of CG: integrate from nose (nodes with x <= s_cut)
            mask = all_span <= s_cut
        else:
            # Aft of CG: integrate from tail (nodes with x >= s_cut)
            mask = all_span >= s_cut

        if not np.any(mask):
            continue

        F_out = all_f6[mask, :3]
        M_out = all_f6[mask, 3:6]
        xyz_out = all_xyz[mask]

        cut_point = np.zeros(3)
        cut_point[span_ax] = s_cut
        cut_point[chord_ax] = ref_chord[i]
        third_ax = 3 - span_ax - chord_ax
        cut_point[third_ax] = xyz_out[:, third_ax].mean()

        r = xyz_out - cut_point
        sum_F = F_out.sum(axis=0)
        sum_M = np.cross(r, F_out).sum(axis=0) + M_out.sum(axis=0)

        V[i] = sum_F[shear_ax]
        M[i] = sum_M[bend_ax]
        T[i] = sum_M[torsion_ax]

        # Aft side: flip sign so the VMT is consistent from CG perspective
        if s_cut > cg_x:
            V[i] = -V[i]
            M[i] = -M[i]
            T[i] = -T[i]

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
    fuselage_cg_x: Optional[float] = None,
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
    fuselage_cg_x : float, optional
        If provided, fuselage components use CG-based forward/aft
        integration instead of single-direction integration.

    Returns
    -------
    VMTResult
    """
    result = VMTResult()
    for comp in components.components:
        if (fuselage_cg_x is not None
                and 'fuselage' in comp.name.lower()
                and comp.span_axis == 0):
            curve = compute_vmt_fuselage_cg(
                model, nodal_forces, comp,
                cg_x=fuselage_cg_x,
                n_stations=n_stations,
                load_type=load_type,
                subcase_id=subcase_id,
            )
        else:
            curve = compute_vmt(model, nodal_forces, comp,
                                n_stations=n_stations,
                                load_type=load_type,
                                subcase_id=subcase_id)
        result.curves.append(curve)
    return result


def component_local_frame(
    all_xyz: np.ndarray,
    component: ComponentDef,
) -> tuple:
    """Build the component-local orthonormal frame (e1, e2, e3).

    e1: dominant principal axis of the node cloud (member/span axis),
        oriented outboard (sign of integration_sign on the declared
        global span axis).
    e3: the component's "up hint" axis (the global axis that is neither
        span nor chord: Z for wings/HTPs/V-tail halves, Y for VTPs and
        fuselages) orthogonalized against e1.
    e2: e3 x e1, completing a right-handed triad (e1 x e2 = e3). Note
        that for mirrored components (left/right wing) e2 is NOT the
        mirror image; the frame is fully reported in
        VMTCurve.local_frame so signs are unambiguous.

    Returns (frame, chord_ax, third_ax) where frame rows are e1, e2, e3.
    """
    span_ax = component.span_axis
    if span_ax in (1, 2):
        chord_ax = 0
    else:
        chord_ax = 2
    third_ax = 3 - span_ax - chord_ax

    ctr = all_xyz.mean(axis=0)
    x = all_xyz - ctr
    cov = x.T @ x
    w, vec = np.linalg.eigh(cov)
    e1 = vec[:, np.argmax(w)].copy()
    if abs(e1[span_ax]) > 1e-9 and e1[span_ax] * component.integration_sign < 0:
        e1 = -e1

    hint = np.zeros(3)
    hint[third_ax] = 1.0
    e3 = hint - (hint @ e1) * e1
    n3 = np.linalg.norm(e3)
    if n3 < 1e-9:
        hint = np.zeros(3)
        hint[chord_ax] = 1.0
        e3 = hint - (hint @ e1) * e1
        n3 = np.linalg.norm(e3)
    e3 = e3 / n3
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3]), chord_ax, third_ax


def _local6_recovery(
    all_xyz: np.ndarray,
    all_f6: np.ndarray,
    component: ComponentDef,
    n_stations: int,
    elastic_axis_frac: float,
) -> Optional[dict]:
    """Component-local 6-component section-load recovery (r3 MC1).

    Cuts are half-spaces s >= s_cut with s = x . e1 (plane normal to the
    member axis). The cut point at each station is the elastic-axis
    estimate of a sliding window (|s - s_cut| < half_bin): window chord
    min + frac * chord range on the global chord axis, window mean on
    the third axis, then projected onto the cut plane. All forces and
    direct nodal moments outboard of the cut are transported to that
    point and projected onto (e1, e2, e3).
    """
    frame, chord_ax, third_ax = component_local_frame(all_xyz, component)
    e1, e2, e3 = frame
    s = all_xyz @ e1
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-6:
        return None

    stations = np.linspace(s_min, s_max, n_stations)
    half_bin = max((s_max - s_min) / (n_stations * 0.8), 1.0)
    # Frame chord/normal coordinates: chord measured IN the section plane
    # (perpendicular to the member axis), so sweep-induced spread along
    # the global chord axis does not contaminate the elastic-axis
    # estimate. The chord-fraction datum stays "from the LE side": when
    # e2 points opposite the global chord hint the fraction is measured
    # from the max-c end instead.
    c_all = all_xyz @ e2
    n_all = all_xyz @ e3
    frac_from_min = e2[chord_ax] >= 0.0

    out = {k: np.zeros(n_stations) for k in ('N', 'Vy', 'Vz', 'Mx', 'My', 'Mz')}
    cuts = np.zeros((n_stations, 3))

    for i, s_cut in enumerate(stations):
        mask = s >= s_cut - 1e-9
        if not np.any(mask):
            continue
        nearby = np.abs(s - s_cut) < half_bin
        if not np.any(nearby):
            nearby = mask
        c_min = c_all[nearby].min()
        c_max = c_all[nearby].max()
        if frac_from_min:
            c_ea = c_min + elastic_axis_frac * (c_max - c_min)
        else:
            c_ea = c_max - elastic_axis_frac * (c_max - c_min)
        cut = s_cut * e1 + c_ea * e2 + n_all[nearby].mean() * e3
        cuts[i] = cut

        f_out = all_f6[mask, :3]
        m_out = all_f6[mask, 3:6]
        r = all_xyz[mask] - cut
        sum_f = f_out.sum(axis=0)
        sum_m = np.cross(r, f_out).sum(axis=0) + m_out.sum(axis=0)

        out['N'][i] = sum_f @ e1
        out['Vy'][i] = sum_f @ e2
        out['Vz'][i] = sum_f @ e3
        out['Mx'][i] = sum_m @ e1
        out['My'][i] = sum_m @ e2
        out['Mz'][i] = sum_m @ e3

    return {
        'stations': stations, 'frame': frame, 'cut_points': cuts, **out,
    }


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
