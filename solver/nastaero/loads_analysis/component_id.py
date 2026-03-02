"""Geometric identification of aircraft structural components.

Automatically detects wing, horizontal tail, vertical tail, and fuselage
from structural node coordinates. Works with both BDFModel and VizModel
(no SPLINE/SET1 data required).

Usage:
    components = identify_components(model)
    for comp in components.components:
        print(f"{comp.name}: {len(comp.node_ids)} nodes")
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentDef:
    """Definition of a structural component for VMT integration."""
    name: str
    node_ids: List[int]
    span_axis: int              # 0=X, 1=Y, 2=Z
    shear_axis: int             # global force axis for V
    bending_axis: int           # global moment axis for M
    torsion_axis: int           # global moment axis for T
    integration_sign: float     # +1.0: outboard = +direction, -1.0: outboard = -direction
    color: str = 'blue'


@dataclass
class ComponentSet:
    """Collection of identified components."""
    components: List[ComponentDef] = field(default_factory=list)

    def get(self, name: str) -> Optional[ComponentDef]:
        """Find component by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for c in self.components:
            if name_lower in c.name.lower():
                return c
        return None

    def names(self) -> List[str]:
        return [c.name for c in self.components]


def identify_components(
    model: Any,
    centerline_tol_frac: float = 0.06,
    vtp_z_frac: float = 0.35,
    min_nodes: int = 20,
) -> ComponentSet:
    """Identify structural components from node geometry.

    Parameters
    ----------
    model : BDFModel or VizModel
        Model with .nodes[nid].xyz_global attribute.
    centerline_tol_frac : float
        Fraction of Y-span used as fuselage half-width tolerance.
    vtp_z_frac : float
        Fraction of Z-span above which nodes are classified as VTP.
    min_nodes : int
        Minimum nodes to qualify as a component.

    Returns
    -------
    ComponentSet with detected components.
    """
    # 1. Collect all node coordinates
    nids = sorted(model.nodes.keys())
    n = len(nids)
    if n < min_nodes:
        return ComponentSet()

    nid_arr = np.array(nids)
    coords = np.array([model.nodes[nid].xyz_global for nid in nids], dtype=np.float64)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    y_span = y_max - y_min
    z_span = z_max - z_min

    if y_span < 1e-6:
        return ComponentSet()

    centerline_tol = centerline_tol_frac * y_span
    vtp_z_thresh = z_min + vtp_z_frac * z_span

    # 2. Classify nodes into regions
    near_center = np.abs(y) < centerline_tol

    # VTP: near centerline AND high Z
    vtp_mask = near_center & (z > vtp_z_thresh)

    # Fuselage: near centerline AND NOT VTP
    fuse_mask = near_center & ~vtp_mask

    # Lateral nodes (wing / HTP candidates)
    lateral_mask = ~near_center

    # 3. Separate lateral nodes into Wing vs HTP by X-position clustering
    lateral_indices = np.where(lateral_mask)[0]
    components = ComponentSet()

    if len(lateral_indices) > 0:
        _classify_lateral(nid_arr, coords, lateral_indices,
                          components, min_nodes)

    # 4. VTP
    vtp_indices = np.where(vtp_mask)[0]
    if len(vtp_indices) >= min_nodes:
        vtp_nids = nid_arr[vtp_indices].tolist()
        components.components.append(ComponentDef(
            name='VTP',
            node_ids=vtp_nids,
            span_axis=2, shear_axis=1, bending_axis=0, torsion_axis=2,
            integration_sign=1.0,  # outboard = +Z (upward)
            color='green',
        ))

    # 5. Fuselage
    fuse_indices = np.where(fuse_mask)[0]
    if len(fuse_indices) >= min_nodes:
        fuse_nids = nid_arr[fuse_indices].tolist()
        components.components.append(ComponentDef(
            name='Fuselage',
            node_ids=fuse_nids,
            span_axis=0, shear_axis=2, bending_axis=1, torsion_axis=0,
            integration_sign=-1.0,  # integrate nose → tail (−X direction)
            color='gray',
        ))

    return components


def _classify_lateral(
    nid_arr: np.ndarray,
    coords: np.ndarray,
    lateral_indices: np.ndarray,
    comp_set: ComponentSet,
    min_nodes: int,
) -> None:
    """Classify lateral nodes into Wing and HTP by X-position clustering."""
    lat_x = coords[lateral_indices, 0]
    lat_y = coords[lateral_indices, 1]

    # Split into right (Y>0) and left (Y<0)
    right_mask = lat_y > 0
    left_mask = lat_y < 0

    for side_mask, side_name, sign in [
        (right_mask, 'Right', 1.0),
        (left_mask, 'Left', -1.0),
    ]:
        side_indices = lateral_indices[side_mask]
        if len(side_indices) < min_nodes:
            continue

        side_x = coords[side_indices, 0]

        # Find gap in X distribution to separate wing from HTP
        groups = _split_by_x_gap(side_indices, side_x, coords)

        if len(groups) == 0:
            continue

        if len(groups) == 1:
            # Only one group — assume it's the wing
            grp = groups[0]
            if len(grp) >= min_nodes:
                comp_set.components.append(ComponentDef(
                    name=f'{side_name} Wing',
                    node_ids=nid_arr[grp].tolist(),
                    span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1,
                    integration_sign=sign,
                    color='blue' if sign > 0 else 'dodgerblue',
                ))
        else:
            # Multiple groups — the one with greater Y-extent is probably wing,
            # the one further aft (higher mean X) with smaller Y-extent is HTP
            groups.sort(key=lambda g: coords[g, 0].mean())
            # Wing: the group with larger spanwise (Y) extent
            y_extents = [np.ptp(coords[g, 1]) for g in groups]
            wing_idx = int(np.argmax(y_extents))
            for gi, grp in enumerate(groups):
                if len(grp) < min_nodes:
                    continue
                if gi == wing_idx:
                    comp_set.components.append(ComponentDef(
                        name=f'{side_name} Wing',
                        node_ids=nid_arr[grp].tolist(),
                        span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1,
                        integration_sign=sign,
                        color='blue' if sign > 0 else 'dodgerblue',
                    ))
                else:
                    comp_set.components.append(ComponentDef(
                        name=f'{side_name} HTP',
                        node_ids=nid_arr[grp].tolist(),
                        span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1,
                        integration_sign=sign,
                        color='red' if sign > 0 else 'salmon',
                    ))


def _split_by_x_gap(
    indices: np.ndarray,
    x_vals: np.ndarray,
    coords: np.ndarray,
    gap_fraction: float = 0.15,
) -> List[np.ndarray]:
    """Split indices into groups based on gaps in X-distribution.

    If there's a significant gap in X-positions (> gap_fraction * x_range),
    split into separate groups (e.g., Wing vs HTP).
    """
    if len(indices) == 0:
        return []

    sorted_order = np.argsort(x_vals)
    sorted_x = x_vals[sorted_order]
    sorted_indices = indices[sorted_order]

    x_range = sorted_x[-1] - sorted_x[0]
    if x_range < 1e-6:
        return [indices]

    gap_threshold = gap_fraction * x_range

    # Find the largest gap
    diffs = np.diff(sorted_x)
    if len(diffs) == 0:
        return [indices]

    max_gap_idx = np.argmax(diffs)
    max_gap = diffs[max_gap_idx]

    if max_gap > gap_threshold:
        # Split at the largest gap
        grp1 = sorted_indices[:max_gap_idx + 1]
        grp2 = sorted_indices[max_gap_idx + 1:]
        return [grp1, grp2]
    else:
        return [indices]


def identify_components_manual(
    model: Any,
    specs: List[dict],
) -> ComponentSet:
    """Create component definitions from user specifications.

    Parameters
    ----------
    model : BDFModel or VizModel
    specs : list of dict
        Each dict must have:
        - 'name': str
        - 'node_range': (min_id, max_id) or 'node_ids': list
        - 'span_axis': int (0=X, 1=Y, 2=Z)
        - 'shear_axis': int
        - 'bending_axis': int
        - 'torsion_axis': int
        - 'integration_sign': float (+1 or -1)
    """
    all_nids = set(model.nodes.keys())
    components = ComponentSet()

    for spec in specs:
        if 'node_ids' in spec:
            node_ids = [n for n in spec['node_ids'] if n in all_nids]
        elif 'node_range' in spec:
            lo, hi = spec['node_range']
            node_ids = [n for n in all_nids if lo <= n <= hi]
        else:
            continue

        components.components.append(ComponentDef(
            name=spec['name'],
            node_ids=sorted(node_ids),
            span_axis=spec.get('span_axis', 1),
            shear_axis=spec.get('shear_axis', 2),
            bending_axis=spec.get('bending_axis', 0),
            torsion_axis=spec.get('torsion_axis', 1),
            integration_sign=spec.get('integration_sign', 1.0),
            color=spec.get('color', 'blue'),
        ))

    return components
