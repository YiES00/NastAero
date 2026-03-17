"""Build PyVista meshes from BDF model and results.

Converts NastAero BDFModel and SubcaseResult objects into PyVista
UnstructuredGrid/PolyData for 3D rendering.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    import pyvista as pv
except ImportError:
    raise ImportError(
        "PyVista is required for visualization. Install with: pip install pyvista"
    )


def _create_grid(cells, cell_types, points):
    """Create UnstructuredGrid from cell list."""
    flat = []
    for c in cells:
        flat.extend(c)
    cells_array = np.array(flat, dtype=np.int64)
    return pv.UnstructuredGrid(cells_array, cell_types, points)


def _estimate_beam_radius(bdf_model, elem) -> float:
    """Estimate a representative cross-section radius for beam rendering.

    Uses the cross-sectional area A from the property card:
        r = sqrt(A / pi)

    Falls back to a fraction of the beam length if A is not available.
    """
    radius = 0.0
    pid = getattr(elem, 'pid', 0)
    prop = bdf_model.properties.get(pid)
    if prop is not None:
        A = getattr(prop, 'A', 0.0)
        if A > 0:
            radius = np.sqrt(A / np.pi)

    # Fallback: 3% of element length
    if radius <= 0:
        nids = elem.node_ids
        if len(nids) >= 2 and nids[0] in bdf_model.nodes and nids[1] in bdf_model.nodes:
            p1 = bdf_model.nodes[nids[0]].xyz_global
            p2 = bdf_model.nodes[nids[1]].xyz_global
            L = np.linalg.norm(p2 - p1)
            radius = 0.03 * L

    return max(radius, 1e-6)


def _collect_beam_segments(bdf_model, displacements=None, scale=1.0):
    """Collect beam endpoints and radii from model.

    Returns list of (p1, p2, radius) tuples and optionally disp_values.
    """
    segments = []
    disp_values = [] if displacements is not None else None

    for eid in sorted(bdf_model.elements.keys()):
        elem = bdf_model.elements[eid]
        if elem.type not in ("CBAR", "CROD"):
            continue

        nids = elem.node_ids
        if len(nids) < 2:
            continue
        if nids[0] not in bdf_model.nodes or nids[1] not in bdf_model.nodes:
            continue

        p1 = bdf_model.nodes[nids[0]].xyz_global.copy()
        p2 = bdf_model.nodes[nids[1]].xyz_global.copy()

        if displacements is not None:
            d1 = displacements.get(nids[0], np.zeros(6))
            d2 = displacements.get(nids[1], np.zeros(6))
            p1 += d1[:3] * scale
            p2 += d2[:3] * scale
            disp_values.append(0.5 * (np.linalg.norm(d1[:3]) + np.linalg.norm(d2[:3])))

        L = np.linalg.norm(p2 - p1)
        if L < 1e-12:
            if disp_values is not None:
                disp_values.pop()
            continue

        radius = _estimate_beam_radius(bdf_model, elem)
        segments.append((p1, p2, radius))

    return segments, disp_values


def _build_tubes_bulk(segments, n_sides=12, disp_values=None):
    """Build tube mesh from beam segments using bulk PolyData + tube filter.

    Groups beams by radius to minimize tube() calls.
    """
    if not segments:
        return None

    # Group segments by radius for bulk tube generation
    from collections import defaultdict
    radius_groups = defaultdict(list)
    radius_disp = defaultdict(list) if disp_values is not None else None
    for i, (p1, p2, r) in enumerate(segments):
        # Quantize radius to reduce groups (round to 4 significant digits)
        rq = float(f'{r:.4g}')
        radius_groups[rq].append((p1, p2))
        if radius_disp is not None:
            radius_disp[rq].append(disp_values[i])

    tube_meshes = []
    tube_disp_arrays = [] if disp_values is not None else None

    for r, segs in radius_groups.items():
        # Build all lines in this radius group as a single PolyData
        n_segs = len(segs)
        points = np.empty((n_segs * 2, 3), dtype=np.float64)
        lines = np.empty((n_segs, 3), dtype=np.int64)
        for j, (p1, p2) in enumerate(segs):
            points[j * 2] = p1
            points[j * 2 + 1] = p2
            lines[j] = [2, j * 2, j * 2 + 1]

        poly = pv.PolyData(points, lines=lines.ravel())
        tube = poly.tube(radius=r, n_sides=n_sides)
        tube_meshes.append(tube)

        if tube_disp_arrays is not None:
            # Assign per-cell displacement from segment averages
            cells_per_seg = tube.n_cells // n_segs if n_segs > 0 else 0
            if cells_per_seg > 0:
                dv = np.array(radius_disp[r])
                tube.cell_data['Displacement_Magnitude'] = np.repeat(dv, cells_per_seg)
            else:
                tube.cell_data['Displacement_Magnitude'] = np.zeros(tube.n_cells)

    if len(tube_meshes) == 1:
        return tube_meshes[0]

    blocks = pv.MultiBlock(tube_meshes)
    return blocks.combine()


def build_beam_tubes(bdf_model, n_sides: int = 12) -> Optional[pv.PolyData]:
    """Build 3D tube representations for CBAR/CROD elements.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model.
    n_sides : int
        Number of sides for the tube cross-section polygon.

    Returns
    -------
    pv.PolyData or None
        Combined tube mesh for all beam elements, or None if no beams.
    """
    segments, _ = _collect_beam_segments(bdf_model)
    return _build_tubes_bulk(segments, n_sides=n_sides)


def build_deformed_beam_tubes(
    bdf_model,
    displacements: Dict[int, np.ndarray],
    scale: float = 1.0,
    n_sides: int = 12,
) -> Optional[pv.PolyData]:
    """Build 3D tube representations for beams with deformed node positions.

    Parameters
    ----------
    bdf_model : BDFModel
    displacements : Dict[int, np.ndarray]
        Node ID -> 6-DOF displacement.
    scale : float
        Deformation scale factor.
    n_sides : int
        Tube polygon sides.

    Returns
    -------
    pv.PolyData or None
    """
    segments, disp_values = _collect_beam_segments(
        bdf_model, displacements=displacements, scale=scale)
    return _build_tubes_bulk(segments, n_sides=n_sides, disp_values=disp_values)


def build_structural_mesh(bdf_model, include_beams: bool = True) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid from BDF model structural elements.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model containing nodes and elements.
    include_beams : bool
        Whether to include beam/rod elements as line cells.

    Returns
    -------
    pv.UnstructuredGrid
        Mesh with cell data 'ElementID', 'PropertyID' and point data 'NodeID'.
    """
    if not bdf_model.nodes:
        raise ValueError("BDF model has no nodes")

    sorted_nids = sorted(bdf_model.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}

    points = np.zeros((len(sorted_nids), 3))
    for i, nid in enumerate(sorted_nids):
        points[i] = bdf_model.nodes[nid].xyz_global

    cells_list = []
    cell_types = []
    elem_ids = []
    prop_ids = []

    VTK_LINE = 3
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    VTK_QUADRATIC_TRIANGLE = 22
    VTK_QUADRATIC_QUAD = 23

    for eid in sorted(bdf_model.elements.keys()):
        elem = bdf_model.elements[eid]
        etype = elem.type

        if etype == "CQUAD4":
            nids = elem.node_ids
            if all(n in nid_to_idx for n in nids):
                idx = [nid_to_idx[n] for n in nids]
                cells_list.append([4] + idx)
                cell_types.append(VTK_QUAD)
                elem_ids.append(eid)
                prop_ids.append(getattr(elem, 'pid', 0))

        elif etype == "CQUAD8":
            nids = elem.node_ids
            if all(n in nid_to_idx for n in nids):
                idx = [nid_to_idx[n] for n in nids]
                cells_list.append([8] + idx)
                cell_types.append(VTK_QUADRATIC_QUAD)
                elem_ids.append(eid)
                prop_ids.append(getattr(elem, 'pid', 0))

        elif etype == "CTRIA3":
            nids = elem.node_ids
            if all(n in nid_to_idx for n in nids):
                idx = [nid_to_idx[n] for n in nids]
                cells_list.append([3] + idx)
                cell_types.append(VTK_TRIANGLE)
                elem_ids.append(eid)
                prop_ids.append(getattr(elem, 'pid', 0))

        elif etype == "CTRIA6":
            nids = elem.node_ids
            if all(n in nid_to_idx for n in nids):
                idx = [nid_to_idx[n] for n in nids]
                cells_list.append([6] + idx)
                cell_types.append(VTK_QUADRATIC_TRIANGLE)
                elem_ids.append(eid)
                prop_ids.append(getattr(elem, 'pid', 0))

        elif etype in ("CBAR", "CROD") and include_beams:
            nids = elem.node_ids
            if all(n in nid_to_idx for n in nids):
                idx = [nid_to_idx[n] for n in nids]
                cells_list.append([2] + idx)
                cell_types.append(VTK_LINE)
                elem_ids.append(eid)
                prop_ids.append(getattr(elem, 'pid', 0))

    if not cells_list:
        # Return an empty grid with points but no cells
        # (happens when include_beams=False and model has only beams)
        grid = pv.UnstructuredGrid()
        grid.points = points
        grid.point_data['NodeID'] = np.array(sorted_nids, dtype=np.int64)
        return grid

    grid = _create_grid(cells_list, np.array(cell_types, dtype=np.uint8), points)
    grid.cell_data['ElementID'] = np.array(elem_ids, dtype=np.int64)
    grid.cell_data['PropertyID'] = np.array(prop_ids, dtype=np.int64)
    grid.point_data['NodeID'] = np.array(sorted_nids, dtype=np.int64)

    return grid


def build_deformed_mesh(
    bdf_model,
    displacements: Dict[int, np.ndarray],
    scale: float = 1.0,
) -> pv.UnstructuredGrid:
    """Build deformed mesh by displacing nodes.

    Parameters
    ----------
    bdf_model : BDFModel
        Original BDF model.
    displacements : Dict[int, np.ndarray]
        Node ID -> displacement vector (6 DOF: T1,T2,T3,R1,R2,R3).
    scale : float
        Displacement magnification factor.

    Returns
    -------
    pv.UnstructuredGrid
        Deformed mesh with displacement magnitude as point data.
    """
    grid = build_structural_mesh(bdf_model)
    sorted_nids = sorted(bdf_model.nodes.keys())

    disp_mag = np.zeros(len(sorted_nids))
    disp_vectors = np.zeros((len(sorted_nids), 3))

    for i, nid in enumerate(sorted_nids):
        if nid in displacements:
            d = displacements[nid]
            disp_vectors[i] = d[:3] * scale
            disp_mag[i] = np.linalg.norm(d[:3])

    grid.points += disp_vectors
    grid.point_data['Displacement_Magnitude'] = disp_mag
    grid.point_data['T1'] = disp_vectors[:, 0] / max(scale, 1e-30)
    grid.point_data['T2'] = disp_vectors[:, 1] / max(scale, 1e-30)
    grid.point_data['T3'] = disp_vectors[:, 2] / max(scale, 1e-30)

    return grid


def build_mode_shape_mesh(
    bdf_model,
    mode_shape: Dict[int, np.ndarray],
    scale: float = 1.0,
) -> pv.UnstructuredGrid:
    """Build mesh deformed by a mode shape.

    Parameters
    ----------
    bdf_model : BDFModel
    mode_shape : Dict[int, np.ndarray]
        Node ID -> mode shape vector (6 DOF).
    scale : float
        Mode shape magnification factor.

    Returns
    -------
    pv.UnstructuredGrid
    """
    grid = build_structural_mesh(bdf_model)
    sorted_nids = sorted(bdf_model.nodes.keys())

    mode_mag = np.zeros(len(sorted_nids))
    mode_vectors = np.zeros((len(sorted_nids), 3))

    for i, nid in enumerate(sorted_nids):
        if nid in mode_shape:
            d = mode_shape[nid]
            mode_vectors[i] = d[:3] * scale
            mode_mag[i] = np.linalg.norm(d[:3])

    grid.points += mode_vectors
    grid.point_data['Mode_Magnitude'] = mode_mag

    return grid


def _rotate_point_about_axis(
    point: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate a point about an arbitrary axis using Rodrigues' formula.

    Parameters
    ----------
    point : (3,) point to rotate
    axis_point : (3,) a point on the rotation axis
    axis_dir : (3,) unit direction of rotation axis
    angle : rotation angle in radians (positive = right-hand rule)

    Returns
    -------
    ndarray (3,)
        Rotated point.
    """
    v = point - axis_point
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    v_rot = (v * cos_a
             + np.cross(axis_dir, v) * sin_a
             + axis_dir * np.dot(axis_dir, v) * (1 - cos_a))
    return axis_point + v_rot


def _apply_control_surface_deflections(
    aero_boxes: list,
    bdf_model,
    trim_variables: Dict[str, float],
) -> np.ndarray:
    """Compute deflected corner coordinates for control surface panels.

    For each control surface (AESURF), finds the corresponding AELIST boxes
    and rotates their trailing-edge corners about the hinge line (LE edge)
    by the trim deflection angle.

    Parameters
    ----------
    aero_boxes : list of AeroBox
    bdf_model : BDFModel
    trim_variables : dict
        Trim variable name -> value in radians.

    Returns
    -------
    corners : ndarray (n_boxes, 4, 3)
        Corner coordinates with control surface panels deflected.
    """
    n = len(aero_boxes)
    corners = np.array([box.corners.copy() for box in aero_boxes])

    if not hasattr(bdf_model, 'aesurfs') or not hasattr(bdf_model, 'aelists'):
        return corners

    # Build box_id -> index mapping
    box_id_to_idx = {box.box_id: i for i, box in enumerate(aero_boxes)}

    for surf in bdf_model.aesurfs.values():
        label = surf.label.upper()
        if label not in trim_variables:
            continue
        delta = float(trim_variables[label])  # deflection in radians
        if abs(delta) < 1e-12:
            continue

        # Get AELIST box IDs for this control surface
        alid = surf.alid1
        if alid not in bdf_model.aelists:
            continue
        cs_box_ids = bdf_model.aelists[alid].elements

        for box_eid in cs_box_ids:
            if box_eid not in box_id_to_idx:
                continue
            idx = box_id_to_idx[box_eid]
            c = corners[idx]  # (4, 3): [LE_in, TE_in, TE_out, LE_out]

            # Hinge line = LE edge of this box: c[0] (inboard LE) to c[3] (outboard LE)
            hinge_pt = c[0].copy()
            hinge_dir = c[3] - c[0]
            hinge_len = np.linalg.norm(hinge_dir)
            if hinge_len < 1e-12:
                continue
            hinge_axis = hinge_dir / hinge_len

            # Rotate TE corners (c[1]=inboard TE, c[2]=outboard TE) about hinge
            c[1] = _rotate_point_about_axis(c[1], hinge_pt, hinge_axis, delta)
            c[2] = _rotate_point_about_axis(c[2], hinge_pt, hinge_axis, delta)

    return corners


def build_aero_mesh(
    aero_boxes: list,
    bdf_model=None,
    trim_variables: Optional[Dict[str, float]] = None,
) -> pv.PolyData:
    """Build aerodynamic panel mesh from AeroBox list.

    Parameters
    ----------
    aero_boxes : list
        List of AeroBox objects with corners (4,3).
    bdf_model : BDFModel, optional
        If provided with trim_variables, control surface panels are deflected.
    trim_variables : dict, optional
        Trim variable name -> value (radians) for control surface deflection.

    Returns
    -------
    pv.PolyData
        Mesh of quadrilateral aerodynamic panels.
    """
    if not aero_boxes:
        raise ValueError("No aerodynamic boxes provided")

    n_boxes = len(aero_boxes)

    # Apply control surface deflections if available
    if bdf_model is not None and trim_variables is not None:
        corners_all = _apply_control_surface_deflections(
            aero_boxes, bdf_model, trim_variables)
    else:
        corners_all = np.array([box.corners for box in aero_boxes])

    points = np.zeros((n_boxes * 4, 3))
    faces = []
    box_ids = []

    for i, box in enumerate(aero_boxes):
        base = i * 4
        points[base:base + 4] = corners_all[i]
        faces.extend([4, base, base + 1, base + 2, base + 3])
        box_ids.append(box.box_id)

    faces_array = np.array(faces, dtype=np.int64)
    mesh = pv.PolyData(points, faces=faces_array)
    mesh.cell_data['BoxID'] = np.array(box_ids, dtype=np.int64)

    return mesh


def build_aero_pressure_mesh(
    aero_boxes: list,
    pressures: np.ndarray,
    bdf_model=None,
    trim_variables: Optional[Dict[str, float]] = None,
) -> pv.PolyData:
    """Build aerodynamic mesh with pressure distribution.

    Parameters
    ----------
    aero_boxes : list
        List of AeroBox objects.
    pressures : np.ndarray
        Pressure values (Cp or dimensional) per box.
    bdf_model : BDFModel, optional
        For control surface deflection.
    trim_variables : dict, optional
        Trim variable name -> value (radians).

    Returns
    -------
    pv.PolyData
        Mesh with 'Pressure' cell data.
    """
    mesh = build_aero_mesh(aero_boxes, bdf_model=bdf_model,
                           trim_variables=trim_variables)
    if pressures is not None and len(pressures) == len(aero_boxes):
        mesh.cell_data['Pressure'] = np.real(pressures).astype(float)
    return mesh


def build_aero_force_arrows(
    aero_boxes: list,
    aero_forces: np.ndarray,
    scale: float = 0.0,
) -> Optional[pv.PolyData]:
    """Build arrow glyphs showing aerodynamic force direction on each panel.

    Parameters
    ----------
    aero_boxes : list
        List of AeroBox objects.
    aero_forces : ndarray (n, 3)
        Force vector (fx, fy, fz) per box.
    scale : float
        Arrow length scale factor. 0 = auto-scale based on panel size.

    Returns
    -------
    pv.PolyData or None
        Arrow glyph mesh, or None if no valid data.
    """
    if aero_forces is None or len(aero_boxes) == 0:
        return None

    n = len(aero_boxes)
    if len(aero_forces) != n:
        return None

    # Use control points (3/4 chord) as arrow origins
    origins = np.array([box.control_point for box in aero_boxes])
    force_vecs = np.real(aero_forces).astype(float)

    # Compute force magnitudes
    magnitudes = np.linalg.norm(force_vecs, axis=1)
    max_mag = np.max(magnitudes)
    if max_mag < 1e-30:
        return None

    # Auto-scale: arrow length ~ average panel chord
    if scale <= 0.0:
        avg_chord = np.mean([box.chord for box in aero_boxes])
        scale = 1.5 * avg_chord / max_mag

    # Normalize directions and compute scaled magnitudes
    directions = np.zeros_like(force_vecs)
    arrow_mags = np.zeros(n)
    for i in range(n):
        if magnitudes[i] > 1e-30:
            directions[i] = force_vecs[i] / magnitudes[i]
            arrow_mags[i] = magnitudes[i] * scale

    # Build as PolyData with vectors for glyph
    points = pv.PolyData(origins)
    points['vectors'] = directions * arrow_mags[:, np.newaxis]
    points['Force_Magnitude'] = magnitudes

    # Create arrow glyphs
    arrows = points.glyph(
        orient='vectors',
        scale='vectors',
        factor=1.0,
        geom=pv.Arrow(
            tip_length=0.3,
            tip_radius=0.12,
            shaft_radius=0.04,
            shaft_resolution=12,
            tip_resolution=12,
        ),
    )

    return arrows


def build_aero_normal_arrows(
    aero_boxes: list,
    scale: float = 0.0,
) -> Optional[pv.PolyData]:
    """Build arrow glyphs showing panel normal directions.

    Useful for verifying panel orientation without force data.

    Parameters
    ----------
    aero_boxes : list
        List of AeroBox objects.
    scale : float
        Arrow length. 0 = auto-scale based on panel chord.

    Returns
    -------
    pv.PolyData or None
    """
    if not aero_boxes:
        return None

    n = len(aero_boxes)
    origins = np.array([box.control_point for box in aero_boxes])
    normals = np.array([box.normal for box in aero_boxes])

    if scale <= 0.0:
        avg_chord = np.mean([box.chord for box in aero_boxes])
        scale = 0.3 * avg_chord

    points = pv.PolyData(origins)
    points['vectors'] = normals * scale
    points['Normal_Z'] = normals[:, 2]  # z-component for coloring

    arrows = points.glyph(
        orient='vectors',
        scale='vectors',
        factor=1.0,
        geom=pv.Arrow(
            tip_length=0.3,
            tip_radius=0.12,
            shaft_radius=0.04,
            shaft_resolution=12,
            tip_resolution=12,
        ),
    )

    return arrows


def build_rbe_lines(bdf_model) -> Optional[pv.PolyData]:
    """Build line segments for RBE2/RBE3 visualization.

    Returns PolyData of line segments connecting master/slave nodes.
    """
    if not bdf_model.rigids:
        return None

    nid_set = set(bdf_model.nodes.keys())

    lines_points = []
    lines_cells = []
    point_idx = 0

    for rid, rbe in bdf_model.rigids.items():
        if hasattr(rbe, 'independent_node') and hasattr(rbe, 'dependent_nodes'):
            # RBE2
            master_nid = rbe.independent_node
            if master_nid not in nid_set:
                continue
            master_xyz = bdf_model.nodes[master_nid].xyz_global
            for dep_nid in rbe.dependent_nodes:
                if dep_nid not in nid_set:
                    continue
                dep_xyz = bdf_model.nodes[dep_nid].xyz_global
                lines_points.extend([master_xyz, dep_xyz])
                lines_cells.extend([2, point_idx, point_idx + 1])
                point_idx += 2

        elif hasattr(rbe, 'refgrid') and hasattr(rbe, 'weight_sets'):
            # RBE3
            ref_nid = rbe.refgrid
            if ref_nid not in nid_set:
                continue
            ref_xyz = bdf_model.nodes[ref_nid].xyz_global
            for ws in rbe.weight_sets:
                weight, comp, grid_ids = ws
                for gid in grid_ids:
                    if gid not in nid_set:
                        continue
                    gid_xyz = bdf_model.nodes[gid].xyz_global
                    lines_points.extend([ref_xyz, gid_xyz])
                    lines_cells.extend([2, point_idx, point_idx + 1])
                    point_idx += 2

    if not lines_points:
        return None

    points = np.array(lines_points)
    cells = np.array(lines_cells, dtype=np.int64)
    mesh = pv.PolyData(points, lines=cells)

    return mesh


def build_nodal_force_arrows(
    bdf_model,
    nodal_forces: Dict[int, np.ndarray],
    scale: float = 0.0,
    min_fraction: float = 0.01,
    max_arrows: int = 500,
) -> Optional[pv.PolyData]:
    """Build arrow glyphs showing force vectors at structural nodes.

    Parameters
    ----------
    bdf_model : BDFModel
    nodal_forces : Dict[int, ndarray(6)]
        Node ID -> force/moment vector.
    scale : float
        Arrow length scale. 0 = auto-scale based on model size.
    min_fraction : float
        Minimum force fraction to display (filters noise).
    max_arrows : int
        Maximum number of arrows to display.  When more nodes have
        non-negligible forces, only the top *max_arrows* by magnitude
        are kept.  This keeps large models readable.

    Returns
    -------
    pv.PolyData or None
        Arrow glyph mesh, or None if no valid data.
    """
    if not nodal_forces:
        return None

    # Collect non-zero force nodes
    origins = []
    force_vecs = []
    magnitudes_list = []

    for nid in sorted(nodal_forces.keys()):
        if nid not in bdf_model.nodes:
            continue
        fv = nodal_forces[nid][:3]
        mag = np.linalg.norm(fv)
        if mag < 1e-30:
            continue
        origins.append(bdf_model.nodes[nid].xyz_global)
        force_vecs.append(fv)
        magnitudes_list.append(mag)

    if not origins:
        return None

    origins = np.array(origins)
    force_vecs = np.array(force_vecs)
    magnitudes = np.array(magnitudes_list)

    max_mag = np.max(magnitudes)

    # Filter out very small forces
    mask = magnitudes > (max_mag * min_fraction)
    if not np.any(mask):
        return None
    origins = origins[mask]
    force_vecs = force_vecs[mask]
    magnitudes = magnitudes[mask]

    # Keep only top-N arrows by magnitude for readability
    if len(magnitudes) > max_arrows:
        top_idx = np.argsort(magnitudes)[-max_arrows:]
        origins = origins[top_idx]
        force_vecs = force_vecs[top_idx]
        magnitudes = magnitudes[top_idx]
        max_mag = np.max(magnitudes)

    # Auto-scale: arrow length relative to model bounding box
    if scale <= 0.0:
        all_pts = np.array([bdf_model.nodes[nid].xyz_global
                            for nid in bdf_model.nodes])
        bbox_diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
        # Adaptive factor: larger arrows for better visibility
        # Use 15% of bbox for max arrow, ensuring clear display
        scale = 0.15 * bbox_diag / max_mag

    # Normalize and scale
    directions = force_vecs / magnitudes[:, np.newaxis]
    arrow_lengths = magnitudes * scale

    points = pv.PolyData(origins)
    points['vectors'] = directions * arrow_lengths[:, np.newaxis]
    points['Force_Magnitude'] = magnitudes

    arrows = points.glyph(
        orient='vectors',
        scale='vectors',
        factor=1.0,
        geom=pv.Arrow(
            tip_length=0.25,
            tip_radius=0.10,
            shaft_radius=0.03,
            shaft_resolution=12,
            tip_resolution=12,
        ),
    )

    return arrows


def add_displacement_data(
    grid: pv.UnstructuredGrid,
    bdf_model,
    displacements: Dict[int, np.ndarray],
) -> None:
    """Add displacement field data to existing mesh (in-place).

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        Mesh to add data to.
    bdf_model : BDFModel
    displacements : Dict[int, np.ndarray]
        Node ID -> 6-DOF displacement vector.
    """
    sorted_nids = sorted(bdf_model.nodes.keys())

    disp_mag = np.zeros(len(sorted_nids))
    t1 = np.zeros(len(sorted_nids))
    t2 = np.zeros(len(sorted_nids))
    t3 = np.zeros(len(sorted_nids))
    r1 = np.zeros(len(sorted_nids))
    r2 = np.zeros(len(sorted_nids))
    r3 = np.zeros(len(sorted_nids))

    for i, nid in enumerate(sorted_nids):
        if nid in displacements:
            d = displacements[nid]
            t1[i], t2[i], t3[i] = d[0], d[1], d[2]
            r1[i], r2[i], r3[i] = d[3], d[4], d[5]
            disp_mag[i] = np.linalg.norm(d[:3])

    grid.point_data['Displacement_Magnitude'] = disp_mag
    grid.point_data['T1'] = t1
    grid.point_data['T2'] = t2
    grid.point_data['T3'] = t3
    grid.point_data['R1'] = r1
    grid.point_data['R2'] = r2
    grid.point_data['R3'] = r3


# ---------------------------------------------------------------------------
# Rotor blade and disk meshes for VTOL visualization
# ---------------------------------------------------------------------------

def _rotor_shaft_frame(shaft_axis: np.ndarray) -> np.ndarray:
    """Build orthonormal rotation matrix from shaft frame to model coords.

    Same convention as rotor_loads_applicator.py:39-51.

    Returns
    -------
    R : ndarray (3, 3)
        Columns are [x_shaft, y_shaft, z_shaft] in model coords.
        z_shaft = normalised shaft_axis.
    """
    z = shaft_axis / np.linalg.norm(shaft_axis)
    v = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(z, v)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def build_rotor_blades(
    vtol_config,
    n_radial: int = 15,
    n_profile: int = 12,
    rotation_filter=None,
) -> Optional[pv.PolyData]:
    """Build 3D blade surface meshes for VTOL rotors.

    Each blade is a lofted surface through NACA 0012 cross-sections
    at each radial station, with chord and twist distribution applied.

    Parameters
    ----------
    vtol_config : VTOLConfig
    n_radial : int
        Radial stations per blade.
    n_profile : int
        Airfoil points per side (total ~2*n_profile - 1).
    rotation_filter : RotationDir or None
        If set, only build blades for rotors matching this direction.

    Returns
    -------
    pv.PolyData or None
    """
    from ..rotor.airfoil import RotorAirfoil

    rotors = vtol_config.rotors
    if rotation_filter is not None:
        rotors = [r for r in rotors if r.rotation_dir == rotation_filter]
    if not rotors:
        return None

    # Base airfoil profile (chord-normalised)
    profile = RotorAirfoil.naca_4digit_profile(t=0.12, n_pts=n_profile)
    n_prof = len(profile)  # 2*n_profile - 1

    all_grids = []

    for rotor in rotors:
        blade = rotor.blade
        R_mm = blade.radius * 1000.0
        root_r = blade.root_cutout
        r_stations = np.linspace(root_r, 1.0, n_radial)

        shaft = rotor.effective_shaft_axis
        R_mat = _rotor_shaft_frame(shaft)
        hub = rotor.hub_position  # mm

        for k in range(rotor.n_blades):
            psi = k * 2.0 * np.pi / rotor.n_blades

            # Rotation matrix for azimuth about z_shaft (in shaft frame)
            cos_p, sin_p = np.cos(psi), np.sin(psi)

            points = np.zeros((n_radial, n_prof, 3))

            for i, r_frac in enumerate(r_stations):
                chord_mm = blade.chord_at(r_frac) * 1000.0
                twist = blade.twist_at(r_frac)
                r_mm = r_frac * R_mm

                for j in range(n_prof):
                    # Airfoil in local blade section coords
                    # x = chordwise (positive TE), y = thickness
                    x_af = (profile[j, 0] - 0.25) * chord_mm
                    y_af = profile[j, 1] * chord_mm

                    # Apply twist (rotation about span axis)
                    cos_t, sin_t = np.cos(twist), np.sin(twist)
                    x_tw = x_af * cos_t + y_af * sin_t
                    y_tw = -x_af * sin_t + y_af * cos_t

                    # In shaft frame: x_s = chordwise, y_s = thickness,
                    # z_s = radial (along shaft axis for lift rotor)
                    # Blade extends in x-y plane at radius r
                    # Position before azimuth: (x_tw, r_mm + y_tw, 0)
                    # Wait — need to think about this more carefully.
                    #
                    # Blade span is radial from hub (perpendicular to shaft).
                    # In shaft frame, z=shaft direction (up for hover).
                    # Blade extends radially outward from hub in x-y plane.
                    # At azimuth=0, blade spans along x_shaft direction.
                    #
                    # Local blade coords:
                    #   span = radial direction (blade long axis)
                    #   chord = perpendicular to span in x-y plane
                    #   thickness = along z_shaft
                    #
                    # At azimuth=0, span is along x_shaft:
                    #   p_shaft = (r_mm, x_tw_rotated, y_tw_rotated)
                    # Actually simpler:
                    #   span direction = x_shaft (at psi=0)
                    #   chordwise = y_shaft
                    #   thickness = z_shaft (along shaft axis)
                    #
                    # So in shaft frame before azimuth rotation:
                    #   p = (r_mm, x_tw, y_tw)  where x_tw is chord, y_tw is thickness

                    p_shaft = np.array([
                        r_mm * cos_p - x_tw * sin_p,
                        r_mm * sin_p + x_tw * cos_p,
                        y_tw,
                    ])

                    # Transform to model coordinates
                    p_model = R_mat @ p_shaft + hub
                    points[i, j, :] = p_model

            # Build structured grid for this blade
            X = points[:, :, 0]
            Y = points[:, :, 1]
            Z = points[:, :, 2]
            grid = pv.StructuredGrid(X, Y, Z)
            all_grids.append(grid.extract_surface())

    if not all_grids:
        return None

    combined = all_grids[0]
    for g in all_grids[1:]:
        combined = combined.merge(g)
    return combined


def build_rotor_disks(
    vtol_config,
    n_circle: int = 48,
) -> Optional[pv.PolyData]:
    """Build semi-transparent rotor disk annuli for all rotors.

    Parameters
    ----------
    vtol_config : VTOLConfig
    n_circle : int
        Points around the disk perimeter.

    Returns
    -------
    pv.PolyData or None
    """
    rotors = vtol_config.rotors
    if not rotors:
        return None

    all_disks = []
    theta = np.linspace(0, 2.0 * np.pi, n_circle, endpoint=False)

    for rotor in rotors:
        R_mm = rotor.blade.radius * 1000.0
        r_inner = rotor.blade.root_cutout * R_mm
        r_outer = R_mm

        shaft = rotor.effective_shaft_axis
        R_mat = _rotor_shaft_frame(shaft)
        hub = rotor.hub_position

        # Annulus: outer ring + inner ring
        pts_outer = np.zeros((n_circle, 3))
        pts_inner = np.zeros((n_circle, 3))

        for i in range(n_circle):
            cos_t, sin_t = np.cos(theta[i]), np.sin(theta[i])
            # In shaft frame: circle in x-y plane
            p_out = np.array([r_outer * cos_t, r_outer * sin_t, 0.0])
            p_in = np.array([r_inner * cos_t, r_inner * sin_t, 0.0])
            pts_outer[i] = R_mat @ p_out + hub
            pts_inner[i] = R_mat @ p_in + hub

        # Build annulus as quad strip
        n = n_circle
        points = np.vstack([pts_outer, pts_inner])  # 2*n points
        faces = []
        for i in range(n):
            i_next = (i + 1) % n
            # Quad: outer[i], outer[i+1], inner[i+1], inner[i]
            faces.extend([4, i, i_next, n + i_next, n + i])

        disk = pv.PolyData(points, np.array(faces))
        all_disks.append(disk)

    if not all_disks:
        return None

    combined = all_disks[0]
    for d in all_disks[1:]:
        combined = combined.merge(d)
    return combined
