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

        elif etype == "CTRIA3":
            nids = elem.node_ids
            if all(n in nid_to_idx for n in nids):
                idx = [nid_to_idx[n] for n in nids]
                cells_list.append([3] + idx)
                cell_types.append(VTK_TRIANGLE)
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
        raise ValueError("No renderable elements found in model")

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


def build_aero_mesh(aero_boxes: list) -> pv.PolyData:
    """Build aerodynamic panel mesh from AeroBox list.

    Parameters
    ----------
    aero_boxes : list
        List of AeroBox objects with corners (4,3).

    Returns
    -------
    pv.PolyData
        Mesh of quadrilateral aerodynamic panels.
    """
    if not aero_boxes:
        raise ValueError("No aerodynamic boxes provided")

    n_boxes = len(aero_boxes)
    points = np.zeros((n_boxes * 4, 3))
    faces = []
    box_ids = []

    for i, box in enumerate(aero_boxes):
        base = i * 4
        points[base:base + 4] = box.corners
        faces.extend([4, base, base + 1, base + 2, base + 3])
        box_ids.append(box.box_id)

    faces_array = np.array(faces, dtype=np.int64)
    mesh = pv.PolyData(points, faces=faces_array)
    mesh.cell_data['BoxID'] = np.array(box_ids, dtype=np.int64)

    return mesh


def build_aero_pressure_mesh(
    aero_boxes: list,
    pressures: np.ndarray,
) -> pv.PolyData:
    """Build aerodynamic mesh with pressure distribution.

    Parameters
    ----------
    aero_boxes : list
        List of AeroBox objects.
    pressures : np.ndarray
        Pressure values (Cp or dimensional) per box.

    Returns
    -------
    pv.PolyData
        Mesh with 'Pressure' cell data.
    """
    mesh = build_aero_mesh(aero_boxes)
    if pressures is not None and len(pressures) == len(aero_boxes):
        mesh.cell_data['Pressure'] = np.real(pressures).astype(float)
    return mesh


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
