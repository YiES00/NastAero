"""Trim loads analysis: compute and verify nodal aerodynamic and inertial loads.

Given a converged SOL 144 trim solution, this module:
1. Transfers panel aerodynamic forces to structural nodes via spline transpose
2. Computes inertial (gravity) forces at each node for the trimmed load factor
3. Combines aero + inertial forces and verifies 6-DOF equilibrium (trim balance)
4. Outputs nodal force cards in Nastran FORCE format
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..bdf.model import BDFModel
from ..aero.panel import AeroBox
from ..config import logger


def compute_node_masses(bdf_model: BDFModel) -> Dict[int, float]:
    """Compute lumped mass at each node from element contributions.

    Uses the same mass lumping as gravity load computation:
    - CBAR/CBEAM/CROD: rho * A * L / 2 per node
    - CQUAD4: rho * t * area / 4 per node
    - CTRIA3: rho * t * area / 3 per node
    - CONM2: direct mass

    Returns
    -------
    node_mass : Dict[int, float]
        Node ID -> lumped mass value.
    """
    node_mass: Dict[int, float] = {}

    for eid, elem in bdf_model.elements.items():
        if not hasattr(elem, 'property_ref') or elem.property_ref is None:
            continue
        prop = elem.property_ref

        if elem.type in ("CBAR", "CBEAM", "CROD"):
            mat = getattr(prop, 'material_ref', None)
            if mat is None or mat.rho <= 0:
                continue
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            em = mat.rho * prop.A * L + getattr(prop, 'nsm', 0.0) * L
            for nid in elem.node_ids:
                node_mass[nid] = node_mass.get(nid, 0.0) + em / 2.0

        elif elem.type == "CQUAD4":
            if hasattr(prop, 'equivalent_isotropic'):
                _, _, t, rho = prop.equivalent_isotropic()
            else:
                mat = getattr(prop, 'material_ref', None)
                if mat is None:
                    continue
                rho = mat.rho
                t = getattr(prop, 't', 0.0)
            if rho <= 0 or t <= 0:
                continue
            nids = elem.node_ids
            coords = np.array([bdf_model.nodes[nid].xyz_global for nid in nids])
            d13 = coords[2] - coords[0]
            d24 = coords[3] - coords[1]
            area = 0.5 * np.linalg.norm(np.cross(d13, d24))
            em = rho * t * area
            m_per_node = em / 4.0
            for nid in nids:
                node_mass[nid] = node_mass.get(nid, 0.0) + m_per_node

        elif elem.type == "CTRIA3":
            if hasattr(prop, 'equivalent_isotropic'):
                _, _, t, rho = prop.equivalent_isotropic()
            else:
                mat = getattr(prop, 'material_ref', None)
                if mat is None:
                    continue
                rho = mat.rho
                t = getattr(prop, 't', 0.0)
            if rho <= 0 or t <= 0:
                continue
            nids = elem.node_ids
            coords = np.array([bdf_model.nodes[nid].xyz_global for nid in nids])
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            em = rho * t * area
            m_per_node = em / 3.0
            for nid in nids:
                node_mass[nid] = node_mass.get(nid, 0.0) + m_per_node

    # CONM2 concentrated masses
    for mid, mass_elem in bdf_model.masses.items():
        nid = mass_elem.node_id
        node_mass[nid] = node_mass.get(nid, 0.0) + mass_elem.mass

    return node_mass


def compute_nodal_aero_forces(
    bdf_model: BDFModel,
    boxes: List[AeroBox],
    aero_forces: np.ndarray,
    G_eff_sparse,
    f_dofs: List[int],
    dof_mgr,
) -> Dict[int, np.ndarray]:
    """Transfer panel aerodynamic forces to structural nodes via spline.

    The spline matrix G_eff maps structural z-displacements to aero panel
    downwash. By virtual work, the transpose G_eff.T maps aero forces to
    structural forces. We use the spline z-weights to distribute the full
    3D force vector from each panel to structural nodes.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    aero_forces : ndarray (n_boxes, 3)
        Aerodynamic force vector (Fx, Fy, Fz) per panel.
    G_eff_sparse : sparse matrix (n_boxes, n_free)
        Spline coupling matrix.
    f_dofs : list of int
        Free DOF indices.
    dof_mgr : DOFManager

    Returns
    -------
    nodal_forces : Dict[int, ndarray(6)]
        Node ID -> [Fx, Fy, Fz, Mx, My, Mz] aerodynamic force.
    """
    import scipy.sparse as sp

    n_boxes = len(boxes)
    n_free = len(f_dofs)
    nodal_forces: Dict[int, np.ndarray] = {}

    # Initialize all nodes to zero
    for nid in dof_mgr.node_ids:
        nodal_forces[nid] = np.zeros(6)

    # Build f_dofs lookup
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}

    # For each aero panel, distribute its force to structural nodes
    # using the spline weights from G_eff
    G_csc = G_eff_sparse.tocsc() if sp.issparse(G_eff_sparse) else sp.csc_matrix(G_eff_sparse)

    # Strategy: For each box j, find which structural DOFs it connects to
    # G_eff[j, :] has nonzero entries at DOF indices corresponding to
    # z-translation and twist DOFs of structural nodes.
    # We use the z-translation weights to distribute the full 3D force.

    # Iterate over each box
    for j in range(n_boxes):
        F_j = aero_forces[j]  # (3,) force vector
        if np.linalg.norm(F_j) < 1e-30:
            continue

        # Get the row of G_eff for this box
        row = G_csc[j, :].toarray().ravel()  # (n_free,)

        # Find z-translation DOF weights only (not twist DOFs)
        # z-translation DOFs are component 3 (index 2 in 0-based within 6-DOF)
        for nid in dof_mgr.node_ids:
            z_dof_global = dof_mgr.get_dof(nid, 3)  # component 3 = z-trans
            if z_dof_global not in f_dof_index:
                continue
            f_idx = f_dof_index[z_dof_global]
            w = row[f_idx]
            if abs(w) < 1e-15:
                continue
            # Distribute full 3D force proportionally
            nodal_forces[nid][:3] += w * F_j

    return nodal_forces


def compute_nodal_aero_forces_fast(
    bdf_model: BDFModel,
    boxes: List[AeroBox],
    aero_forces: np.ndarray,
    G_eff_sparse,
    f_dofs: List[int],
    dof_mgr,
) -> Dict[int, np.ndarray]:
    """Vectorized version of aero force transfer via spline transpose.

    Uses G_eff.T to map each component of aero force to structural DOFs.
    For the z-component, this is exact (G_eff.T @ Fz).
    For x and y components, we use the z-weight distribution pattern.
    """
    import scipy.sparse as sp

    n_boxes = len(boxes)
    n_free = len(f_dofs)
    f_dof_set = set(f_dofs)

    # Initialize force vector in free-DOF space
    F_struct = np.zeros(n_free)

    G_csc = G_eff_sparse.tocsc() if sp.issparse(G_eff_sparse) else sp.csc_matrix(G_eff_sparse)

    # Build mapping: for each node, which f_dof index is its z-translation?
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}
    nid_to_z_fidx = {}
    for nid in dof_mgr.node_ids:
        z_dof = dof_mgr.get_dof(nid, 3)
        if z_dof in f_dof_index:
            nid_to_z_fidx[nid] = f_dof_index[z_dof]

    # Collect z-weight per (box, node) pair
    # For each box, the G_eff row gives normalwash contribution from each DOF.
    # The z-translation DOFs give us the pure displacement weight.

    # Build a z-only spline matrix: (n_boxes x n_nodes) where entry [j,i]
    # is the weight of node i's z-displacement on box j's downwash
    sorted_nids = dof_mgr.node_ids
    nid_to_node_idx = {nid: i for i, nid in enumerate(sorted_nids)}
    n_nodes = len(sorted_nids)

    # Extract z-DOF columns from G_eff
    z_fidx_list = []
    node_idx_list = []
    for nid in sorted_nids:
        if nid in nid_to_z_fidx:
            z_fidx_list.append(nid_to_z_fidx[nid])
            node_idx_list.append(nid_to_node_idx[nid])

    if not z_fidx_list:
        return {nid: np.zeros(6) for nid in sorted_nids}

    z_fidx_arr = np.array(z_fidx_list)
    node_idx_arr = np.array(node_idx_list)

    # G_z = G_eff[:, z_fidx_arr] → (n_boxes x n_z_nodes)
    G_z = G_csc[:, z_fidx_arr].toarray()  # (n_boxes, n_z_nodes)

    # For each force component, compute nodal forces via transpose:
    # F_node_comp = G_z.T @ F_aero_comp
    nodal_forces: Dict[int, np.ndarray] = {}
    for nid in sorted_nids:
        nodal_forces[nid] = np.zeros(6)

    for comp in range(3):  # Fx, Fy, Fz
        f_comp = np.real(aero_forces[:, comp]).astype(float)
        # G_z.T @ f_comp → (n_z_nodes,)
        f_nodal = G_z.T @ f_comp
        for k, nidx in enumerate(node_idx_arr):
            nid = sorted_nids[nidx]
            nodal_forces[nid][comp] += f_nodal[k]

    return nodal_forces


def compute_nodal_inertial_forces(
    bdf_model: BDFModel,
    nz: float,
    g: float,
    ny: float = 0.0,
) -> Dict[int, np.ndarray]:
    """Compute inertial (gravity) forces at each node for given load factors.

    F_inertia_z = -m_node * nz * g * k_hat  (negative z for +nz)
    F_inertia_y = -m_node * ny * g * j_hat  (negative y for +ny)

    For 1g level flight: nz = 1.0, ny = 0.0.
    For yaw maneuvers: ny ≠ 0 from lateral acceleration.

    Parameters
    ----------
    bdf_model : BDFModel
    nz : float
        Vertical load factor (1.0 for 1g level flight).
    g : float
        Gravitational acceleration in model units.
    ny : float
        Lateral load factor (0.0 for symmetric flight).

    Returns
    -------
    nodal_forces : Dict[int, ndarray(6)]
        Node ID -> [Fx, Fy, Fz, Mx, My, Mz] inertial force.
    """
    node_mass = compute_node_masses(bdf_model)
    nodal_forces: Dict[int, np.ndarray] = {}

    for nid in bdf_model.nodes:
        f = np.zeros(6)
        m = node_mass.get(nid, 0.0)
        if m > 0:
            # Inertial load in -z direction for positive nz (upward acceleration)
            f[2] = -m * nz * g
            # Lateral inertial load for yaw/roll maneuvers
            if abs(ny) > 1e-6:
                f[1] = -m * ny * g
        nodal_forces[nid] = f

    return nodal_forces


def compute_trim_nodal_loads(
    bdf_model: BDFModel,
    boxes: List[AeroBox],
    aero_forces: np.ndarray,
    G_eff_sparse,
    f_dofs: List[int],
    dof_mgr,
    nz: float = 1.0,
    g: float = 9810.0,
    ny: float = 0.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute all three nodal force sets for a trimmed condition.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    aero_forces : ndarray (n_boxes, 3)
    G_eff_sparse : sparse matrix
    f_dofs : list of int
    dof_mgr : DOFManager
    nz : float
        Vertical load factor.
    g : float
        Gravitational acceleration.
    ny : float
        Lateral load factor (0.0 for symmetric flight).

    Returns
    -------
    aero_nodal : Dict[int, ndarray(6)]
    inertial_nodal : Dict[int, ndarray(6)]
    combined_nodal : Dict[int, ndarray(6)]
    """
    logger.info("Computing nodal trim loads...")

    # Aerodynamic forces
    aero_nodal = compute_nodal_aero_forces_fast(
        bdf_model, boxes, aero_forces, G_eff_sparse, f_dofs, dof_mgr)

    # --- Post-spline force conservation ---
    # The spline transpose G_z.T may not perfectly conserve total force
    # if the z-DOF interpolation weights don't partition unity (e.g. IPS
    # spline with non-trivial geometry). Scale each force component
    # independently so that total nodal force matches total panel force.
    total_panel = np.zeros(3)
    for comp in range(3):
        total_panel[comp] = float(np.sum(np.real(aero_forces[:, comp])))

    total_nodal_raw = np.zeros(3)
    for f in aero_nodal.values():
        total_nodal_raw += f[:3]

    for comp in range(3):
        if abs(total_nodal_raw[comp]) > 1.0 and abs(total_panel[comp]) > 1.0:
            scale = total_panel[comp] / total_nodal_raw[comp]
            if abs(scale - 1.0) > 1e-6:
                logger.info("  Spline force conservation [%s]: "
                            "panel=%.1f, nodal=%.1f, scale=%.6f",
                            "XYZ"[comp], total_panel[comp],
                            total_nodal_raw[comp], scale)
                for nid in aero_nodal:
                    aero_nodal[nid][comp] *= scale

    total_aero = np.zeros(3)
    for f in aero_nodal.values():
        total_aero += f[:3]
    logger.info("  Nodal aero forces: Fx=%.2f, Fy=%.2f, Fz=%.2f",
                total_aero[0], total_aero[1], total_aero[2])

    # Inertial forces
    inertial_nodal = compute_nodal_inertial_forces(bdf_model, nz, g, ny=ny)
    total_inertial = np.zeros(3)
    for f in inertial_nodal.values():
        total_inertial += f[:3]
    logger.info("  Nodal inertial forces: Fx=%.2f, Fy=%.2f, Fz=%.2f",
                total_inertial[0], total_inertial[1], total_inertial[2])

    # Combined
    combined_nodal: Dict[int, np.ndarray] = {}
    all_nids = set(aero_nodal.keys()) | set(inertial_nodal.keys())
    for nid in all_nids:
        f_aero = aero_nodal.get(nid, np.zeros(6))
        f_inertia = inertial_nodal.get(nid, np.zeros(6))
        combined_nodal[nid] = f_aero + f_inertia

    total_combined = np.zeros(3)
    for f in combined_nodal.values():
        total_combined += f[:3]
    logger.info("  Combined forces: Fx=%.2f, Fy=%.2f, Fz=%.2f",
                total_combined[0], total_combined[1], total_combined[2])

    return aero_nodal, inertial_nodal, combined_nodal


def verify_trim_balance(
    bdf_model: BDFModel,
    combined_forces: Dict[int, np.ndarray],
    ref_point: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Check 6-DOF equilibrium of combined nodal forces.

    Computes the resultant force and moment about a reference point.
    For a properly trimmed condition, all 6 components should be near zero.

    Parameters
    ----------
    bdf_model : BDFModel
    combined_forces : Dict[int, ndarray(6)]
    ref_point : ndarray(3), optional
        Moment reference point. Defaults to CG.

    Returns
    -------
    balance : Dict[str, float]
        Keys: 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'
    """
    if ref_point is None:
        ref_point = np.zeros(3)

    total_force = np.zeros(3)
    total_moment = np.zeros(3)

    for nid, f in combined_forces.items():
        if nid not in bdf_model.nodes:
            continue
        pos = bdf_model.nodes[nid].xyz_global
        total_force += f[:3]
        # Moment about reference point
        r = pos - ref_point
        total_moment += np.cross(r, f[:3])
        # Add direct moments if any
        total_moment += f[3:6]

    balance = {
        'Fx': float(total_force[0]),
        'Fy': float(total_force[1]),
        'Fz': float(total_force[2]),
        'Mx': float(total_moment[0]),
        'My': float(total_moment[1]),
        'Mz': float(total_moment[2]),
    }

    logger.info("  Trim balance (6-DOF resultant):")
    logger.info("    Forces:  Fx=%+.4e  Fy=%+.4e  Fz=%+.4e",
                balance['Fx'], balance['Fy'], balance['Fz'])
    logger.info("    Moments: Mx=%+.4e  My=%+.4e  Mz=%+.4e",
                balance['Mx'], balance['My'], balance['Mz'])

    return balance


def write_force_cards(
    nodal_forces: Dict[int, np.ndarray],
    filepath: str,
    load_sid: int = 1,
    label: str = "COMBINED",
    cid: int = 0,
) -> None:
    """Write nodal forces in Nastran FORCE card format.

    Output format (fixed-8):
    FORCE   SID     G       CID     F       N1      N2      N3

    Parameters
    ----------
    nodal_forces : Dict[int, ndarray(6)]
    filepath : str
    load_sid : int
        Load set ID.
    label : str
        Comment label for the force set.
    cid : int
        Coordinate system ID (0 = basic).
    """
    with open(filepath, 'w') as f:
        f.write(f"$ {label} NODAL FORCES\n")
        f.write(f"$ Generated by NastAero SOL 144 Trim Loads Analysis\n")
        f.write("$\n")

        for nid in sorted(nodal_forces.keys()):
            fv = nodal_forces[nid]
            f_mag = np.linalg.norm(fv[:3])
            if f_mag < 1e-20:
                continue
            # Direction cosines
            n1, n2, n3 = fv[:3] / f_mag

            # Use Nastran fixed-16 format for precision
            f.write("FORCE*  %16d%16d%16d%16.8E\n" %
                    (load_sid, nid, cid, f_mag))
            f.write("*       %16.8E%16.8E%16.8E\n" %
                    (n1, n2, n3))

            # Write MOMENT card if rotational DOFs have values
            m_mag = np.linalg.norm(fv[3:6])
            if m_mag > 1e-20:
                mn1, mn2, mn3 = fv[3:6] / m_mag
                f.write("MOMENT* %16d%16d%16d%16.8E\n" %
                        (load_sid, nid, cid, m_mag))
                f.write("*       %16.8E%16.8E%16.8E\n" %
                        (mn1, mn2, mn3))

    logger.info("  FORCE cards written to: %s", filepath)
