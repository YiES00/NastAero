"""Global stiffness and mass matrix assembly.

Optimized with vectorized assembly for large-scale models.
"""
from __future__ import annotations
from typing import Dict, List
import time
import numpy as np
import scipy.sparse as sp
from .dof_manager import DOFManager
from ..bdf.model import BDFModel
from ..elements.bar import CBarElement
from ..elements.quad4 import CQuad4Element
from ..elements.tria3 import CTria3Element
from ..config import logger


def assemble_global_matrices(model: BDFModel, dof_mgr: DOFManager):
    """Assemble global K and M matrices using vectorized operations.

    Strategy:
    1. Group elements by type
    2. Batch-compute element stiffness/mass matrices
    3. Vectorized COO triplet construction (no Python inner loops)
    """
    ndof = dof_mgr.total_dof
    t_start = time.perf_counter()

    # Collect element data by type
    cbar_elems = []
    cbeam_elems = []
    cquad4_elems = []
    ctria3_elems = []
    crod_elems = []

    for eid, elem in model.elements.items():
        if elem.type == "CBAR":
            cbar_elems.append((eid, elem))
        elif elem.type == "CBEAM":
            cbeam_elems.append((eid, elem))
        elif elem.type == "CQUAD4":
            cquad4_elems.append((eid, elem))
        elif elem.type == "CTRIA3":
            ctria3_elems.append((eid, elem))
        elif elem.type == "CROD":
            crod_elems.append((eid, elem))

    n_total = (len(cbar_elems) + len(cbeam_elems) + len(cquad4_elems) +
               len(ctria3_elems) + len(crod_elems))
    logger.info("Assembly: %d CBAR, %d CBEAM, %d CQUAD4, %d CTRIA3, %d CROD = %d total elements",
                len(cbar_elems), len(cbeam_elems), len(cquad4_elems),
                len(ctria3_elems), len(crod_elems), n_total)

    # Pre-allocate COO arrays with known sizes
    # CBAR/CBEAM: 12x12 = 144 entries per element
    # CQUAD4: 24x24 = 576 entries per element
    # CTRIA3: 18x18 = 324 entries per element
    # CROD: 12x12 = 144 entries per element
    max_nnz = ((len(cbar_elems) + len(cbeam_elems)) * 144 +
               len(cquad4_elems) * 576 +
               len(ctria3_elems) * 324 + len(crod_elems) * 144)

    rows_k = np.empty(max_nnz, dtype=np.int64)
    cols_k = np.empty(max_nnz, dtype=np.int64)
    vals_k = np.empty(max_nnz, dtype=np.float64)
    rows_m = np.empty(max_nnz, dtype=np.int64)
    cols_m = np.empty(max_nnz, dtype=np.int64)
    vals_m = np.empty(max_nnz, dtype=np.float64)

    ptr_k = 0  # current position in K arrays
    ptr_m = 0  # current position in M arrays
    n_assembled = 0

    # --- Batch process CQUAD4 elements (typically the largest group) ---
    if cquad4_elems:
        t_q = time.perf_counter()
        pk, pm = _assemble_cquad4_batch(cquad4_elems, model, dof_mgr,
                                         rows_k, cols_k, vals_k,
                                         rows_m, cols_m, vals_m,
                                         ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(cquad4_elems)
        logger.info("  CQUAD4 batch: %.3f s (%d elements)", time.perf_counter() - t_q, len(cquad4_elems))

    # --- Batch process CTRIA3 elements ---
    if ctria3_elems:
        t_t = time.perf_counter()
        pk, pm = _assemble_ctria3_batch(ctria3_elems, model, dof_mgr,
                                         rows_k, cols_k, vals_k,
                                         rows_m, cols_m, vals_m,
                                         ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(ctria3_elems)
        logger.info("  CTRIA3 batch: %.3f s (%d elements)", time.perf_counter() - t_t, len(ctria3_elems))

    # --- Batch process CBAR elements ---
    if cbar_elems:
        t_b = time.perf_counter()
        pk, pm = _assemble_cbar_batch(cbar_elems, model, dof_mgr,
                                       rows_k, cols_k, vals_k,
                                       rows_m, cols_m, vals_m,
                                       ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(cbar_elems)
        logger.info("  CBAR batch: %.3f s (%d elements)", time.perf_counter() - t_b, len(cbar_elems))

    # --- Batch process CBEAM elements (reuse CBAR logic) ---
    if cbeam_elems:
        t_bm = time.perf_counter()
        pk, pm = _assemble_cbar_batch(cbeam_elems, model, dof_mgr,
                                       rows_k, cols_k, vals_k,
                                       rows_m, cols_m, vals_m,
                                       ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(cbeam_elems)
        logger.info("  CBEAM batch: %.3f s (%d elements)", time.perf_counter() - t_bm, len(cbeam_elems))

    # --- Batch process CROD elements ---
    if crod_elems:
        t_r = time.perf_counter()
        pk, pm = _assemble_crod_batch(crod_elems, model, dof_mgr,
                                       rows_k, cols_k, vals_k,
                                       rows_m, cols_m, vals_m,
                                       ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(crod_elems)
        logger.info("  CROD batch: %.3f s (%d elements)", time.perf_counter() - t_r, len(crod_elems))

    # Trim arrays to actual size
    rows_k = rows_k[:ptr_k]
    cols_k = cols_k[:ptr_k]
    vals_k = vals_k[:ptr_k]
    rows_m = rows_m[:ptr_m]
    cols_m = cols_m[:ptr_m]
    vals_m = vals_m[:ptr_m]

    # --- Concentrated masses (CONM2) ---
    conm2_rows, conm2_cols, conm2_vals = [], [], []
    for mid, mass_elem in model.masses.items():
        nid = mass_elem.node_id
        if nid not in dof_mgr._nid_to_index:
            continue
        node_dofs = dof_mgr.get_node_dofs(nid)
        for i in range(3):
            conm2_rows.append(node_dofs[i])
            conm2_cols.append(node_dofs[i])
            conm2_vals.append(mass_elem.mass)
        if mass_elem.I11 > 0:
            conm2_rows.append(node_dofs[3]); conm2_cols.append(node_dofs[3]); conm2_vals.append(mass_elem.I11)
        if mass_elem.I22 > 0:
            conm2_rows.append(node_dofs[4]); conm2_cols.append(node_dofs[4]); conm2_vals.append(mass_elem.I22)
        if mass_elem.I33 > 0:
            conm2_rows.append(node_dofs[5]); conm2_cols.append(node_dofs[5]); conm2_vals.append(mass_elem.I33)

    if conm2_rows:
        rows_m = np.concatenate([rows_m, np.array(conm2_rows, dtype=np.int64)])
        cols_m = np.concatenate([cols_m, np.array(conm2_cols, dtype=np.int64)])
        vals_m = np.concatenate([vals_m, np.array(conm2_vals, dtype=np.float64)])

    # --- Step 1: Build K and M from elements (BASIC frame) ---
    t_coo = time.perf_counter()
    K = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(ndof, ndof)).tocsc()
    M = sp.coo_matrix((vals_m, (rows_m, cols_m)), shape=(ndof, ndof)).tocsc()
    t_sparse = time.perf_counter() - t_coo

    # --- Step 2: Apply CD (displacement coordinate system) transforms ---
    # Must be done BEFORE adding RBE2/MPC/CELAS penalties because those
    # reference DOF components in the CD frame, not BASIC frame.
    K, M = _apply_cd_transforms(K, M, model, dof_mgr)

    # --- Step 3: Add constraint/spring penalties (in CD-frame DOFs) ---

    # --- RBE2 rigid elements (penalty approach) ---
    rbe2_rows, rbe2_cols, rbe2_vals = [], [], []
    for rid, rbe in model.rigids.items():
        if rbe.type == "RBE2":
            _assemble_rbe2(rbe, model, dof_mgr, rbe2_rows, rbe2_cols, rbe2_vals)

    if rbe2_rows:
        rbe2_K = sp.coo_matrix((rbe2_vals, (rbe2_rows, rbe2_cols)),
                                shape=(ndof, ndof)).tocsc()
        K = K + rbe2_K

    # --- Spring elements (CELAS1/CELAS2) ---
    spring_rows, spring_cols, spring_vals = [], [], []
    for sid, spring in model.springs.items():
        _assemble_spring(spring, model, dof_mgr, spring_rows, spring_cols, spring_vals)

    if spring_rows:
        spring_K = sp.coo_matrix((spring_vals, (spring_rows, spring_cols)),
                                  shape=(ndof, ndof)).tocsc()
        K = K + spring_K
        logger.info("  Assembled %d spring elements", len(model.springs))

    # --- MPC constraints (penalty method) ---
    mpc_rows, mpc_cols, mpc_vals = [], [], []
    for mpc_sid, mpc_list in model.mpcs.items():
        for mpc in mpc_list:
            _assemble_mpc(mpc, dof_mgr, mpc_rows, mpc_cols, mpc_vals)

    if mpc_rows:
        mpc_K = sp.coo_matrix((mpc_vals, (mpc_rows, mpc_cols)),
                               shape=(ndof, ndof)).tocsc()
        K = K + mpc_K
        n_mpc_total = sum(len(v) for v in model.mpcs.values())
        logger.info("  Assembled %d MPC constraints (penalty method)", n_mpc_total)

    t_total = time.perf_counter() - t_start
    logger.info("Assembled %d elements into global matrices (%d DOFs) in %.3f s (sparse convert: %.3f s)",
                n_assembled, ndof, t_total, t_sparse)
    return K, M


# ============================================================
# CD (Displacement Coordinate System) transformation
# ============================================================

def _apply_cd_transforms(K, M, model: BDFModel, dof_mgr: DOFManager):
    """Apply CD coordinate system rotations to global K and M.

    In Nastran, when a GRID has CD != 0, the DOFs at that node are defined
    in the CD coordinate system rather than BASIC. Element stiffness matrices
    are computed in BASIC, so we must rotate them to align with the CD frame.

    The transformation is: K_cd = T^T @ K_basic @ T, M_cd = T^T @ M_basic @ T
    where T is a block-diagonal orthogonal matrix with 3x3 rotation blocks
    for each node with CD != 0 (identity for CD = 0 nodes).

    Since T is orthogonal and very sparse (only a few nodes have CD != 0),
    we build T as a sparse matrix and apply the similarity transform.
    """
    # Collect nodes with non-zero CD
    cd_nodes = []
    for nid in dof_mgr.node_ids:
        if nid in model.nodes:
            grid = model.nodes[nid]
            if grid.cd != 0 and grid.cd in model.coords:
                cd_nodes.append((nid, grid.cd))

    if not cd_nodes:
        return K, M

    logger.info("  Applying CD transforms for %d nodes", len(cd_nodes))

    # Build sparse transformation matrix T (ndof x ndof)
    # T is identity everywhere except at CD != 0 nodes where it has
    # the 3x3 rotation blocks on the diagonal
    ndof = dof_mgr.total_dof

    # Start with identity - build as COO for efficiency
    # We only need to modify the 6x6 blocks for CD != 0 nodes
    # For CD node: replace the 6x6 identity block with [[R, 0], [0, R]]
    t_rows = []
    t_cols = []
    t_vals = []

    # Set of DOFs that belong to CD != 0 nodes
    cd_dof_set = set()
    for nid, cd in cd_nodes:
        node_dofs = dof_mgr.get_node_dofs(nid)
        cd_dof_set.update(node_dofs)

        R = model.coords[cd].transform  # 3x3 rotation: BASIC → CD local
        # T transforms from CD-local DOFs to BASIC DOFs: u_basic = T @ u_cd
        # So T's columns at this node contain the CD basis vectors
        # T = [[R, 0], [0, R]] for this node's 6 DOFs
        for block in range(2):  # translations (0-2) and rotations (3-5)
            for i in range(3):
                for j in range(3):
                    row = node_dofs[block * 3 + i]
                    col = node_dofs[block * 3 + j]
                    t_rows.append(row)
                    t_cols.append(col)
                    t_vals.append(R[i, j])

    # Add identity entries for all other DOFs
    for dof in range(ndof):
        if dof not in cd_dof_set:
            t_rows.append(dof)
            t_cols.append(dof)
            t_vals.append(1.0)

    T = sp.coo_matrix((t_vals, (t_rows, t_cols)), shape=(ndof, ndof)).tocsc()

    # Apply transformation: K_new = T^T @ K @ T, M_new = T^T @ M @ T
    t_start = time.perf_counter()
    K_new = T.T @ K @ T
    M_new = T.T @ M @ T
    logger.info("  CD transform applied in %.3f s", time.perf_counter() - t_start)

    return K_new.tocsc(), M_new.tocsc()


# ============================================================
# Vectorized batch assembly for CQUAD4
# ============================================================

def _assemble_cquad4_batch(elems, model, dof_mgr,
                            rows_k, cols_k, vals_k,
                            rows_m, cols_m, vals_m,
                            ptr_k, ptr_m):
    """Batch assemble CQUAD4 elements with vectorized COO construction."""
    n_elem = len(elems)
    ndof_e = 24

    # Pre-compute all element DOF indices as a (n_elem, 24) array
    all_edofs = np.empty((n_elem, ndof_e), dtype=np.int64)
    for idx, (eid, elem) in enumerate(elems):
        all_edofs[idx, :] = dof_mgr.get_element_dofs(elem.node_ids)

    # Create the (i,j) index pairs for a 24x24 matrix (576 pairs)
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()  # (576,)
    jj_flat = jj_local.ravel()  # (576,)

    # For each element, compute ke, me and fill COO arrays
    for idx, (eid, elem) in enumerate(elems):
        try:
            prop = elem.property_ref
            # PCOMP: use equivalent isotropic properties
            if hasattr(prop, 'equivalent_isotropic'):
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = prop.material_ref
                E = mat.E; nu = mat.nu; t = prop.t; rho = mat.rho
            node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            q = CQuad4Element(node_xyz, E, nu, t, rho)
            ke = q.stiffness_matrix()
            me = q.mass_matrix()
        except Exception as exc:
            logger.warning("Error assembling CQUAD4 %d: %s", eid, exc)
            continue

        edofs = all_edofs[idx]
        # Vectorized COO fill: map local indices to global DOFs
        global_rows = edofs[ii_flat]  # (576,)
        global_cols = edofs[jj_flat]  # (576,)
        ke_flat = ke.ravel()
        me_flat = me.ravel()

        # Filter near-zero entries for K
        mask_k = np.abs(ke_flat) > 1e-30
        nk = mask_k.sum()
        rows_k[ptr_k:ptr_k+nk] = global_rows[mask_k]
        cols_k[ptr_k:ptr_k+nk] = global_cols[mask_k]
        vals_k[ptr_k:ptr_k+nk] = ke_flat[mask_k]
        ptr_k += nk

        # Filter near-zero entries for M
        mask_m = np.abs(me_flat) > 1e-30
        nm = mask_m.sum()
        rows_m[ptr_m:ptr_m+nm] = global_rows[mask_m]
        cols_m[ptr_m:ptr_m+nm] = global_cols[mask_m]
        vals_m[ptr_m:ptr_m+nm] = me_flat[mask_m]
        ptr_m += nm

    return ptr_k, ptr_m


def _assemble_ctria3_batch(elems, model, dof_mgr,
                            rows_k, cols_k, vals_k,
                            rows_m, cols_m, vals_m,
                            ptr_k, ptr_m):
    """Batch assemble CTRIA3 elements with vectorized COO construction."""
    ndof_e = 18
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()
    jj_flat = jj_local.ravel()

    for idx, (eid, elem) in enumerate(elems):
        try:
            prop = elem.property_ref
            # PCOMP: use equivalent isotropic properties
            if hasattr(prop, 'equivalent_isotropic'):
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = prop.material_ref
                E = mat.E; nu = mat.nu; t = prop.t; rho = mat.rho
            node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            tri = CTria3Element(node_xyz, E, nu, t, rho)
            ke = tri.stiffness_matrix()
            me = tri.mass_matrix()
        except Exception as exc:
            logger.warning("Error assembling CTRIA3 %d: %s", eid, exc)
            continue

        edofs = np.array(dof_mgr.get_element_dofs(elem.node_ids), dtype=np.int64)
        global_rows = edofs[ii_flat]
        global_cols = edofs[jj_flat]
        ke_flat = ke.ravel()
        me_flat = me.ravel()

        mask_k = np.abs(ke_flat) > 1e-30
        nk = mask_k.sum()
        rows_k[ptr_k:ptr_k+nk] = global_rows[mask_k]
        cols_k[ptr_k:ptr_k+nk] = global_cols[mask_k]
        vals_k[ptr_k:ptr_k+nk] = ke_flat[mask_k]
        ptr_k += nk

        mask_m = np.abs(me_flat) > 1e-30
        nm = mask_m.sum()
        rows_m[ptr_m:ptr_m+nm] = global_rows[mask_m]
        cols_m[ptr_m:ptr_m+nm] = global_cols[mask_m]
        vals_m[ptr_m:ptr_m+nm] = me_flat[mask_m]
        ptr_m += nm

    return ptr_k, ptr_m


def _assemble_cbar_batch(elems, model, dof_mgr,
                          rows_k, cols_k, vals_k,
                          rows_m, cols_m, vals_m,
                          ptr_k, ptr_m):
    """Batch assemble CBAR elements with vectorized COO construction."""
    ndof_e = 12
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()
    jj_flat = jj_local.ravel()

    for idx, (eid, elem) in enumerate(elems):
        try:
            prop = elem.property_ref
            mat = prop.material_ref
            n1 = model.nodes[elem.node_ids[0]]
            n2 = model.nodes[elem.node_ids[1]]
            if elem.g0 > 0 and elem.g0 in model.nodes:
                v_vector = model.nodes[elem.g0].xyz_global - n1.xyz_global
            else:
                v_vector = elem.x.copy()
                if np.linalg.norm(v_vector) < 1e-12:
                    v_vector = np.array([0., 0., 1.])
            bar = CBarElement(n1.xyz_global, n2.xyz_global, v_vector,
                              mat.E, mat.G, prop.A, prop.I1, prop.I2, prop.J,
                              mat.rho, prop.nsm)
            ke = bar.stiffness_matrix()
            me = bar.mass_matrix()
        except Exception as exc:
            logger.warning("Error assembling CBAR %d: %s", eid, exc)
            continue

        edofs = np.array(dof_mgr.get_element_dofs(elem.node_ids), dtype=np.int64)
        global_rows = edofs[ii_flat]
        global_cols = edofs[jj_flat]
        ke_flat = ke.ravel()
        me_flat = me.ravel()

        mask_k = np.abs(ke_flat) > 1e-30
        nk = mask_k.sum()
        rows_k[ptr_k:ptr_k+nk] = global_rows[mask_k]
        cols_k[ptr_k:ptr_k+nk] = global_cols[mask_k]
        vals_k[ptr_k:ptr_k+nk] = ke_flat[mask_k]
        ptr_k += nk

        mask_m = np.abs(me_flat) > 1e-30
        nm = mask_m.sum()
        rows_m[ptr_m:ptr_m+nm] = global_rows[mask_m]
        cols_m[ptr_m:ptr_m+nm] = global_cols[mask_m]
        vals_m[ptr_m:ptr_m+nm] = me_flat[mask_m]
        ptr_m += nm

    return ptr_k, ptr_m


def _assemble_crod_batch(elems, model, dof_mgr,
                          rows_k, cols_k, vals_k,
                          rows_m, cols_m, vals_m,
                          ptr_k, ptr_m):
    """Batch assemble CROD elements."""
    ndof_e = 12
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()
    jj_flat = jj_local.ravel()

    for idx, (eid, elem) in enumerate(elems):
        try:
            prop = elem.property_ref
            mat = prop.material_ref
            n1 = model.nodes[elem.node_ids[0]]
            n2 = model.nodes[elem.node_ids[1]]
            diff = n2.xyz_global - n1.xyz_global
            L = np.linalg.norm(diff)
            if L < 1e-12:
                raise ValueError(f"Zero-length CROD {elem.eid}")
            ex = diff / L
            ea_l = mat.E * prop.A / L
            ke = np.zeros((12, 12))
            me = np.zeros((12, 12))
            ex_out = np.outer(ex, ex)
            ke[0:3, 0:3] = ea_l * ex_out
            ke[0:3, 6:9] = -ea_l * ex_out
            ke[6:9, 0:3] = -ea_l * ex_out
            ke[6:9, 6:9] = ea_l * ex_out
            m_half = (mat.rho * prop.A * L) / 2.0
            for i in range(3):
                me[i, i] = m_half
                me[i + 6, i + 6] = m_half
        except Exception as exc:
            logger.warning("Error assembling CROD %d: %s", eid, exc)
            continue

        edofs = np.array(dof_mgr.get_element_dofs(elem.node_ids), dtype=np.int64)
        global_rows = edofs[ii_flat]
        global_cols = edofs[jj_flat]
        ke_flat = ke.ravel()
        me_flat = me.ravel()

        mask_k = np.abs(ke_flat) > 1e-30
        nk = mask_k.sum()
        rows_k[ptr_k:ptr_k+nk] = global_rows[mask_k]
        cols_k[ptr_k:ptr_k+nk] = global_cols[mask_k]
        vals_k[ptr_k:ptr_k+nk] = ke_flat[mask_k]
        ptr_k += nk

        mask_m = np.abs(me_flat) > 1e-30
        nm = mask_m.sum()
        rows_m[ptr_m:ptr_m+nm] = global_rows[mask_m]
        cols_m[ptr_m:ptr_m+nm] = global_cols[mask_m]
        vals_m[ptr_m:ptr_m+nm] = me_flat[mask_m]
        ptr_m += nm

    return ptr_k, ptr_m


def _assemble_rbe2(rbe, model, dof_mgr, rows_k, cols_k, vals_k):
    """Assemble RBE2 using penalty method with rigid body kinematics.

    The RBE2 constraint for dependent node d relative to independent node i:
      u_d = u_i + theta_i × r   (translations, r = xyz_dep - xyz_ind)
      theta_d = theta_i          (rotations)

    In component form for translations:
      u_d1 - u_i1 - theta_i2*r3 + theta_i3*r2 = 0
      u_d2 - u_i2 - theta_i3*r1 + theta_i1*r3 = 0
      u_d3 - u_i3 - theta_i1*r2 + theta_i2*r1 = 0

    For rotations (same DOF coupling):
      theta_d4 - theta_i4 = 0
      theta_d5 - theta_i5 = 0
      theta_d6 - theta_i6 = 0

    Each constraint g(u)=0 is enforced via penalty: K += penalty * c * c^T
    where c is the constraint coefficient vector.
    """
    penalty = 1e12
    ind_nid = rbe.independent_node
    if ind_nid not in dof_mgr._nid_to_index:
        return
    if ind_nid not in model.nodes:
        return

    ind_xyz = model.nodes[ind_nid].xyz_global

    # Parse which DOF components are constrained
    cm_set = set()
    for ch in rbe.components:
        comp = int(ch)
        if 1 <= comp <= 6:
            cm_set.add(comp)

    # Check if we need rotation coupling for translations
    has_trans = bool(cm_set & {1, 2, 3})
    has_rot = bool(cm_set & {4, 5, 6})

    for dep_nid in rbe.dependent_nodes:
        if dep_nid not in dof_mgr._nid_to_index:
            continue
        if dep_nid not in model.nodes:
            continue

        dep_xyz = model.nodes[dep_nid].xyz_global
        r = dep_xyz - ind_xyz  # offset vector

        # Get DOF indices for independent and dependent nodes
        ind_dofs = dof_mgr.get_node_dofs(ind_nid)  # [u1,u2,u3,r1,r2,r3]
        dep_dofs = dof_mgr.get_node_dofs(dep_nid)

        # --- Translational constraints with rigid body kinematics ---
        if has_trans:
            offset_mag = np.linalg.norm(r)

            if offset_mag < 1e-10:
                # Coincident nodes: simple coupling (no rotation coupling needed)
                for comp in [1, 2, 3]:
                    if comp not in cm_set:
                        continue
                    ind_dof = ind_dofs[comp - 1]
                    dep_dof = dep_dofs[comp - 1]
                    rows_k.extend([ind_dof, ind_dof, dep_dof, dep_dof])
                    cols_k.extend([ind_dof, dep_dof, ind_dof, dep_dof])
                    vals_k.extend([penalty, -penalty, -penalty, penalty])
            else:
                # Non-coincident: full rigid body coupling
                # Constraint for comp 1 (X-translation):
                #   u_d1 - u_i1 - theta_i2*r3 + theta_i3*r2 = 0
                # Constraint vector c = [dep_dof1: +1, ind_dof1: -1,
                #                        ind_dof5: -r3, ind_dof6: +r2]
                #
                # Constraint for comp 2 (Y-translation):
                #   u_d2 - u_i2 - theta_i3*r1 + theta_i1*r3 = 0
                # c = [dep_dof2: +1, ind_dof2: -1, ind_dof6: -r1, ind_dof4: +r3]
                #
                # Constraint for comp 3 (Z-translation):
                #   u_d3 - u_i3 - theta_i1*r2 + theta_i2*r1 = 0
                # c = [dep_dof3: +1, ind_dof3: -1, ind_dof4: -r2, ind_dof5: +r1]

                # Build constraint coefficient vectors for each constrained
                # translation DOF. Each constraint: c^T u = 0
                # K += penalty * c * c^T (outer product)

                # Cross product mapping: theta × r
                # (theta1, theta2, theta3) × (r1, r2, r3) =
                #   (theta2*r3 - theta3*r2, theta3*r1 - theta1*r3, theta1*r2 - theta2*r1)

                for comp in [1, 2, 3]:
                    if comp not in cm_set:
                        continue

                    # Build constraint: dep_trans - ind_trans - (theta × r)_comp = 0
                    # c is a list of (global_dof, coefficient) pairs
                    c_terms = []
                    c_terms.append((dep_dofs[comp - 1], 1.0))   # +u_dep
                    c_terms.append((ind_dofs[comp - 1], -1.0))  # -u_ind

                    # Cross product contributions from independent rotations
                    if comp == 1:
                        # -(theta2*r3) + (theta3*r2)
                        if abs(r[2]) > 1e-15:
                            c_terms.append((ind_dofs[4], -r[2]))  # -theta_i2 * r3
                        if abs(r[1]) > 1e-15:
                            c_terms.append((ind_dofs[5], r[1]))   # +theta_i3 * r2
                    elif comp == 2:
                        # -(theta3*r1) + (theta1*r3)
                        if abs(r[0]) > 1e-15:
                            c_terms.append((ind_dofs[5], -r[0]))  # -theta_i3 * r1
                        if abs(r[2]) > 1e-15:
                            c_terms.append((ind_dofs[3], r[2]))   # +theta_i1 * r3
                    elif comp == 3:
                        # -(theta1*r2) + (theta2*r1)
                        if abs(r[1]) > 1e-15:
                            c_terms.append((ind_dofs[3], -r[1]))  # -theta_i1 * r2
                        if abs(r[0]) > 1e-15:
                            c_terms.append((ind_dofs[4], r[0]))   # +theta_i2 * r1

                    # Add penalty contribution: K += penalty * c * c^T
                    for dof_i, ci in c_terms:
                        for dof_j, cj in c_terms:
                            rows_k.append(dof_i)
                            cols_k.append(dof_j)
                            vals_k.append(penalty * ci * cj)

        # --- Rotational constraints (simple coupling) ---
        for comp in [4, 5, 6]:
            if comp not in cm_set:
                continue
            ind_dof = ind_dofs[comp - 1]
            dep_dof = dep_dofs[comp - 1]
            rows_k.extend([ind_dof, ind_dof, dep_dof, dep_dof])
            cols_k.extend([ind_dof, dep_dof, ind_dof, dep_dof])
            vals_k.extend([penalty, -penalty, -penalty, penalty])


def _assemble_spring(spring, model, dof_mgr, rows_k, cols_k, vals_k):
    """Assemble CELAS1/CELAS2 scalar spring element."""
    # Get spring stiffness
    if spring.type == "CELAS2":
        k = spring.k
    elif spring.type == "CELAS1":
        if hasattr(spring, 'property_ref') and spring.property_ref is not None:
            k = spring.property_ref.k
        else:
            return
    else:
        return

    if abs(k) < 1e-30:
        return

    g1 = spring.g1
    c1 = spring.c1
    g2 = spring.g2
    c2 = spring.c2

    if g1 > 0 and g1 in dof_mgr._nid_to_index:
        dof1 = dof_mgr.get_dof(g1, c1)
    else:
        dof1 = None

    if g2 > 0 and g2 in dof_mgr._nid_to_index:
        dof2 = dof_mgr.get_dof(g2, c2)
    else:
        dof2 = None

    if dof1 is not None and dof2 is not None:
        # Two-node spring
        rows_k.extend([dof1, dof1, dof2, dof2])
        cols_k.extend([dof1, dof2, dof1, dof2])
        vals_k.extend([k, -k, -k, k])
    elif dof1 is not None:
        # Grounded spring (g2 = 0)
        rows_k.append(dof1)
        cols_k.append(dof1)
        vals_k.append(k)
    elif dof2 is not None:
        # Grounded spring (g1 = 0)
        rows_k.append(dof2)
        cols_k.append(dof2)
        vals_k.append(k)


def _assemble_mpc(mpc, dof_mgr, rows_k, cols_k, vals_k):
    """Assemble MPC constraint using penalty method.

    MPC equation: sum(Ai * ui_ci) = 0
    Penalty: K += penalty * (a * a^T) where a is the coefficient vector
    mapped to global DOFs.
    """
    penalty = 1e12

    # Build DOF-coefficient pairs
    dof_coeff = []
    for nid, comp, coeff in mpc.terms:
        if nid in dof_mgr._nid_to_index and abs(coeff) > 1e-30:
            dof = dof_mgr.get_dof(nid, comp)
            dof_coeff.append((dof, coeff))

    if len(dof_coeff) < 2:
        return

    # Add penalty contribution: K += penalty * a * a^T
    for i, (dof_i, ai) in enumerate(dof_coeff):
        for j, (dof_j, aj) in enumerate(dof_coeff):
            rows_k.append(dof_i)
            cols_k.append(dof_j)
            vals_k.append(penalty * ai * aj)
