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
from ..elements.quad8 import CQuad8Element
from ..elements.tria6 import CTria6Element
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
    cquad8_elems = []
    ctria6_elems = []

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
        elif elem.type == "CQUAD8":
            cquad8_elems.append((eid, elem))
        elif elem.type == "CTRIA6":
            ctria6_elems.append((eid, elem))

    n_total = (len(cbar_elems) + len(cbeam_elems) + len(cquad4_elems) +
               len(ctria3_elems) + len(crod_elems) +
               len(cquad8_elems) + len(ctria6_elems))
    logger.info("Assembly: %d CBAR, %d CBEAM, %d CQUAD4, %d CTRIA3, %d CROD, "
                "%d CQUAD8, %d CTRIA6 = %d total elements",
                len(cbar_elems), len(cbeam_elems), len(cquad4_elems),
                len(ctria3_elems), len(crod_elems),
                len(cquad8_elems), len(ctria6_elems), n_total)

    # Pre-allocate COO arrays with known sizes
    # CBAR/CBEAM: 12x12 = 144 entries per element
    # CQUAD4: 24x24 = 576 entries per element
    # CTRIA3: 18x18 = 324 entries per element
    # CROD: 12x12 = 144 entries per element
    # CQUAD8: 48x48 = 2304 entries per element
    # CTRIA6: 36x36 = 1296 entries per element
    max_nnz = ((len(cbar_elems) + len(cbeam_elems)) * 144 +
               len(cquad4_elems) * 576 +
               len(ctria3_elems) * 324 + len(crod_elems) * 144 +
               len(cquad8_elems) * 2304 + len(ctria6_elems) * 1296)

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

    # --- Batch process CQUAD8 elements ---
    if cquad8_elems:
        t_q8 = time.perf_counter()
        pk, pm = _assemble_cquad8_batch(cquad8_elems, model, dof_mgr,
                                         rows_k, cols_k, vals_k,
                                         rows_m, cols_m, vals_m,
                                         ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(cquad8_elems)
        logger.info("  CQUAD8 batch: %.3f s (%d elements)", time.perf_counter() - t_q8, len(cquad8_elems))

    # --- Batch process CTRIA6 elements ---
    if ctria6_elems:
        t_t6 = time.perf_counter()
        pk, pm = _assemble_ctria6_batch(ctria6_elems, model, dof_mgr,
                                         rows_k, cols_k, vals_k,
                                         rows_m, cols_m, vals_m,
                                         ptr_k, ptr_m)
        ptr_k = pk
        ptr_m = pm
        n_assembled += len(ctria6_elems)
        logger.info("  CTRIA6 batch: %.3f s (%d elements)", time.perf_counter() - t_t6, len(ctria6_elems))

    # Trim arrays to actual size
    rows_k = rows_k[:ptr_k]
    cols_k = cols_k[:ptr_k]
    vals_k = vals_k[:ptr_k]
    rows_m = rows_m[:ptr_m]
    cols_m = cols_m[:ptr_m]
    vals_m = vals_m[:ptr_m]

    # --- Concentrated masses (CONM2) ---
    # Full 6x6 mass matrix at grid point including:
    # - Translational mass (3x3 diagonal)
    # - Full 3x3 symmetric inertia tensor (with off-diagonal terms)
    # - Offset handling via parallel axis theorem
    # - Translation-rotation coupling from offset
    # - CID coordinate transform for offset and inertia
    conm2_rows, conm2_cols, conm2_vals = [], [], []
    for mid, mass_elem in model.masses.items():
        nid = mass_elem.node_id
        if nid not in dof_mgr._nid_to_index:
            continue
        node_dofs = dof_mgr.get_node_dofs(nid)
        m = mass_elem.mass
        offset = mass_elem.offset.copy()

        # Inertia tensor at CG (symmetric, Nastran lower-triangular convention):
        #   [[I11, I21, I31],
        #    [I21, I22, I32],
        #    [I31, I32, I33]]
        I_cg = np.array([[mass_elem.I11, mass_elem.I21, mass_elem.I31],
                         [mass_elem.I21, mass_elem.I22, mass_elem.I32],
                         [mass_elem.I31, mass_elem.I32, mass_elem.I33]])

        # CID coordinate transform: rotate offset and inertia to basic
        cid = mass_elem.cid
        if cid > 0 and cid in model.coords:
            R = model.coords[cid].transform  # 3x3 rotation
            offset = R @ offset
            I_cg = R @ I_cg @ R.T

        # Translational mass
        for i in range(3):
            conm2_rows.append(node_dofs[i])
            conm2_cols.append(node_dofs[i])
            conm2_vals.append(m)

        # Parallel axis theorem: I_node = I_cg + m*(r·r*I - r⊗r)
        r = offset
        r_sq = np.dot(r, r)
        I_node = I_cg + m * (r_sq * np.eye(3) - np.outer(r, r))

        # Rotational inertia (full symmetric 3x3)
        for i in range(3):
            for j in range(3):
                val = I_node[i, j]
                if abs(val) > 1e-30:
                    conm2_rows.append(node_dofs[3 + i])
                    conm2_cols.append(node_dofs[3 + j])
                    conm2_vals.append(val)

        # Translation-rotation coupling from offset: M_tr = m * skew(r)
        # M[trans, rot] = m * skew(r), M[rot, trans] = -m * skew(r) = (m*skew(r))^T
        # skew(r) = [[0, -r3, r2], [r3, 0, -r1], [-r2, r1, 0]]
        if m > 0 and np.linalg.norm(r) > 1e-15:
            S = np.array([[0, -r[2], r[1]],
                          [r[2], 0, -r[0]],
                          [-r[1], r[0], 0]])
            mS = m * S
            for i in range(3):
                for j in range(3):
                    val = mS[i, j]
                    if abs(val) > 1e-30:
                        # Upper-right block: trans-rot
                        conm2_rows.append(node_dofs[i])
                        conm2_cols.append(node_dofs[3 + j])
                        conm2_vals.append(val)
                        # Lower-left block: rot-trans (transpose)
                        conm2_rows.append(node_dofs[3 + j])
                        conm2_cols.append(node_dofs[i])
                        conm2_vals.append(val)

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

# 2x2 Gauss quadrature
_GP2 = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])


def _assemble_cquad4_batch(elems, model, dof_mgr,
                            rows_k, cols_k, vals_k,
                            rows_m, cols_m, vals_m,
                            ptr_k, ptr_m):
    """Fully vectorized CQUAD4 assembly — all elements computed in batch.

    Computes stiffness and mass matrices for all CQUAD4 elements simultaneously
    using NumPy broadcasting and einsum, with no Python per-element loops for
    the numerical computation.
    """
    n_elem = len(elems)
    ndof_e = 24

    # --- Collect element data into contiguous arrays ---
    all_edofs = np.empty((n_elem, ndof_e), dtype=np.int64)
    all_xyz = np.empty((n_elem, 4, 3))
    all_E = np.empty(n_elem)
    all_nu = np.empty(n_elem)
    all_t = np.empty(n_elem)
    all_rho = np.empty(n_elem)
    valid = np.ones(n_elem, dtype=bool)

    for idx, (eid, elem) in enumerate(elems):
        try:
            all_edofs[idx, :] = dof_mgr.get_element_dofs(elem.node_ids)
            for k, nid in enumerate(elem.node_ids):
                all_xyz[idx, k] = model.nodes[nid].xyz_global
            prop = elem.property_ref
            if hasattr(prop, 'equivalent_isotropic'):
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = prop.material_ref
                E = mat.E; nu = mat.nu; t = prop.t; rho = mat.rho
            all_E[idx] = E; all_nu[idx] = nu; all_t[idx] = t; all_rho[idx] = rho
        except Exception as exc:
            logger.warning("Error collecting CQUAD4 %d: %s", eid, exc)
            valid[idx] = False

    # Filter valid elements
    if not np.all(valid):
        mask = valid
        all_edofs = all_edofs[mask]
        all_xyz = all_xyz[mask]
        all_E = all_E[mask]; all_nu = all_nu[mask]
        all_t = all_t[mask]; all_rho = all_rho[mask]
        n_elem = int(mask.sum())

    if n_elem == 0:
        return ptr_k, ptr_m

    # --- Build local coordinate systems (vectorized) ---
    # ne = n_elem
    p = all_xyz  # (ne, 4, 3)
    center = p.mean(axis=1)  # (ne, 3)
    d13 = p[:, 2] - p[:, 0]  # (ne, 3)
    d24 = p[:, 3] - p[:, 1]  # (ne, 3)
    ez = np.cross(d13, d24)   # (ne, 3)
    ez_norm = np.linalg.norm(ez, axis=1, keepdims=True)  # (ne, 1)
    ez = ez / np.maximum(ez_norm, 1e-30)

    ex = p[:, 1] - p[:, 0]  # (ne, 3)
    ex = ex - np.sum(ex * ez, axis=1, keepdims=True) * ez
    ex = ex / np.maximum(np.linalg.norm(ex, axis=1, keepdims=True), 1e-30)
    ey = np.cross(ez, ex)    # (ne, 3)

    # T_local (ne, 3, 3) — rows are ex, ey, ez
    T_local = np.stack([ex, ey, ez], axis=1)  # (ne, 3, 3)

    # Project nodes to local 2D: xy_local (ne, 4, 2)
    d = p - center[:, np.newaxis, :]  # (ne, 4, 3)
    xy_local = np.empty((n_elem, 4, 2))
    xy_local[:, :, 0] = np.einsum('nij,nj->ni', d, ex)  # x = d · ex
    xy_local[:, :, 1] = np.einsum('nij,nj->ni', d, ey)  # y = d · ey

    # --- Constitutive matrices (ne,) ---
    E_ = all_E; nu_ = all_nu; t_ = all_t

    # --- Compute all ke in batch ---
    ke_all = _batch_cquad4_stiffness(xy_local, E_, nu_, t_, n_elem)  # (ne, 24, 24)

    # --- Transform to global: ke_global = T24.T @ ke_local @ T24 ---
    # Instead of building full (ne, 24, 24) T24 matrix and doing triple einsum,
    # use block structure: T24 is block-diagonal with 8 copies of T_local (3x3).
    # Apply rotation block-by-block: for each 3x3 sub-block (i,j) of ke (24x24),
    # ke_global[3i:3i+3, 3j:3j+3] = R^T @ ke_local[3i:3i+3, 3j:3j+3] @ R
    ke_global = np.empty_like(ke_all)
    RT = T_local.transpose(0, 2, 1)  # (ne, 3, 3) — R^T
    for bi in range(8):
        si = 3 * bi
        for bj in range(8):
            sj = 3 * bj
            # block = R^T @ ke[si:si+3, sj:sj+3] @ R
            tmp = np.einsum('nij,njk->nik', RT, ke_all[:, si:si+3, sj:sj+3])
            ke_global[:, si:si+3, sj:sj+3] = np.einsum('nij,njk->nik', tmp, T_local)

    # --- Lumped mass matrices ---
    dl13 = xy_local[:, 2] - xy_local[:, 0]
    dl24 = xy_local[:, 3] - xy_local[:, 1]
    area = 0.5 * np.abs(dl13[:, 0]*dl24[:, 1] - dl13[:, 1]*dl24[:, 0])

    total_mass = all_rho * t_ * area           # (ne,)
    m_per_node = total_mass / 4.0              # (ne,)
    rot_inertia = m_per_node * t_**2 / 12.0   # (ne,)

    # Mass matrix is diagonal in local coords. Under block rotation R^T diag R,
    # diagonal blocks become R^T @ (m*I) @ R = m*I (since R is orthogonal).
    # So lumped mass matrix is the same in global coords — no transform needed.
    me_global = np.zeros((n_elem, 24, 24))
    for nd in range(4):
        base = 6 * nd
        for i in range(3):
            me_global[:, base+i, base+i] = m_per_node
        for i in range(3, 6):
            me_global[:, base+i, base+i] = rot_inertia

    # --- Assemble into COO arrays ---
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()  # (576,)
    jj_flat = jj_local.ravel()  # (576,)

    # Map local DOF indices to global DOFs for all elements at once
    global_rows_all = all_edofs[:, ii_flat]  # (ne, 576)
    global_cols_all = all_edofs[:, jj_flat]  # (ne, 576)
    ke_flat_all = ke_global.reshape(n_elem, -1)  # (ne, 576)
    me_flat_all = me_global.reshape(n_elem, -1)  # (ne, 576)

    # Filter near-zero entries
    mask_k_all = np.abs(ke_flat_all) > 1e-30  # (ne, 576)
    nk_total = mask_k_all.sum()
    rows_k[ptr_k:ptr_k+nk_total] = global_rows_all[mask_k_all]
    cols_k[ptr_k:ptr_k+nk_total] = global_cols_all[mask_k_all]
    vals_k[ptr_k:ptr_k+nk_total] = ke_flat_all[mask_k_all]
    ptr_k += nk_total

    mask_m_all = np.abs(me_flat_all) > 1e-30
    nm_total = mask_m_all.sum()
    rows_m[ptr_m:ptr_m+nm_total] = global_rows_all[mask_m_all]
    cols_m[ptr_m:ptr_m+nm_total] = global_cols_all[mask_m_all]
    vals_m[ptr_m:ptr_m+nm_total] = me_flat_all[mask_m_all]
    ptr_m += nm_total

    return ptr_k, ptr_m


def _batch_cquad4_stiffness(xy_local, E_, nu_, t_, n_elem):
    """Compute 24x24 local stiffness for all CQUAD4 elements simultaneously.

    Uses fully vectorized Gauss integration over all elements in parallel.

    Parameters
    ----------
    xy_local : (ne, 4, 2)  - local 2D coordinates
    E_, nu_, t_ : (ne,) - material/thickness arrays
    n_elem : int

    Returns
    -------
    ke : (ne, 24, 24) - local stiffness matrices
    """
    ne = n_elem

    # Constitutive matrices (scalars per element)
    # Membrane: Dm = E*t/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
    fac_m = E_ * t_ / (1.0 - nu_**2)   # (ne,)
    # Bending:  Db = E*t^3/(12*(1-nu^2)) * same pattern
    fac_b = E_ * t_**3 / (12.0 * (1.0 - nu_**2))  # (ne,)
    # Shear:    Ds = kappa * E*t / (2*(1+nu)) * I_2
    kappa = 5.0 / 6.0
    fac_s = kappa * E_ * t_ / (2.0 * (1.0 + nu_))  # (ne,)

    ke = np.zeros((ne, 24, 24))

    # DOF index arrays
    mem_idx = np.array([0,1, 6,7, 12,13, 18,19])  # u,v for 4 nodes
    bend_idx = np.array([3,4, 9,10, 15,16, 21,22])  # rx,ry for 4 nodes
    shear_idx = np.array([2,3,4, 8,9,10, 14,15,16, 20,21,22])  # w,rx,ry

    # 2x2 Gauss integration for membrane and bending
    for gi in range(2):
        for gj in range(2):
            xi = _GP2[gi]; eta = _GP2[gj]

            # Shape function derivatives
            dNdxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
            dNdeta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
            N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                                 (1+xi)*(1+eta), (1-xi)*(1+eta)])

            # Jacobian: J[ne, 2, 2]
            # J[0,0] = dNdxi . x, J[0,1] = dNdxi . y, etc.
            J = np.empty((ne, 2, 2))
            J[:, 0, 0] = dNdxi @ xy_local[:, :, 0].T  # (ne,) ← (4,) @ (4, ne)
            J[:, 0, 1] = dNdxi @ xy_local[:, :, 1].T
            J[:, 1, 0] = dNdeta @ xy_local[:, :, 0].T
            J[:, 1, 1] = dNdeta @ xy_local[:, :, 1].T

            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]  # (ne,)

            # Inverse Jacobian (2x2 analytic)
            inv_det = 1.0 / np.maximum(np.abs(detJ), 1e-30)
            Jinv = np.empty((ne, 2, 2))
            Jinv[:, 0, 0] = J[:, 1, 1] * inv_det
            Jinv[:, 0, 1] = -J[:, 0, 1] * inv_det
            Jinv[:, 1, 0] = -J[:, 1, 0] * inv_det
            Jinv[:, 1, 1] = J[:, 0, 0] * inv_det

            # dN/dx, dN/dy (ne, 4)
            dNdx = np.outer(Jinv[:, 0, 0], dNdxi).reshape(ne, 4) + \
                   np.outer(Jinv[:, 0, 1], dNdeta).reshape(ne, 4)
            dNdy = np.outer(Jinv[:, 1, 0], dNdxi).reshape(ne, 4) + \
                   np.outer(Jinv[:, 1, 1], dNdeta).reshape(ne, 4)

            # --- Membrane: Bm (ne, 3, 8) ---
            Bm = np.zeros((ne, 3, 8))
            for nd in range(4):
                Bm[:, 0, 2*nd] = dNdx[:, nd]
                Bm[:, 1, 2*nd+1] = dNdy[:, nd]
                Bm[:, 2, 2*nd] = dNdy[:, nd]
                Bm[:, 2, 2*nd+1] = dNdx[:, nd]

            # Dm (ne, 3, 3) * Bm (ne, 3, 8) → DmBm (ne, 3, 8)
            # Dm = fac_m * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
            Dm_Bm = np.empty((ne, 3, 8))
            Dm_Bm[:, 0] = fac_m[:, None] * (Bm[:, 0] + nu_[:, None] * Bm[:, 1])
            Dm_Bm[:, 1] = fac_m[:, None] * (nu_[:, None] * Bm[:, 0] + Bm[:, 1])
            Dm_Bm[:, 2] = fac_m[:, None] * ((1 - nu_[:, None]) / 2) * Bm[:, 2]

            # km = Bm^T @ Dm @ Bm * detJ = Bm^T @ Dm_Bm * detJ
            km = np.einsum('nai,naj->nij', Bm, Dm_Bm) * detJ[:, None, None]  # (ne, 8, 8)
            ke[:, mem_idx[:, None], mem_idx[None, :]] += km

            # --- Bending: Bb (ne, 3, 8) ---
            Bb = np.zeros((ne, 3, 8))
            for nd in range(4):
                Bb[:, 0, 2*nd+1] = -dNdx[:, nd]
                Bb[:, 1, 2*nd] = dNdy[:, nd]
                Bb[:, 2, 2*nd] = dNdx[:, nd]
                Bb[:, 2, 2*nd+1] = -dNdy[:, nd]

            Db_Bb = np.empty((ne, 3, 8))
            Db_Bb[:, 0] = fac_b[:, None] * (Bb[:, 0] + nu_[:, None] * Bb[:, 1])
            Db_Bb[:, 1] = fac_b[:, None] * (nu_[:, None] * Bb[:, 0] + Bb[:, 1])
            Db_Bb[:, 2] = fac_b[:, None] * ((1 - nu_[:, None]) / 2) * Bb[:, 2]

            kb = np.einsum('nai,naj->nij', Bb, Db_Bb) * detJ[:, None, None]
            ke[:, bend_idx[:, None], bend_idx[None, :]] += kb

    # --- 1-point shear integration ---
    dNdxi_c = 0.25 * np.array([-1.0, 1.0, 1.0, -1.0])
    dNdeta_c = 0.25 * np.array([-1.0, -1.0, 1.0, 1.0])
    N_c = np.array([0.25, 0.25, 0.25, 0.25])

    J_c = np.empty((ne, 2, 2))
    J_c[:, 0, 0] = dNdxi_c @ xy_local[:, :, 0].T
    J_c[:, 0, 1] = dNdxi_c @ xy_local[:, :, 1].T
    J_c[:, 1, 0] = dNdeta_c @ xy_local[:, :, 0].T
    J_c[:, 1, 1] = dNdeta_c @ xy_local[:, :, 1].T

    detJ_c = J_c[:, 0, 0]*J_c[:, 1, 1] - J_c[:, 0, 1]*J_c[:, 1, 0]
    inv_det_c = 1.0 / np.maximum(np.abs(detJ_c), 1e-30)
    Jinv_c = np.empty((ne, 2, 2))
    Jinv_c[:, 0, 0] = J_c[:, 1, 1] * inv_det_c
    Jinv_c[:, 0, 1] = -J_c[:, 0, 1] * inv_det_c
    Jinv_c[:, 1, 0] = -J_c[:, 1, 0] * inv_det_c
    Jinv_c[:, 1, 1] = J_c[:, 0, 0] * inv_det_c

    dNdx_c = np.outer(Jinv_c[:, 0, 0], dNdxi_c).reshape(ne, 4) + \
             np.outer(Jinv_c[:, 0, 1], dNdeta_c).reshape(ne, 4)
    dNdy_c = np.outer(Jinv_c[:, 1, 0], dNdxi_c).reshape(ne, 4) + \
             np.outer(Jinv_c[:, 1, 1], dNdeta_c).reshape(ne, 4)

    Bs = np.zeros((ne, 2, 12))
    for nd in range(4):
        Bs[:, 0, 3*nd] = dNdx_c[:, nd]
        Bs[:, 0, 3*nd+2] = -N_c[nd]
        Bs[:, 1, 3*nd] = dNdy_c[:, nd]
        Bs[:, 1, 3*nd+1] = N_c[nd]

    # Ds @ Bs = fac_s * Bs (isotropic shear)
    Ds_Bs = fac_s[:, None, None] * Bs  # (ne, 2, 12)
    ks = np.einsum('nai,naj->nij', Bs, Ds_Bs) * (detJ_c * 4.0)[:, None, None]
    ke[:, shear_idx[:, None], shear_idx[None, :]] += ks

    # Drilling stabilization
    dl13 = xy_local[:, 2] - xy_local[:, 0]
    dl24 = xy_local[:, 3] - xy_local[:, 1]
    area = 0.5 * np.abs(dl13[:, 0]*dl24[:, 1] - dl13[:, 1]*dl24[:, 0])
    alpha_drill = E_ * t_ * area * 1e-6  # (ne,)
    for nd in range(4):
        rz_dof = 6 * nd + 5
        ke[:, rz_dof, rz_dof] += alpha_drill

    return ke


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


def _assemble_cquad8_batch(elems, model, dof_mgr,
                            rows_k, cols_k, vals_k,
                            rows_m, cols_m, vals_m,
                            ptr_k, ptr_m):
    """Batch assemble CQUAD8 elements (per-element loop)."""
    ndof_e = 48
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()
    jj_flat = jj_local.ravel()

    for idx, (eid, elem) in enumerate(elems):
        try:
            prop = elem.property_ref
            if hasattr(prop, 'equivalent_isotropic'):
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = prop.material_ref
                E = mat.E; nu = mat.nu; t = prop.t; rho = mat.rho
            node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            q8 = CQuad8Element(node_xyz, E, nu, t, rho)
            ke = q8.stiffness_matrix()
            me = q8.mass_matrix()
        except Exception as exc:
            logger.warning("Error assembling CQUAD8 %d: %s", eid, exc)
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


def _assemble_ctria6_batch(elems, model, dof_mgr,
                            rows_k, cols_k, vals_k,
                            rows_m, cols_m, vals_m,
                            ptr_k, ptr_m):
    """Batch assemble CTRIA6 elements (per-element loop)."""
    ndof_e = 36
    ii_local, jj_local = np.meshgrid(np.arange(ndof_e), np.arange(ndof_e), indexing='ij')
    ii_flat = ii_local.ravel()
    jj_flat = jj_local.ravel()

    for idx, (eid, elem) in enumerate(elems):
        try:
            prop = elem.property_ref
            if hasattr(prop, 'equivalent_isotropic'):
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = prop.material_ref
                E = mat.E; nu = mat.nu; t = prop.t; rho = mat.rho
            node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            t6 = CTria6Element(node_xyz, E, nu, t, rho)
            ke = t6.stiffness_matrix()
            me = t6.mass_matrix()
        except Exception as exc:
            logger.warning("Error assembling CTRIA6 %d: %s", eid, exc)
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
