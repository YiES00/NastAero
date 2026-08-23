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

        # Inertia tensor at CG. MSC의 I21/I31/I32는 관성곱의 크기
        # (양의 적분 integral xi*xj dm)이고, 텐서로 조립할 때 음부호를
        # 자동으로 붙인다:
        #   [[ I11, -I21, -I31],
        #    [-I21,  I22, -I32],
        #    [-I31, -I32,  I33]]
        # 아래 평행축 항 m*(r.r*I - r x r)의 비대각 성분이 -m*ri*rj로
        # 이미 물리 텐서 규약이므로, 카드 항도 같은 규약이어야 한다.
        I_cg = np.array([[mass_elem.I11, -mass_elem.I21, -mass_elem.I31],
                         [-mass_elem.I21, mass_elem.I22, -mass_elem.I32],
                         [-mass_elem.I31, -mass_elem.I32, mass_elem.I33]])

        # CID coordinate transform: rotate offset and inertia to basic
        cid = mass_elem.cid
        if cid > 0 and cid in model.coords:
            R = model.coords[cid].transform  # 3x3 rotation
            offset = R @ offset
            I_cg = R @ I_cg @ R.T
        elif cid == -1:
            # QRG: X1~X3는 오프셋이 아니라 기본좌표계 기준 질량 CG의
            # 절대좌표다. 관성은 이미 그 CG 기준 기본축이므로 회전 없이
            # 절점 기준 오프셋으로만 환산한다.
            node = model.nodes.get(nid)
            if node is not None:
                offset = offset - node.xyz_global

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

        # Translation-rotation coupling from offset: M_tr = -m * skew(r)
        # v_cg = u_dot + omega x r = u_dot - skew(r) omega 이므로 운동
        # 에너지의 교차항은 -m u_dot^T skew(r) omega 다. 따라서
        # M[trans, rot] = -m*skew(r), M[rot, trans] = +m*skew(r).
        # 부호가 반대면 오프셋이 절점을 통해 반전된 것과 같아진다.
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
                        # Upper-right block: trans-rot = -m*skew(r)
                        conm2_rows.append(node_dofs[i])
                        conm2_cols.append(node_dofs[3 + j])
                        conm2_vals.append(-val)
                        # Lower-left block: rot-trans = +m*skew(r)
                        conm2_rows.append(node_dofs[3 + j])
                        conm2_cols.append(node_dofs[i])
                        conm2_vals.append(-val)

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

    # --- Step 3: Constraints and springs (in CD-frame DOFs) ---

    # --- RBE2 rigid elements (elimination method, Nastran-style) ---
    slave_deps = {}  # {slave_dof: [(master_dof, coeff), ...]}
    for rid, rbe in model.rigids.items():
        if rbe.type == "RBE2":
            _build_rbe2_slave_deps(rbe, model, dof_mgr, slave_deps)
    for rid, rbe in model.rigids.items():
        if rbe.type == "RBE3":
            _build_rbe3_slave_deps(rbe, model, dof_mgr, slave_deps)

    # --- MPC constraints (elimination method) ---
    for mpc_sid, mpc_list in model.mpcs.items():
        for mpc in mpc_list:
            _build_mpc_slave_deps(mpc, dof_mgr, slave_deps)

    if slave_deps:
        n_rbe2 = sum(1 for r in model.rigids.values() if r.type == "RBE2")
        n_rbe3 = sum(1 for r in model.rigids.values() if r.type == "RBE3")
        n_mpc = sum(len(v) for v in model.mpcs.values())
        logger.info("  RBE2/RBE3/MPC elimination: %d slave DOFs "
                     "(%d RBE2, %d RBE3, %d MPC)",
                     len(slave_deps), n_rbe2, n_rbe3, n_mpc)

        # Build transformation matrix G and apply elimination
        K, M = _apply_elimination(K, M, slave_deps, ndof)

    # --- Spring elements (CELAS1/CELAS2) ---
    spring_rows, spring_cols, spring_vals = [], [], []
    for sid, spring in model.springs.items():
        _assemble_spring(spring, model, dof_mgr, spring_rows, spring_cols, spring_vals)

    if spring_rows:
        spring_K = sp.coo_matrix((spring_vals, (spring_rows, spring_cols)),
                                  shape=(ndof, ndof)).tocsc()
        K = K + spring_K
        logger.info("  Assembled %d spring elements", len(model.springs))

    t_total = time.perf_counter() - t_start
    logger.info("Assembled %d elements into global matrices (%d DOFs) in %.3f s (sparse convert: %.3f s)",
                n_assembled, ndof, t_total, t_sparse)
    return K, M, slave_deps


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
    all_r12 = np.ones(n_elem)    # PSHELL 12I/T^3 (굽힘 관성비)
    all_nsm = np.zeros(n_elem)   # 단위면적당 비구조 질량
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
            all_r12[idx] = float(getattr(prop, 'ratio_12it3', 1.0) or 1.0)
            all_nsm[idx] = float(getattr(prop, 'nsm', 0.0) or 0.0)
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
        all_r12 = all_r12[mask]; all_nsm = all_nsm[mask]
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
    ke_all = _batch_cquad4_stiffness(xy_local, E_, nu_, t_, n_elem,
                                     r12_=all_r12)  # (ne, 24, 24)

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

    # 비구조 질량(단위면적당)은 구조 질량과 같은 경로로 실린다
    total_mass = (all_rho * t_ + all_nsm) * area   # (ne,)
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


def _batch_cquad4_stiffness(xy_local, E_, nu_, t_, n_elem, r12_=1.0):
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
    # PSHELL 12I/T^3: 실제 굽힘 관성 / (T^3/12) 비율 (기본 1.0)
    fac_b = r12_ * E_ * t_**3 / (12.0 * (1.0 - nu_**2))  # (ne,)
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

    # 전단 변형률 (Nastran 절점회전 규약): gxz = dw/dx + theta_y,
    # gyz = dw/dy - theta_x. 이전 부호(-theta_y/+theta_x)는 회전이 Nastran의
    # 음수 규약이라 판 단독으로는 등가지만 보/강체와 절점을 공유하는 혼합
    # 구조에서 전단 잠금을 일으켜 강성이 수백 배 과대였다 (MSC 대조로 확인).
    Bs = np.zeros((ne, 2, 12))
    for nd in range(4):
        Bs[:, 0, 3*nd] = dNdx_c[:, nd]
        Bs[:, 0, 3*nd+2] = N_c[nd]
        Bs[:, 1, 3*nd] = dNdy_c[:, nd]
        Bs[:, 1, 3*nd+1] = -N_c[nd]

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
            tri = CTria3Element(node_xyz, E, nu, t, rho,
                                r12=float(getattr(prop, 'ratio_12it3', 1.0) or 1.0),
                                nsm=float(getattr(prop, 'nsm', 0.0) or 0.0))
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


def _resolve_bar_orientation(elem, model, n1):
    """CBAR/CBEAM 방향 벡터 v를 기본좌표계로 환산한다.

    MSC 규약(QRG): X1~X3는 OFFT 첫 글자가 'B'가 아닌 한 GA의 변위
    좌표계(CD) 성분이다(기본값 OFFT=GGG). G0 분기는 절점 좌표
    차이라 이미 기본좌표계다.
    """
    if elem.g0 > 0 and elem.g0 in model.nodes:
        return model.nodes[elem.g0].xyz_global - n1.xyz_global
    v = np.asarray(elem.x, dtype=float).copy()
    if np.linalg.norm(v) < 1e-12:
        return np.array([0., 0., 1.])
    offt = (getattr(elem, 'offt', None) or 'GGG')
    cd = int(getattr(n1, 'cd', 0) or 0)
    if str(offt)[:1].upper() != 'B' and cd != 0 and cd in model.coords:
        v = model.coords[cd].transform @ v
    return v


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
            v_vector = _resolve_bar_orientation(elem, model, n1)
            bar = CBarElement(n1.xyz_global, n2.xyz_global, v_vector,
                              mat.E, mat.G, prop.A, prop.I1, prop.I2, prop.J,
                              mat.rho, prop.nsm,
                              pa=getattr(elem, 'pa', 0),
                              pb=getattr(elem, 'pb', 0),
                              wa=getattr(elem, 'wa', None),
                              wb=getattr(elem, 'wb', None))
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


def _cd_rotation(nid, model):
    """절점의 CD 좌표계 -> 기본좌표계 3x3 회전 (CD=0이면 항등)."""
    node = model.nodes.get(nid)
    cd = int(getattr(node, 'cd', 0) or 0) if node is not None else 0
    if cd and cd in model.coords:
        return np.asarray(model.coords[cd].transform, dtype=float)
    return np.eye(3)


def _build_rbe2_slave_deps(rbe, model, dof_mgr, slave_deps):
    """Build slave DOF dependency map for RBE2 (elimination method).

    RBE2 rigid body kinematics:
      u_dep = u_ind + theta_ind × r   (r = xyz_dep - xyz_ind)

    Each dependent (slave) DOF is expressed as a linear combination of
    independent (master) DOFs.  No penalty parameter needed.

    Translation dependencies (via cross product theta × r):
      u_d1 = u_i1 + theta_i5*r3 - theta_i6*r2
      u_d2 = u_i2 + theta_i6*r1 - theta_i4*r3
      u_d3 = u_i3 + theta_i4*r2 - theta_i5*r1

    Rotation dependencies (direct coupling):
      theta_d4 = theta_i4
      theta_d5 = theta_i5
      theta_d6 = theta_i6

    Args:
        slave_deps: Dict to update — {slave_dof: [(master_dof, coeff), ...]}
    """
    ind_nid = rbe.independent_node
    if ind_nid not in dof_mgr._nid_to_index:
        return
    if ind_nid not in model.nodes:
        return

    ind_xyz = model.nodes[ind_nid].xyz_global

    cm_set = set()
    for ch in rbe.components:
        comp = int(ch)
        if 1 <= comp <= 6:
            cm_set.add(comp)

    has_trans = bool(cm_set & {1, 2, 3})

    for dep_nid in rbe.dependent_nodes:
        if dep_nid not in dof_mgr._nid_to_index:
            continue
        if dep_nid not in model.nodes:
            continue

        dep_xyz = model.nodes[dep_nid].xyz_global
        r = dep_xyz - ind_xyz

        ind_dofs = dof_mgr.get_node_dofs(ind_nid)
        dep_dofs = dof_mgr.get_node_dofs(dep_nid)

        # 강체 운동학은 기본좌표계에서 정의되지만, 조립된 K/M의
        # 자유도는 CD!=0 절점에서 CD 성분이다(_apply_cd_transforms).
        # 따라서 계수는 두 절점의 CD 회전을 함께 태워야 한다:
        #   D = blkdiag(R_dep,R_dep)^T [[I, -S(r)],[0, I]] blkdiag(R_ind,R_ind)
        R_ind = _cd_rotation(ind_nid, model)
        R_dep = _cd_rotation(dep_nid, model)
        S = np.array([[0.0, -r[2], r[1]],
                      [r[2], 0.0, -r[0]],
                      [-r[1], r[0], 0.0]])
        block = np.zeros((6, 6))
        block[0:3, 0:3] = np.eye(3)
        block[0:3, 3:6] = -S
        block[3:6, 3:6] = np.eye(3)
        T_ind = np.zeros((6, 6)); T_dep = np.zeros((6, 6))
        T_ind[0:3, 0:3] = R_ind; T_ind[3:6, 3:6] = R_ind
        T_dep[0:3, 0:3] = R_dep; T_dep[3:6, 3:6] = R_dep
        D = T_dep.T @ block @ T_ind

        for comp in range(1, 7):
            if comp not in cm_set:
                continue
            row = comp - 1
            terms = [(ind_dofs[col], D[row, col])
                     for col in range(6) if abs(D[row, col]) > 1e-14]
            if terms:
                slave_deps[dep_dofs[row]] = terms


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


def _build_rbe3_slave_deps(rbe, model, dof_mgr, slave_deps):
    """RBE3의 REFGRID 자유도를 독립 절점 병진의 가중 최소자승 보간으로
    표현한다 (소거법, RBE2와 같은 slave_deps 구조).

    가중 도심 r_c = sum(w_i r_i)/W 를 기준으로
      u_bar   = sum(w_i u_i)/W
      theta_bar = J^-1 sum(w_i rho_i x u_i),  rho_i = r_i - r_c
      J = sum(w_i (|rho_i|^2 I - rho_i rho_i^T))
    이고 REFGRID 값은
      u_ref = u_bar - S(d) theta_bar   (d = r_ref - r_c)
      theta_ref = theta_bar
    이다. 독립 절점의 회전 성분은 참여시키지 않는다(Ci=123 관행).
    J가 특이한 배치(단일/공선 절점)에서는 유사역행렬을 쓴다.
    """
    ref_nid = rbe.refgrid
    if ref_nid not in dof_mgr._nid_to_index or ref_nid not in model.nodes:
        return

    nids, weights = [], []
    for wt, comp, grids in rbe.weight_sets:
        for gid in grids:
            if gid in dof_mgr._nid_to_index and gid in model.nodes:
                nids.append(gid)
                weights.append(float(wt))
    if not nids:
        logger.warning("  RBE3 %d: 독립 절점 없음 — 건너뜀", rbe.eid)
        return

    w = np.array(weights, dtype=float)
    W = w.sum()
    if W <= 0:
        return
    xyz = np.array([model.nodes[n].xyz_global for n in nids], dtype=float)
    r_c = (w[:, None] * xyz).sum(axis=0) / W
    rho = xyz - r_c
    d = model.nodes[ref_nid].xyz_global - r_c

    J = np.zeros((3, 3))
    for wi, p in zip(w, rho):
        J += wi * (float(p @ p) * np.eye(3) - np.outer(p, p))
    J_inv = np.linalg.pinv(J)

    def skew(v):
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    Sd = skew(d)
    ref_comps = {int(c) for c in rbe.refc if c.isdigit() and 1 <= int(c) <= 6}

    # 성분별 계수: u_ref = sum_i C_i u_i, theta_ref = sum_i D_i u_i
    for k_local, (nid, wi, p) in enumerate(zip(nids, w, rho)):
        Sp = skew(p)
        D_i = wi * (J_inv @ Sp)              # theta_ref <- u_i
        C_i = (wi / W) * np.eye(3) - Sd @ D_i  # u_ref   <- u_i
        for row in range(3):
            if (row + 1) in ref_comps:
                dof_s = dof_mgr.get_dof(ref_nid, row + 1)
                terms = slave_deps.setdefault(dof_s, [])
                for col in range(3):
                    c = C_i[row, col]
                    if abs(c) > 1e-14:
                        terms.append((dof_mgr.get_dof(nid, col + 1), c))
            if (row + 4) in ref_comps:
                dof_s = dof_mgr.get_dof(ref_nid, row + 4)
                terms = slave_deps.setdefault(dof_s, [])
                for col in range(3):
                    c = D_i[row, col]
                    if abs(c) > 1e-14:
                        terms.append((dof_mgr.get_dof(nid, col + 1), c))


def _build_mpc_slave_deps(mpc, dof_mgr, slave_deps):
    """Build slave DOF dependency for an MPC constraint (elimination method).

    MPC equation: A0*u0 + A1*u1 + ... = 0
    The first term (largest |coefficient|) is chosen as slave:
        u_slave = -(A1/A0)*u1 - (A2/A0)*u2 - ...

    No penalty parameter needed.
    """
    dof_coeff = []
    for nid, comp, coeff in mpc.terms:
        if nid in dof_mgr._nid_to_index and abs(coeff) > 1e-30:
            dof = dof_mgr.get_dof(nid, comp)
            dof_coeff.append((dof, coeff))

    if len(dof_coeff) < 2:
        return

    # Choose slave as the term with largest absolute coefficient
    idx_max = max(range(len(dof_coeff)), key=lambda i: abs(dof_coeff[i][1]))
    slave_dof, slave_coeff = dof_coeff[idx_max]

    # Don't overwrite if already a slave from RBE2
    if slave_dof in slave_deps:
        return

    # u_slave = -sum(A_j/A_slave * u_j) for j != slave
    terms = []
    for i, (dof, coeff) in enumerate(dof_coeff):
        if i != idx_max:
            terms.append((dof, -coeff / slave_coeff))

    slave_deps[slave_dof] = terms


def apply_load_elimination(F, slave_deps):
    """소거된 종속 자유도의 하중을 주 자유도로 옮긴다 (F_new = G^T F).

    _apply_elimination이 K, M을 G^T(.)G로 줄이는 것과 짝이 되는
    연산이다. 이걸 빼면 RBE2/RBE3/MPC 종속 자유도에 실린 하중이
    구속 집합에 들어가면서 그대로 사라진다(MSC는 P_n = P_n + G_m^T P_m).

    Parameters
    ----------
    F : ndarray
        전체 하중 벡터 (in-place로 수정하지 않고 사본을 반환).
    slave_deps : dict
        {slave_dof: [(master_dof, coeff), ...]}
    """
    if not slave_deps:
        return F
    F_new = np.asarray(F, dtype=float).copy()
    for s_dof, terms in slave_deps.items():
        fs = F_new[s_dof]
        if fs == 0.0:
            continue
        for m_dof, coeff in terms:
            F_new[m_dof] += coeff * fs
        F_new[s_dof] = 0.0
    return F_new


def _apply_elimination(K, M, slave_deps, ndof):
    """Apply constraint elimination via transformation matrix G.

    For each slave DOF s with dependency u_s = sum(a_j * u_j):
        G[s, s] = 0  (remove slave self-coupling)
        G[s, j] = a_j (slave expressed as master combination)
        G[i, i] = 1  for all non-slave DOFs i

    Then: K_new = G^T @ K @ G,  M_new = G^T @ M @ G

    This is the Nastran-style elimination method: exact constraint
    enforcement with no penalty parameter, no ill-conditioning.
    """
    import time
    t0 = time.perf_counter()

    # Build sparse G matrix (mostly identity)
    g_rows, g_cols, g_vals = [], [], []

    # Identity for all DOFs first
    all_dofs = np.arange(ndof)
    g_rows.extend(all_dofs)
    g_cols.extend(all_dofs)
    g_vals.extend(np.ones(ndof))

    # Override slave rows: remove identity, add master dependencies
    for s_dof, terms in slave_deps.items():
        # Remove identity entry (set G[s,s] = 0)
        g_rows.append(s_dof)
        g_cols.append(s_dof)
        g_vals.append(-1.0)  # cancels the +1.0 from identity above

        # Add dependency terms: G[s, master_j] = a_j
        for m_dof, coeff in terms:
            g_rows.append(s_dof)
            g_cols.append(m_dof)
            g_vals.append(coeff)

    G = sp.coo_matrix(
        (g_vals, (g_rows, g_cols)), shape=(ndof, ndof)).tocsc()

    # K_new = G^T @ K @ G,  M_new = G^T @ M @ G
    Gt = G.T.tocsc()
    K_new = Gt @ K @ G
    M_new = Gt @ M @ G

    dt = time.perf_counter() - t0
    logger.info("  Elimination G^T K G: %.3f s (%d slave DOFs)", dt, len(slave_deps))
    return K_new.tocsc(), M_new.tocsc()


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
            q8 = CQuad8Element(node_xyz, E, nu, t, rho,
                     r12=float(getattr(prop, 'ratio_12it3', 1.0) or 1.0),
                     nsm=float(getattr(prop, 'nsm', 0.0) or 0.0))
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
            t6 = CTria6Element(node_xyz, E, nu, t, rho,
                     r12=float(getattr(prop, 'ratio_12it3', 1.0) or 1.0),
                     nsm=float(getattr(prop, 'nsm', 0.0) or 0.0))
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
