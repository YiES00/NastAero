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
    cquad4_elems = []
    ctria3_elems = []
    crod_elems = []

    for eid, elem in model.elements.items():
        if elem.type == "CBAR":
            cbar_elems.append((eid, elem))
        elif elem.type == "CQUAD4":
            cquad4_elems.append((eid, elem))
        elif elem.type == "CTRIA3":
            ctria3_elems.append((eid, elem))
        elif elem.type == "CROD":
            crod_elems.append((eid, elem))

    n_total = len(cbar_elems) + len(cquad4_elems) + len(ctria3_elems) + len(crod_elems)
    logger.info("Assembly: %d CBAR, %d CQUAD4, %d CTRIA3, %d CROD = %d total elements",
                len(cbar_elems), len(cquad4_elems), len(ctria3_elems), len(crod_elems), n_total)

    # Pre-allocate COO arrays with known sizes
    # CBAR: 12x12 = 144 entries per element
    # CQUAD4: 24x24 = 576 entries per element
    # CTRIA3: 18x18 = 324 entries per element
    # CROD: 12x12 = 144 entries per element
    max_nnz = (len(cbar_elems) * 144 + len(cquad4_elems) * 576 +
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

    # --- RBE2 rigid elements (penalty approach) ---
    rbe2_rows, rbe2_cols, rbe2_vals = [], [], []
    for rid, rbe in model.rigids.items():
        if rbe.type == "RBE2":
            _assemble_rbe2(rbe, model, dof_mgr, rbe2_rows, rbe2_cols, rbe2_vals)

    if rbe2_rows:
        rows_k = np.concatenate([rows_k, np.array(rbe2_rows, dtype=np.int64)])
        cols_k = np.concatenate([cols_k, np.array(rbe2_cols, dtype=np.int64)])
        vals_k = np.concatenate([vals_k, np.array(rbe2_vals, dtype=np.float64)])

    t_coo = time.perf_counter()
    K = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(ndof, ndof)).tocsc()
    M = sp.coo_matrix((vals_m, (rows_m, cols_m)), shape=(ndof, ndof)).tocsc()
    t_sparse = time.perf_counter() - t_coo

    t_total = time.perf_counter() - t_start
    logger.info("Assembled %d elements into global matrices (%d DOFs) in %.3f s (sparse convert: %.3f s)",
                n_assembled, ndof, t_total, t_sparse)
    return K, M


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
            mat = prop.material_ref
            node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            q = CQuad4Element(node_xyz, mat.E, mat.nu, prop.t, mat.rho)
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
            mat = prop.material_ref
            node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            tri = CTria3Element(node_xyz, mat.E, mat.nu, prop.t, mat.rho)
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
    """Assemble RBE2 using penalty method."""
    penalty = 1e12
    ind_nid = rbe.independent_node
    if ind_nid not in dof_mgr._nid_to_index:
        return
    for dep_nid in rbe.dependent_nodes:
        if dep_nid not in dof_mgr._nid_to_index:
            continue
        for ch in rbe.components:
            comp = int(ch)
            if comp < 1 or comp > 6:
                continue
            ind_dof = dof_mgr.get_dof(ind_nid, comp)
            dep_dof = dof_mgr.get_dof(dep_nid, comp)
            rows_k.extend([ind_dof, ind_dof, dep_dof, dep_dof])
            cols_k.extend([ind_dof, dep_dof, ind_dof, dep_dof])
            vals_k.extend([penalty, -penalty, -penalty, penalty])
