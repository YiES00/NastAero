"""Global stiffness and mass matrix assembly."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import scipy.sparse as sp
from .dof_manager import DOFManager
from ..bdf.model import BDFModel
from ..elements.bar import CBarElement
from ..elements.quad4 import CQuad4Element
from ..elements.tria3 import CTria3Element
from ..config import logger

def assemble_global_matrices(model: BDFModel, dof_mgr: DOFManager):
    ndof = dof_mgr.total_dof
    rows_k, cols_k, vals_k = [], [], []
    rows_m, cols_m, vals_m = [], [], []
    n_assembled = 0

    for eid, elem in model.elements.items():
        try:
            if elem.type == "CBAR":
                ke, me, edofs = _assemble_cbar(elem, model, dof_mgr)
            elif elem.type == "CROD":
                ke, me, edofs = _assemble_crod(elem, model, dof_mgr)
            elif elem.type == "CQUAD4":
                ke, me, edofs = _assemble_cquad4(elem, model, dof_mgr)
            elif elem.type == "CTRIA3":
                ke, me, edofs = _assemble_ctria3(elem, model, dof_mgr)
            else:
                logger.debug("Element type %s not supported for assembly", elem.type)
                continue
        except Exception as exc:
            logger.warning("Error assembling element %d (%s): %s", eid, elem.type, exc)
            continue

        nd = len(edofs)
        for i in range(nd):
            for j in range(nd):
                if abs(ke[i,j]) > 1e-30:
                    rows_k.append(edofs[i]); cols_k.append(edofs[j]); vals_k.append(ke[i,j])
                if abs(me[i,j]) > 1e-30:
                    rows_m.append(edofs[i]); cols_m.append(edofs[j]); vals_m.append(me[i,j])
        n_assembled += 1

    # Concentrated masses (CONM2)
    for mid, mass_elem in model.masses.items():
        nid = mass_elem.node_id
        if nid not in dof_mgr._nid_to_index: continue
        node_dofs = dof_mgr.get_node_dofs(nid)
        for i in range(3):
            rows_m.append(node_dofs[i]); cols_m.append(node_dofs[i])
            vals_m.append(mass_elem.mass)
        # Rotational inertia if provided
        if mass_elem.I11 > 0:
            rows_m.append(node_dofs[3]); cols_m.append(node_dofs[3]); vals_m.append(mass_elem.I11)
        if mass_elem.I22 > 0:
            rows_m.append(node_dofs[4]); cols_m.append(node_dofs[4]); vals_m.append(mass_elem.I22)
        if mass_elem.I33 > 0:
            rows_m.append(node_dofs[5]); cols_m.append(node_dofs[5]); vals_m.append(mass_elem.I33)

    # RBE2 rigid elements (penalty approach)
    for rid, rbe in model.rigids.items():
        if rbe.type == "RBE2":
            _assemble_rbe2(rbe, model, dof_mgr, rows_k, cols_k, vals_k)

    logger.info("Assembled %d elements into global matrices (%d DOFs)", n_assembled, ndof)
    K = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(ndof, ndof)).tocsc()
    M = sp.coo_matrix((vals_m, (rows_m, cols_m)), shape=(ndof, ndof)).tocsc()
    return K, M

def _assemble_cbar(elem, model, dof_mgr):
    prop = elem.property_ref; mat = prop.material_ref
    n1 = model.nodes[elem.node_ids[0]]; n2 = model.nodes[elem.node_ids[1]]
    if elem.g0 > 0 and elem.g0 in model.nodes:
        v_vector = model.nodes[elem.g0].xyz_global - n1.xyz_global
    else:
        v_vector = elem.x.copy()
        if np.linalg.norm(v_vector) < 1e-12: v_vector = np.array([0., 0., 1.])
    bar = CBarElement(n1.xyz_global, n2.xyz_global, v_vector, mat.E, mat.G, prop.A, prop.I1, prop.I2, prop.J, mat.rho, prop.nsm)
    return bar.stiffness_matrix(), bar.mass_matrix(), dof_mgr.get_element_dofs(elem.node_ids)

def _assemble_crod(elem, model, dof_mgr):
    prop = elem.property_ref; mat = prop.material_ref
    n1 = model.nodes[elem.node_ids[0]]; n2 = model.nodes[elem.node_ids[1]]
    diff = n2.xyz_global - n1.xyz_global; L = np.linalg.norm(diff)
    if L < 1e-12: raise ValueError(f"Zero-length CROD {elem.eid}")
    ex = diff / L; ea_l = mat.E * prop.A / L
    ke = np.zeros((12, 12)); me = np.zeros((12, 12))
    ex_out = np.outer(ex, ex)
    ke[0:3,0:3] = ea_l*ex_out; ke[0:3,6:9] = -ea_l*ex_out
    ke[6:9,0:3] = -ea_l*ex_out; ke[6:9,6:9] = ea_l*ex_out
    m_half = (mat.rho * prop.A * L) / 2.0
    for i in range(3): me[i,i] = m_half; me[i+6,i+6] = m_half
    return ke, me, dof_mgr.get_element_dofs(elem.node_ids)

def _assemble_cquad4(elem, model, dof_mgr):
    prop = elem.property_ref; mat = prop.material_ref
    node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
    q = CQuad4Element(node_xyz, mat.E, mat.nu, prop.t, mat.rho)
    return q.stiffness_matrix(), q.mass_matrix(), dof_mgr.get_element_dofs(elem.node_ids)

def _assemble_ctria3(elem, model, dof_mgr):
    prop = elem.property_ref; mat = prop.material_ref
    node_xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
    tri = CTria3Element(node_xyz, mat.E, mat.nu, prop.t, mat.rho)
    return tri.stiffness_matrix(), tri.mass_matrix(), dof_mgr.get_element_dofs(elem.node_ids)

def _assemble_rbe2(rbe, model, dof_mgr, rows_k, cols_k, vals_k):
    """Assemble RBE2 using penalty method. Large stiffness coupling independent to dependent DOFs."""
    penalty = 1e12  # large penalty stiffness
    ind_nid = rbe.independent_node
    if ind_nid not in dof_mgr._nid_to_index: return
    for dep_nid in rbe.dependent_nodes:
        if dep_nid not in dof_mgr._nid_to_index: continue
        for ch in rbe.components:
            comp = int(ch)
            if comp < 1 or comp > 6: continue
            ind_dof = dof_mgr.get_dof(ind_nid, comp)
            dep_dof = dof_mgr.get_dof(dep_nid, comp)
            # K_penalty: penalty * [[1, -1], [-1, 1]]
            rows_k.extend([ind_dof, ind_dof, dep_dof, dep_dof])
            cols_k.extend([ind_dof, dep_dof, ind_dof, dep_dof])
            vals_k.extend([penalty, -penalty, -penalty, penalty])
