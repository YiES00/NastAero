"""Assemble the global load vector from parsed load cards."""
from __future__ import annotations
from typing import Dict
import numpy as np
from .dof_manager import DOFManager
from ..bdf.model import BDFModel, Subcase
from ..config import logger

def assemble_load_vector(model: BDFModel, subcase: Subcase, dof_mgr: DOFManager) -> np.ndarray:
    ndof = dof_mgr.total_dof; F = np.zeros(ndof)
    load_id = subcase.load_id
    if load_id == 0: return F
    if load_id in model.load_combinations:
        lc = model.load_combinations[load_id]
        for sf, lid in zip(lc.scale_factors, lc.load_ids):
            if lid in model.loads:
                _add_load_set(F, model, model.loads[lid], lc.scale * sf, dof_mgr)
    elif load_id in model.loads:
        _add_load_set(F, model, model.loads[load_id], 1.0, dof_mgr)
    else:
        logger.warning("Load set %d not found", load_id)
    return F

def _add_load_set(F, model, load_list, scale, dof_mgr):
    for load in load_list:
        if load.type == "FORCE":
            fv = load.get_force_vector() * scale
            nd = dof_mgr.get_node_dofs(load.node_id)
            F[nd[0]] += fv[0]; F[nd[1]] += fv[1]; F[nd[2]] += fv[2]
        elif load.type == "MOMENT":
            mv = load.get_moment_vector() * scale
            nd = dof_mgr.get_node_dofs(load.node_id)
            F[nd[3]] += mv[0]; F[nd[4]] += mv[1]; F[nd[5]] += mv[2]
        elif load.type == "GRAV":
            _add_gravity_load(F, model, load, scale, dof_mgr)

def _add_gravity_load(F, model, grav_load, scale, dof_mgr):
    """Optimized gravity load: uses vectorized node mass computation for shells."""
    accel = grav_load.get_acceleration_vector() * scale

    # Use numpy array for node masses (indexed by DOF manager index)
    n_nodes = dof_mgr.n_nodes
    node_mass_array = np.zeros(n_nodes)

    # Process beam/rod elements
    for eid, elem in model.elements.items():
        if not hasattr(elem, "property_ref") or elem.property_ref is None:
            continue
        prop = elem.property_ref
        mat = getattr(prop, "material_ref", None)
        if mat is None or mat.rho <= 0:
            continue

        if elem.type in ("CBAR", "CROD"):
            n1 = model.nodes[elem.node_ids[0]]
            n2 = model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            em = mat.rho * prop.A * L + getattr(prop, "nsm", 0) * L
            for nid in elem.node_ids:
                if nid in dof_mgr._nid_to_index:
                    node_mass_array[dof_mgr._nid_to_index[nid]] += em / 2.0

        elif elem.type == "CQUAD4":
            # Vectorized area computation
            nids = elem.node_ids
            p0 = model.nodes[nids[0]].xyz_global
            p1 = model.nodes[nids[1]].xyz_global
            p2 = model.nodes[nids[2]].xyz_global
            p3 = model.nodes[nids[3]].xyz_global
            d13 = p2 - p0
            d24 = p3 - p1
            area = 0.5 * np.linalg.norm(np.cross(d13, d24))
            em = mat.rho * prop.t * area
            m_per_node = em / 4.0
            for nid in nids:
                if nid in dof_mgr._nid_to_index:
                    node_mass_array[dof_mgr._nid_to_index[nid]] += m_per_node

        elif elem.type == "CTRIA3":
            nids = elem.node_ids
            p0 = model.nodes[nids[0]].xyz_global
            p1 = model.nodes[nids[1]].xyz_global
            p2 = model.nodes[nids[2]].xyz_global
            v1 = p1 - p0
            v2 = p2 - p0
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            em = mat.rho * prop.t * area
            m_per_node = em / 3.0
            for nid in nids:
                if nid in dof_mgr._nid_to_index:
                    node_mass_array[dof_mgr._nid_to_index[nid]] += m_per_node

    # Concentrated masses (CONM2)
    for mid, me in model.masses.items():
        if me.node_id in dof_mgr._nid_to_index:
            node_mass_array[dof_mgr._nid_to_index[me.node_id]] += me.mass

    # Vectorized force application
    # F[nid*6 + 0] += mass * accel[0], etc.
    for idx in range(n_nodes):
        m = node_mass_array[idx]
        if m > 0:
            base = idx * 6
            F[base] += m * accel[0]
            F[base + 1] += m * accel[1]
            F[base + 2] += m * accel[2]
