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
    accel = grav_load.get_acceleration_vector() * scale
    node_mass: Dict[int, float] = {}
    for eid, elem in model.elements.items():
        if not hasattr(elem, "property_ref") or elem.property_ref is None: continue
        prop = elem.property_ref; mat = getattr(prop, "material_ref", None)
        if mat is None or mat.rho <= 0: continue
        if elem.type in ("CBAR", "CROD"):
            n1 = model.nodes[elem.node_ids[0]]; n2 = model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            em = mat.rho * prop.A * L + getattr(prop, "nsm", 0) * L
            for nid in elem.node_ids: node_mass[nid] = node_mass.get(nid, 0.) + em/2
        elif elem.type in ("CQUAD4", "CTRIA3"):
            xyz = np.array([model.nodes[nid].xyz_global for nid in elem.node_ids])
            if elem.type == "CQUAD4":
                d13 = xyz[2]-xyz[0]; d24 = xyz[3]-xyz[1]
                area = 0.5*np.linalg.norm(np.cross(d13, d24))
            else:
                v1 = xyz[1]-xyz[0]; v2 = xyz[2]-xyz[0]
                area = 0.5*np.linalg.norm(np.cross(v1, v2))
            em = mat.rho * prop.t * area
            nn = len(elem.node_ids)
            for nid in elem.node_ids: node_mass[nid] = node_mass.get(nid, 0.) + em/nn
    for mid, me in model.masses.items():
        node_mass[me.node_id] = node_mass.get(me.node_id, 0.) + me.mass
    for nid, m in node_mass.items():
        if nid not in dof_mgr._nid_to_index: continue
        nd = dof_mgr.get_node_dofs(nid)
        F[nd[0]] += m*accel[0]; F[nd[1]] += m*accel[1]; F[nd[2]] += m*accel[2]
