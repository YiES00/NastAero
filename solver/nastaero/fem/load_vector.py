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

def _cid_to_basic(vec, cid, model):
    """하중 카드의 방향 성분(CID 기준)을 기본좌표계로 환산한다."""
    if cid and cid in model.coords:
        return np.asarray(model.coords[cid].transform) @ np.asarray(vec)
    return np.asarray(vec)


def _to_cd(vec, nid, model, ):
    """기본좌표계 하중 성분을 그 절점의 변위 좌표계(CD) 성분으로 옮긴다.

    조립된 K/M은 CD!=0 절점에서 CD 성분 자유도로 표현돼 있으므로
    (assembly._apply_cd_transforms), 하중도 같은 기저여야 한다.
    가상일 u_basic^T F_basic = u_cd^T (R^T F_basic) 에서 F_cd = R^T F_basic.
    """
    node = model.nodes.get(nid)
    cd = int(getattr(node, 'cd', 0) or 0) if node is not None else 0
    if cd and cd in model.coords:
        return np.asarray(model.coords[cd].transform).T @ np.asarray(vec)
    return np.asarray(vec)


def _add_load_set(F, model, load_list, scale, dof_mgr):
    for load in load_list:
        if load.type == "FORCE":
            fv = _cid_to_basic(load.get_force_vector(), load.cid, model) * scale
            fv = _to_cd(fv, load.node_id, model)
            nd = dof_mgr.get_node_dofs(load.node_id)
            F[nd[0]] += fv[0]; F[nd[1]] += fv[1]; F[nd[2]] += fv[2]
        elif load.type == "MOMENT":
            mv = _cid_to_basic(load.get_moment_vector(), load.cid, model) * scale
            mv = _to_cd(mv, load.node_id, model)
            nd = dof_mgr.get_node_dofs(load.node_id)
            F[nd[3]] += mv[0]; F[nd[4]] += mv[1]; F[nd[5]] += mv[2]
        elif load.type == "GRAV":
            _add_gravity_load(F, model, load, scale, dof_mgr)

def _add_gravity_load(F, model, grav_load, scale, dof_mgr):
    """Optimized gravity load: uses vectorized node mass computation for shells."""
    accel = _cid_to_basic(grav_load.get_acceleration_vector(),
                          grav_load.cid, model) * scale

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
            em = (mat.rho * prop.t
                  + float(getattr(prop, 'nsm', 0.0) or 0.0)) * area
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
            em = (mat.rho * prop.t
                  + float(getattr(prop, 'nsm', 0.0) or 0.0)) * area
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
    node_ids = dof_mgr.node_ids
    for idx in range(n_nodes):
        m = node_mass_array[idx]
        if m > 0:
            a = _to_cd(accel, node_ids[idx], model)
            base = idx * 6
            F[base] += m * a[0]
            F[base + 1] += m * a[1]
            F[base + 2] += m * a[2]
