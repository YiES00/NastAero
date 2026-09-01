"""Element stress recovery from nodal displacements.

Computes per-element stress components (membrane + bending for shells,
axial + bending for bars) from the free-DOF displacement vector, then
averages to nodes for visualization.

The stress components are split into membrane and bending contributions
so that von Mises can be evaluated at both top (z = +t/2) and bottom
(z = -t/2) surfaces:

    σ_top = σ_mem + σ_bend
    σ_bot = σ_mem - σ_bend
    vm = max(vm(σ_top), vm(σ_bot))

This separation preserves linearity for ROM superposition: each
component can be scaled and summed independently.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional

from ..config import logger
from ..elements.quad4 import CQuad4Element
from ..elements.tria3 import CTria3Element


def recover_stresses_to_nodes(
    model,
    dof_mgr,
    f_dofs: List[int],
    u_free: np.ndarray,
    sorted_nids: List[int],
) -> np.ndarray:
    """Recover per-element stress and average to nodes.

    Parameters
    ----------
    model : BDFModel
        Cross-referenced BDF model.
    dof_mgr : DOFManager
        DOF manager from FEM assembly.
    f_dofs : list of int
        Free DOF indices (global numbering).
    u_free : ndarray (n_free,)
        Free-DOF displacement vector.
    sorted_nids : list of int
        Sorted node IDs (defines output ordering).

    Returns
    -------
    S_nodal : ndarray (n_nodes, 6)
        Node-averaged stress components:
        [σxx_mem, σyy_mem, σxy_mem, σxx_bend, σyy_bend, σxy_bend]
    """
    f_dof_index = {d: i for i, d in enumerate(f_dofs)}
    nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
    n_nodes = len(sorted_nids)

    S_sum = np.zeros((n_nodes, 6))
    count = np.zeros(n_nodes)
    n_recovered = 0
    n_skipped = 0

    for eid in sorted(model.elements.keys()):
        elem = model.elements[eid]
        etype = elem.type

        try:
            if etype == "CQUAD4":
                s_mem, s_bend = _recover_cquad4(
                    elem, model, dof_mgr, f_dof_index, u_free)
            elif etype == "CTRIA3":
                s_mem, s_bend = _recover_ctria3(
                    elem, model, dof_mgr, f_dof_index, u_free)
            elif etype in ("CBAR", "CBEAM"):
                s_mem, s_bend = _recover_cbar(
                    elem, model, dof_mgr, f_dof_index, u_free)
            elif etype == "CROD":
                s_mem, s_bend = _recover_crod(
                    elem, model, dof_mgr, f_dof_index, u_free)
            else:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        # Accumulate at connected nodes
        for nid in elem.node_ids:
            idx = nid_to_idx.get(nid)
            if idx is not None:
                S_sum[idx, :3] += s_mem
                S_sum[idx, 3:] += s_bend
                count[idx] += 1
        n_recovered += 1

    # Average
    mask = count > 0
    S_sum[mask] /= count[mask, np.newaxis]

    if n_skipped > 0:
        logger.debug("  Stress recovery: %d recovered, %d skipped",
                     n_recovered, n_skipped)

    return S_sum


def compute_von_mises(S: np.ndarray) -> np.ndarray:
    """Compute von Mises stress from (n, 6) stress component array.

    Evaluates at both top (mem + bend) and bottom (mem - bend) surfaces
    and returns the maximum.

    Parameters
    ----------
    S : ndarray (n, 6)
        [σxx_mem, σyy_mem, σxy_mem, σxx_bend, σyy_bend, σxy_bend]

    Returns
    -------
    vm : ndarray (n,)
        Von Mises stress at each point.
    """
    mem = S[:, :3]
    bend = S[:, 3:]

    top = mem + bend
    bot = mem - bend

    vm_top = _von_mises_2d(top[:, 0], top[:, 1], top[:, 2])
    vm_bot = _von_mises_2d(bot[:, 0], bot[:, 1], bot[:, 2])

    return np.maximum(vm_top, vm_bot)


def _von_mises_2d(sxx, syy, sxy):
    """Von Mises for 2D plane stress: sqrt(σxx² + σyy² - σxx·σyy + 3·σxy²)."""
    return np.sqrt(np.maximum(
        sxx * sxx + syy * syy - sxx * syy + 3.0 * sxy * sxy,
        0.0,
    ))


# =====================================================================
# Per-element stress recovery functions
# =====================================================================

def _get_shell_material(elem, model):
    """Extract (E, nu, t) for a shell element.  Returns None on failure."""
    pid = elem.pid
    prop = model.properties.get(pid)
    if prop is None:
        return None
    # PCOMP: use smeared equivalent
    if hasattr(prop, 'equivalent_isotropic'):
        E, nu, t, _ = prop.equivalent_isotropic(model.materials)
    else:
        mat = (prop.material_ref
               if hasattr(prop, 'material_ref') and prop.material_ref
               else model.materials.get(getattr(prop, 'mid', 0)))
        if mat is None:
            return None
        E = mat.E
        nu = mat.nu
        t = getattr(prop, 't', 0.0)
    if E <= 0 or t <= 0:
        return None
    return E, nu, t


def _extract_elem_u(node_ids, n_dof_per_node, dof_mgr, f_dof_index, u_free):
    """Extract element DOF displacement vector from global u_free."""
    n_nodes = len(node_ids)
    u = np.zeros(n_nodes * n_dof_per_node)
    for i, nid in enumerate(node_ids):
        for comp in range(n_dof_per_node):
            gdof = dof_mgr.get_dof(nid, comp + 1)
            idx = f_dof_index.get(gdof)
            if idx is not None:
                u[i * n_dof_per_node + comp] = u_free[idx]
    return u


def _recover_cquad4(elem, model, dof_mgr, f_dof_index, u_free):
    """Recover stress at CQUAD4 element center.

    Returns (sigma_mem, sigma_bend) each (3,): [σxx, σyy, σxy].
    sigma_bend is at z = +t/2 (top surface).
    """
    props = _get_shell_material(elem, model)
    if props is None:
        return np.zeros(3), np.zeros(3)
    E, nu, t = props

    nids = elem.node_ids
    node_xyz = np.array([model.nodes[n].xyz_global for n in nids])
    q4 = CQuad4Element(node_xyz, E, nu, t)

    # Extract and transform to local
    u_global = _extract_elem_u(nids, 6, dof_mgr, f_dof_index, u_free)
    T = q4._build_transform_24x24()
    u_local = T @ u_global

    # Stress constitutive (no thickness factor — gives true stress in N/mm²)
    c = E / (1.0 - nu * nu)
    D_stress = c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    # Shape function derivatives at center (ξ=0, η=0)
    _, dNdxi, dNdeta = q4._shape_functions(0.0, 0.0)
    J = q4._jacobian(dNdxi, dNdeta)
    Jinv = np.linalg.inv(J)
    dNdx = Jinv[0, 0] * dNdxi + Jinv[0, 1] * dNdeta
    dNdy = Jinv[1, 0] * dNdxi + Jinv[1, 1] * dNdeta

    # Membrane B-matrix (3 × 8) at center
    Bm = np.zeros((3, 8))
    for n_idx in range(4):
        Bm[0, 2 * n_idx] = dNdx[n_idx]
        Bm[1, 2 * n_idx + 1] = dNdy[n_idx]
        Bm[2, 2 * n_idx] = dNdy[n_idx]
        Bm[2, 2 * n_idx + 1] = dNdx[n_idx]

    mem_dofs = []
    for n_idx in range(4):
        mem_dofs.extend([6 * n_idx, 6 * n_idx + 1])
    u_mem = u_local[mem_dofs]
    sigma_mem = D_stress @ (Bm @ u_mem)

    # Bending B-matrix (3 × 8) at center
    Bb = np.zeros((3, 8))
    for n_idx in range(4):
        Bb[0, 2 * n_idx + 1] = -dNdx[n_idx]      # -d(ry)/dx
        Bb[1, 2 * n_idx] = dNdy[n_idx]             # d(rx)/dy
        Bb[2, 2 * n_idx] = dNdx[n_idx]             # d(rx)/dx
        Bb[2, 2 * n_idx + 1] = -dNdy[n_idx]        # -d(ry)/dy

    bend_dofs = []
    for n_idx in range(4):
        bend_dofs.extend([6 * n_idx + 3, 6 * n_idx + 4])
    u_bend = u_local[bend_dofs]
    kappa = Bb @ u_bend
    sigma_bend = D_stress @ kappa * (t / 2.0)

    return sigma_mem, sigma_bend


def _recover_ctria3(elem, model, dof_mgr, f_dof_index, u_free):
    """Recover stress for CTRIA3 (constant strain triangle).

    Returns (sigma_mem, sigma_bend) each (3,): [σxx, σyy, σxy].
    """
    props = _get_shell_material(elem, model)
    if props is None:
        return np.zeros(3), np.zeros(3)
    E, nu, t = props

    nids = elem.node_ids
    node_xyz = np.array([model.nodes[n].xyz_global for n in nids])
    t3 = CTria3Element(node_xyz, E, nu, t)

    u_global = _extract_elem_u(nids, 6, dof_mgr, f_dof_index, u_free)
    T = t3._build_transform_18x18()
    u_local = T @ u_global

    c = E / (1.0 - nu * nu)
    D_stress = c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    xy = t3.xy_local
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    A2 = 2.0 * t3.area

    # CST membrane B-matrix (constant)
    Bm = (1.0 / A2) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    mem_dofs = [0, 1, 6, 7, 12, 13]
    u_mem = u_local[mem_dofs]
    sigma_mem = D_stress @ (Bm @ u_mem)

    # Bending B-matrix (constant curvature triangle)
    Bb = (1.0 / A2) * np.array([
        [0, 0, -b1, 0, 0, -b2, 0, 0, -b3],
        [0, c1, 0, 0, c2, 0, 0, c3, 0],
        [0, b1, -c1, 0, b2, -c2, 0, b3, -c3],
    ])
    bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    u_bend = u_local[bend_dofs]
    kappa = Bb @ u_bend
    sigma_bend = D_stress @ kappa * (t / 2.0)

    return sigma_mem, sigma_bend


def _extreme_fiber_distances(prop):
    """보 단면의 각 굽힘 평면 최외곽 거리 (c1: 평면1, c2: 평면2).

    PBAR 응력점이 있으면 그 최대 절대좌표를, PBARL/PBEAML이면 형상
    치수에서 계산한다. 아무 정보도 없으면 (0, 0)을 돌려주고 호출부가
    경고 후 굽힘응력을 0으로 남긴다.
    """
    # PBAR 응력점: (C1,C2) (D1,D2) (E1,E2) (F1,F2) — 1은 평면1 좌표
    pts = [(getattr(prop, a, 0.0) or 0.0, getattr(prop, b, 0.0) or 0.0)
           for a, b in (("c1", "c2"), ("d1", "d2"),
                        ("e1", "e2"), ("f1", "f2"))]
    c1 = max(abs(p[0]) for p in pts)
    c2 = max(abs(p[1]) for p in pts)
    if c1 > 0.0 or c2 > 0.0:
        return c1, c2

    tname = getattr(prop, "type_name", "")
    dims = list(getattr(prop, "dims", []) or [])
    if tname and dims:
        if tname == "ROD":
            return dims[0], dims[0]
        if tname in ("TUBE", "TUBE2"):
            return dims[0], dims[0]
        if tname == "BAR":
            # BAR: DIM1=폭(평면2 방향), DIM2=깊이(평면1 방향)
            return dims[1] / 2.0, dims[0] / 2.0
        if tname == "BOX" and len(dims) >= 2:
            return dims[1] / 2.0, dims[0] / 2.0
        if tname in ("I", "CHAN", "CHAN1", "CHAN2", "I1", "Z", "HAT",
                     "T", "L") and dims:
            # 깊이 방향 최외곽만 신뢰 (약축은 형상별 도심 위치가 달라
            # 보수적으로 깊이 절반을 함께 사용)
            depth = {"I": dims[0], "CHAN": dims[1], "CHAN1": dims[3],
                     "CHAN2": dims[2], "I1": dims[3], "Z": dims[3],
                     "HAT": dims[0], "T": dims[1] if len(dims) > 1 else 0.0,
                     "L": max(dims[0], dims[1]) if len(dims) > 1 else 0.0,
                     }.get(tname, 0.0)
            return depth / 2.0, depth / 2.0
    return 0.0, 0.0


def _recover_cbar(elem, model, dof_mgr, f_dof_index, u_free):
    """Recover stress for CBAR/CBEAM element.

    Returns (sigma_mem, sigma_bend) each (3,): [σxx, σyy, σxy].
    For beams, all stress is axial: σyy = σxy = 0.
    sigma_mem[0] = axial stress (P/A).
    sigma_bend[0] = bending stress at extreme fiber (combined planes).
    """
    from ..fem.coordinate_systems import build_beam_transform, build_transform_12x12

    nids = elem.node_ids
    if len(nids) < 2:
        return np.zeros(3), np.zeros(3)

    pid = elem.pid
    prop = model.properties.get(pid)
    if prop is None:
        return np.zeros(3), np.zeros(3)

    mat = (prop.material_ref
           if hasattr(prop, 'material_ref') and prop.material_ref
           else model.materials.get(getattr(prop, 'mid', 0)))
    if mat is None:
        return np.zeros(3), np.zeros(3)

    E = mat.E
    A = getattr(prop, 'A', 0.0)
    I1 = getattr(prop, 'I1', 0.0)
    I2 = getattr(prop, 'I2', 0.0)
    if E <= 0 or A <= 0:
        return np.zeros(3), np.zeros(3)

    n1_xyz = model.nodes[nids[0]].xyz_global
    n2_xyz = model.nodes[nids[1]].xyz_global
    diff = n2_xyz - n1_xyz
    L = np.linalg.norm(diff)
    if L < 1e-12:
        return np.zeros(3), np.zeros(3)

    # Orientation vector — 조립부와 동일하게 GA의 CD 좌표계로 해석
    from .assembly import _resolve_bar_orientation
    n1_node = model.nodes[nids[0]]
    v_vec = _resolve_bar_orientation(elem, model, n1_node)

    Lambda = build_beam_transform(n1_xyz, n2_xyz, v_vec)
    T12 = build_transform_12x12(Lambda)

    u_global = _extract_elem_u(nids, 6, dof_mgr, f_dof_index, u_free)
    u_local = T12 @ u_global

    # Axial stress: σ = E · (u2 - u1) / L
    sigma_axial = E * (u_local[6] - u_local[0]) / L

    # Bending: approximate extreme fiber distance from I and A
    # 최외곽 거리는 단면 정보에서 얻는다. 우선순위:
    #  1) PBAR 응력 회수점 C/D/E/F (각 평면의 최대 |좌표|)
    #  2) PBARL/PBEAML 형상 치수(타입별 반깊이)
    #  3) 없으면 회수 불가 — 0 응력 + 경고 (sqrt(I/A)는 원형에서도
    #     실제 최외곽의 절반이라 굽힘응력을 50% 과소평가한다)
    c1_fiber, c2_fiber = _extreme_fiber_distances(prop)
    if (I1 > 0 and c1_fiber <= 0.0) or (I2 > 0 and c2_fiber <= 0.0):
        logger.warning(
            "CBAR/CBEAM %s: 응력 회수점(C1..F2)도 단면 형상도 없어 "
            "굽힘응력을 회수하지 않는다 (PBAR 연속행에 응력점을 "
            "지정할 것)", getattr(elem, 'eid', '?'))

    # Curvature at beam center from Hermite interpolation:
    #   d²v/dx² at x=L/2 = (θ₂ - θ₁) / L
    # xz-plane bending (v, θz): DOFs 1,5,7,11
    kappa_z = (u_local[11] - u_local[5]) / L
    # xy-plane bending (w, θy): DOFs 2,4,8,10
    kappa_y = (u_local[10] - u_local[4]) / L

    # Bending stress at extreme fiber (combined both planes)
    sigma_bend_z = E * c1_fiber * kappa_z
    sigma_bend_y = E * c2_fiber * kappa_y

    sigma_mem = np.array([sigma_axial, 0.0, 0.0])
    sigma_bend = np.array([sigma_bend_z + sigma_bend_y, 0.0, 0.0])

    return sigma_mem, sigma_bend


def _recover_crod(elem, model, dof_mgr, f_dof_index, u_free):
    """Recover stress for CROD element (axial only).

    Returns (sigma_mem, sigma_bend) each (3,).
    """
    nids = elem.node_ids
    if len(nids) < 2:
        return np.zeros(3), np.zeros(3)

    pid = elem.pid
    prop = model.properties.get(pid)
    if prop is None:
        return np.zeros(3), np.zeros(3)
    mat = (prop.material_ref
           if hasattr(prop, 'material_ref') and prop.material_ref
           else model.materials.get(getattr(prop, 'mid', 0)))
    if mat is None:
        return np.zeros(3), np.zeros(3)

    E = mat.E
    A = getattr(prop, 'A', 0.0)
    if E <= 0 or A <= 0:
        return np.zeros(3), np.zeros(3)

    n1_xyz = model.nodes[nids[0]].xyz_global
    n2_xyz = model.nodes[nids[1]].xyz_global
    diff = n2_xyz - n1_xyz
    L = np.linalg.norm(diff)
    if L < 1e-12:
        return np.zeros(3), np.zeros(3)

    # Rod is purely axial: project global DOFs onto rod axis
    ex = diff / L
    u1 = _extract_elem_u([nids[0]], 6, dof_mgr, f_dof_index, u_free)[:3]
    u2 = _extract_elem_u([nids[1]], 6, dof_mgr, f_dof_index, u_free)[:3]

    # Axial displacement = projection onto rod axis
    du_axial = np.dot(u2 - u1, ex)
    sigma_axial = E * du_axial / L

    sigma_mem = np.array([sigma_axial, 0.0, 0.0])
    sigma_bend = np.zeros(3)

    return sigma_mem, sigma_bend
