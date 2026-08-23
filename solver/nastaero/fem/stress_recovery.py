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
    #   c ≈ sqrt(I / A) (exact for circular section, approximate otherwise)
    c1_fiber = np.sqrt(I1 / A) if I1 > 0 else 0.0
    c2_fiber = np.sqrt(I2 / A) if I2 > 0 else 0.0

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
