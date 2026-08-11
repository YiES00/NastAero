"""Structural-aerodynamic interpolation splines.

Implements:
- Infinite Plate Spline (IPS) for SPLINE1
- Beam Spline for SPLINE2
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .panel import AeroBox


def _assemble_ips_system(struct_nodes: np.ndarray, dz: float = 0.0) -> np.ndarray:
    """Assemble and invert the augmented IPS system matrix.

    [K  P] [w]   [fs]
    [P' 0] [a] = [0 ]
    where K_ij = G(r_ij) + dz * delta_ij and P = [1, x, y].

    Returns A_inv of shape (ns+3, ns+3). Shared by the displacement and
    slope interpolation builders so both use identical spline weights.
    """
    ns = struct_nodes.shape[0]
    K = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            r = np.linalg.norm(struct_nodes[i, :2] - struct_nodes[j, :2])
            K[i, j] = _green_function(r)
        K[i, i] += dz  # flexibility/smoothing

    P = np.ones((ns, 3))
    P[:, 1] = struct_nodes[:, 0]
    P[:, 2] = struct_nodes[:, 1]

    n_aug = ns + 3
    A = np.zeros((n_aug, n_aug))
    A[:ns, :ns] = K
    A[:ns, ns:] = P
    A[ns:, :ns] = P.T

    try:
        return np.linalg.solve(A, np.eye(n_aug))
    except np.linalg.LinAlgError:
        A += np.eye(n_aug) * 1e-10
        return np.linalg.solve(A, np.eye(n_aug))


def build_ips_spline(struct_nodes: np.ndarray, aero_points: np.ndarray,
                     dz: float = 0.0) -> np.ndarray:
    """Build Infinite Plate Spline (IPS) interpolation matrix.

    The IPS uses the thin-plate Green's function to interpolate
    displacements from structural grid points to aerodynamic points.

    Green's function: G(r) = r^2 * (ln(r) - 1) / (8*pi)

    The spline satisfies:
    - Exact interpolation at structural nodes
    - Smooth interpolation elsewhere
    - Augmented with polynomial (linear) terms for rigid body exactness

    Parameters
    ----------
    struct_nodes : ndarray (ns, 3)
        Structural grid point coordinates.
    aero_points : ndarray (na, 3)
        Aerodynamic point coordinates.
    dz : float
        Flexibility parameter (0 = rigid spline).

    Returns
    -------
    G_ka : ndarray (na, ns)
        Interpolation matrix mapping structural z-displacements
        to aerodynamic z-displacements.
    """
    ns = struct_nodes.shape[0]
    na = aero_points.shape[0]
    if ns == 0 or na == 0:
        return np.zeros((na, ns))

    A_inv = _assemble_ips_system(struct_nodes, dz)
    n_aug = ns + 3

    G_ka = np.zeros((na, ns))
    for k in range(na):
        rhs = np.zeros(n_aug)
        for j in range(ns):
            r = np.linalg.norm(aero_points[k, :2] - struct_nodes[j, :2])
            rhs[j] = _green_function(r)
        rhs[ns] = 1.0
        rhs[ns + 1] = aero_points[k, 0]
        rhs[ns + 2] = aero_points[k, 1]
        weights = A_inv @ rhs
        G_ka[k, :] = weights[:ns]

    return G_ka


def build_ips_spline_slope(struct_nodes: np.ndarray,
                           aero_points: np.ndarray,
                           dz: float = 0.0) -> np.ndarray:
    """Build the streamwise-slope (dz/dx) IPS interpolation matrix.

    Differentiates the interpolated IPS surface analytically in x, which
    is the SPLINE1 (surface spline) semantics used by MSC Nastran: the
    panel slope comes from the derivative of the displacement field
    built from structural z-translations, not from nodal rotations.

    Derivative kernel:  dG/dx = x * (2 ln r - 1) / (8 pi),
    which tends to 0 as r -> 0 (no singularity, since |x| <= r).
    The linear polynomial term contributes its x-coefficient.

    Parameters
    ----------
    struct_nodes : ndarray (ns, 3)
        Structural grid point coordinates.
    aero_points : ndarray (na, 3)
        Aerodynamic point coordinates (slope evaluation points).
    dz : float
        Flexibility parameter (0 = rigid spline). Must match the value
        used for the displacement matrix so both derive from the same
        spline surface.

    Returns
    -------
    G_ka_x : ndarray (na, ns)
        Matrix mapping structural z-displacements to the streamwise
        surface slope dz/dx at the aerodynamic points.
    """
    ns = struct_nodes.shape[0]
    na = aero_points.shape[0]
    if ns == 0 or na == 0:
        return np.zeros((na, ns))

    A_inv = _assemble_ips_system(struct_nodes, dz)
    n_aug = ns + 3

    G_ka_x = np.zeros((na, ns))
    for k in range(na):
        rhs = np.zeros(n_aug)
        for j in range(ns):
            dx = aero_points[k, 0] - struct_nodes[j, 0]
            dy = aero_points[k, 1] - struct_nodes[j, 1]
            r = np.hypot(dx, dy)
            rhs[j] = _green_function_dx(dx, r)
        # d/dx of [1, x, y] = [0, 1, 0]
        rhs[ns + 1] = 1.0
        weights = A_inv @ rhs
        G_ka_x[k, :] = weights[:ns]

    return G_ka_x


def build_beam_spline(struct_nodes: np.ndarray, aero_points: np.ndarray,
                      axis: int = 1) -> np.ndarray:
    """Build beam spline interpolation matrix (1D).

    Uses cubic spline interpolation along a single axis (typically span).

    Parameters
    ----------
    struct_nodes : ndarray (ns, 3)
        Structural node coordinates.
    aero_points : ndarray (na, 3)
        Aerodynamic point coordinates.
    axis : int
        Spanwise axis index (0=x, 1=y, 2=z). Default y.

    Returns
    -------
    G_ka : ndarray (na, ns)
        Interpolation matrix.
    """
    ns = struct_nodes.shape[0]
    na = aero_points.shape[0]

    if ns < 2 or na == 0:
        return np.zeros((na, ns))

    # Sort structural nodes by the axis coordinate
    s_coords = struct_nodes[:, axis]
    sort_idx = np.argsort(s_coords)
    s_sorted = s_coords[sort_idx]

    a_coords = aero_points[:, axis]

    # Linear interpolation weights
    G_ka_sorted = np.zeros((na, ns))
    for k in range(na):
        eta = a_coords[k]
        # Find interval
        idx = np.searchsorted(s_sorted, eta) - 1
        idx = max(0, min(idx, ns - 2))

        s0 = s_sorted[idx]
        s1 = s_sorted[idx + 1]
        ds = s1 - s0
        if abs(ds) < 1e-12:
            G_ka_sorted[k, idx] = 1.0
        else:
            t = (eta - s0) / ds
            t = max(0.0, min(1.0, t))
            G_ka_sorted[k, idx] = 1.0 - t
            G_ka_sorted[k, idx + 1] = t

    # Un-sort: map back to original structural node ordering
    G_ka = np.zeros((na, ns))
    for j_sorted in range(ns):
        j_orig = sort_idx[j_sorted]
        G_ka[:, j_orig] = G_ka_sorted[:, j_sorted]

    return G_ka


def build_spline_matrix(bdf_model, boxes: List[AeroBox],
                        struct_node_ids: List[int]) -> np.ndarray:
    """Build the full structural-to-aero interpolation matrix.

    Maps structural z-displacements at SETG nodes to
    aerodynamic z-displacements at box control points.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    struct_node_ids : list of int
        Structural node IDs in the spline set.

    Returns
    -------
    G_ka : ndarray (n_aero, n_struct)
        Interpolation matrix.
    """
    # Collect structural node coordinates
    struct_xyz = np.array([bdf_model.nodes[nid].xyz_global
                           for nid in struct_node_ids])

    # Collect aerodynamic control points
    aero_pts = np.array([box.control_point for box in boxes])

    # Determine spline type from model splines
    # Default to IPS
    use_ips = True
    dz = 0.0
    for sid, spline in bdf_model.splines.items():
        if hasattr(spline, 'method'):
            if spline.method == "IPS" or spline.method == "FPS":
                use_ips = True
            dz = getattr(spline, 'dz', 0.0)
            break

    if use_ips:
        return build_ips_spline(struct_xyz, aero_pts, dz)
    else:
        return build_beam_spline(struct_xyz, aero_pts)


def _green_function(r: float) -> float:
    """Thin-plate Green's function: G(r) = r^2 * (ln(r) - 1) / (8*pi)."""
    if r < 1e-12:
        return 0.0
    return r * r * (np.log(r) - 1.0) / (8.0 * np.pi)


def _green_function_dx(dx: float, r: float) -> float:
    """x-derivative of the thin-plate Green's function.

    dG/dx = dG/dr * dr/dx = [r (2 ln r - 1) / (8 pi)] * (dx / r)
          = dx * (2 ln r - 1) / (8 pi)

    Finite limit 0 as r -> 0 because |dx| <= r and r ln r -> 0.
    """
    if r < 1e-12:
        return 0.0
    return dx * (2.0 * np.log(r) - 1.0) / (8.0 * np.pi)
