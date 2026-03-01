"""Doublet-Lattice Method (DLM) kernel and AIC matrix computation.

Implements the steady (k=0) DLM for SOL 144 static aeroelastic analysis.
The kernel is based on the Landahl planar lifting surface theory.

Performance: AIC build is fully vectorized using NumPy broadcasting —
all n×n Biot-Savart interactions computed in a single pass with no Python loops.
"""
from __future__ import annotations
from typing import List
import numpy as np
from .panel import AeroBox


def build_aic_matrix(boxes: List[AeroBox], mach: float = 0.0,
                     reduced_freq: float = 0.0) -> np.ndarray:
    """Build the Aerodynamic Influence Coefficient (AIC) matrix.

    For steady flow (k=0), this computes the normalwash influence
    coefficients using the Vortex-Lattice Method (VLM) approach:
    horseshoe vortex with bound segment at 1/4 chord and trailing
    legs extending downstream to infinity.

    Parameters
    ----------
    boxes : list of AeroBox
        Aerodynamic boxes from panel mesh generation.
    mach : float
        Freestream Mach number (used for Prandtl-Glauert correction).
    reduced_freq : float
        Reduced frequency k = omega*c/(2*V). k=0 for steady.

    Returns
    -------
    D : ndarray (n, n)
        AIC matrix such that {w} = [D]{cp} or equivalently
        {delta_cp} = [D]^{-1}{w/V} depending on formulation.
        Here we use: w_j/V = sum_i D_ji * Gamma_i  convention.
    """
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))

    # Prandtl-Glauert compressibility factor
    beta = np.sqrt(max(1.0 - mach**2, 0.01))

    if reduced_freq < 1e-10:
        return _build_steady_aic_vectorized(boxes, beta)
    else:
        return _build_steady_aic_vectorized(boxes, beta)


def _build_steady_aic_vectorized(boxes: List[AeroBox], beta: float) -> np.ndarray:
    """Build steady AIC using fully vectorized horseshoe vortex method.

    All n×n Biot-Savart interactions are computed in one pass using
    NumPy broadcasting. No Python for-loops over box pairs.
    """
    n = len(boxes)

    # Pre-extract all geometry into contiguous arrays
    corners = np.array([b.corners for b in boxes])  # (n, 4, 3)
    c0 = corners[:, 0]  # (n, 3) inboard LE
    c1 = corners[:, 1]  # (n, 3) inboard TE
    c2 = corners[:, 2]  # (n, 3) outboard TE
    c3 = corners[:, 3]  # (n, 3) outboard LE

    # 1/4 chord points for horseshoe vortex
    a_pts = c0 + 0.25 * (c1 - c0)  # (n, 3) inboard quarter-chord
    b_pts = c3 + 0.25 * (c2 - c3)  # (n, 3) outboard quarter-chord

    # Control points and normals
    xc_all = np.array([b.control_point for b in boxes])  # (n, 3)
    nrm_all = np.array([b.normal for b in boxes])         # (n, 3)

    # Prandtl-Glauert transformation: scale y, z by 1/beta
    pg_scale = np.array([1.0, 1.0 / beta, 1.0 / beta])
    xc_pg = xc_all * pg_scale   # (n, 3)
    a_pg = a_pts * pg_scale     # (n, 3)
    b_pg = b_pts * pg_scale     # (n, 3)

    # Broadcast: receiving (i) = axis 0, sending (j) = axis 1
    # xc_pg[i] shape: (n, 1, 3) after expand; a_pg[j] shape: (1, n, 3)
    xc_i = xc_pg[:, np.newaxis, :]  # (n, 1, 3)
    a_j = a_pg[np.newaxis, :, :]    # (1, n, 3)
    b_j = b_pg[np.newaxis, :, :]    # (1, n, 3)

    # ============================================================
    # 1. Bound vortex segment: a → b (Biot-Savart)
    # ============================================================
    v_bound = _biot_savart_segment_vec(xc_i, a_j, b_j)  # (n, n, 3)

    # ============================================================
    # 2. Semi-infinite trailing legs (direction = +x)
    # ============================================================
    v_trail_a = _semi_infinite_vortex_vec(xc_i, a_j)  # (n, n, 3)
    v_trail_b = _semi_infinite_vortex_vec(xc_i, b_j)  # (n, n, 3)

    # Total induced velocity (horseshoe: bound - trail_a + trail_b)
    v_total = v_bound - v_trail_a + v_trail_b  # (n, n, 3)

    # Normalwash: dot product with receiving panel normal
    # nrm_all[i]: (n, 1, 3)
    nrm_i = nrm_all[:, np.newaxis, :]  # (n, 1, 3)
    D = np.sum(v_total * nrm_i, axis=2)  # (n, n)

    return D


def _biot_savart_segment_vec(xc, p1, p2):
    """Vectorized Biot-Savart for finite vortex segment p1→p2.

    Parameters: all broadcastable to (n_recv, n_send, 3)
    Returns: induced velocity (n_recv, n_send, 3)
    """
    r1 = xc - p1       # (ni, nj, 3)
    r2 = xc - p2       # (ni, nj, 3)
    r0 = p2 - p1       # (1, nj, 3) or (ni, nj, 3)

    # Cross product r1 × r2
    cross = np.cross(r1, r2)                         # (ni, nj, 3)
    cross_sq = np.sum(cross * cross, axis=2)          # (ni, nj)

    # Magnitudes
    r1_mag = np.sqrt(np.sum(r1 * r1, axis=2))        # (ni, nj)
    r2_mag = np.sqrt(np.sum(r2 * r2, axis=2))        # (ni, nj)

    # Avoid division by zero
    safe = (cross_sq > 1e-20) & (r1_mag > 1e-12) & (r2_mag > 1e-12)

    # r0 · (r1/|r1| - r2/|r2|)
    r1_hat = r1 / np.maximum(r1_mag[..., np.newaxis], 1e-30)
    r2_hat = r2 / np.maximum(r2_mag[..., np.newaxis], 1e-30)
    factor = np.sum(r0 * (r1_hat - r2_hat), axis=2)  # (ni, nj)

    # V = (1/4π) * cross / cross_sq * factor
    coeff = np.where(safe, factor / np.maximum(cross_sq, 1e-30), 0.0)
    v = (1.0 / (4.0 * np.pi)) * cross * coeff[..., np.newaxis]

    return v


def _semi_infinite_vortex_vec(xc, origin):
    """Vectorized semi-infinite vortex (direction = +x).

    Parameters: all broadcastable to (n_recv, n_send, 3)
    Returns: induced velocity (n_recv, n_send, 3)
    """
    r = xc - origin                                   # (ni, nj, 3)
    r_mag = np.sqrt(np.sum(r * r, axis=2))             # (ni, nj)

    # direction × r where direction = [1, 0, 0]
    # cross([1,0,0], [rx, ry, rz]) = [0*rz - 0*ry, 0*rx - 1*rz, 1*ry - 0*rx]
    #                               = [0, -rz, ry]
    cross = np.empty_like(r)
    cross[..., 0] = 0.0
    cross[..., 1] = -r[..., 2]
    cross[..., 2] = r[..., 1]

    cross_sq = np.sum(cross * cross, axis=2)           # (ni, nj)

    safe = (r_mag > 1e-12) & (cross_sq > 1e-20)

    # cos_theta = r · [1,0,0] / |r| = rx / |r|
    cos_theta = r[..., 0] / np.maximum(r_mag, 1e-30)

    coeff = np.where(safe,
                     (1.0 + cos_theta) / np.maximum(cross_sq, 1e-30),
                     0.0)
    v = (1.0 / (4.0 * np.pi)) * cross * coeff[..., np.newaxis]

    return v


# ============================================================
# Scalar reference functions (kept for testing/debugging)
# ============================================================

def _horseshoe_normalwash(xc, a, b, normal):
    """Scalar: normalwash at xc due to unit horseshoe vortex."""
    v_bound = _biot_savart_segment(xc, a, b)
    v_trail_a = _semi_infinite_vortex(xc, a, np.array([1.0, 0.0, 0.0]))
    v_trail_b = _semi_infinite_vortex(xc, b, np.array([1.0, 0.0, 0.0]))
    v_total = v_bound - v_trail_a + v_trail_b
    return np.dot(v_total, normal)


def _biot_savart_segment(xc, p1, p2):
    """Scalar: Biot-Savart for finite vortex segment p1→p2."""
    r1 = xc - p1
    r2 = xc - p2
    r0 = p2 - p1
    cross = np.cross(r1, r2)
    cross_sq = np.dot(cross, cross)
    if cross_sq < 1e-20:
        return np.zeros(3)
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    if r1_mag < 1e-12 or r2_mag < 1e-12:
        return np.zeros(3)
    factor = np.dot(r0, r1 / r1_mag - r2 / r2_mag)
    return (1.0 / (4.0 * np.pi)) * (cross / cross_sq) * factor


def _semi_infinite_vortex(xc, origin, direction):
    """Scalar: semi-infinite vortex induced velocity."""
    r = xc - origin
    r_mag = np.linalg.norm(r)
    if r_mag < 1e-12:
        return np.zeros(3)
    cross = np.cross(direction, r)
    cross_sq = np.dot(cross, cross)
    if cross_sq < 1e-20:
        return np.zeros(3)
    cos_theta = np.dot(r, direction) / r_mag
    return (1.0 / (4.0 * np.pi)) * (cross / cross_sq) * (1.0 + cos_theta)


# ============================================================
# Post-processing helpers (also vectorized)
# ============================================================

def compute_aero_forces(boxes: List[AeroBox], delta_cp: np.ndarray,
                        q: float) -> np.ndarray:
    """Compute aerodynamic forces from pressure difference coefficients."""
    areas = np.array([b.area for b in boxes])        # (n,)
    normals = np.array([b.normal for b in boxes])     # (n, 3)
    return q * (delta_cp * areas)[:, np.newaxis] * normals


def circulation_to_delta_cp(boxes: List[AeroBox], gamma: np.ndarray) -> np.ndarray:
    """Convert VLM normalized circulation to pressure difference coefficient."""
    chords = np.array([b.chord for b in boxes])
    safe = chords > 1e-12
    delta_cp = np.where(safe, 2.0 * gamma / np.maximum(chords, 1e-30), 0.0)
    return delta_cp
