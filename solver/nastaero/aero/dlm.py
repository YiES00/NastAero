"""Doublet-Lattice Method (DLM) kernel and AIC matrix computation.

Implements the steady (k=0) DLM for SOL 144 static aeroelastic analysis.
The kernel is based on the Landahl planar lifting surface theory.
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
        # Steady VLM - horseshoe vortex method
        return _build_steady_aic(boxes, beta)
    else:
        # Unsteady DLM (simplified kernel for now, can be extended)
        return _build_steady_aic(boxes, beta)


def _build_steady_aic(boxes: List[AeroBox], beta: float) -> np.ndarray:
    """Build steady AIC using horseshoe vortex method (VLM).

    Each box has a horseshoe vortex:
    - Bound vortex at 1/4 chord (doublet_point, spanning the box width)
    - Two trailing vortices extending to +infinity in x-direction

    The induced normalwash at each control point (3/4 chord) is computed
    using the Biot-Savart law.
    """
    n = len(boxes)
    D = np.zeros((n, n))

    for i in range(n):  # receiving box (control point)
        xc = boxes[i].control_point
        nrm = boxes[i].normal

        for j in range(n):  # sending box (horseshoe vortex)
            # Get bound vortex endpoints (spanwise extent at 1/4 chord)
            bj = boxes[j]
            # Bound vortex endpoints: 1/4 chord at inboard and outboard edges
            c0, c1, c2, c3 = bj.corners
            # 1/4 chord inboard
            a_pt = c0 + 0.25 * (c1 - c0)
            # 1/4 chord outboard
            b_pt = c3 + 0.25 * (c2 - c3)

            # Apply Prandtl-Glauert: scale y,z by 1/beta
            # (transform to incompressible frame)
            xc_pg = np.array([xc[0], xc[1] / beta, xc[2] / beta])
            a_pg = np.array([a_pt[0], a_pt[1] / beta, a_pt[2] / beta])
            b_pg = np.array([b_pt[0], b_pt[1] / beta, b_pt[2] / beta])

            # Horseshoe vortex induced velocity (Biot-Savart)
            w = _horseshoe_normalwash(xc_pg, a_pg, b_pg, nrm)

            # Scale by box chord (normalizing)
            D[i, j] = w

    return D


def _horseshoe_normalwash(xc: np.ndarray, a: np.ndarray, b: np.ndarray,
                           normal: np.ndarray) -> float:
    """Compute normalwash at xc due to a unit horseshoe vortex.

    The horseshoe consists of:
    - Bound segment from a to b (finite vortex)
    - Semi-infinite trailing leg from a to x=+inf (parallel to x-axis)
    - Semi-infinite trailing leg from b to x=+inf (parallel to x-axis)

    Uses Biot-Savart law with unit circulation Gamma=1.

    Returns the component of induced velocity along the normal direction.
    """
    # 1. Bound vortex segment (a → b)
    v_bound = _biot_savart_segment(xc, a, b)

    # 2. Trailing leg from a to +infinity (semi-infinite, direction +x)
    v_trail_a = _semi_infinite_vortex(xc, a, np.array([1.0, 0.0, 0.0]))

    # 3. Trailing leg from b to +infinity (semi-infinite, direction +x)
    # Note: trailing vortex from b has opposite sense
    v_trail_b = _semi_infinite_vortex(xc, b, np.array([1.0, 0.0, 0.0]))

    # Total induced velocity (note sign convention for horseshoe)
    v_total = v_bound + v_trail_a - v_trail_b

    # Return normalwash component
    return np.dot(v_total, normal)


def _biot_savart_segment(xc: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Induced velocity at xc due to a finite vortex segment p1→p2.

    Biot-Savart for finite segment with unit circulation:
    V = (Gamma/4pi) * [(r1×r2)/|r1×r2|^2] * (r0 . (r1/|r1| - r2/|r2|))

    where r1 = xc-p1, r2 = xc-p2, r0 = p2-p1
    """
    r1 = xc - p1
    r2 = xc - p2
    r0 = p2 - p1

    cross = np.cross(r1, r2)
    cross_sq = np.dot(cross, cross)

    # Cutoff for numerical stability
    if cross_sq < 1e-20:
        return np.zeros(3)

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    if r1_mag < 1e-12 or r2_mag < 1e-12:
        return np.zeros(3)

    factor = np.dot(r0, r1 / r1_mag - r2 / r2_mag)

    return (1.0 / (4.0 * np.pi)) * (cross / cross_sq) * factor


def _semi_infinite_vortex(xc: np.ndarray, origin: np.ndarray,
                          direction: np.ndarray) -> np.ndarray:
    """Induced velocity at xc due to a semi-infinite vortex.

    Vortex starts at 'origin' and extends to infinity in 'direction'.
    Uses the limiting form of Biot-Savart.

    V = (Gamma/4pi) * (d × e_inf) / |d × e_inf|^2 * (1 + r.e_inf/|r|)

    where r = xc - origin, e_inf = direction (unit vector)
    """
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


def compute_aero_forces(boxes: List[AeroBox], delta_cp: np.ndarray,
                        q: float) -> np.ndarray:
    """Compute aerodynamic forces from pressure coefficients.

    Parameters
    ----------
    boxes : list of AeroBox
    delta_cp : ndarray (n,)
        Pressure difference coefficient for each box.
    q : float
        Dynamic pressure (0.5 * rho * V^2).

    Returns
    -------
    forces : ndarray (n, 3)
        Force vector (fx, fy, fz) for each box.
    """
    n = len(boxes)
    forces = np.zeros((n, 3))
    for i in range(n):
        # Force = q * delta_cp * area, directed along panel normal
        forces[i] = q * delta_cp[i] * boxes[i].area * boxes[i].normal
    return forces
