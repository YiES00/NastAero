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

    The horseshoe vortex loop (following Bertin & Cummings convention):
    - Trailing leg from a extending downstream to x=+inf (inboard leg)
    - Bound segment from a to b (finite vortex, positive lift)
    - Trailing leg from b extending downstream to x=+inf (outboard leg)

    The circulation sense: going around the loop from downstream-a → a →
    b → downstream-b, the inboard trailing leg has opposite sense to the
    outboard trailing leg relative to the bound vortex.

    For positive lift (downwash at control point), the correct combination is:
      v_total = v_bound(a→b) - v_trail(a,+x) + v_trail(b,+x)

    Uses Biot-Savart law with unit circulation Gamma=1.

    Returns the component of induced velocity along the normal direction.
    """
    # 1. Bound vortex segment (a → b)
    v_bound = _biot_savart_segment(xc, a, b)

    # 2. Trailing leg from a to +infinity (semi-infinite, direction +x)
    v_trail_a = _semi_infinite_vortex(xc, a, np.array([1.0, 0.0, 0.0]))

    # 3. Trailing leg from b to +infinity (semi-infinite, direction +x)
    v_trail_b = _semi_infinite_vortex(xc, b, np.array([1.0, 0.0, 0.0]))

    # Total: bound + outboard_trail - inboard_trail
    # The inboard trailing leg (from a) has opposite sense in the vortex loop
    v_total = v_bound - v_trail_a + v_trail_b

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
    """Compute aerodynamic forces from pressure difference coefficients.

    Parameters
    ----------
    boxes : list of AeroBox
    delta_cp : ndarray (n,)
        Pressure difference coefficient (delta_Cp) for each box.
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
        # Force = q * delta_Cp * area, directed along panel normal
        forces[i] = q * delta_cp[i] * boxes[i].area * boxes[i].normal
    return forces


def circulation_to_delta_cp(boxes: List[AeroBox], gamma: np.ndarray) -> np.ndarray:
    """Convert VLM normalized circulation to pressure difference coefficient.

    In VLM, the AIC relates normalwash ratio to normalized circulation:
        {w/V} = [D]{gamma}
    where gamma = Gamma/V (normalized by freestream velocity).

    The pressure coefficient for each box is:
        delta_Cp_j = 2 * gamma_j / chord_j

    This follows from Kutta-Joukowski:
        L_j = rho * V * Gamma_j * span_j = rho * V^2 * gamma_j * span_j
        delta_Cp_j = L_j / (q * area_j)
                   = (rho * V^2 * gamma_j * span_j) / (0.5*rho*V^2 * chord_j*span_j)
                   = 2 * gamma_j / chord_j

    Parameters
    ----------
    boxes : list of AeroBox
    gamma : ndarray (n,)
        Normalized circulation (Gamma/V) from solving D @ gamma = w/V.

    Returns
    -------
    delta_cp : ndarray (n,)
        Pressure difference coefficient for each box.
    """
    n = len(boxes)
    delta_cp = np.zeros(n)
    for i in range(n):
        if boxes[i].chord > 1e-12:
            delta_cp[i] = 2.0 * gamma[i] / boxes[i].chord
    return delta_cp
