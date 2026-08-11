"""CTRIA3: 3-node triangular plate element (18 DOFs).

CST (Constant Strain Triangle) membrane + Mindlin-Reissner plate bending.

Uses constant-strain membrane, constant-curvature bending, and
1-point reduced integration for transverse shear (prevents shear locking).
"""
from __future__ import annotations
import numpy as np
from .base import BaseElement

class CTria3Element(BaseElement):
    def __init__(self, node_xyz: np.ndarray, E: float, nu: float, t: float, rho: float = 0.0):
        self.nodes = node_xyz  # (3, 3)
        self.E = E; self.nu = nu; self.t = t; self.rho = rho
        self._build_local_system()

    def _build_local_system(self):
        p = self.nodes
        # Element normal
        v1 = p[1] - p[0]; v2 = p[2] - p[0]
        ez = np.cross(v1, v2)
        n = np.linalg.norm(ez)
        if n < 1e-15: raise ValueError("Degenerate CTRIA3")
        self.area = n / 2.0
        ez = ez / n
        ex = v1 / np.linalg.norm(v1)
        ey = np.cross(ez, ex)
        self.T_local = np.array([ex, ey, ez])
        self.center = np.mean(p, axis=0)
        # Local 2D coordinates
        self.xy_local = np.zeros((3, 2))
        for i in range(3):
            d = p[i] - self.center
            self.xy_local[i, 0] = np.dot(d, ex)
            self.xy_local[i, 1] = np.dot(d, ey)

    def dof_count(self): return 18

    def stiffness_matrix(self) -> np.ndarray:
        k_local = self._local_stiffness()
        T = self._build_transform_18x18()
        return T.T @ k_local @ T

    def mass_matrix(self) -> np.ndarray:
        m_local = self._local_mass()
        T = self._build_transform_18x18()
        return T.T @ m_local @ T

    def _local_stiffness(self):
        """Build 18x18 local stiffness (DOF order: u,v,w,rx,ry,rz per node).

        Uses selective reduced integration:
        - Full integration for membrane (CST, constant B-matrix)
        - Full integration for bending (constant curvature)
        - 1-point (centroid) for transverse shear (prevents shear locking)
        """
        E, nu, t = self.E, self.nu, self.t
        xy = self.xy_local
        x1, y1 = xy[0]; x2, y2 = xy[1]; x3, y3 = xy[2]

        # CST membrane
        Dm = (E * t / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        # B-matrix for CST
        b1 = y2-y3; b2 = y3-y1; b3 = y1-y2
        c1 = x3-x2; c2 = x1-x3; c3 = x2-x1
        A2 = 2 * self.area
        Bm = (1/A2) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]])
        km = self.area * Bm.T @ Dm @ Bm

        # Plate bending (constant curvature triangle)
        Db = (E * t**3 / (12 * (1 - nu**2))) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        Bb = (1/A2) * np.array([
            [0, 0, -b1, 0, 0, -b2, 0, 0, -b3],
            [0, c1, 0, 0, c2, 0, 0, c3, 0],
            [0, b1, -c1, 0, b2, -c2, 0, b3, -c3]])
        kb = self.area * Bb.T @ Db @ Bb

        # Transverse shear (Mindlin-Reissner, 1-point at centroid)
        # gamma_xz = dw/dx - ry, gamma_yz = dw/dy + rx
        kappa = 5.0 / 6.0
        Ds = kappa * (E * t / (2 * (1 + nu))) * np.eye(2)
        dNdx = np.array([b1, b2, b3]) / A2
        dNdy = np.array([c1, c2, c3]) / A2
        N_c = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])  # centroid
        Bs = np.zeros((2, 9))
        for i in range(3):
            Bs[0, 3*i] = dNdx[i]       # dw/dx
            Bs[0, 3*i+2] = -N_c[i]     # -ry
            Bs[1, 3*i] = dNdy[i]       # dw/dy
            Bs[1, 3*i+1] = N_c[i]      # +rx
        ks = self.area * Bs.T @ Ds @ Bs

        # Assemble into 18x18
        k = np.zeros((18, 18))
        # Membrane: u,v -> DOFs 0,1, 6,7, 12,13
        mem_dofs = [0,1, 6,7, 12,13]
        for i in range(6):
            for j in range(6):
                k[mem_dofs[i], mem_dofs[j]] += km[i, j]

        # Bending + shear: w,rx,ry -> DOFs 2,3,4, 8,9,10, 14,15,16
        bend_dofs = [2,3,4, 8,9,10, 14,15,16]
        for i in range(9):
            for j in range(9):
                k[bend_dofs[i], bend_dofs[j]] += kb[i, j] + ks[i, j]

        # Drilling DOF (rz): small penalty
        alpha = E * t * self.area * 1e-6
        for n_idx in range(3):
            k[6*n_idx+5, 6*n_idx+5] += alpha

        return k

    def _local_mass(self):
        total_mass = self.rho * self.t * self.area
        m_per_node = total_mass / 3.0
        m = np.zeros((18, 18))
        for n_idx in range(3):
            base = 6 * n_idx
            for i in range(3): m[base+i, base+i] = m_per_node
            rot_inertia = m_per_node * self.t**2 / 12.0
            for i in range(3, 6): m[base+i, base+i] = rot_inertia
        return m

    def _build_transform_18x18(self):
        T = np.zeros((18, 18))
        R = self.T_local
        for i in range(6): T[3*i:3*(i+1), 3*i:3*(i+1)] = R
        return T
