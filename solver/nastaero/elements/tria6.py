"""CTRIA6: 6-node quadratic triangular shell element (36 DOFs).

Uses 3-point Gauss triangle integration for membrane and bending.
6 DOFs per node: u, v, w, rx, ry, rz.
"""
from __future__ import annotations
import numpy as np
from .base import BaseElement

# 3-point Gauss quadrature for triangle (in area coordinates L1, L2, L3)
# Points: (1/6, 1/6, 2/3), (1/6, 2/3, 1/6), (2/3, 1/6, 1/6)
# Weights: 1/3 each (integrate over reference triangle with area 1/2)
_TRI_GP = np.array([
    [1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0],
    [2.0 / 3.0, 1.0 / 6.0],
])
_TRI_GW = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


class CTria6Element(BaseElement):
    def __init__(self, node_xyz: np.ndarray, E: float, nu: float, t: float,
                 rho: float = 0.0):
        """
        Args:
            node_xyz: (6, 3) array of node coordinates in global frame.
                      Corners [0..2], then midside [3..5].
                      Node 3 between 0-1, node 4 between 1-2, node 5 between 2-0.
            E: Young's modulus. nu: Poisson's ratio. t: thickness. rho: density.
        """
        self.nodes = node_xyz  # (6, 3)
        self.E = E; self.nu = nu; self.t = t; self.rho = rho
        self._build_local_system()

    def _build_local_system(self):
        p = self.nodes
        v1 = p[1] - p[0]; v2 = p[2] - p[0]
        ez = np.cross(v1, v2)
        n = np.linalg.norm(ez)
        if n < 1e-15:
            raise ValueError("Degenerate CTRIA6")
        self.area_approx = n / 2.0
        ez = ez / n
        ex = v1 / np.linalg.norm(v1)
        ey = np.cross(ez, ex)
        self.T_local = np.array([ex, ey, ez])
        self.center = np.mean(p[:3], axis=0)
        # Local 2D coordinates
        self.xy_local = np.zeros((6, 2))
        for i in range(6):
            d = p[i] - self.center
            self.xy_local[i, 0] = np.dot(d, ex)
            self.xy_local[i, 1] = np.dot(d, ey)

    def dof_count(self):
        return 36

    @staticmethod
    def shape_functions(L1, L2):
        """6-node quadratic triangle shape functions in area coordinates.

        L3 = 1 - L1 - L2.
        Nodes: 0,1,2 = corners at (1,0,0),(0,1,0),(0,0,1)
               3 = midside 0-1, 4 = midside 1-2, 5 = midside 2-0

        Returns N(6,), dNdL1(6,), dNdL2(6,).
        """
        L3 = 1.0 - L1 - L2

        N = np.array([
            L1 * (2 * L1 - 1),   # node 0
            L2 * (2 * L2 - 1),   # node 1
            L3 * (2 * L3 - 1),   # node 2
            4 * L1 * L2,         # node 3 (midside 0-1)
            4 * L2 * L3,         # node 4 (midside 1-2)
            4 * L3 * L1,         # node 5 (midside 2-0)
        ])

        dNdL1 = np.array([
            4 * L1 - 1,          # dN0/dL1
            0.0,                  # dN1/dL1
            -(4 * L3 - 1),       # dN2/dL1 = d/dL1[L3*(2L3-1)] = -1*(2L3-1) + L3*(-2) = -(4L3-1)
            4 * L2,              # dN3/dL1
            -4 * L2,             # dN4/dL1
            4 * (L3 - L1),       # dN5/dL1 = 4*(L3*1 + L1*(-1)) = 4*(L3-L1)
        ])

        dNdL2 = np.array([
            0.0,                  # dN0/dL2
            4 * L2 - 1,          # dN1/dL2
            -(4 * L3 - 1),       # dN2/dL2
            4 * L1,              # dN3/dL2
            4 * (L3 - L2),       # dN4/dL2
            -4 * L1,             # dN5/dL2
        ])

        return N, dNdL1, dNdL2

    def stiffness_matrix(self) -> np.ndarray:
        k_local = self._local_stiffness()
        T = self._build_transform()
        return T.T @ k_local @ T

    def mass_matrix(self) -> np.ndarray:
        m_local = self._local_mass()
        T = self._build_transform()
        return T.T @ m_local @ T

    def _local_stiffness(self):
        E, nu, t = self.E, self.nu, self.t
        xy = self.xy_local

        # Constitutive matrices
        Dm = (E * t / (1 - nu**2)) * np.array(
            [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        Db = (E * t**3 / (12 * (1 - nu**2))) * np.array(
            [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

        k = np.zeros((36, 36))

        # DOF index arrays
        mem_dofs = []
        bend_dofs = []
        for nd in range(6):
            mem_dofs.extend([6 * nd, 6 * nd + 1])
            bend_dofs.extend([6 * nd + 3, 6 * nd + 4])

        # 3-point Gauss integration on triangle
        for gp in range(3):
            L1, L2 = _TRI_GP[gp]
            w = _TRI_GW[gp]
            N, dNdL1, dNdL2 = self.shape_functions(L1, L2)

            # Jacobian: x = sum(N_i * x_i), but derivatives are wrt L1, L2
            # J = [[dx/dL1, dy/dL1], [dx/dL2, dy/dL2]]
            J = np.zeros((2, 2))
            J[0, 0] = dNdL1 @ xy[:, 0]
            J[0, 1] = dNdL1 @ xy[:, 1]
            J[1, 0] = dNdL2 @ xy[:, 0]
            J[1, 1] = dNdL2 @ xy[:, 1]
            detJ = np.linalg.det(J)
            Jinv = np.linalg.inv(J)

            dNdx = Jinv[0, 0] * dNdL1 + Jinv[0, 1] * dNdL2
            dNdy = Jinv[1, 0] * dNdL1 + Jinv[1, 1] * dNdL2

            # Membrane B-matrix (3 x 12)
            Bm = np.zeros((3, 12))
            for nd in range(6):
                Bm[0, 2 * nd] = dNdx[nd]
                Bm[1, 2 * nd + 1] = dNdy[nd]
                Bm[2, 2 * nd] = dNdy[nd]
                Bm[2, 2 * nd + 1] = dNdx[nd]

            # Triangle integration: integral = sum(w_i * f(L1,L2) * detJ * 0.5)
            # The 0.5 comes from reference triangle area
            km = Bm.T @ Dm @ Bm * detJ * w * 0.5
            for ii in range(12):
                for jj in range(12):
                    k[mem_dofs[ii], mem_dofs[jj]] += km[ii, jj]

            # Bending B-matrix (3 x 12)
            Bb = np.zeros((3, 12))
            for nd in range(6):
                Bb[0, 2 * nd + 1] = -dNdx[nd]
                Bb[1, 2 * nd] = dNdy[nd]
                Bb[2, 2 * nd] = dNdx[nd]
                Bb[2, 2 * nd + 1] = -dNdy[nd]

            kb = Bb.T @ Db @ Bb * detJ * w * 0.5
            for ii in range(12):
                for jj in range(12):
                    k[bend_dofs[ii], bend_dofs[jj]] += kb[ii, jj]

        # Drilling DOF stabilization
        area = self._compute_area()
        alpha_drill = E * t * area * 1e-6
        for nd in range(6):
            rz_dof = 6 * nd + 5
            k[rz_dof, rz_dof] += alpha_drill

        return k

    def _local_mass(self):
        area = self._compute_area()
        total_mass = self.rho * self.t * area
        m_per_node = total_mass / 6.0
        m = np.zeros((36, 36))
        for nd in range(6):
            base = 6 * nd
            for i in range(3):
                m[base + i, base + i] = m_per_node
            rot_inertia = m_per_node * self.t**2 / 12.0
            for i in range(3, 6):
                m[base + i, base + i] = rot_inertia
        return m

    def _compute_area(self):
        """Compute element area using 3-point Gauss integration."""
        xy = self.xy_local
        area = 0.0
        for gp in range(3):
            L1, L2 = _TRI_GP[gp]
            w = _TRI_GW[gp]
            _, dNdL1, dNdL2 = self.shape_functions(L1, L2)
            J = np.zeros((2, 2))
            J[0, 0] = dNdL1 @ xy[:, 0]
            J[0, 1] = dNdL1 @ xy[:, 1]
            J[1, 0] = dNdL2 @ xy[:, 0]
            J[1, 1] = dNdL2 @ xy[:, 1]
            area += np.linalg.det(J) * w * 0.5
        return area

    def _build_transform(self):
        """Build 36x36 block-diagonal transformation matrix."""
        T = np.zeros((36, 36))
        R = self.T_local
        for i in range(12):  # 6 nodes x 2 (trans + rot)
            T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
        return T
