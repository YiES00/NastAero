"""CQUAD8: 8-node serendipity quadrilateral shell element (48 DOFs).

Uses 3x3 Gauss integration for membrane and bending, 2x2 reduced
integration for transverse shear (prevents shear locking).
6 DOFs per node: u, v, w, rx, ry, rz.
"""
from __future__ import annotations
import numpy as np
from .base import BaseElement

# 3x3 Gauss quadrature
_GP3 = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GW3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

# 2x2 Gauss quadrature (for shear)
_GP2 = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
_GW2 = np.array([1.0, 1.0])


class CQuad8Element(BaseElement):
    def __init__(self, node_xyz: np.ndarray, E: float, nu: float, t: float,
                 rho: float = 0.0):
        """
        Args:
            node_xyz: (8, 3) array of node coordinates in global frame.
                      Corners [0..3], then midside [4..7].
            E: Young's modulus. nu: Poisson's ratio. t: thickness. rho: density.
        """
        self.nodes = node_xyz  # (8, 3)
        self.E = E; self.nu = nu; self.t = t; self.rho = rho
        self._build_local_system()

    def _build_local_system(self):
        p = self.nodes
        center = np.mean(p[:4], axis=0)  # use corner nodes for center
        d13 = p[2] - p[0]; d24 = p[3] - p[1]
        ez = np.cross(d13, d24)
        n = np.linalg.norm(ez)
        if n < 1e-15:
            raise ValueError("Degenerate CQUAD8")
        ez = ez / n
        ex = p[1] - p[0]
        ex = ex - np.dot(ex, ez) * ez
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        self.T_local = np.array([ex, ey, ez])  # 3x3
        self.center = center
        # Project all 8 nodes to local 2D
        self.xy_local = np.zeros((8, 2))
        for i in range(8):
            d = p[i] - center
            self.xy_local[i, 0] = np.dot(d, ex)
            self.xy_local[i, 1] = np.dot(d, ey)

    def dof_count(self):
        return 48

    @staticmethod
    def shape_functions(xi, eta):
        """8-node serendipity shape functions and derivatives.

        Nodes: corners 0-3 at (-1,-1),(1,-1),(1,1),(-1,1)
               midside 4-7 at (0,-1),(1,0),(0,1),(-1,0)

        Returns N(8,), dNdxi(8,), dNdeta(8,).
        """
        # Corner nodes
        xi_c = np.array([-1, 1, 1, -1], dtype=float)
        eta_c = np.array([-1, -1, 1, 1], dtype=float)

        N = np.zeros(8)
        dNdxi = np.zeros(8)
        dNdeta = np.zeros(8)

        # Corner nodes: N_i = 0.25*(1+xi_i*xi)*(1+eta_i*eta)*(xi_i*xi+eta_i*eta-1)
        for i in range(4):
            xi_i, eta_i = xi_c[i], eta_c[i]
            xp = 1 + xi_i * xi
            ep = 1 + eta_i * eta
            s = xi_i * xi + eta_i * eta - 1
            N[i] = 0.25 * xp * ep * s
            dNdxi[i] = 0.25 * (xi_i * ep * s + xp * ep * xi_i)
            dNdeta[i] = 0.25 * (xp * eta_i * s + xp * ep * eta_i)

        # Midside node 4: xi=0, eta=-1 → N = 0.5*(1-xi^2)*(1-eta)
        N[4] = 0.5 * (1 - xi**2) * (1 - eta)
        dNdxi[4] = -xi * (1 - eta)
        dNdeta[4] = -0.5 * (1 - xi**2)

        # Midside node 5: xi=1, eta=0 → N = 0.5*(1+xi)*(1-eta^2)
        N[5] = 0.5 * (1 + xi) * (1 - eta**2)
        dNdxi[5] = 0.5 * (1 - eta**2)
        dNdeta[5] = -(1 + xi) * eta

        # Midside node 6: xi=0, eta=1 → N = 0.5*(1-xi^2)*(1+eta)
        N[6] = 0.5 * (1 - xi**2) * (1 + eta)
        dNdxi[6] = -xi * (1 + eta)
        dNdeta[6] = 0.5 * (1 - xi**2)

        # Midside node 7: xi=-1, eta=0 → N = 0.5*(1-xi)*(1-eta^2)
        N[7] = 0.5 * (1 - xi) * (1 - eta**2)
        dNdxi[7] = -0.5 * (1 - eta**2)
        dNdeta[7] = -(1 - xi) * eta

        return N, dNdxi, dNdeta

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
        kappa = 5.0 / 6.0
        Ds = kappa * (E * t / (2 * (1 + nu))) * np.eye(2)

        k = np.zeros((48, 48))

        # DOF index arrays
        mem_dofs = []
        bend_dofs = []
        shear_dofs = []
        for nd in range(8):
            mem_dofs.extend([6 * nd, 6 * nd + 1])
            bend_dofs.extend([6 * nd + 3, 6 * nd + 4])
            shear_dofs.extend([6 * nd + 2, 6 * nd + 3, 6 * nd + 4])

        # 3x3 Gauss for membrane and bending
        for gi in range(3):
            for gj in range(3):
                xi, eta = _GP3[gi], _GP3[gj]
                w = _GW3[gi] * _GW3[gj]
                N, dNdxi, dNdeta = self.shape_functions(xi, eta)

                J = np.zeros((2, 2))
                J[0, 0] = dNdxi @ xy[:, 0]
                J[0, 1] = dNdxi @ xy[:, 1]
                J[1, 0] = dNdeta @ xy[:, 0]
                J[1, 1] = dNdeta @ xy[:, 1]
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)

                dNdx = Jinv[0, 0] * dNdxi + Jinv[0, 1] * dNdeta
                dNdy = Jinv[1, 0] * dNdxi + Jinv[1, 1] * dNdeta

                # Membrane B-matrix (3 x 16)
                Bm = np.zeros((3, 16))
                for nd in range(8):
                    Bm[0, 2 * nd] = dNdx[nd]
                    Bm[1, 2 * nd + 1] = dNdy[nd]
                    Bm[2, 2 * nd] = dNdy[nd]
                    Bm[2, 2 * nd + 1] = dNdx[nd]

                km = Bm.T @ Dm @ Bm * detJ * w
                for ii in range(16):
                    for jj in range(16):
                        k[mem_dofs[ii], mem_dofs[jj]] += km[ii, jj]

                # Bending B-matrix (3 x 16)
                Bb = np.zeros((3, 16))
                for nd in range(8):
                    Bb[0, 2 * nd + 1] = -dNdx[nd]
                    Bb[1, 2 * nd] = dNdy[nd]
                    Bb[2, 2 * nd] = dNdx[nd]
                    Bb[2, 2 * nd + 1] = -dNdy[nd]

                kb = Bb.T @ Db @ Bb * detJ * w
                for ii in range(16):
                    for jj in range(16):
                        k[bend_dofs[ii], bend_dofs[jj]] += kb[ii, jj]

        # 2x2 reduced integration for transverse shear
        for gi in range(2):
            for gj in range(2):
                xi, eta = _GP2[gi], _GP2[gj]
                w = _GW2[gi] * _GW2[gj]
                N, dNdxi, dNdeta = self.shape_functions(xi, eta)

                J = np.zeros((2, 2))
                J[0, 0] = dNdxi @ xy[:, 0]
                J[0, 1] = dNdxi @ xy[:, 1]
                J[1, 0] = dNdeta @ xy[:, 0]
                J[1, 1] = dNdeta @ xy[:, 1]
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)

                dNdx = Jinv[0, 0] * dNdxi + Jinv[0, 1] * dNdeta
                dNdy = Jinv[1, 0] * dNdxi + Jinv[1, 1] * dNdeta

                # Shear B-matrix (2 x 24)
                Bs = np.zeros((2, 24))
                for nd in range(8):
                    Bs[0, 3 * nd] = dNdx[nd]      # dw/dx
                    Bs[0, 3 * nd + 2] = -N[nd]    # -ry
                    Bs[1, 3 * nd] = dNdy[nd]       # dw/dy
                    Bs[1, 3 * nd + 1] = N[nd]     # rx

                ks = Bs.T @ Ds @ Bs * detJ * w
                for ii in range(24):
                    for jj in range(24):
                        k[shear_dofs[ii], shear_dofs[jj]] += ks[ii, jj]

        # Drilling DOF stabilization (rz DOFs)
        area = self._compute_area()
        alpha_drill = E * t * area * 1e-6
        for nd in range(8):
            rz_dof = 6 * nd + 5
            k[rz_dof, rz_dof] += alpha_drill

        return k

    def _local_mass(self):
        area = self._compute_area()
        total_mass = self.rho * self.t * area
        m_per_node = total_mass / 8.0
        m = np.zeros((48, 48))
        for nd in range(8):
            base = 6 * nd
            for i in range(3):
                m[base + i, base + i] = m_per_node
            rot_inertia = m_per_node * self.t**2 / 12.0
            for i in range(3, 6):
                m[base + i, base + i] = rot_inertia
        return m

    def _compute_area(self):
        """Compute element area using 3x3 Gauss integration."""
        xy = self.xy_local
        area = 0.0
        for gi in range(3):
            for gj in range(3):
                xi, eta = _GP3[gi], _GP3[gj]
                w = _GW3[gi] * _GW3[gj]
                _, dNdxi, dNdeta = self.shape_functions(xi, eta)
                J = np.zeros((2, 2))
                J[0, 0] = dNdxi @ xy[:, 0]
                J[0, 1] = dNdxi @ xy[:, 1]
                J[1, 0] = dNdeta @ xy[:, 0]
                J[1, 1] = dNdeta @ xy[:, 1]
                area += np.linalg.det(J) * w
        return area

    def _build_transform(self):
        """Build 48x48 block-diagonal transformation matrix."""
        T = np.zeros((48, 48))
        R = self.T_local
        for i in range(16):  # 8 nodes x 2 (trans + rot)
            T[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
        return T
