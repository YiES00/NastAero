"""CQUAD4: 4-node bilinear plate element (24 DOFs).

Combines membrane (CST-like bilinear), bending (Mindlin plate), and drilling DOF.
Uses 2x2 Gauss integration for membrane and bending, 1-point for shear (selective reduced integration).
"""
from __future__ import annotations
import numpy as np
from .base import BaseElement

# 2x2 Gauss points
_GP2 = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
_GW2 = np.array([1.0, 1.0])

class CQuad4Element(BaseElement):
    def __init__(self, node_xyz: np.ndarray, E: float, nu: float, t: float, rho: float = 0.0):
        """
        Args:
            node_xyz: (4, 3) array of node coordinates in global frame.
            E: Young's modulus. nu: Poisson's ratio. t: thickness. rho: density.
        """
        self.nodes = node_xyz  # (4, 3)
        self.E = E; self.nu = nu; self.t = t; self.rho = rho
        # Build local coordinate system from element geometry
        self._build_local_system()

    def _build_local_system(self):
        """Build local coordinate system with z normal to the element."""
        p = self.nodes
        # Element center and approximate normal
        center = np.mean(p, axis=0)
        d13 = p[2] - p[0]; d24 = p[3] - p[1]
        ez = np.cross(d13, d24)
        n = np.linalg.norm(ez)
        if n < 1e-15: raise ValueError("Degenerate CQUAD4")
        ez = ez / n
        # ex along side 1-2 projected
        ex = p[1] - p[0]
        ex = ex - np.dot(ex, ez) * ez
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        self.T_local = np.array([ex, ey, ez])  # 3x3
        self.center = center
        # Project nodes to local 2D
        self.xy_local = np.zeros((4, 2))
        for i in range(4):
            d = p[i] - center
            self.xy_local[i, 0] = np.dot(d, ex)
            self.xy_local[i, 1] = np.dot(d, ey)

    def dof_count(self): return 24

    def stiffness_matrix(self) -> np.ndarray:
        """24x24 stiffness matrix in global coordinates."""
        k_local = self._local_stiffness()
        T = self._build_transform_24x24()
        return T.T @ k_local @ T

    def mass_matrix(self) -> np.ndarray:
        """24x24 lumped mass matrix in global coordinates."""
        m_local = self._local_mass()
        T = self._build_transform_24x24()
        return T.T @ m_local @ T

    def _shape_functions(self, xi, eta):
        """Bilinear shape functions and derivatives."""
        N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
        dNdxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
        dNdeta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
        return N, dNdxi, dNdeta

    def _jacobian(self, dNdxi, dNdeta):
        xy = self.xy_local
        J = np.zeros((2, 2))
        J[0, 0] = dNdxi @ xy[:, 0]; J[0, 1] = dNdxi @ xy[:, 1]
        J[1, 0] = dNdeta @ xy[:, 0]; J[1, 1] = dNdeta @ xy[:, 1]
        return J

    def _local_stiffness(self):
        """Build 24x24 local stiffness (DOF order: u,v,w,rx,ry,rz per node).

        Uses selective reduced integration:
        - 2x2 Gauss for membrane and bending
        - 1-point (center) for transverse shear (prevents shear locking)
        """
        E, nu, t = self.E, self.nu, self.t
        # Membrane constitutive (plane stress)
        Dm = (E * t / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        # Bending constitutive
        Db = (E * t**3 / (12 * (1 - nu**2))) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        # Shear constitutive
        kappa = 5.0/6.0  # shear correction factor
        Ds = kappa * (E * t / (2 * (1 + nu))) * np.eye(2)

        k = np.zeros((24, 24))

        # DOF index arrays (precompute)
        mem_dofs = []
        bend_dofs = []
        shear_dofs = []
        for n_idx in range(4):
            mem_dofs.extend([6*n_idx, 6*n_idx+1])
            bend_dofs.extend([6*n_idx+3, 6*n_idx+4])
            shear_dofs.extend([6*n_idx+2, 6*n_idx+3, 6*n_idx+4])

        # 2x2 integration for membrane and bending
        for i in range(2):
            for j in range(2):
                xi, eta = _GP2[i], _GP2[j]
                w = _GW2[i] * _GW2[j]
                N, dNdxi, dNdeta = self._shape_functions(xi, eta)
                J = self._jacobian(dNdxi, dNdeta)
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)
                dNdx = Jinv[0,0]*dNdxi + Jinv[0,1]*dNdeta
                dNdy = Jinv[1,0]*dNdxi + Jinv[1,1]*dNdeta

                # Membrane B-matrix (3 x 8)
                Bm = np.zeros((3, 8))
                for n_idx in range(4):
                    Bm[0, 2*n_idx] = dNdx[n_idx]
                    Bm[1, 2*n_idx+1] = dNdy[n_idx]
                    Bm[2, 2*n_idx] = dNdy[n_idx]
                    Bm[2, 2*n_idx+1] = dNdx[n_idx]

                km = Bm.T @ Dm @ Bm * detJ * w
                for ii in range(8):
                    for jj in range(8):
                        k[mem_dofs[ii], mem_dofs[jj]] += km[ii, jj]

                # Bending B-matrix (3 x 8)
                Bb = np.zeros((3, 8))
                for n_idx in range(4):
                    Bb[0, 2*n_idx+1] = -dNdx[n_idx]  # -d(ry)/dx
                    Bb[1, 2*n_idx] = dNdy[n_idx]      # d(rx)/dy
                    Bb[2, 2*n_idx] = dNdx[n_idx]      # d(rx)/dx
                    Bb[2, 2*n_idx+1] = -dNdy[n_idx]   # -d(ry)/dy

                kb = Bb.T @ Db @ Bb * detJ * w
                for ii in range(8):
                    for jj in range(8):
                        k[bend_dofs[ii], bend_dofs[jj]] += kb[ii, jj]

        # 1-point integration for transverse shear (selective reduced integration)
        xi_c, eta_c = 0.0, 0.0
        w_c = 4.0  # weight for 1-point rule over [-1,1]^2
        N_c, dNdxi_c, dNdeta_c = self._shape_functions(xi_c, eta_c)
        J_c = self._jacobian(dNdxi_c, dNdeta_c)
        detJ_c = np.linalg.det(J_c)
        Jinv_c = np.linalg.inv(J_c)
        dNdx_c = Jinv_c[0,0]*dNdxi_c + Jinv_c[0,1]*dNdeta_c
        dNdy_c = Jinv_c[1,0]*dNdxi_c + Jinv_c[1,1]*dNdeta_c

        Bs = np.zeros((2, 12))
        for n_idx in range(4):
            Bs[0, 3*n_idx] = dNdx_c[n_idx]       # dw/dx
            Bs[0, 3*n_idx+2] = -N_c[n_idx]        # -ry
            Bs[1, 3*n_idx] = dNdy_c[n_idx]         # dw/dy
            Bs[1, 3*n_idx+1] = N_c[n_idx]          # rx

        ks = Bs.T @ Ds @ Bs * detJ_c * w_c
        for ii in range(12):
            for jj in range(12):
                k[shear_dofs[ii], shear_dofs[jj]] += ks[ii, jj]

        # Drilling DOF stabilization (rz DOFs: 5, 11, 17, 23)
        area = self._compute_area()
        alpha_drill = E * t * area * 1e-6
        for n_idx in range(4):
            rz_dof = 6 * n_idx + 5
            k[rz_dof, rz_dof] += alpha_drill

        return k

    def _local_mass(self):
        """24x24 lumped mass matrix in local coordinates."""
        area = self._compute_area()
        total_mass = self.rho * self.t * area
        m_per_node = total_mass / 4.0
        m = np.zeros((24, 24))
        for n_idx in range(4):
            base = 6 * n_idx
            for i in range(3):  # translational DOFs
                m[base+i, base+i] = m_per_node
            # Small rotational inertia
            rot_inertia = m_per_node * self.t**2 / 12.0
            for i in range(3, 6):
                m[base+i, base+i] = rot_inertia
        return m

    def _compute_area(self):
        xy = self.xy_local
        # Shoelace formula for quadrilateral
        d13 = xy[2] - xy[0]; d24 = xy[3] - xy[1]
        return 0.5 * abs(d13[0]*d24[1] - d13[1]*d24[0])

    def _build_transform_24x24(self):
        """Build 24x24 block-diagonal transformation matrix."""
        T = np.zeros((24, 24))
        R = self.T_local  # 3x3
        for i in range(8):  # 8 blocks of 3x3 (4 nodes x 2 groups: trans + rot)
            T[3*i:3*(i+1), 3*i:3*(i+1)] = R
        return T
