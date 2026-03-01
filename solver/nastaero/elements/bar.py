"""CBAR element: 2-node Euler-Bernoulli beam with 12 DOFs."""
from __future__ import annotations
import numpy as np
from .base import BaseElement
from ..fem.coordinate_systems import build_beam_transform, build_transform_12x12

class CBarElement(BaseElement):
    def __init__(self, node1_xyz, node2_xyz, v_vector, E, G, A, I1, I2, J, rho=0.0, nsm=0.0):
        self.E = E; self.G = G; self.A = A; self.I1 = I1; self.I2 = I2; self.J = J
        self.rho = rho; self.nsm = nsm
        diff = node2_xyz - node1_xyz
        self.L = np.linalg.norm(diff)
        if self.L < 1e-12: raise ValueError("Zero-length CBAR")
        self.Lambda = build_beam_transform(node1_xyz, node2_xyz, v_vector)
        self.T = build_transform_12x12(self.Lambda)

    def dof_count(self): return 12

    def stiffness_matrix(self):
        return self.T.T @ self._local_stiffness() @ self.T

    def mass_matrix(self):
        return self.T.T @ self._local_mass() @ self.T

    def _local_stiffness(self):
        E, G, A, L = self.E, self.G, self.A, self.L
        Iz, Iy, J = self.I1, self.I2, self.J
        k = np.zeros((12, 12))
        ea_l = E*A/L; gj_l = G*J/L
        L2 = L*L; L3 = L2*L
        # Axial
        k[0,0] = ea_l; k[0,6] = -ea_l; k[6,0] = -ea_l; k[6,6] = ea_l
        # Torsion
        k[3,3] = gj_l; k[3,9] = -gj_l; k[9,3] = -gj_l; k[9,9] = gj_l
        # Bending xz-plane (v, rz): DOFs 1,5,7,11
        eiz = E*Iz
        k[1,1]=12*eiz/L3; k[1,5]=6*eiz/L2; k[1,7]=-12*eiz/L3; k[1,11]=6*eiz/L2
        k[5,1]=6*eiz/L2; k[5,5]=4*eiz/L; k[5,7]=-6*eiz/L2; k[5,11]=2*eiz/L
        k[7,1]=-12*eiz/L3; k[7,5]=-6*eiz/L2; k[7,7]=12*eiz/L3; k[7,11]=-6*eiz/L2
        k[11,1]=6*eiz/L2; k[11,5]=2*eiz/L; k[11,7]=-6*eiz/L2; k[11,11]=4*eiz/L
        # Bending xy-plane (w, ry): DOFs 2,4,8,10
        eiy = E*Iy
        k[2,2]=12*eiy/L3; k[2,4]=-6*eiy/L2; k[2,8]=-12*eiy/L3; k[2,10]=-6*eiy/L2
        k[4,2]=-6*eiy/L2; k[4,4]=4*eiy/L; k[4,8]=6*eiy/L2; k[4,10]=2*eiy/L
        k[8,2]=-12*eiy/L3; k[8,4]=6*eiy/L2; k[8,8]=12*eiy/L3; k[8,10]=6*eiy/L2
        k[10,2]=-6*eiy/L2; k[10,4]=2*eiy/L; k[10,8]=6*eiy/L2; k[10,10]=4*eiy/L
        return k

    def _local_mass(self):
        L = self.L; mL = (self.rho * self.A + self.nsm) * L
        m = np.zeros((12, 12))
        # Axial
        m[0,0]=mL/3; m[0,6]=mL/6; m[6,0]=mL/6; m[6,6]=mL/3
        # Torsion
        Ip = self.I1 + self.I2; rIpL = self.rho * Ip * L
        m[3,3]=rIpL/3; m[3,9]=rIpL/6; m[9,3]=rIpL/6; m[9,9]=rIpL/3
        # Bending xz
        m[1,1]=13*mL/35; m[1,5]=11*mL*L/210; m[1,7]=9*mL/70; m[1,11]=-13*mL*L/420
        m[5,1]=11*mL*L/210; m[5,5]=mL*L*L/105; m[5,7]=13*mL*L/420; m[5,11]=-mL*L*L/140
        m[7,1]=9*mL/70; m[7,5]=13*mL*L/420; m[7,7]=13*mL/35; m[7,11]=-11*mL*L/210
        m[11,1]=-13*mL*L/420; m[11,5]=-mL*L*L/140; m[11,7]=-11*mL*L/210; m[11,11]=mL*L*L/105
        # Bending xy
        m[2,2]=13*mL/35; m[2,4]=-11*mL*L/210; m[2,8]=9*mL/70; m[2,10]=13*mL*L/420
        m[4,2]=-11*mL*L/210; m[4,4]=mL*L*L/105; m[4,8]=-13*mL*L/420; m[4,10]=-mL*L*L/140
        m[8,2]=9*mL/70; m[8,4]=-13*mL*L/420; m[8,8]=13*mL/35; m[8,10]=11*mL*L/210
        m[10,2]=13*mL*L/420; m[10,4]=-mL*L*L/140; m[10,8]=11*mL*L/210; m[10,10]=mL*L*L/105
        return m
