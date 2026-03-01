"""Material card parsers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..field_parser import nastran_int, nastran_float

@dataclass
class MAT1:
    mid: int = 0
    E: float = 0.0
    G: float = 0.0
    nu: float = 0.0
    rho: float = 0.0
    a: float = 0.0
    tref: float = 0.0
    ge: float = 0.0

    @classmethod
    def from_fields(cls, fields: List[str]) -> MAT1:
        mat = cls()
        mat.mid = nastran_int(fields[1])
        mat.E = nastran_float(fields[2])
        mat.G = nastran_float(fields[3])
        mat.nu = nastran_float(fields[4])
        mat.rho = nastran_float(fields[5]) if len(fields) > 5 else 0.0
        mat.a = nastran_float(fields[6]) if len(fields) > 6 else 0.0
        mat.tref = nastran_float(fields[7]) if len(fields) > 7 else 0.0
        mat.ge = nastran_float(fields[8]) if len(fields) > 8 else 0.0
        if mat.E > 0 and mat.G > 0 and mat.nu == 0.0:
            mat.nu = mat.E / (2.0 * mat.G) - 1.0
        elif mat.E > 0 and mat.nu > 0 and mat.G == 0.0:
            mat.G = mat.E / (2.0 * (1.0 + mat.nu))
        elif mat.G > 0 and mat.nu > 0 and mat.E == 0.0:
            mat.E = 2.0 * mat.G * (1.0 + mat.nu)
        return mat


@dataclass
class MAT8:
    """Orthotropic material for shell elements (composite plies).
    MAT8  MID  E1  E2  NU12  G12  G1Z  G2Z  RHO
    """
    mid: int = 0
    E1: float = 0.0
    E2: float = 0.0
    nu12: float = 0.0
    G12: float = 0.0
    G1Z: float = 0.0
    G2Z: float = 0.0
    rho: float = 0.0
    # Derived for compatibility with MAT1 interface
    E: float = 0.0
    G: float = 0.0
    nu: float = 0.0

    @classmethod
    def from_fields(cls, fields: List[str]) -> MAT8:
        mat = cls()
        mat.mid = nastran_int(fields[1])
        mat.E1 = nastran_float(fields[2])
        mat.E2 = nastran_float(fields[3])
        mat.nu12 = nastran_float(fields[4])
        mat.G12 = nastran_float(fields[5])
        mat.G1Z = nastran_float(fields[6]) if len(fields) > 6 else 0.0
        mat.G2Z = nastran_float(fields[7]) if len(fields) > 7 else 0.0
        mat.rho = nastran_float(fields[8]) if len(fields) > 8 else 0.0
        # Set approximate isotropic values for compatibility
        mat.E = mat.E1
        mat.G = mat.G12
        mat.nu = mat.nu12
        return mat

    def plane_stress_Q(self):
        """Return the plane stress stiffness matrix Q (3x3).

        Q relates {sigma_1, sigma_2, tau_12} to {eps_1, eps_2, gamma_12}.
        """
        import numpy as np
        nu21 = self.nu12 * self.E2 / self.E1 if self.E1 > 0 else 0.0
        denom = 1.0 - self.nu12 * nu21
        if abs(denom) < 1e-30:
            denom = 1.0
        Q11 = self.E1 / denom
        Q22 = self.E2 / denom
        Q12 = self.nu12 * self.E2 / denom
        Q66 = self.G12
        return np.array([[Q11, Q12, 0.0],
                         [Q12, Q22, 0.0],
                         [0.0, 0.0, Q66]])
