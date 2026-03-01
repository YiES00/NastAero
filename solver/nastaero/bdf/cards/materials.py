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
