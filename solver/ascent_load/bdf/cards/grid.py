"""GRID card parser."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from ..field_parser import nastran_int, nastran_float

@dataclass
class GRID:
    nid: int = 0
    cp: int = 0
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cd: int = 0
    ps: str = ""
    seid: int = 0
    xyz_global: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_fields(cls, fields: List[str]) -> GRID:
        grid = cls()
        grid.nid = nastran_int(fields[1])
        grid.cp = nastran_int(fields[2])
        grid.xyz = np.array([nastran_float(fields[3]), nastran_float(fields[4]), nastran_float(fields[5])])
        if len(fields) > 6: grid.cd = nastran_int(fields[6])
        if len(fields) > 7: grid.ps = fields[7].strip()
        if len(fields) > 8: grid.seid = nastran_int(fields[8])
        grid.xyz_global = grid.xyz.copy()
        return grid
