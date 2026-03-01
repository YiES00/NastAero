"""EIGRL card parser - eigenvalue extraction parameters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..field_parser import nastran_int, nastran_float

@dataclass
class EIGRL:
    """Real eigenvalue extraction (Lanczos method).
    EIGRL   SID     V1      V2      ND      MSGLVL  MAXSET  SHFSCL  NORM
    """
    sid: int = 0
    v1: float = 0.0      # Lower frequency bound (Hz if NORM=MAX)
    v2: float = 0.0      # Upper frequency bound
    nd: int = 0           # Number of desired roots
    msglvl: int = 0
    maxset: int = 0
    shfscl: float = 0.0
    norm: str = "MAX"     # Normalization method

    @classmethod
    def from_fields(cls, fields: List[str]) -> EIGRL:
        e = cls(); e.sid = nastran_int(fields[1])
        e.v1 = nastran_float(fields[2]) if len(fields) > 2 else 0.0
        e.v2 = nastran_float(fields[3]) if len(fields) > 3 else 0.0
        e.nd = nastran_int(fields[4]) if len(fields) > 4 else 0
        if len(fields) > 8:
            norm_str = fields[8].strip().upper()
            if norm_str: e.norm = norm_str
        return e
