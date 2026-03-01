"""Property card parsers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
from ..field_parser import nastran_int, nastran_float

@dataclass
class PBAR:
    pid: int = 0
    mid: int = 0
    A: float = 0.0
    I1: float = 0.0
    I2: float = 0.0
    J: float = 0.0
    nsm: float = 0.0
    c1: float = 0.0; c2: float = 0.0; d1: float = 0.0; d2: float = 0.0
    e1: float = 0.0; e2: float = 0.0; f1: float = 0.0; f2: float = 0.0
    material_ref: Optional[Any] = None

    @classmethod
    def from_fields(cls, fields: List[str]) -> PBAR:
        p = cls()
        p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.A = nastran_float(fields[3]); p.I1 = nastran_float(fields[4])
        p.I2 = nastran_float(fields[5]); p.J = nastran_float(fields[6])
        p.nsm = nastran_float(fields[7]) if len(fields) > 7 else 0.0
        return p

@dataclass
class PROD:
    pid: int = 0; mid: int = 0; A: float = 0.0; J: float = 0.0
    c: float = 0.0; nsm: float = 0.0; material_ref: Optional[Any] = None
    @classmethod
    def from_fields(cls, fields: List[str]) -> PROD:
        p = cls(); p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.A = nastran_float(fields[3])
        p.J = nastran_float(fields[4]) if len(fields) > 4 else 0.0
        return p

@dataclass
class PSHELL:
    pid: int = 0; mid: int = 0; t: float = 0.0; mid2: int = 0
    ratio_12it3: float = 1.0; mid3: int = 0; ts_t: float = 0.833333; nsm: float = 0.0
    material_ref: Optional[Any] = None
    @classmethod
    def from_fields(cls, fields: List[str]) -> PSHELL:
        p = cls(); p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.t = nastran_float(fields[3])
        p.mid2 = nastran_int(fields[4]) if len(fields) > 4 and fields[4].strip() else p.mid
        p.ratio_12it3 = nastran_float(fields[5], default=1.0) if len(fields) > 5 else 1.0
        p.mid3 = nastran_int(fields[6]) if len(fields) > 6 and fields[6].strip() else 0
        p.ts_t = nastran_float(fields[7], default=0.833333) if len(fields) > 7 else 0.833333
        p.nsm = nastran_float(fields[8]) if len(fields) > 8 else 0.0
        return p

@dataclass
class PSOLID:
    pid: int = 0; mid: int = 0; cordm: int = 0; material_ref: Optional[Any] = None
    @classmethod
    def from_fields(cls, fields: List[str]) -> PSOLID:
        p = cls(); p.pid = nastran_int(fields[1]); p.mid = nastran_int(fields[2])
        p.cordm = nastran_int(fields[3]) if len(fields) > 3 else 0
        return p
