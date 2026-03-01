"""Structural element card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
from ..field_parser import nastran_int, nastran_float

@dataclass
class CBAR:
    eid: int = 0; pid: int = 0
    node_ids: List[int] = field(default_factory=list)
    g0: int = 0
    x: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    offt: str = "GGG"; pa: int = 0; pb: int = 0
    wa: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)
    @property
    def type(self) -> str: return "CBAR"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CBAR:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.node_ids = [nastran_int(fields[3]), nastran_int(fields[4])]
        f5 = fields[5].strip() if len(fields) > 5 else ""
        if f5:
            try:
                if "." not in f5 and "+" not in f5 and "-" not in f5[1:]:
                    e.g0 = int(f5)
                else:
                    raise ValueError()
            except (ValueError, IndexError):
                e.x = np.array([nastran_float(fields[5]),
                    nastran_float(fields[6]) if len(fields) > 6 else 0.0,
                    nastran_float(fields[7]) if len(fields) > 7 else 0.0])
        if len(fields) > 8:
            offt_str = fields[8].strip()
            if offt_str: e.offt = offt_str
        if len(fields) > 9: e.pa = nastran_int(fields[9])
        if len(fields) > 10: e.pb = nastran_int(fields[10])
        return e

@dataclass
class CROD:
    eid: int = 0; pid: int = 0; node_ids: List[int] = field(default_factory=list)
    property_ref: Optional[Any] = None; node_refs: List[Any] = field(default_factory=list)
    @property
    def type(self) -> str: return "CROD"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CROD:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.node_ids = [nastran_int(fields[3]), nastran_int(fields[4])]
        return e

@dataclass
class CQUAD4:
    eid: int = 0; pid: int = 0
    node_ids: List[int] = field(default_factory=list)  # [G1, G2, G3, G4]
    theta_mcid: float = 0.0  # Material angle or coord system ID
    zoffs: float = 0.0       # Z offset from reference plane
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)
    @property
    def type(self) -> str: return "CQUAD4"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CQUAD4:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.node_ids = [nastran_int(fields[3]), nastran_int(fields[4]),
                      nastran_int(fields[5]), nastran_int(fields[6])]
        if len(fields) > 7: e.theta_mcid = nastran_float(fields[7])
        if len(fields) > 8: e.zoffs = nastran_float(fields[8])
        return e

@dataclass
class CTRIA3:
    eid: int = 0; pid: int = 0
    node_ids: List[int] = field(default_factory=list)  # [G1, G2, G3]
    theta_mcid: float = 0.0
    zoffs: float = 0.0
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)
    @property
    def type(self) -> str: return "CTRIA3"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CTRIA3:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.node_ids = [nastran_int(fields[3]), nastran_int(fields[4]), nastran_int(fields[5])]
        if len(fields) > 6: e.theta_mcid = nastran_float(fields[6])
        if len(fields) > 7: e.zoffs = nastran_float(fields[7])
        return e
