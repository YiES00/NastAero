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


@dataclass
class CQUAD8:
    """8-node serendipity quadrilateral shell element (48 DOFs).

    CQUAD8 EID PID G1 G2 G3 G4 G5 G6
           G7 G8 T1 T2 T3 T4 THETA ZOFFS
    """
    eid: int = 0; pid: int = 0
    node_ids: List[int] = field(default_factory=list)  # [G1..G8]
    theta_mcid: float = 0.0
    zoffs: float = 0.0
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)
    @property
    def type(self) -> str: return "CQUAD8"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CQUAD8:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.node_ids = [nastran_int(fields[i]) for i in range(3, 11)]
        if len(fields) > 15: e.theta_mcid = nastran_float(fields[15])
        if len(fields) > 16: e.zoffs = nastran_float(fields[16])
        return e


@dataclass
class CTRIA6:
    """6-node quadratic triangular shell element (36 DOFs).

    CTRIA6 EID PID G1 G2 G3 G4 G5 G6
           THETA ZOFFS
    """
    eid: int = 0; pid: int = 0
    node_ids: List[int] = field(default_factory=list)  # [G1..G6]
    theta_mcid: float = 0.0
    zoffs: float = 0.0
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)
    @property
    def type(self) -> str: return "CTRIA6"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CTRIA6:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.node_ids = [nastran_int(fields[i]) for i in range(3, 9)]
        if len(fields) > 9: e.theta_mcid = nastran_float(fields[9])
        if len(fields) > 10: e.zoffs = nastran_float(fields[10])
        return e


@dataclass
class CBEAM:
    """Beam element (treated same as CBAR for SOL 144).

    CBEAM  EID PID GA GB X1/G0 X2 X3 OFFT
    """
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
    def type(self) -> str: return "CBEAM"
    @classmethod
    def from_fields(cls, fields: List[str]) -> CBEAM:
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
class CELAS1:
    """Scalar spring element referencing PELAS property.

    CELAS1 EID PID G1 C1 G2 C2
    """
    eid: int = 0; pid: int = 0
    g1: int = 0; c1: int = 0
    g2: int = 0; c2: int = 0
    property_ref: Optional[Any] = None
    @property
    def type(self) -> str: return "CELAS1"
    @property
    def node_ids(self) -> List[int]:
        ids = []
        if self.g1 > 0: ids.append(self.g1)
        if self.g2 > 0: ids.append(self.g2)
        return ids
    @classmethod
    def from_fields(cls, fields: List[str]) -> CELAS1:
        e = cls(); e.eid = nastran_int(fields[1]); e.pid = nastran_int(fields[2])
        e.g1 = nastran_int(fields[3]); e.c1 = nastran_int(fields[4])
        e.g2 = nastran_int(fields[5]) if len(fields) > 5 else 0
        e.c2 = nastran_int(fields[6]) if len(fields) > 6 else 0
        return e


@dataclass
class CELAS2:
    """Scalar spring element with direct stiffness value.

    CELAS2 EID K G1 C1 G2 C2
    """
    eid: int = 0; k: float = 0.0
    g1: int = 0; c1: int = 0
    g2: int = 0; c2: int = 0
    @property
    def type(self) -> str: return "CELAS2"
    @property
    def node_ids(self) -> List[int]:
        ids = []
        if self.g1 > 0: ids.append(self.g1)
        if self.g2 > 0: ids.append(self.g2)
        return ids
    @classmethod
    def from_fields(cls, fields: List[str]) -> CELAS2:
        e = cls(); e.eid = nastran_int(fields[1])
        e.k = nastran_float(fields[2])
        e.g1 = nastran_int(fields[3]); e.c1 = nastran_int(fields[4])
        e.g2 = nastran_int(fields[5]) if len(fields) > 5 else 0
        e.c2 = nastran_int(fields[6]) if len(fields) > 6 else 0
        return e
