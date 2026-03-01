"""Aerodynamic and aeroelastic BDF card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..field_parser import nastran_int, nastran_float, nastran_string


@dataclass
class AEFACT:
    """Non-uniform division factors for CAERO1 panels.

    AEFACT SID  D1 D2 D3 D4 D5 D6 D7
           D8   D9 ...
    """
    sid: int = 0
    factors: List[float] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> AEFACT:
        a = cls()
        a.sid = nastran_int(fields[1])
        for f in fields[2:]:
            s = f.strip()
            if s:
                try:
                    a.factors.append(nastran_float(s))
                except (ValueError, TypeError):
                    pass
        return a


@dataclass
class AELIST:
    """Aerodynamic element (box) ID list for control surfaces.
    AELIST  SID  E1  E2  E3  E4  E5  E6  E7
            E8   E9  ...  or THRU notation
    """
    sid: int = 0
    elements: List[int] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> AELIST:
        a = cls()
        a.sid = nastran_int(fields[1])
        raw = [f.strip() for f in fields[2:] if f.strip()]
        i = 0
        while i < len(raw):
            token = raw[i].upper()
            if token == "THRU" and i >= 1 and i + 1 < len(raw):
                start = a.elements[-1]
                end = int(raw[i + 1])
                a.elements.extend(range(start + 1, end + 1))
                i += 2
            else:
                try:
                    a.elements.append(int(token))
                except ValueError:
                    pass
                i += 1
        return a


@dataclass
class AERO:
    """Aerodynamic physical data for dynamic aeroelastic analysis.
    AERO  ACSID  VELOCITY  REFC  RHOREF  SYMXZ  SYMXY
    """
    acsid: int = 0
    velocity: float = 0.0
    refc: float = 1.0
    rhoref: float = 1.225
    symxz: int = 0
    symxy: int = 0

    @classmethod
    def from_fields(cls, fields: List[str]) -> AERO:
        a = cls()
        a.acsid = nastran_int(fields[1])
        a.velocity = nastran_float(fields[2])
        if len(fields) > 3: a.refc = nastran_float(fields[3], 1.0)
        if len(fields) > 4: a.rhoref = nastran_float(fields[4], 1.225)
        if len(fields) > 5: a.symxz = nastran_int(fields[5])
        if len(fields) > 6: a.symxy = nastran_int(fields[6])
        return a


@dataclass
class AEROS:
    """Aerodynamic physical data for static aeroelastic analysis.
    AEROS  ACSID  RCSID  REFC  REFB  REFS  SYMXZ  SYMXY
    """
    acsid: int = 0
    rcsid: int = 0
    refc: float = 1.0
    refb: float = 1.0
    refs: float = 1.0
    symxz: int = 0
    symxy: int = 0

    @classmethod
    def from_fields(cls, fields: List[str]) -> AEROS:
        a = cls()
        a.acsid = nastran_int(fields[1])
        a.rcsid = nastran_int(fields[2])
        a.refc = nastran_float(fields[3], 1.0)
        a.refb = nastran_float(fields[4], 1.0)
        a.refs = nastran_float(fields[5], 1.0)
        if len(fields) > 6: a.symxz = nastran_int(fields[6])
        if len(fields) > 7: a.symxy = nastran_int(fields[7])
        return a


@dataclass
class CAERO1:
    """Aerodynamic panel element definition (Doublet-Lattice).
    CAERO1 EID  PID  CP  NSPAN NCHORD LSPAN LCHORD IGID
           X1   Y1   Z1  X12   X4    Y4    Z4     X43
    """
    eid: int = 0
    pid: int = 0
    cp: int = 0
    nspan: int = 1
    nchord: int = 1
    lspan: int = 0
    lchord: int = 0
    igid: int = 0
    p1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    chord1: float = 0.0
    p4: np.ndarray = field(default_factory=lambda: np.zeros(3))
    chord4: float = 0.0

    @property
    def type(self) -> str: return "CAERO1"

    @classmethod
    def from_fields(cls, fields: List[str]) -> CAERO1:
        c = cls()
        c.eid = nastran_int(fields[1])
        c.pid = nastran_int(fields[2])
        c.cp = nastran_int(fields[3])
        c.nspan = nastran_int(fields[4], 1)
        c.nchord = nastran_int(fields[5], 1)
        c.lspan = nastran_int(fields[6])
        c.lchord = nastran_int(fields[7])
        if len(fields) > 8: c.igid = nastran_int(fields[8])
        # Second line: corner points
        c.p1 = np.array([nastran_float(fields[9]), nastran_float(fields[10]),
                          nastran_float(fields[11])])
        c.chord1 = nastran_float(fields[12])
        c.p4 = np.array([nastran_float(fields[13]), nastran_float(fields[14]),
                          nastran_float(fields[15])])
        c.chord4 = nastran_float(fields[16])
        return c


@dataclass
class PAERO1:
    """Aerodynamic panel property.
    PAERO1 PID  B1  B2  B3  B4  B5  B6
    """
    pid: int = 0
    bodies: List[int] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> PAERO1:
        p = cls()
        p.pid = nastran_int(fields[1])
        for f in fields[2:]:
            s = f.strip()
            if s:
                try: p.bodies.append(int(s))
                except ValueError: pass
        return p


@dataclass
class SPLINE1:
    """Surface spline (infinite plate spline).
    SPLINE1 EID  CAERO BOX1 BOX2 SETG  DZ  METHOD USAGE
            NELEM MELEM
    """
    eid: int = 0
    caero: int = 0
    box1: int = 0
    box2: int = 0
    setg: int = 0
    dz: float = 0.0
    method: str = "IPS"
    usage: str = "BOTH"

    @classmethod
    def from_fields(cls, fields: List[str]) -> SPLINE1:
        s = cls()
        s.eid = nastran_int(fields[1])
        s.caero = nastran_int(fields[2])
        s.box1 = nastran_int(fields[3])
        s.box2 = nastran_int(fields[4])
        s.setg = nastran_int(fields[5])
        if len(fields) > 6: s.dz = nastran_float(fields[6])
        if len(fields) > 7:
            m = fields[7].strip()
            if m: s.method = m
        if len(fields) > 8:
            u = fields[8].strip()
            if u: s.usage = u
        return s


@dataclass
class SPLINE2:
    """Beam spline for 1D interpolation.
    SPLINE2 EID  CAERO ID1  ID2  SETG  DZ  DTOR  CID
            DTHX DTHY
    """
    eid: int = 0
    caero: int = 0
    id1: int = 0
    id2: int = 0
    setg: int = 0
    dz: float = 0.0
    dtor: float = 1.0
    cid: int = 0
    dthx: float = 0.0
    dthy: float = 0.0

    @classmethod
    def from_fields(cls, fields: List[str]) -> SPLINE2:
        s = cls()
        s.eid = nastran_int(fields[1])
        s.caero = nastran_int(fields[2])
        s.id1 = nastran_int(fields[3])
        s.id2 = nastran_int(fields[4])
        s.setg = nastran_int(fields[5])
        if len(fields) > 6: s.dz = nastran_float(fields[6])
        if len(fields) > 7: s.dtor = nastran_float(fields[7], 1.0)
        if len(fields) > 8: s.cid = nastran_int(fields[8])
        if len(fields) > 9: s.dthx = nastran_float(fields[9])
        if len(fields) > 10: s.dthy = nastran_float(fields[10])
        return s


@dataclass
class AESTAT:
    """Rigid body aerodynamic trim variable.
    AESTAT ID  LABEL
    """
    id: int = 0
    label: str = ""

    @classmethod
    def from_fields(cls, fields: List[str]) -> AESTAT:
        a = cls()
        a.id = nastran_int(fields[1])
        a.label = fields[2].strip().upper() if len(fields) > 2 else ""
        return a


@dataclass
class AESURF:
    """Aerodynamic control surface definition.
    AESURF ID  LABEL  CID1  ALID1  CID2  ALID2  EFF
    """
    id: int = 0
    label: str = ""
    cid1: int = 0
    alid1: int = 0
    cid2: int = 0
    alid2: int = 0
    eff: float = 1.0

    @classmethod
    def from_fields(cls, fields: List[str]) -> AESURF:
        a = cls()
        a.id = nastran_int(fields[1])
        a.label = fields[2].strip().upper() if len(fields) > 2 else ""
        if len(fields) > 3: a.cid1 = nastran_int(fields[3])
        if len(fields) > 4: a.alid1 = nastran_int(fields[4])
        if len(fields) > 5: a.cid2 = nastran_int(fields[5])
        if len(fields) > 6: a.alid2 = nastran_int(fields[6])
        if len(fields) > 7: a.eff = nastran_float(fields[7], 1.0)
        return a


@dataclass
class TRIM:
    """Static aeroelastic trim condition.
    TRIM  ID  MACH  Q  LABEL1 UX1  LABEL2 UX2  ...
    """
    tid: int = 0
    mach: float = 0.0
    q: float = 0.0
    aeqr: float = 1.0
    variables: List[Tuple[str, float]] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> TRIM:
        t = cls()
        t.tid = nastran_int(fields[1])
        t.mach = nastran_float(fields[2])
        t.q = nastran_float(fields[3])
        # Parse label-value pairs
        i = 4
        while i + 1 < len(fields):
            label = fields[i].strip().upper()
            if not label:
                i += 1; continue
            val = nastran_float(fields[i + 1])
            t.variables.append((label, val))
            i += 2
        return t


@dataclass
class FLFACT:
    """Factor list for flutter analysis.
    FLFACT SID  F1  F2  F3  F4  F5  F6  F7
           F8   F9  ...
    """
    sid: int = 0
    factors: List[float] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> FLFACT:
        f = cls()
        f.sid = nastran_int(fields[1])
        for s in fields[2:]:
            s_strip = s.strip()
            if s_strip:
                try:
                    f.factors.append(nastran_float(s_strip))
                except ValueError:
                    pass
        return f


@dataclass
class MKAERO1:
    """Mach-reduced frequency combinations for AIC generation.
    MKAERO1 M1  M2  M3  M4  M5  M6  M7  M8
            K1  K2  K3  K4  K5  K6  K7  K8
    """
    machs: List[float] = field(default_factory=list)
    reduced_freqs: List[float] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> MKAERO1:
        m = cls()
        # First 8 data fields = Mach numbers
        for f in fields[1:9]:
            s = f.strip()
            if s:
                try: m.machs.append(nastran_float(s))
                except ValueError: pass
        # Next 8 data fields = reduced frequencies
        for f in fields[9:17]:
            s = f.strip()
            if s:
                try: m.reduced_freqs.append(nastran_float(s))
                except ValueError: pass
        return m
