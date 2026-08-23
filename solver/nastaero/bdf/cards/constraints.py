"""Constraint card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
from ..field_parser import nastran_int, nastran_float, expand_thru

@dataclass
class SPC:
    sid: int = 0
    constraints: List[Tuple[int, str, float]] = field(default_factory=list)
    @property
    def type(self) -> str: return "SPC"
    @classmethod
    def from_fields(cls, fields: List[str]) -> SPC:
        spc = cls(); spc.sid = nastran_int(fields[1])
        i = 2
        while i + 2 < len(fields):
            g_str = fields[i].strip()
            if not g_str: break
            nid = nastran_int(fields[i]); comp = fields[i+1].strip()
            disp = nastran_float(fields[i+2])
            spc.constraints.append((nid, comp, disp)); i += 3
        return spc

@dataclass
class SPC1:
    sid: int = 0; components: str = ""; node_ids: List[int] = field(default_factory=list)
    @property
    def type(self) -> str: return "SPC1"
    @classmethod
    def from_fields(cls, fields: List[str]) -> SPC1:
        spc = cls(); spc.sid = nastran_int(fields[1]); spc.components = fields[2].strip()
        # THRU는 목록 어디에나 올 수 있다 (선두만 처리하면 중간
        # 범위의 내부 ID가 통째로 빠진다)
        spc.node_ids = expand_thru(fields[3:])
        return spc


@dataclass
class MPC:
    """Multi-point constraint: A1*u(G1,C1) + A2*u(G2,C2) + ... = 0

    MPC  SID  G1 C1 A1  G2 C2 A2 ...
    """
    sid: int = 0
    terms: List[Tuple[int, int, float]] = field(default_factory=list)
    # Each term: (node_id, component, coefficient)
    @property
    def type(self) -> str: return "MPC"
    @classmethod
    def from_fields(cls, fields: List[str]) -> MPC:
        mpc = cls()
        mpc.sid = nastran_int(fields[1])
        i = 2
        while i + 2 < len(fields):
            g_str = fields[i].strip()
            if not g_str:
                i += 3; continue
            try:
                nid = nastran_int(fields[i])
                comp = nastran_int(fields[i+1])
                coeff = nastran_float(fields[i+2])
                mpc.terms.append((nid, comp, coeff))
            except (ValueError, IndexError):
                pass
            i += 3
        return mpc


@dataclass
class MPCADD:
    """Combines multiple MPC sets.

    MPCADD SID  S1 S2 S3 ...
    """
    sid: int = 0
    mpc_ids: List[int] = field(default_factory=list)
    @property
    def type(self) -> str: return "MPCADD"
    @classmethod
    def from_fields(cls, fields: List[str]) -> MPCADD:
        m = cls()
        m.sid = nastran_int(fields[1])
        for f in fields[2:]:
            s = f.strip()
            if s:
                try:
                    m.mpc_ids.append(int(s))
                except ValueError:
                    pass
        return m


@dataclass
class SPCADD:
    """Combines multiple SPC sets.

    SPCADD SID  S1 S2 S3 ...
    """
    sid: int = 0
    spc_ids: List[int] = field(default_factory=list)
    @property
    def type(self) -> str: return "SPCADD"
    @classmethod
    def from_fields(cls, fields: List[str]) -> SPCADD:
        s = cls()
        s.sid = nastran_int(fields[1])
        for f in fields[2:]:
            fs = f.strip()
            if fs:
                try:
                    s.spc_ids.append(int(fs))
                except ValueError:
                    pass
        return s


@dataclass
class SUPORT:
    """자유-자유 해석의 기준(지지) 자유도.

    SUPORT  ID1 C1  ID2 C2  ID3 C3  ID4 C4

    강체 운동을 확정적으로 억제하는 기준점을 지정한다. 관성 릴리프가
    적용된 평형 하중에서는 이 자유도의 반력이 ~0이며, 변위는 이 기준에
    대한 상대 변형이 된다. 따라서 두 솔버를 대조할 때 같은 SUPORT를
    쓰는 것이 비교 가능성의 전제다.
    """
    entries: List[Tuple[int, str]] = field(default_factory=list)
    # 각 항목: (절점 ID, 성분 문자열 예 "123")

    @property
    def type(self) -> str: return "SUPORT"

    @property
    def dof_count(self) -> int:
        """지정된 총 자유도 수 (3-2-1이면 6)."""
        return sum(len([c for c in comp if c in "123456"])
                   for _, comp in self.entries)

    @classmethod
    def from_fields(cls, fields: List[str]) -> SUPORT:
        s = cls()
        i = 1
        while i + 1 < len(fields):
            g_str = fields[i].strip()
            if not g_str:
                i += 2
                continue
            try:
                nid = nastran_int(fields[i])
            except (ValueError, TypeError):
                break
            comp = fields[i + 1].strip() or "123456"
            s.entries.append((nid, comp))
            i += 2
        return s
