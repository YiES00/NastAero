"""Constraint card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
from ..field_parser import nastran_int, nastran_float

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
        node_strs = [f.strip() for f in fields[3:] if f.strip()]
        if len(node_strs) >= 3 and node_strs[1].upper() == "THRU":
            spc.node_ids = list(range(int(node_strs[0]), int(node_strs[2]) + 1))
        else:
            for ns in node_strs:
                if ns and ns.upper() != "THRU":
                    try: spc.node_ids.append(int(ns))
                    except ValueError: pass
        return spc
