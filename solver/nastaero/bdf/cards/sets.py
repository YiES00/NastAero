"""SET1 card parser."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from ..field_parser import nastran_int, expand_thru


@dataclass
class SET1:
    """Defines a list of structural grid points or element IDs.
    SET1  SID  G1  G2  G3  G4  G5  G6  G7
          G8   G9  ...  or THRU notation
    """
    sid: int = 0
    ids: List[int] = field(default_factory=list)

    @classmethod
    def from_fields(cls, fields: List[str]) -> SET1:
        s = cls()
        s.sid = nastran_int(fields[1])
        s.ids = expand_thru(fields[2:])
        return s
