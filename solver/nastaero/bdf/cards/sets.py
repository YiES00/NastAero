"""SET1 card parser."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from ..field_parser import nastran_int


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
        raw = [f.strip() for f in fields[2:] if f.strip()]
        i = 0
        while i < len(raw):
            token = raw[i].upper()
            if token == "THRU" and i >= 1 and i + 1 < len(raw):
                start = s.ids[-1]  # already appended
                end = int(raw[i + 1])
                s.ids.extend(range(start + 1, end + 1))
                i += 2
            else:
                try:
                    s.ids.append(int(token))
                except ValueError:
                    pass
                i += 1
        return s
