"""PARAM card parser."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple
from ..field_parser import nastran_float

@dataclass
class PARAM:
    name: str = ""
    @classmethod
    def from_fields(cls, fields: List[str]) -> Tuple[str, Any]:
        name = fields[1].strip().upper()
        val_str = fields[2].strip() if len(fields) > 2 else ""
        if not val_str: return name, None
        try:
            val = nastran_float(val_str)
            if "." not in val_str: return name, int(val)
            return name, val
        except (ValueError, IndexError):
            pass
        return name, val_str
