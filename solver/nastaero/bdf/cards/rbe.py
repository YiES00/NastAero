"""Rigid element card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
from ..field_parser import nastran_int

@dataclass
class RBE2:
    """Rigid body element connecting dependent nodes to an independent node.
    RBE2    EID     GN      CM      GM1     GM2     GM3     ...
    """
    eid: int = 0
    independent_node: int = 0      # GN
    components: str = ""            # CM (components constrained)
    dependent_nodes: List[int] = field(default_factory=list)  # GM1, GM2, ...
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)

    @property
    def type(self) -> str: return "RBE2"
    @property
    def node_ids(self) -> List[int]:
        return [self.independent_node] + self.dependent_nodes

    @classmethod
    def from_fields(cls, fields: List[str]) -> RBE2:
        r = cls(); r.eid = nastran_int(fields[1])
        r.independent_node = nastran_int(fields[2])
        r.components = fields[3].strip()
        for f in fields[4:]:
            s = f.strip()
            if s:
                try: r.dependent_nodes.append(int(s))
                except ValueError: pass
        return r
