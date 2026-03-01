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


@dataclass
class RBE3:
    """Weighted rigid body element for load distribution.
    RBE3  EID  REFGRID  REFC  WT1  C1  G1,1 G1,2 ...
                               WT2  C2  G2,1 ...
    """
    eid: int = 0
    refgrid: int = 0
    refc: str = "123456"
    weight_sets: List = field(default_factory=list)  # [(wt, comp_str, [grid_ids])]
    property_ref: Optional[Any] = None
    node_refs: List[Any] = field(default_factory=list)

    @property
    def type(self) -> str: return "RBE3"

    @property
    def node_ids(self) -> List[int]:
        ids = [self.refgrid]
        for wt, c, grids in self.weight_sets:
            ids.extend(grids)
        return ids

    @classmethod
    def from_fields(cls, fields: List[str]) -> RBE3:
        r = cls()
        r.eid = nastran_int(fields[1])
        # fields[2] is blank in standard format
        r.refgrid = nastran_int(fields[3])
        r.refc = fields[4].strip()
        # Parse weight-component-grid sets
        i = 5
        while i < len(fields):
            s = fields[i].strip()
            if not s:
                i += 1; continue
            try:
                wt = float(s)
            except ValueError:
                i += 1; continue
            if i + 1 >= len(fields): break
            comp = fields[i + 1].strip()
            grids = []
            j = i + 2
            while j < len(fields):
                gs = fields[j].strip()
                if not gs:
                    j += 1; continue
                try:
                    gid = int(gs)
                    grids.append(gid)
                    j += 1
                except ValueError:
                    # Next token is not a grid ID - might be next weight
                    break
            if grids:
                r.weight_sets.append((wt, comp, grids))
            i = j
        return r
