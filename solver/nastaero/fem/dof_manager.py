"""DOF (Degree of Freedom) manager."""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np

class DOFManager:
    def __init__(self, node_ids: List[int]) -> None:
        self._sorted_nids = sorted(node_ids)
        self._nid_to_index: Dict[int, int] = {nid: idx for idx, nid in enumerate(self._sorted_nids)}
        self._ndof_per_node = 6
        self._total_dof = len(self._sorted_nids) * self._ndof_per_node

    @property
    def total_dof(self) -> int: return self._total_dof
    @property
    def n_nodes(self) -> int: return len(self._sorted_nids)
    @property
    def node_ids(self) -> List[int]: return self._sorted_nids

    def get_dof(self, node_id: int, component: int) -> int:
        return self._nid_to_index[node_id] * self._ndof_per_node + (component - 1)

    def get_node_dofs(self, node_id: int) -> List[int]:
        base = self._nid_to_index[node_id] * self._ndof_per_node
        return list(range(base, base + 6))

    def get_element_dofs(self, node_ids: List[int]) -> List[int]:
        dofs: List[int] = []
        for nid in node_ids:
            dofs.extend(self.get_node_dofs(nid))
        return dofs

    def get_constrained_dofs(self, spc_list: list, model_nodes: dict) -> Tuple[Set[int], Dict[int, float]]:
        constrained: Set[int] = set()
        enforced: Dict[int, float] = {}
        for nid, grid in model_nodes.items():
            if nid not in self._nid_to_index: continue
            if grid.ps:
                for ch in grid.ps:
                    comp = int(ch)
                    if 1 <= comp <= 6:
                        constrained.add(self.get_dof(nid, comp))
        for spc_obj in spc_list:
            if hasattr(spc_obj, "constraints"):
                for nid, comp_str, disp_val in spc_obj.constraints:
                    if nid not in self._nid_to_index: continue
                    for ch in comp_str:
                        comp = int(ch)
                        if 1 <= comp <= 6:
                            dof = self.get_dof(nid, comp)
                            constrained.add(dof)
                            if abs(disp_val) > 0: enforced[dof] = disp_val
            elif hasattr(spc_obj, "components"):
                for nid in spc_obj.node_ids:
                    if nid not in self._nid_to_index: continue
                    for ch in spc_obj.components:
                        comp = int(ch)
                        if 1 <= comp <= 6:
                            constrained.add(self.get_dof(nid, comp))
        return constrained, enforced
