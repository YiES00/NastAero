"""BDFModel: container for all parsed BDF data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Subcase:
    id: int = 0
    spc_id: int = 0
    load_id: int = 0
    method_id: int = 0
    flutter_id: int = 0
    trim_id: int = 0
    label: str = ""
    output_requests: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BDFModel:
    sol: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    subcases: List[Subcase] = field(default_factory=list)
    global_case: Subcase = field(default_factory=Subcase)
    nodes: Dict[int, Any] = field(default_factory=dict)
    coords: Dict[int, Any] = field(default_factory=dict)
    elements: Dict[int, Any] = field(default_factory=dict)
    properties: Dict[int, Any] = field(default_factory=dict)
    materials: Dict[int, Any] = field(default_factory=dict)
    loads: Dict[int, list] = field(default_factory=dict)
    load_combinations: Dict[int, Any] = field(default_factory=dict)
    spcs: Dict[int, list] = field(default_factory=dict)
    mpcs: Dict[int, list] = field(default_factory=dict)
    masses: Dict[int, Any] = field(default_factory=dict)
    rigids: Dict[int, Any] = field(default_factory=dict)
    eigrls: Dict[int, Any] = field(default_factory=dict)
    sets: Dict[int, Any] = field(default_factory=dict)
    # Aero placeholders
    aero: Optional[Any] = None
    aeros: Optional[Any] = None
    caero_panels: Dict[int, Any] = field(default_factory=dict)
    splines: Dict[int, Any] = field(default_factory=dict)
    flutter_entries: Dict[int, Any] = field(default_factory=dict)
    flfacts: Dict[int, Any] = field(default_factory=dict)
    trims: Dict[int, Any] = field(default_factory=dict)
    mkaeros: list = field(default_factory=list)
    aestats: Dict[int, Any] = field(default_factory=dict)
    aesurfs: Dict[int, Any] = field(default_factory=dict)

    def cross_reference(self) -> None:
        for eid, elem in self.elements.items():
            if hasattr(elem, "pid") and elem.pid in self.properties:
                elem.property_ref = self.properties[elem.pid]
                prop = elem.property_ref
                if hasattr(prop, "mid") and prop.mid in self.materials:
                    prop.material_ref = self.materials[prop.mid]
            if hasattr(elem, "node_ids"):
                elem.node_refs = [self.nodes[nid] for nid in elem.node_ids if nid in self.nodes]

    def get_subcase(self, subcase_id: int) -> Subcase:
        for sc in self.subcases:
            if sc.id == subcase_id:
                return sc
        return self.global_case

    def get_effective_subcase(self, subcase: Subcase) -> Subcase:
        effective = Subcase(id=subcase.id)
        effective.spc_id = self.global_case.spc_id
        effective.load_id = self.global_case.load_id
        effective.method_id = self.global_case.method_id
        effective.flutter_id = self.global_case.flutter_id
        effective.trim_id = self.global_case.trim_id
        effective.output_requests = dict(self.global_case.output_requests)
        if subcase.spc_id:
            effective.spc_id = subcase.spc_id
        if subcase.load_id:
            effective.load_id = subcase.load_id
        if subcase.method_id:
            effective.method_id = subcase.method_id
        if subcase.flutter_id:
            effective.flutter_id = subcase.flutter_id
        if subcase.trim_id:
            effective.trim_id = subcase.trim_id
        effective.output_requests.update(subcase.output_requests)
        effective.label = subcase.label
        return effective
