"""BDFModel: container for all parsed BDF data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Subcase:
    id: int = 0
    spc_id: int = 0
    mpc_id: int = 0
    load_id: int = 0
    method_id: int = 0
    flutter_id: int = 0
    trim_id: int = 0
    dload_id: int = 0
    freq_id: int = 0
    tstep_id: int = 0
    sdamp_id: int = 0
    gust_id: int = 0
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
    spcadds: Dict[int, Any] = field(default_factory=dict)
    mpcadds: Dict[int, Any] = field(default_factory=dict)
    masses: Dict[int, Any] = field(default_factory=dict)
    rigids: Dict[int, Any] = field(default_factory=dict)
    eigrls: Dict[int, Any] = field(default_factory=dict)
    sets: Dict[int, Any] = field(default_factory=dict)
    # Aero data
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
    aelists: Dict[int, Any] = field(default_factory=dict)
    aefacts: Dict[int, Any] = field(default_factory=dict)
    aelinks: List[Any] = field(default_factory=list)
    # Dynamic analysis cards (SOL 112/146)
    tloads: Dict[int, Any] = field(default_factory=dict)
    dloads: Dict[int, Any] = field(default_factory=dict)
    tabled1s: Dict[int, Any] = field(default_factory=dict)
    gusts: Dict[int, Any] = field(default_factory=dict)
    dareas: Dict[int, Any] = field(default_factory=dict)
    freq_entries: Dict[int, Any] = field(default_factory=dict)
    tsteps: Dict[int, Any] = field(default_factory=dict)
    tabdmp1s: Dict[int, Any] = field(default_factory=dict)
    # Spring elements (stored separately from elements dict)
    springs: Dict[int, Any] = field(default_factory=dict)

    @property
    def conm2s(self) -> Dict[int, Any]:
        """CONM2 elements filtered from masses dict.

        Returns dict of references to the same objects in self.masses,
        so attribute writes (e.g., model.conm2s[eid].mass = x) persist.
        """
        return {eid: m for eid, m in self.masses.items()
                if hasattr(m, 'type') and m.type == 'CONM2'}

    def cross_reference(self) -> None:
        """Cross-reference elements → properties → materials."""
        for eid, elem in self.elements.items():
            if hasattr(elem, "pid") and elem.pid in self.properties:
                elem.property_ref = self.properties[elem.pid]
                prop = elem.property_ref
                if hasattr(prop, "mid") and prop.mid in self.materials:
                    prop.material_ref = self.materials[prop.mid]
                # PCOMP: cross-reference ply materials
                if hasattr(prop, 'plies') and hasattr(prop, 'ply_materials'):
                    prop.ply_materials = []
                    for mid, t, theta, sout in prop.plies:
                        if mid in self.materials:
                            prop.ply_materials.append(self.materials[mid])
                        else:
                            prop.ply_materials.append(None)
                    # Pre-compute equivalent isotropic with materials
                    if hasattr(prop, 'equivalent_isotropic'):
                        prop.equivalent_isotropic(self.materials)
            if hasattr(elem, "node_ids"):
                elem.node_refs = [self.nodes[nid] for nid in elem.node_ids if nid in self.nodes]
        # Cross-reference spring elements
        for eid, spring in self.springs.items():
            if hasattr(spring, 'pid') and spring.pid in self.properties:
                spring.property_ref = self.properties[spring.pid]

    def get_subcase(self, subcase_id: int) -> Subcase:
        for sc in self.subcases:
            if sc.id == subcase_id:
                return sc
        return self.global_case

    def get_effective_subcase(self, subcase: Subcase) -> Subcase:
        effective = Subcase(id=subcase.id)
        effective.spc_id = self.global_case.spc_id
        effective.mpc_id = self.global_case.mpc_id
        effective.load_id = self.global_case.load_id
        effective.method_id = self.global_case.method_id
        effective.flutter_id = self.global_case.flutter_id
        effective.trim_id = self.global_case.trim_id
        effective.dload_id = self.global_case.dload_id
        effective.freq_id = self.global_case.freq_id
        effective.tstep_id = self.global_case.tstep_id
        effective.sdamp_id = self.global_case.sdamp_id
        effective.gust_id = self.global_case.gust_id
        effective.output_requests = dict(self.global_case.output_requests)
        if subcase.spc_id:
            effective.spc_id = subcase.spc_id
        if subcase.mpc_id:
            effective.mpc_id = subcase.mpc_id
        if subcase.load_id:
            effective.load_id = subcase.load_id
        if subcase.method_id:
            effective.method_id = subcase.method_id
        if subcase.flutter_id:
            effective.flutter_id = subcase.flutter_id
        if subcase.trim_id:
            effective.trim_id = subcase.trim_id
        if subcase.dload_id:
            effective.dload_id = subcase.dload_id
        if subcase.freq_id:
            effective.freq_id = subcase.freq_id
        if subcase.tstep_id:
            effective.tstep_id = subcase.tstep_id
        if subcase.sdamp_id:
            effective.sdamp_id = subcase.sdamp_id
        if subcase.gust_id:
            effective.gust_id = subcase.gust_id
        effective.output_requests.update(subcase.output_requests)
        effective.label = subcase.label
        return effective

    def resolve_spc_ids(self, spc_id: int) -> list:
        """Resolve SPCADD to get all SPC/SPC1 entries."""
        all_spcs = []
        if spc_id in self.spcadds:
            spcadd = self.spcadds[spc_id]
            for child_id in spcadd.spc_ids:
                all_spcs.extend(self.spcs.get(child_id, []))
        else:
            all_spcs.extend(self.spcs.get(spc_id, []))
        return all_spcs

    def resolve_mpc_ids(self, mpc_id: int) -> list:
        """Resolve MPCADD to get all MPC entries."""
        all_mpcs = []
        if mpc_id in self.mpcadds:
            mpcadd = self.mpcadds[mpc_id]
            for child_id in mpcadd.mpc_ids:
                all_mpcs.extend(self.mpcs.get(child_id, []))
        else:
            all_mpcs.extend(self.mpcs.get(mpc_id, []))
        return all_mpcs
