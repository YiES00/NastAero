"""Result file I/O for NastAero (.naero format).

Saves analysis results and model geometry to a single compressed archive
for later visualization without re-running the solver.

File format: ZIP archive containing numpy arrays (.npy) and JSON metadata.
No additional dependencies beyond numpy (standard library: zipfile, json).

Usage:
    # Save after solving
    save_results(results, bdf_model, "model.naero")

    # Load for visualization (no BDF parsing or solving needed)
    results, viz_model = load_results("model.naero")
    viewer = NastAeroViewer(viz_model, results)
"""
from __future__ import annotations
import json
import zipfile
import io
import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .result_data import ResultData, SubcaseResult
from ..config import logger

FORMAT_VERSION = 1
__version__ = "0.3.0"


# ---------------------------------------------------------------------------
# VizModel: lightweight BDFModel proxy for visualization
# ---------------------------------------------------------------------------

@dataclass
class VizNode:
    """Minimal node proxy for visualization."""
    nid: int = 0
    xyz_global: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class VizElement:
    """Minimal element proxy for visualization."""
    type: str = ""
    pid: int = 0
    node_ids: List[int] = field(default_factory=list)


@dataclass
class VizProperty:
    """Minimal property proxy (beam tube radius uses A)."""
    A: float = 0.0


@dataclass
class VizRBE2:
    """Minimal RBE2 proxy for build_rbe_lines()."""
    independent_node: int = 0
    components: str = ""
    dependent_nodes: List[int] = field(default_factory=list)


@dataclass
class VizRBE3:
    """Minimal RBE3 proxy for build_rbe_lines()."""
    refgrid: int = 0
    refc: str = ""
    weight_sets: list = field(default_factory=list)  # [(wt, comp, [grids]), ...]


class VizModel:
    """Lightweight BDFModel proxy for visualization from saved results.

    Provides the minimal interface required by NastAeroViewer and
    mesh_builder functions, without needing the full BDF parser.
    VizModel is a drop-in replacement for BDFModel in all visualization code.
    """
    def __init__(self):
        self.sol: int = 0
        self.nodes: Dict[int, VizNode] = {}
        self.elements: Dict[int, VizElement] = {}
        self.properties: Dict[int, VizProperty] = {}
        self.rigids: Dict[int, Any] = {}
        self.caero_panels: Dict[int, Any] = {}
        self.aesurfs: Dict[int, Any] = {}
        self.aelists: Dict[int, Any] = {}
        self.aefacts: Dict[int, Any] = {}
        # Additional attributes that viewer.py may check existence of
        self.splines: Dict[int, Any] = {}
        self.sets: Dict[int, Any] = {}
        self.subcases: list = []


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    results: ResultData,
    bdf_model: Any,
    filepath: str,
    bdf_file: str = "",
) -> None:
    """Save analysis results and model geometry to .naero file.

    Parameters
    ----------
    results : ResultData
        Analysis results from solver.
    bdf_model : BDFModel
        Parsed BDF model (for geometry data).
    filepath : str
        Output file path (recommended extension: .naero).
    bdf_file : str
        Original BDF filename (for metadata only).
    """
    filepath = str(filepath)
    if not filepath.endswith('.naero'):
        filepath += '.naero'

    sorted_nids = sorted(bdf_model.nodes.keys())

    # Build metadata
    metadata = {
        'version': FORMAT_VERSION,
        'nastaero_version': __version__,
        'sol': getattr(bdf_model, 'sol', 0),
        'title': results.title,
        'bdf_file': bdf_file,
        'created': datetime.datetime.now().isoformat(),
        'n_subcases': len(results.subcases),
        'model': _extract_model_metadata(bdf_model, sorted_nids),
        'subcases': [],
    }

    # Collect all arrays
    all_arrays = {}

    # Node coordinates
    node_xyz = np.array([bdf_model.nodes[nid].xyz_global for nid in sorted_nids],
                        dtype=np.float64)
    all_arrays['node_xyz'] = node_xyz

    # Per-subcase arrays and metadata
    for idx, sc in enumerate(results.subcases):
        sc_meta = {
            'subcase_id': sc.subcase_id,
            'trim_variables': sc.trim_variables,
            'trim_balance': sc.trim_balance,
            'has_displacements': bool(sc.displacements),
            'has_spc_forces': bool(sc.spc_forces),
            'has_aero_pressures': sc.aero_pressures is not None,
            'has_aero_forces': sc.aero_forces is not None,
            'has_aero_boxes': sc.aero_boxes is not None and len(sc.aero_boxes) > 0,
            'has_nodal_aero_forces': sc.nodal_aero_forces is not None,
            'has_nodal_inertial_forces': sc.nodal_inertial_forces is not None,
            'has_nodal_combined_forces': sc.nodal_combined_forces is not None,
            'has_eigenvalues': sc.eigenvalues is not None,
            'has_frequencies': sc.frequencies is not None,
            'has_mode_shapes': bool(sc.mode_shapes),
            'n_modes': len(sc.mode_shapes) if sc.mode_shapes else 0,
        }
        metadata['subcases'].append(sc_meta)

        sc_arrays = _subcase_to_arrays(sc, sorted_nids, idx)
        all_arrays.update(sc_arrays)

    # Write ZIP archive
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Metadata as JSON
        meta_str = json.dumps(metadata, indent=2, ensure_ascii=False)
        zf.writestr('metadata.json', meta_str)

        # Each array as .npy inside the zip
        for name, arr in all_arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(f'{name}.npy', buf.getvalue())

    import os
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info("Results saved to %s (%.1f MB, %d subcases)",
                filepath, size_mb, len(results.subcases))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(filepath: str) -> Tuple[ResultData, VizModel]:
    """Load results and model proxy from .naero file.

    Parameters
    ----------
    filepath : str
        Path to .naero file.

    Returns
    -------
    results : ResultData
        Restored analysis results.
    viz_model : VizModel
        Lightweight model proxy with geometry data for visualization.
    """
    with zipfile.ZipFile(filepath, 'r') as zf:
        # Load metadata
        metadata = json.loads(zf.read('metadata.json'))

        version = metadata.get('version', 0)
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Unsupported .naero format version {version} "
                f"(max supported: {FORMAT_VERSION})")

        # Load node coordinates
        node_xyz = _load_npy(zf, 'node_xyz')

        # Reconstruct VizModel
        viz_model = _reconstruct_viz_model(metadata, node_xyz)

        # Reconstruct ResultData
        sorted_nids = metadata['model']['node_ids']
        results = ResultData(title=metadata.get('title', ''))

        for idx, sc_meta in enumerate(metadata['subcases']):
            sc = _reconstruct_subcase(zf, sc_meta, sorted_nids, idx)
            results.subcases.append(sc)

    logger.info("Loaded %s: %d subcases, %d nodes, %d elements",
                filepath, len(results.subcases),
                len(viz_model.nodes), len(viz_model.elements))

    return results, viz_model


# ---------------------------------------------------------------------------
# Internal: metadata extraction
# ---------------------------------------------------------------------------

def _extract_model_metadata(bdf_model: Any, sorted_nids: List[int]) -> dict:
    """Extract visualization-relevant subset of bdf_model into JSON-safe dict."""
    model = {}

    # Node IDs (sorted)
    model['node_ids'] = sorted_nids
    model['n_nodes'] = len(sorted_nids)

    # Elements: only type, pid, node_ids needed
    elements = {}
    for eid, elem in bdf_model.elements.items():
        elements[str(eid)] = {
            'type': elem.type,
            'pid': getattr(elem, 'pid', 0),
            'node_ids': list(elem.node_ids),
        }
    model['elements'] = elements
    model['n_elements'] = len(elements)

    # Properties: only A needed (for beam tube radius estimation)
    properties = {}
    for pid, prop in bdf_model.properties.items():
        pdata = {'A': float(getattr(prop, 'A', 0.0))}
        properties[str(pid)] = pdata
    model['properties'] = properties

    # Rigids: full connectivity for RBE visualization
    rigids = {}
    for rid, rbe in bdf_model.rigids.items():
        if hasattr(rbe, 'independent_node') and hasattr(rbe, 'dependent_nodes'):
            # RBE2
            rigids[str(rid)] = {
                'type': 'RBE2',
                'independent_node': rbe.independent_node,
                'components': str(getattr(rbe, 'components', '123456')),
                'dependent_nodes': list(rbe.dependent_nodes),
            }
        elif hasattr(rbe, 'refgrid') and hasattr(rbe, 'weight_sets'):
            # RBE3
            weight_sets = []
            for ws in rbe.weight_sets:
                wt, comp, grids = ws
                weight_sets.append([float(wt), str(comp), list(grids)])
            rigids[str(rid)] = {
                'type': 'RBE3',
                'refgrid': rbe.refgrid,
                'refc': str(getattr(rbe, 'refc', '')),
                'weight_sets': weight_sets,
            }
    model['rigids'] = rigids

    # CAERO1 panels (for aero panel mesh generation)
    caero_panels = {}
    for eid, caero in getattr(bdf_model, 'caero_panels', {}).items():
        caero_panels[str(eid)] = {
            'eid': caero.eid,
            'pid': caero.pid,
            'cp': getattr(caero, 'cp', 0),
            'nspan': caero.nspan,
            'nchord': caero.nchord,
            'lspan': caero.lspan,
            'lchord': caero.lchord,
            'igid': getattr(caero, 'igid', 0),
            'p1': caero.p1.tolist(),
            'chord1': float(caero.chord1),
            'p4': caero.p4.tolist(),
            'chord4': float(caero.chord4),
        }
    model['caero_panels'] = caero_panels

    # AESURF (control surface definitions)
    aesurfs = {}
    for sid, surf in getattr(bdf_model, 'aesurfs', {}).items():
        aesurfs[str(sid)] = {
            'id': surf.id,
            'label': surf.label,
            'cid1': getattr(surf, 'cid1', 0),
            'alid1': getattr(surf, 'alid1', 0),
            'cid2': getattr(surf, 'cid2', 0),
            'alid2': getattr(surf, 'alid2', 0),
            'eff': float(getattr(surf, 'eff', 1.0)),
        }
    model['aesurfs'] = aesurfs

    # AELIST (control surface box lists)
    aelists = {}
    for sid, alist in getattr(bdf_model, 'aelists', {}).items():
        aelists[str(sid)] = {
            'sid': alist.sid,
            'elements': list(alist.elements),
        }
    model['aelists'] = aelists

    # AEFACT (non-uniform panel divisions)
    aefacts = {}
    for sid, afact in getattr(bdf_model, 'aefacts', {}).items():
        aefacts[str(sid)] = {
            'sid': afact.sid,
            'factors': [float(f) for f in afact.factors],
        }
    model['aefacts'] = aefacts

    return model


# ---------------------------------------------------------------------------
# Internal: subcase array conversion
# ---------------------------------------------------------------------------

def _subcase_to_arrays(sc: SubcaseResult, node_ids: List[int], idx: int) -> dict:
    """Convert SubcaseResult to dict of named numpy arrays."""
    prefix = f'sc{idx}_'
    arrays = {}
    n = len(node_ids)

    # Displacements: Dict[int, ndarray(6)] -> (n_nodes, 6)
    if sc.displacements:
        disp = np.zeros((n, 6), dtype=np.float64)
        for i, nid in enumerate(node_ids):
            if nid in sc.displacements:
                disp[i] = np.real(sc.displacements[nid])
        arrays[prefix + 'displacements'] = disp

    # SPC forces
    if sc.spc_forces:
        spc = np.zeros((n, 6), dtype=np.float64)
        for i, nid in enumerate(node_ids):
            if nid in sc.spc_forces:
                spc[i] = np.real(sc.spc_forces[nid])
        arrays[prefix + 'spc_forces'] = spc

    # Aero pressures: ndarray(n_boxes,)
    if sc.aero_pressures is not None:
        arrays[prefix + 'aero_pressures'] = np.real(sc.aero_pressures).astype(np.float64)

    # Aero forces: ndarray(n_boxes, 3)
    if sc.aero_forces is not None:
        arrays[prefix + 'aero_forces'] = np.real(sc.aero_forces).astype(np.float64)

    # AeroBox data: decompose into numpy arrays
    if sc.aero_boxes and len(sc.aero_boxes) > 0:
        nb = len(sc.aero_boxes)
        corners = np.array([box.corners for box in sc.aero_boxes], dtype=np.float64)
        cp = np.array([box.control_point for box in sc.aero_boxes], dtype=np.float64)
        dp = np.array([box.doublet_point for box in sc.aero_boxes], dtype=np.float64)
        normals = np.array([box.normal for box in sc.aero_boxes], dtype=np.float64)
        scalars = np.array([[box.area, box.chord, box.span, float(box.box_id)]
                            for box in sc.aero_boxes], dtype=np.float64)
        arrays[prefix + 'aero_box_corners'] = corners
        arrays[prefix + 'aero_box_cp'] = cp
        arrays[prefix + 'aero_box_dp'] = dp
        arrays[prefix + 'aero_box_normals'] = normals
        arrays[prefix + 'aero_box_scalars'] = scalars

    # Nodal forces: Dict[int, ndarray(6)] -> (n_nodes, 6)
    for name in ('nodal_aero_forces', 'nodal_inertial_forces', 'nodal_combined_forces'):
        forces = getattr(sc, name)
        if forces is not None:
            arr = np.zeros((n, 6), dtype=np.float64)
            for i, nid in enumerate(node_ids):
                if nid in forces:
                    arr[i] = np.real(forces[nid])
            arrays[prefix + name] = arr

    # SOL 103: eigenvalues, frequencies
    if sc.eigenvalues is not None:
        arrays[prefix + 'eigenvalues'] = np.real(sc.eigenvalues).astype(np.float64)
    if sc.frequencies is not None:
        arrays[prefix + 'frequencies'] = np.real(sc.frequencies).astype(np.float64)

    # Mode shapes: List[Dict[int, ndarray(6)]]
    if sc.mode_shapes:
        for m, mode in enumerate(sc.mode_shapes):
            ms = np.zeros((n, 6), dtype=np.float64)
            for i, nid in enumerate(node_ids):
                if nid in mode:
                    ms[i] = np.real(mode[nid])
            arrays[prefix + f'mode_shape_{m}'] = ms

    return arrays


# ---------------------------------------------------------------------------
# Internal: reconstruction from loaded data
# ---------------------------------------------------------------------------

def _load_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Load a .npy file from inside the ZIP."""
    buf = io.BytesIO(zf.read(f'{name}.npy'))
    return np.load(buf)


def _reconstruct_viz_model(metadata: dict, node_xyz: np.ndarray) -> VizModel:
    """Reconstruct VizModel from metadata and coordinates."""
    model_meta = metadata['model']
    vm = VizModel()
    vm.sol = metadata.get('sol', 0)

    # Nodes
    for i, nid in enumerate(model_meta['node_ids']):
        vm.nodes[nid] = VizNode(nid=nid, xyz_global=node_xyz[i].copy())

    # Elements
    for eid_str, edata in model_meta.get('elements', {}).items():
        eid = int(eid_str)
        vm.elements[eid] = VizElement(
            type=edata['type'],
            pid=edata.get('pid', 0),
            node_ids=edata['node_ids'],
        )

    # Properties (for beam tube radius)
    for pid_str, pdata in model_meta.get('properties', {}).items():
        pid = int(pid_str)
        vm.properties[pid] = VizProperty(A=pdata.get('A', 0.0))

    # Rigids
    for rid_str, rdata in model_meta.get('rigids', {}).items():
        rid = int(rid_str)
        if rdata['type'] == 'RBE2':
            vm.rigids[rid] = VizRBE2(
                independent_node=rdata['independent_node'],
                components=rdata.get('components', '123456'),
                dependent_nodes=rdata['dependent_nodes'],
            )
        elif rdata['type'] == 'RBE3':
            ws = [(w, c, g) for w, c, g in rdata['weight_sets']]
            vm.rigids[rid] = VizRBE3(
                refgrid=rdata['refgrid'],
                refc=rdata.get('refc', ''),
                weight_sets=ws,
            )

    # Reconstruct actual CAERO1/AESURF/AELIST/AEFACT card instances
    # These are needed by generate_all_panels() and control surface visualization
    _reconstruct_aero_cards(vm, model_meta)

    return vm


def _reconstruct_aero_cards(vm: VizModel, model_meta: dict) -> None:
    """Reconstruct actual aero card instances for VizModel."""
    from ..bdf.cards.aero import CAERO1, AESURF, AELIST, AEFACT

    for eid_str, cdata in model_meta.get('caero_panels', {}).items():
        eid = int(eid_str)
        c = CAERO1()
        c.eid = cdata['eid']
        c.pid = cdata.get('pid', 0)
        c.cp = cdata.get('cp', 0)
        c.nspan = cdata.get('nspan', 1)
        c.nchord = cdata.get('nchord', 1)
        c.lspan = cdata.get('lspan', 0)
        c.lchord = cdata.get('lchord', 0)
        c.igid = cdata.get('igid', 0)
        c.p1 = np.array(cdata['p1'], dtype=np.float64)
        c.chord1 = float(cdata['chord1'])
        c.p4 = np.array(cdata['p4'], dtype=np.float64)
        c.chord4 = float(cdata['chord4'])
        vm.caero_panels[eid] = c

    for sid_str, sdata in model_meta.get('aesurfs', {}).items():
        sid = int(sid_str)
        s = AESURF()
        s.id = sdata['id']
        s.label = sdata.get('label', '')
        s.cid1 = sdata.get('cid1', 0)
        s.alid1 = sdata.get('alid1', 0)
        s.cid2 = sdata.get('cid2', 0)
        s.alid2 = sdata.get('alid2', 0)
        s.eff = float(sdata.get('eff', 1.0))
        vm.aesurfs[sid] = s

    for sid_str, adata in model_meta.get('aelists', {}).items():
        sid = int(sid_str)
        a = AELIST()
        a.sid = adata['sid']
        a.elements = adata['elements']
        vm.aelists[sid] = a

    for sid_str, fdata in model_meta.get('aefacts', {}).items():
        sid = int(sid_str)
        f = AEFACT()
        f.sid = fdata['sid']
        f.factors = fdata['factors']
        vm.aefacts[sid] = f


def _reconstruct_subcase(
    zf: zipfile.ZipFile,
    sc_meta: dict,
    sorted_nids: List[int],
    idx: int,
) -> SubcaseResult:
    """Reconstruct a SubcaseResult from arrays in the ZIP."""
    from ..aero.panel import AeroBox

    prefix = f'sc{idx}_'
    sc = SubcaseResult()
    sc.subcase_id = sc_meta['subcase_id']
    sc.trim_variables = sc_meta.get('trim_variables')
    sc.trim_balance = sc_meta.get('trim_balance')

    n = len(sorted_nids)

    # Displacements: (n_nodes, 6) -> Dict[int, ndarray(6)]
    if sc_meta.get('has_displacements', False):
        disp_arr = _load_npy(zf, prefix + 'displacements')
        sc.displacements = {}
        for i, nid in enumerate(sorted_nids):
            sc.displacements[nid] = disp_arr[i]

    # SPC forces
    if sc_meta.get('has_spc_forces', False):
        spc_arr = _load_npy(zf, prefix + 'spc_forces')
        sc.spc_forces = {}
        for i, nid in enumerate(sorted_nids):
            if np.any(spc_arr[i] != 0):
                sc.spc_forces[nid] = spc_arr[i]

    # Aero data
    if sc_meta.get('has_aero_pressures', False):
        sc.aero_pressures = _load_npy(zf, prefix + 'aero_pressures')
    if sc_meta.get('has_aero_forces', False):
        sc.aero_forces = _load_npy(zf, prefix + 'aero_forces')
    if sc_meta.get('has_aero_boxes', False):
        corners = _load_npy(zf, prefix + 'aero_box_corners')
        cp = _load_npy(zf, prefix + 'aero_box_cp')
        dp = _load_npy(zf, prefix + 'aero_box_dp')
        normals = _load_npy(zf, prefix + 'aero_box_normals')
        scalars = _load_npy(zf, prefix + 'aero_box_scalars')
        sc.aero_boxes = []
        for j in range(len(corners)):
            box = AeroBox(
                corners=corners[j],
                control_point=cp[j],
                doublet_point=dp[j],
                normal=normals[j],
                area=float(scalars[j, 0]),
                chord=float(scalars[j, 1]),
                span=float(scalars[j, 2]),
                box_id=int(scalars[j, 3]),
            )
            sc.aero_boxes.append(box)

    # Nodal forces: (n_nodes, 6) -> Dict[int, ndarray(6)]
    for name in ('nodal_aero_forces', 'nodal_inertial_forces', 'nodal_combined_forces'):
        key = f'has_{name}'
        if sc_meta.get(key, False):
            arr = _load_npy(zf, prefix + name)
            forces = {}
            for i, nid in enumerate(sorted_nids):
                forces[nid] = arr[i]
            setattr(sc, name, forces)

    # SOL 103: eigenvalues, frequencies, mode shapes
    if sc_meta.get('has_eigenvalues', False):
        sc.eigenvalues = _load_npy(zf, prefix + 'eigenvalues')
    if sc_meta.get('has_frequencies', False):
        sc.frequencies = _load_npy(zf, prefix + 'frequencies')
    if sc_meta.get('has_mode_shapes', False):
        sc.mode_shapes = []
        for m in range(sc_meta.get('n_modes', 0)):
            ms_arr = _load_npy(zf, prefix + f'mode_shape_{m}')
            mode = {}
            for i, nid in enumerate(sorted_nids):
                mode[nid] = ms_arr[i]
            sc.mode_shapes.append(mode)

    return sc
