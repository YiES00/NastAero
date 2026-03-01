"""Aerodynamic panel mesh generation from CAERO1 cards."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class AeroBox:
    """Single aerodynamic doublet-lattice box."""
    corners: np.ndarray      # (4, 3) corner coordinates
    control_point: np.ndarray  # 3/4 chord midspan point (downwash location)
    doublet_point: np.ndarray  # 1/4 chord midspan point (doublet line)
    normal: np.ndarray        # panel normal vector
    area: float = 0.0
    chord: float = 0.0
    span: float = 0.0
    box_id: int = 0


def generate_panel_mesh(caero1, start_id: int = 0, use_nastran_eid: bool = False,
                        aefacts: dict = None) -> List[AeroBox]:
    """Generate DLM boxes from a CAERO1 card.

    The CAERO1 defines a trapezoidal panel via:
    - p1 (inboard LE), chord1 (inboard chord length)
    - p4 (outboard LE), chord4 (outboard chord length)
    - nspan (spanwise divisions), nchord (chordwise divisions)
    - lspan > 0: use AEFACT for non-uniform spanwise divisions
    - lchord > 0: use AEFACT for non-uniform chordwise divisions

    DLM convention:
    - Doublet line at 1/4 chord of each box
    - Control point (downwash point) at 3/4 chord midspan

    Parameters
    ----------
    caero1 : CAERO1
        The CAERO1 card definition.
    start_id : int
        Sequential box index (used for matrix indexing).
    use_nastran_eid : bool
        If True, box_id uses Nastran convention: CAERO1.eid + sequential offset.
        If False, box_id = start_id + sequential offset.
    aefacts : dict, optional
        Dictionary of AEFACT cards keyed by SID. Used for non-uniform divisions.
    """
    nspan = max(caero1.nspan, 1)
    nchord = max(caero1.nchord, 1)

    p1 = np.asarray(caero1.p1, dtype=float)
    p4 = np.asarray(caero1.p4, dtype=float)

    # Build spanwise eta fractions (0 to 1)
    span_etas = None
    if aefacts and caero1.lspan > 0 and caero1.lspan in aefacts:
        factors = aefacts[caero1.lspan].factors
        if factors:
            # AEFACT gives division fractions (typically 0.0 to 1.0)
            # or absolute values that need normalizing
            span_etas = np.array(factors, dtype=float)
            if span_etas[-1] > 1.5:
                # Absolute values - normalize to 0..1
                span_etas = (span_etas - span_etas[0]) / (span_etas[-1] - span_etas[0])
            nspan = len(span_etas) - 1

    if span_etas is None:
        span_etas = np.linspace(0.0, 1.0, nspan + 1)

    # Build chordwise xi fractions (0 to 1)
    chord_xis = None
    if aefacts and caero1.lchord > 0 and caero1.lchord in aefacts:
        factors = aefacts[caero1.lchord].factors
        if factors:
            chord_xis = np.array(factors, dtype=float)
            if chord_xis[-1] > 1.5:
                chord_xis = (chord_xis - chord_xis[0]) / (chord_xis[-1] - chord_xis[0])
            nchord = len(chord_xis) - 1

    if chord_xis is None:
        chord_xis = np.linspace(0.0, 1.0, nchord + 1)

    boxes = []
    box_id = start_id
    nastran_eid = getattr(caero1, 'eid', start_id)

    for j in range(nspan):
        # Spanwise interpolation using eta fractions
        eta0 = span_etas[j]
        eta1 = span_etas[j + 1]

        # Inboard and outboard LE/TE for this strip
        le_in = p1 + eta0 * (p4 - p1)
        le_out = p1 + eta1 * (p4 - p1)
        chord_in = caero1.chord1 + eta0 * (caero1.chord4 - caero1.chord1)
        chord_out = caero1.chord1 + eta1 * (caero1.chord4 - caero1.chord1)

        # Stream direction (assumed +X)
        stream = np.array([1.0, 0.0, 0.0])

        for i in range(nchord):
            # Chordwise interpolation using xi fractions
            xi0 = chord_xis[i]
            xi1 = chord_xis[i + 1]

            # Four corners of this box (LE-in, TE-in, TE-out, LE-out)
            c0 = le_in + xi0 * chord_in * stream   # inboard LE side
            c1 = le_in + xi1 * chord_in * stream   # inboard TE side
            c2 = le_out + xi1 * chord_out * stream  # outboard TE side
            c3 = le_out + xi0 * chord_out * stream  # outboard LE side

            # Normal (cross product of diagonals)
            d1 = c2 - c0
            d2 = c3 - c1
            normal = np.cross(d1, d2)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-12:
                normal = normal / norm_mag

            area = 0.5 * norm_mag

            # Ensure consistent normal direction (Nastran DLM convention):
            # Panel normals must be consistent so that positive circulation
            # produces positive lift. For horizontal surfaces (wing, tail),
            # nz > 0; for vertical surfaces (fin), ny > 0.
            # When the normal is flipped, reverse corner winding so that
            # the bound-vortex direction (a→b from corner ordering)
            # stays consistent with the new normal sign.
            if abs(normal[2]) >= abs(normal[1]):
                if normal[2] < 0:
                    normal = -normal
                    c0, c1, c2, c3 = c3, c2, c1, c0
            else:
                if normal[1] < 0:
                    normal = -normal
                    c0, c1, c2, c3 = c3, c2, c1, c0

            corners = np.array([c0, c1, c2, c3])

            # Box chord and span
            mid_in = 0.5 * (c0 + c1)
            mid_out = 0.5 * (c2 + c3)
            box_span_vec = mid_out - mid_in
            box_span = np.linalg.norm(box_span_vec)

            box_chord = 0.5 * (np.linalg.norm(c1 - c0) + np.linalg.norm(c2 - c3))

            # 1/4 chord doublet line midspan point
            qc_in = c0 + 0.25 * (c1 - c0)
            qc_out = c3 + 0.25 * (c2 - c3)
            doublet_pt = 0.5 * (qc_in + qc_out)

            # 3/4 chord control point at midspan
            tqc_in = c0 + 0.75 * (c1 - c0)
            tqc_out = c3 + 0.75 * (c2 - c3)
            control_pt = 0.5 * (tqc_in + tqc_out)

            # box_id for matrix indexing is sequential (start_id based)
            # nastran_box_id follows Nastran convention: CAERO1 EID + offset
            seq_offset = box_id - start_id
            actual_box_id = (nastran_eid + seq_offset) if use_nastran_eid else box_id

            box = AeroBox(
                corners=corners,
                control_point=control_pt,
                doublet_point=doublet_pt,
                normal=normal,
                area=area,
                chord=box_chord,
                span=box_span,
                box_id=actual_box_id,
            )
            boxes.append(box)
            box_id += 1

    return boxes


def generate_all_panels(bdf_model, use_nastran_eid: bool = True) -> List[AeroBox]:
    """Generate all DLM boxes from all CAERO1 cards in the model.

    Parameters
    ----------
    bdf_model : BDFModel
        The BDF model containing CAERO1 cards.
    use_nastran_eid : bool
        If True, box_id uses Nastran convention (CAERO1 EID + offset).

    Returns
    -------
    all_boxes : list of AeroBox
        All generated aerodynamic boxes, in order.
    """
    all_boxes = []
    box_id = 0
    aefacts = getattr(bdf_model, 'aefacts', None)
    for eid in sorted(bdf_model.caero_panels.keys()):
        caero = bdf_model.caero_panels[eid]
        boxes = generate_panel_mesh(caero, start_id=box_id,
                                     use_nastran_eid=use_nastran_eid,
                                     aefacts=aefacts)
        all_boxes.extend(boxes)
        box_id += len(boxes)
    return all_boxes


def get_box_index_map(boxes: List[AeroBox]) -> dict:
    """Build mapping from Nastran box_id to sequential index.

    Parameters
    ----------
    boxes : list of AeroBox
        All aerodynamic boxes.

    Returns
    -------
    box_id_to_index : dict
        Mapping from box_id (Nastran EID) to sequential index in the boxes list.
    """
    return {box.box_id: i for i, box in enumerate(boxes)}
