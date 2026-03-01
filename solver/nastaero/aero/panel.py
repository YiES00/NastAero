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


def generate_panel_mesh(caero1, start_id: int = 0, use_nastran_eid: bool = False) -> List[AeroBox]:
    """Generate DLM boxes from a CAERO1 card.

    The CAERO1 defines a trapezoidal panel via:
    - p1 (inboard LE), chord1 (inboard chord length)
    - p4 (outboard LE), chord4 (outboard chord length)
    - nspan (spanwise divisions), nchord (chordwise divisions)

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
    """
    nspan = max(caero1.nspan, 1)
    nchord = max(caero1.nchord, 1)

    p1 = np.asarray(caero1.p1, dtype=float)
    p4 = np.asarray(caero1.p4, dtype=float)

    boxes = []
    box_id = start_id
    nastran_eid = getattr(caero1, 'eid', start_id)

    for j in range(nspan):
        # Spanwise interpolation
        eta0 = j / nspan
        eta1 = (j + 1) / nspan

        # Inboard and outboard LE/TE for this strip
        le_in = p1 + eta0 * (p4 - p1)
        le_out = p1 + eta1 * (p4 - p1)
        chord_in = caero1.chord1 + eta0 * (caero1.chord4 - caero1.chord1)
        chord_out = caero1.chord1 + eta1 * (caero1.chord4 - caero1.chord1)

        # Stream direction (assumed +X)
        stream = np.array([1.0, 0.0, 0.0])

        for i in range(nchord):
            # Chordwise interpolation
            xi0 = i / nchord
            xi1 = (i + 1) / nchord

            # Four corners of this box (LE-in, TE-in, TE-out, LE-out)
            c0 = le_in + xi0 * chord_in * stream   # inboard LE side
            c1 = le_in + xi1 * chord_in * stream   # inboard TE side
            c2 = le_out + xi1 * chord_out * stream  # outboard TE side
            c3 = le_out + xi0 * chord_out * stream  # outboard LE side

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

            # Normal (cross product of diagonals)
            d1 = c2 - c0
            d2 = c3 - c1
            normal = np.cross(d1, d2)
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-12:
                normal = normal / norm_mag

            area = 0.5 * norm_mag

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
    for eid in sorted(bdf_model.caero_panels.keys()):
        caero = bdf_model.caero_panels[eid]
        boxes = generate_panel_mesh(caero, start_id=box_id,
                                     use_nastran_eid=use_nastran_eid)
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
