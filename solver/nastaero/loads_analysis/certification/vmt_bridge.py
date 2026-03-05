"""VMT bridge: convert BatchResult nodal forces into VMT data for EnvelopeProcessor.

Connects the BatchRunner output (per-case nodal_forces dicts) to the VMT
integration pipeline (compute_vmt_all) and feeds results into EnvelopeProcessor.

Data flow:
    BatchResult.case_results[i].nodal_forces  (Dict[node_id, ndarray(6)])
        → compute_vmt_all(model, forces, components)
        → VMTResult with VMTCurve per component
        → vmt_data dict in EnvelopeProcessor format

Usage:
    from .vmt_bridge import compute_vmt_for_batch

    vmt_data = compute_vmt_for_batch(bdf_model, batch_result)
    proc = EnvelopeProcessor(batch_result, vmt_data)
    proc.compute_envelopes()
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..component_id import ComponentSet, identify_components
from ..vmt import compute_vmt_all
from .batch_runner import BatchResult

logger = logging.getLogger(__name__)


def compute_vmt_for_batch(
    model: Any,
    batch_result: BatchResult,
    components: Optional[ComponentSet] = None,
    n_stations: int = 50,
    fuselage_cg_x: Optional[float] = None,
) -> Dict[int, Dict[str, dict]]:
    """Compute VMT curves for all converged cases in a BatchResult.

    For each converged case with nodal forces, calls compute_vmt_all()
    to integrate shear/bending/torsion along all structural components.
    Returns data in the exact format expected by EnvelopeProcessor.

    Parameters
    ----------
    model : BDFModel
        The structural model (for node positions and component ID).
    batch_result : BatchResult
        Results from BatchRunner.run().
    components : ComponentSet, optional
        Structural components. If None, auto-identified from model.
    n_stations : int
        Number of span stations for VMT integration.
    fuselage_cg_x : float, optional
        Aircraft CG X position (mm). If provided, the fuselage VMT is
        computed by integrating forward and aft from the CG separately,
        producing a distribution that peaks at the CG.

    Returns
    -------
    dict of {case_id: {component_name: {"stations": ndarray,
             "shear": ndarray, "bending": ndarray, "torsion": ndarray}}}
        Ready for ``EnvelopeProcessor(batch_result, vmt_data)``.
    """
    if components is None:
        components = identify_components(model)
        logger.info("Identified %d structural components: %s",
                    len(components.components),
                    ', '.join(components.names()))

    vmt_data: Dict[int, Dict[str, dict]] = {}
    n_computed = 0
    n_skipped = 0

    for cr in batch_result.case_results:
        if not cr.converged or cr.nodal_forces is None:
            n_skipped += 1
            continue

        vmt_result = compute_vmt_all(
            model, cr.nodal_forces, components,
            n_stations=n_stations,
            subcase_id=cr.case_id,
            fuselage_cg_x=fuselage_cg_x,
        )

        case_vmt: Dict[str, dict] = {}
        for curve in vmt_result.curves:
            case_vmt[curve.component_name] = {
                "stations": curve.stations,
                "shear": curve.shear,
                "bending": curve.bending_moment,
                "torsion": curve.torsion,
            }

        vmt_data[cr.case_id] = case_vmt
        n_computed += 1

    logger.info("VMT computed for %d cases (%d skipped)",
                n_computed, n_skipped)

    return vmt_data
