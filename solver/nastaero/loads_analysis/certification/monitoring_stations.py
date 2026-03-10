"""Critical monitoring station identification for VMT envelope analysis.

Automatically identifies key span/fuselage stations where VMT potato plots
and critical load selection should be performed, based on:
- Control surface hinge-line boundaries (AESURF/CAERO)
- Landing gear attachment positions
- Engine/pylon locations (large concentrated masses)
- Wing/tail root and tip locations
- Fuselage large mass items, wing/tail reference points

Usage
-----
>>> from nastaero.loads_analysis.certification.monitoring_stations import (
...     identify_monitoring_stations,
... )
>>> stations = identify_monitoring_stations(model, config, components)
>>> for comp, sta_list in stations.items():
...     print(f"{comp}: {[s.label for s in sta_list]}")
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..component_id import ComponentDef, ComponentSet


@dataclass
class MonitoringStation:
    """A critical monitoring station for VMT analysis.

    Attributes
    ----------
    position : float
        Station coordinate along the span axis (mm).
    label : str
        Human-readable description.
    reason : str
        Why this station is critical (e.g., "aileron_boundary",
        "landing_gear", "engine_pylon", "root", "tip", "large_mass").
    component : str
        Name of the structural component.
    """
    position: float
    label: str
    reason: str
    component: str


def identify_monitoring_stations(
    model: Any,
    config: Any = None,
    components: Optional[ComponentSet] = None,
    n_equidistant: int = 0,
    mass_threshold_kg: float = 5.0,
    offset_mm: float = 50.0,
    vtol_config: Any = None,
) -> Dict[str, List[MonitoringStation]]:
    """Identify critical monitoring stations for all structural components.

    Parameters
    ----------
    model : BDFModel
        Parsed structural model.
    config : AircraftConfig, optional
        Aircraft configuration (for landing gear positions, CG).
    components : ComponentSet, optional
        Structural components. Auto-identified if None.
    n_equidistant : int
        Additional equidistant stations to add between identified points.
    mass_threshold_kg : float
        Minimum CONM2 mass to consider as a "large mass item" (in kg).
    offset_mm : float
        Offset distance before/after feature locations (mm).
    vtol_config : VTOLConfig, optional
        VTOL rotor configuration. If provided, adds monitoring stations
        at rotor hub span positions on wing components.

    Returns
    -------
    dict of {component_name: list of MonitoringStation}
        Monitoring stations sorted by position for each component.
    """
    from ..component_id import identify_components

    if components is None:
        components = identify_components(model)

    result: Dict[str, List[MonitoringStation]] = {}

    for comp in components.components:
        stations = []
        span_ax = comp.span_axis
        comp_name = comp.name

        # Get span range of this component
        span_coords = []
        for nid in comp.node_ids:
            if nid in model.nodes:
                span_coords.append(model.nodes[nid].xyz_global[span_ax])
        if not span_coords:
            result[comp_name] = []
            continue

        span_min = min(span_coords)
        span_max = max(span_coords)
        span_range = span_max - span_min

        # ── Root and tip ──
        stations.append(MonitoringStation(
            position=span_min + offset_mm,
            label=f"{comp_name} Root",
            reason="root",
            component=comp_name,
        ))
        stations.append(MonitoringStation(
            position=span_max - offset_mm,
            label=f"{comp_name} Tip",
            reason="tip",
            component=comp_name,
        ))

        # ── Control surface boundaries ──
        if hasattr(model, 'aesurfs') and model.aesurfs:
            cs_boundaries = _find_control_surface_boundaries(
                model, comp, span_ax)
            for pos, cs_name in cs_boundaries:
                if span_min < pos < span_max:
                    # Inboard boundary
                    pos_in = max(pos - offset_mm, span_min + offset_mm)
                    stations.append(MonitoringStation(
                        position=pos_in,
                        label=f"Inboard of {cs_name}",
                        reason="ctrl_surface_boundary",
                        component=comp_name,
                    ))
                    # Outboard boundary
                    pos_out = min(pos + offset_mm, span_max - offset_mm)
                    stations.append(MonitoringStation(
                        position=pos_out,
                        label=f"Outboard of {cs_name}",
                        reason="ctrl_surface_boundary",
                        component=comp_name,
                    ))

        # ── Landing gear ──
        if config and config.landing_gear:
            gear = config.landing_gear
            _add_gear_stations(stations, comp, span_ax, span_min,
                                span_max, gear, model, offset_mm)

        # ── Large mass items ──
        _add_mass_stations(stations, comp, span_ax, span_min, span_max,
                            model, mass_threshold_kg, offset_mm)

        # ── Rotor hub positions (VTOL) ──
        if vtol_config is not None:
            _add_rotor_hub_stations(stations, comp, span_ax, span_min,
                                     span_max, vtol_config, offset_mm)

        # ── Fuselage-specific: wing/tail reference points ──
        if 'fuselage' in comp_name.lower() and span_ax == 0:
            _add_fuselage_feature_stations(
                stations, comp, components, model, config, offset_mm)

        # ── Mid-span ──
        mid = (span_min + span_max) / 2.0
        stations.append(MonitoringStation(
            position=mid,
            label=f"{comp_name} Mid-span",
            reason="mid_span",
            component=comp_name,
        ))

        # ── Equidistant stations ──
        if n_equidistant > 0:
            for i in range(1, n_equidistant + 1):
                frac = i / (n_equidistant + 1)
                pos = span_min + frac * span_range
                stations.append(MonitoringStation(
                    position=pos,
                    label=f"{comp_name} {frac*100:.0f}% span",
                    reason="equidistant",
                    component=comp_name,
                ))

        # ── De-duplicate and sort ──
        stations = _deduplicate_stations(stations, min_spacing=offset_mm)
        stations.sort(key=lambda s: abs(s.position))

        result[comp_name] = stations

    return result


def _find_control_surface_boundaries(
    model: Any, comp: ComponentDef, span_ax: int,
) -> List[tuple]:
    """Find control surface inboard/outboard boundaries in span coordinates.

    Returns list of (span_position, surface_name) at the boundary locations.
    """
    boundaries = []
    comp_nids = set(comp.node_ids)

    # Map AESURF names
    aesurf_names = {}
    for aid, aesurf in model.aesurfs.items():
        label = getattr(aesurf, 'label', '') or getattr(aesurf, 'name', '')
        aesurf_names[aid] = label.strip()

    # Build set of box IDs per AESURF from AELISTs
    aesurf_boxes = {}  # {aesurf_id: set of box IDs}
    if hasattr(model, 'aelists'):
        for aid, aesurf in model.aesurfs.items():
            boxes = set()
            for alid in [getattr(aesurf, 'alid1', 0),
                         getattr(aesurf, 'alid2', 0)]:
                if alid and alid in model.aelists:
                    boxes.update(model.aelists[alid].elements)
            if boxes:
                aesurf_boxes[aid] = boxes

    # Examine CAERO panels to find control surface extents
    caero_spans = {}  # {aesurf_label: (min_span, max_span)}

    for eid, caero in model.caero_panels.items():
        p1 = getattr(caero, 'p1', None)
        p4 = getattr(caero, 'p4', None)
        if p1 is None or p4 is None:
            continue

        # Box IDs generated by this CAERO1: eid .. eid+nspan*nchord-1
        ns = max(getattr(caero, 'nspan', 1), 1)
        nc = max(getattr(caero, 'nchord', 1), 1)
        caero_box_ids = set(range(eid, eid + ns * nc))

        span_vals = [p1[span_ax], p4[span_ax]]
        s_min, s_max = min(span_vals), max(span_vals)

        # Check for associated AESURF via box ID overlap
        for aid, boxes in aesurf_boxes.items():
            if caero_box_ids & boxes:  # non-empty intersection
                name = aesurf_names.get(aid, f"CS_{aid}")
                if name not in caero_spans:
                    caero_spans[name] = (s_min, s_max)
                else:
                    old_min, old_max = caero_spans[name]
                    caero_spans[name] = (
                        min(old_min, s_min),
                        max(old_max, s_max),
                    )

    # Generate boundary points from control surface extents
    for cs_name, (s_min, s_max) in caero_spans.items():
        boundaries.append((s_min, cs_name))
        boundaries.append((s_max, cs_name))

    return boundaries


def _add_gear_stations(
    stations: List[MonitoringStation],
    comp: ComponentDef,
    span_ax: int,
    span_min: float,
    span_max: float,
    gear: Any,
    model: Any,
    offset_mm: float,
):
    """Add monitoring stations near landing gear attachment points."""
    comp_name = comp.name

    # For wings (span_ax=Y), check if gear nodes have Y overlap
    # For fuselage (span_ax=X), use gear X positions
    if span_ax == 0:  # fuselage
        for x_pos, label in [
            (gear.main_gear_x, "Main Gear"),
            (gear.nose_gear_x, "Nose Gear"),
        ]:
            if x_pos and span_min < x_pos < span_max:
                stations.append(MonitoringStation(
                    position=max(x_pos - offset_mm, span_min + offset_mm),
                    label=f"Fwd of {label}",
                    reason="landing_gear",
                    component=comp_name,
                ))
                stations.append(MonitoringStation(
                    position=min(x_pos + offset_mm, span_max - offset_mm),
                    label=f"Aft of {label}",
                    reason="landing_gear",
                    component=comp_name,
                ))
    else:  # wings
        # Check if any gear nodes fall within this wing component
        all_gear_nids = list(gear.main_gear_node_ids or [])
        for nid in all_gear_nids:
            if nid in model.nodes:
                y_pos = model.nodes[nid].xyz_global[span_ax]
                if span_min < y_pos < span_max:
                    stations.append(MonitoringStation(
                        position=max(y_pos - offset_mm,
                                      span_min + offset_mm),
                        label="Inboard of Gear",
                        reason="landing_gear",
                        component=comp_name,
                    ))
                    stations.append(MonitoringStation(
                        position=min(y_pos + offset_mm,
                                      span_max - offset_mm),
                        label="Outboard of Gear",
                        reason="landing_gear",
                        component=comp_name,
                    ))


def _add_mass_stations(
    stations: List[MonitoringStation],
    comp: ComponentDef,
    span_ax: int,
    span_min: float,
    span_max: float,
    model: Any,
    mass_threshold_kg: float,
    offset_mm: float,
):
    """Add monitoring stations near large concentrated masses."""
    comp_nids = set(comp.node_ids)
    comp_name = comp.name

    # Find large masses belonging to this component
    mass_items = []
    for mid, mass_obj in model.masses.items():
        m_kg = getattr(mass_obj, 'mass', 0.0) * 1000.0  # Mg → kg
        nid = getattr(mass_obj, 'nid', 0)
        if m_kg >= mass_threshold_kg and nid in comp_nids and nid in model.nodes:
            pos = model.nodes[nid].xyz_global[span_ax]
            mass_items.append((m_kg, pos, nid))

    # Sort by mass descending, take top items
    mass_items.sort(reverse=True)

    # Limit to top 5 per component to avoid clutter
    for m_kg, pos, nid in mass_items[:5]:
        if span_min + offset_mm < pos < span_max - offset_mm:
            stations.append(MonitoringStation(
                position=max(pos - offset_mm, span_min + offset_mm),
                label=f"Fwd of mass {m_kg:.1f}kg (N{nid})",
                reason="large_mass",
                component=comp_name,
            ))
            stations.append(MonitoringStation(
                position=min(pos + offset_mm, span_max - offset_mm),
                label=f"Aft of mass {m_kg:.1f}kg (N{nid})",
                reason="large_mass",
                component=comp_name,
            ))


def _add_fuselage_feature_stations(
    stations: List[MonitoringStation],
    fuse_comp: ComponentDef,
    all_components: ComponentSet,
    model: Any,
    config: Any,
    offset_mm: float,
):
    """Add fuselage stations at wing/tail attachment regions and CG."""
    comp_name = fuse_comp.name
    span_ax = fuse_comp.span_axis  # X=0

    fuse_nids = set(fuse_comp.node_ids)
    fuse_coords = []
    for nid in fuse_comp.node_ids:
        if nid in model.nodes:
            fuse_coords.append(model.nodes[nid].xyz_global[0])
    if not fuse_coords:
        return

    span_min = min(fuse_coords)
    span_max = max(fuse_coords)

    # Wing attachment region: find X range where wing root nodes are
    for comp in all_components.components:
        if 'wing' in comp.name.lower() and comp.span_axis == 1:
            # Find X positions of wing root nodes (smallest |Y|)
            root_x_vals = []
            for nid in comp.node_ids:
                if nid in model.nodes:
                    xyz = model.nodes[nid].xyz_global
                    y_abs = abs(xyz[1])
                    if y_abs < 1000.0:  # near fuselage junction
                        root_x_vals.append(xyz[0])
            if root_x_vals:
                wing_x_min = min(root_x_vals)
                wing_x_max = max(root_x_vals)
                wing_x_mid = (wing_x_min + wing_x_max) / 2.0

                if span_min < wing_x_mid < span_max:
                    stations.append(MonitoringStation(
                        position=max(wing_x_min - offset_mm,
                                      span_min + offset_mm),
                        label="Fwd of Wing Attachment",
                        reason="wing_attachment",
                        component=comp_name,
                    ))
                    stations.append(MonitoringStation(
                        position=min(wing_x_max + offset_mm,
                                      span_max - offset_mm),
                        label="Aft of Wing Attachment",
                        reason="wing_attachment",
                        component=comp_name,
                    ))
            break  # only first wing (left/right have same X)

    # HTP attachment region
    for comp in all_components.components:
        if 'htp' in comp.name.lower() and comp.span_axis == 1:
            root_x_vals = []
            for nid in comp.node_ids:
                if nid in model.nodes:
                    xyz = model.nodes[nid].xyz_global
                    if abs(xyz[1]) < 1000.0:
                        root_x_vals.append(xyz[0])
            if root_x_vals:
                htp_x_min = min(root_x_vals)
                htp_x_max = max(root_x_vals)
                htp_x_mid = (htp_x_min + htp_x_max) / 2.0

                if span_min < htp_x_mid < span_max:
                    stations.append(MonitoringStation(
                        position=max(htp_x_min - offset_mm,
                                      span_min + offset_mm),
                        label="Fwd of HTP Attachment",
                        reason="htp_attachment",
                        component=comp_name,
                    ))
                    stations.append(MonitoringStation(
                        position=min(htp_x_max + offset_mm,
                                      span_max - offset_mm),
                        label="Aft of HTP Attachment",
                        reason="htp_attachment",
                        component=comp_name,
                    ))
            break

    # CG position
    if config and config.weight_cg_conditions:
        cg_x = config.weight_cg_conditions[0].cg_x
        if span_min < cg_x < span_max:
            stations.append(MonitoringStation(
                position=cg_x,
                label="At CG",
                reason="cg_position",
                component=comp_name,
            ))


def _add_rotor_hub_stations(
    stations: List[MonitoringStation],
    comp: ComponentDef,
    span_ax: int,
    span_min: float,
    span_max: float,
    vtol_config: Any,
    offset_mm: float,
):
    """Add monitoring stations at rotor hub positions on this component.

    Rotor hubs mounted on wing components create critical load introduction
    points. Adds stations inboard and outboard of each hub position.
    """
    comp_name = comp.name

    for rotor in vtol_config.rotors:
        hub_pos = rotor.hub_position[span_ax]

        # Check if this hub falls within this component's span range
        if span_min + offset_mm < hub_pos < span_max - offset_mm:
            stations.append(MonitoringStation(
                position=max(hub_pos - offset_mm, span_min + offset_mm),
                label=f"Inboard of {rotor.label}",
                reason="rotor_hub",
                component=comp_name,
            ))
            stations.append(MonitoringStation(
                position=min(hub_pos + offset_mm, span_max - offset_mm),
                label=f"Outboard of {rotor.label}",
                reason="rotor_hub",
                component=comp_name,
            ))
        elif abs(hub_pos) < offset_mm and span_ax == 1:
            # Hub near centerline on a Y-span component (like fuselage rotor)
            pass  # Skip, not relevant for wing components


def _deduplicate_stations(
    stations: List[MonitoringStation],
    min_spacing: float = 30.0,
) -> List[MonitoringStation]:
    """Remove stations too close together, keeping higher-priority ones.

    Priority: root > landing_gear > ctrl_surface_boundary > wing_attachment
    > large_mass > cg_position > mid_span > equidistant > tip
    """
    priority = {
        "root": 0,
        "landing_gear": 1,
        "rotor_hub": 1,
        "ctrl_surface_boundary": 2,
        "wing_attachment": 3,
        "htp_attachment": 3,
        "large_mass": 4,
        "cg_position": 5,
        "mid_span": 6,
        "equidistant": 7,
        "tip": 8,
    }

    # Sort by position
    sorted_stations = sorted(stations, key=lambda s: s.position)

    # Greedy dedup: keep the higher-priority station when two are close
    if not sorted_stations:
        return []

    result = [sorted_stations[0]]
    for sta in sorted_stations[1:]:
        last = result[-1]
        if abs(sta.position - last.position) < min_spacing:
            # Keep the one with higher priority (lower number)
            p_sta = priority.get(sta.reason, 99)
            p_last = priority.get(last.reason, 99)
            if p_sta < p_last:
                result[-1] = sta
        else:
            result.append(sta)

    return result
