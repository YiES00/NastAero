"""Stability and control derivative computation via VLM perturbation.

Computes aerodynamic derivatives for 6-DOF flight simulation by:
1. VLM perturbation: geometry-dependent derivatives (CLα, CLδe, Cmα, etc.)
   from the model's panel mesh, solving D·γ = w for each perturbation.
2. Empirical defaults: rate/damping derivatives (Cmq, Clp, Cnr, etc.)
   from Cessna 172 UIUC/FlightGear data cross-validated with Navion.
3. Inertia tensor: directly computed from CONM2 mass distribution.

References
----------
- FlightGear UIUC Cessna 172 Linear Model
- Nelson "Flight Stability and Automatic Control" Appendix B (Navion)
- JSBSim MassProps Database
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...aero.dlm import build_aic_matrix
from ...aero.panel import generate_all_panels, AeroBox, get_box_index_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AeroDerivativeSet — complete stability/control derivative set
# ---------------------------------------------------------------------------

@dataclass
class AeroDerivativeSet:
    """Complete set of stability and control derivatives.

    Fields marked '(VLM)' are computed from the VLM panel model.
    Fields marked '(empirical)' use Cessna 172 / Navion default values.

    All angular derivatives are per-radian. Moment derivatives are about
    the aircraft CG in body axes.
    """
    # Reference quantities (model units)
    S_ref: float = 0.0       # Reference wing area (mm²)
    b_ref: float = 0.0       # Reference span (mm)
    c_bar: float = 0.0       # Mean aerodynamic chord (mm)

    # --- Longitudinal (VLM) ---
    CLalpha: float = 0.0     # Lift curve slope [/rad]
    CLdelta_e: float = 0.0   # Elevator lift effectiveness [/rad]
    Cmalpha: float = 0.0     # Pitch moment vs alpha [/rad]
    Cmdelta_e: float = 0.0   # Pitch moment vs elevator [/rad]

    # --- Longitudinal (empirical, C172/Navion defaults) ---
    CD0: float = 0.032       # Zero-lift drag coefficient
    CDalpha: float = 0.13    # Drag curve slope [/rad] (linear approx)
    Cmq: float = -12.4       # Pitch damping [/rad] (C172: -12.4, Navion: -9.96)
    Cmalpha_dot: float = -5.2  # Downwash lag [/rad] (C172: -5.2, Navion: -4.36)
    CLq: float = 3.9         # Pitch-rate lift [/rad] (C172: 3.9)

    # --- Lateral-directional (VLM) ---
    CYbeta: float = 0.0      # Sideforce vs sideslip [/rad]
    Clbeta: float = 0.0      # Roll moment vs sideslip [/rad]
    Cnbeta: float = 0.0      # Yaw moment vs sideslip [/rad]
    Cldelta_a: float = 0.0   # Roll moment vs aileron [/rad]
    Cndelta_a: float = 0.0   # Yaw moment vs aileron [/rad]
    CYdelta_r: float = 0.0   # Sideforce vs rudder [/rad]
    Cldelta_r: float = 0.0   # Roll moment vs rudder [/rad]
    Cndelta_r: float = 0.0   # Yaw moment vs rudder [/rad]

    # --- Half-wing lift from aileron (VLM, for structural load limiting) ---
    CL_aileron_halfwing: float = 0.0  # Z-force on loaded wing / (q·S) [/rad]

    # --- Lateral-directional (empirical, C172/Navion defaults) ---
    Clp: float = -0.47       # Roll damping [/rad] (C172: -0.47, Navion: -0.41)
    Cnr: float = -0.099      # Yaw damping [/rad] (C172: -0.099, Navion: -0.125)
    Clr: float = 0.096       # Yaw rate → roll coupling [/rad] (C172: 0.096)
    Cnp: float = -0.03       # Roll rate → yaw coupling [/rad] (C172: -0.03)
    CYp: float = -0.037      # Roll rate → sideforce [/rad] (C172: -0.037)
    CYr: float = 0.21        # Yaw rate → sideforce [/rad] (C172: 0.21)

    # --- Tracking: which derivatives were VLM-computed vs empirical ---
    vlm_computed: List[str] = field(default_factory=list)
    empirical: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VLM perturbation computation
# ---------------------------------------------------------------------------

def compute_vlm_derivatives(
    bdf_model,
    mach: float,
    ref_area_mm2: float,
    cg_x_mm: float,
) -> Dict[str, float]:
    """Compute all available stability/control derivatives via VLM perturbation.

    Builds the AIC matrix once, then solves for each perturbation mode
    (alpha, elevator, aileron, rudder, sideslip) to extract force and
    moment coefficients.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model with CAERO panels and AESURF definitions.
    mach : float
        Freestream Mach number.
    ref_area_mm2 : float
        Reference wing area in mm².
    cg_x_mm : float
        CG x-position in model coordinates (mm).

    Returns
    -------
    dict
        Mapping derivative name → value (per-radian).
    """
    boxes = generate_all_panels(bdf_model, use_nastran_eid=True)
    n = len(boxes)
    if n == 0:
        logger.warning("No aero boxes — cannot compute VLM derivatives")
        return {}

    box_id_map = get_box_index_map(boxes)

    # Reference quantities
    aeros = getattr(bdf_model, 'aeros', None)
    if aeros is None:
        logger.error("No AEROS card — cannot compute VLM derivatives")
        return {}

    S = ref_area_mm2 if ref_area_mm2 > 0 else aeros.refs
    b = aeros.refb
    c_bar = aeros.refc

    # Pre-extract panel geometry
    areas = np.array([bx.area for bx in boxes])
    chords = np.array([bx.chord for bx in boxes])
    normals = np.array([bx.normal for bx in boxes])
    cps = np.array([bx.control_point for bx in boxes])
    safe_chords = np.maximum(chords, 1e-30)

    # Moment arms from CG (in model coords, mm)
    # x-arm for pitching moment, y-arm for rolling/yawing
    dx = cps[:, 0] - cg_x_mm   # positive aft
    dy = cps[:, 1]              # positive right (starboard)

    # Build AIC and invert once
    D = build_aic_matrix(boxes, mach=mach)
    D_inv = np.linalg.inv(D)

    derivs = {}

    # ------------------------------------------------------------------
    # Helper: solve for a normalwash vector, return CL, Cm, CY, Cl, Cn
    # ------------------------------------------------------------------
    def _solve_derivs(w_vec, label):
        """Solve D·γ = w, compute force/moment coefficients."""
        gamma = D_inv @ w_vec
        dCp = 2.0 * gamma / safe_chords

        # Panel forces: F_panel = dCp * area * normal (unit q)
        f_panels = (dCp * areas)[:, np.newaxis] * normals  # (n, 3)

        # Lift = sum of z-forces / S (positive up)
        CL = np.sum(f_panels[:, 2]) / S

        # Pitching moment = sum of (z-force * dx_arm) / (S * c_bar)
        # Positive nose-up convention: M = -Fz * dx
        Cm = -np.sum(f_panels[:, 2] * dx) / (S * c_bar)

        # Sideforce = sum of y-forces / S
        CY = np.sum(f_panels[:, 1]) / S

        # Rolling moment = sum of (-z-force * dy) / (S * b)
        Cl = -np.sum(f_panels[:, 2] * dy) / (S * b)

        # Yawing moment = sum of (-y-force * dx) / (S * b)
        Cn = -np.sum(f_panels[:, 1] * dx) / (S * b)

        return CL, Cm, CY, Cl, Cn

    # ------------------------------------------------------------------
    # 1. Angle of attack (α = 1 rad)
    # ------------------------------------------------------------------
    w_alpha = -np.ones(n)  # uniform downwash for unit alpha
    CL_a, Cm_a, _, _, _ = _solve_derivs(w_alpha, "ANGLEA")
    derivs["CLalpha"] = CL_a
    derivs["Cmalpha"] = Cm_a
    logger.info("  CLα = %.3f/rad (VLM)", CL_a)
    logger.info("  Cmα = %.3f/rad (VLM)", Cm_a)

    # ------------------------------------------------------------------
    # 2. Elevator (ELEV)
    # ------------------------------------------------------------------
    w_elev = _control_surface_normalwash("ELEV", bdf_model, boxes, box_id_map)
    if np.any(w_elev != 0):
        CL_de, Cm_de, _, _, _ = _solve_derivs(w_elev, "ELEV")
        derivs["CLdelta_e"] = CL_de
        derivs["Cmdelta_e"] = Cm_de
        logger.info("  CLδe = %.3f/rad (VLM)", CL_de)
        logger.info("  Cmδe = %.3f/rad (VLM)", Cm_de)

    # ------------------------------------------------------------------
    # 3. Aileron (ARON) — antisymmetric deflection
    # ------------------------------------------------------------------
    w_ail = _control_surface_normalwash("ARON", bdf_model, boxes, box_id_map,
                                        antisymmetric=True)
    if np.any(w_ail != 0):
        _, _, _, Cl_da, Cn_da = _solve_derivs(w_ail, "ARON")
        derivs["Cldelta_a"] = Cl_da
        derivs["Cndelta_a"] = Cn_da
        logger.info("  Clδa = %.4f/rad (VLM)", Cl_da)
        logger.info("  Cnδa = %.4f/rad (VLM)", Cn_da)

        # Half-wing CL from aileron: sum z-force on the LOADED wing side.
        # During aileron deflection the two wing halves see opposite
        # incremental lift — one wing gains lift, the other loses it.
        # The "loaded" side determines the wing root shear used for
        # nz-based deflection limiting.  This is much larger than what
        # the rolling moment coefficient Clδa implies because Cl is a
        # span-weighted average while the structural loads come from
        # the raw one-sided z-force integral.
        gamma_ail = D_inv @ w_ail
        dCp_ail = 2.0 * gamma_ail / safe_chords
        fz_ail = dCp_ail * areas * normals[:, 2]  # z-force per panel

        # Select the wing side that gains lift (positive z-force sum)
        right_mask = dy > 0.0
        left_mask = dy < 0.0
        fz_right = np.sum(fz_ail[right_mask])
        fz_left = np.sum(fz_ail[left_mask])
        CL_ail_halfwing = max(abs(fz_right), abs(fz_left)) / S
        derivs["CL_aileron_halfwing"] = CL_ail_halfwing
        logger.info("  CL_ail_halfwing = %.3f/rad (VLM, one-sided)",
                    CL_ail_halfwing)

    # ------------------------------------------------------------------
    # 4. Rudder (RUD)
    # ------------------------------------------------------------------
    w_rud = _control_surface_normalwash("RUD", bdf_model, boxes, box_id_map)
    if np.any(w_rud != 0):
        _, _, CY_dr, Cl_dr, Cn_dr = _solve_derivs(w_rud, "RUD")
        derivs["CYdelta_r"] = CY_dr
        derivs["Cldelta_r"] = Cl_dr
        derivs["Cndelta_r"] = Cn_dr
        logger.info("  CYδr = %.4f/rad (VLM)", CY_dr)
        logger.info("  Clδr = %.4f/rad (VLM)", Cl_dr)
        logger.info("  Cnδr = %.4f/rad (VLM)", Cn_dr)

    # ------------------------------------------------------------------
    # 5. Sideslip (β = 1 rad) — approximate via panel normal y-component
    #    VTP panels have significant n_y component; wing panels ~0.
    #    Unit β induces normalwash w_j = -n_y_j on each panel.
    # ------------------------------------------------------------------
    w_beta = -normals[:, 1].copy()  # n_y component of panel normals
    CY_b, _, _, Cl_b, Cn_b = _solve_derivs(w_beta, "SIDES")
    derivs["CYbeta"] = CY_b
    derivs["Clbeta"] = Cl_b
    derivs["Cnbeta"] = Cn_b
    logger.info("  CYβ = %.4f/rad (VLM)", CY_b)
    logger.info("  Clβ = %.4f/rad (VLM)", Cl_b)
    logger.info("  Cnβ = %.4f/rad (VLM)", Cn_b)

    return derivs


def _control_surface_normalwash(
    label: str,
    bdf_model,
    boxes: List[AeroBox],
    box_id_map: Dict[int, int],
    antisymmetric: bool = False,
) -> np.ndarray:
    """Build normalwash vector for unit control surface deflection.

    Parameters
    ----------
    label : str
        AESURF label (e.g. "ARON", "ELEV", "RUD").
    antisymmetric : bool
        If True, the second AELIST (alid2) gets opposite normalwash sign.
        Use for ailerons: alid1 trailing-edge-down, alid2 trailing-edge-up.
        This produces a rolling-moment–generating deflection pattern.
        For symmetric surfaces (elevator), leave False.
    """
    n = len(boxes)
    w = np.zeros(n)

    for aid, aesurf in bdf_model.aesurfs.items():
        if aesurf.label == label:
            eff = aesurf.eff
            for i, alid in enumerate([aesurf.alid1, aesurf.alid2]):
                # For antisymmetric surfaces, flip sign on 2nd AELIST
                sign = -eff if (i == 0 or not antisymmetric) else +eff
                if alid > 0 and alid in bdf_model.aelists:
                    aelist = bdf_model.aelists[alid]
                    for box_eid in aelist.elements:
                        if box_eid in box_id_map:
                            w[box_id_map[box_eid]] = sign
            if np.any(w != 0):
                mode = "antisymmetric" if antisymmetric else "symmetric"
                logger.info("    %s: %d boxes, eff=%.3f (%s)",
                            label, int(np.sum(w != 0)), eff, mode)
            return w

    return w


# ---------------------------------------------------------------------------
# Inertia tensor from CONM2 masses
# ---------------------------------------------------------------------------

def compute_inertia_from_conm2(
    bdf_model,
    cg_xyz: np.ndarray,
    mass_to_kg: float = 1000.0,
    length_to_m: float = 1e-3,
) -> Dict[str, float]:
    """Compute inertia tensor from CONM2 mass distribution.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model with CONM2 mass elements.
    cg_xyz : ndarray (3,)
        CG position in model coordinates.
    mass_to_kg : float
        Conversion factor from model mass units to kg.
        Default 1000.0 for mm/N/Mg models (1 Mg = 1000 kg).
    length_to_m : float
        Conversion factor from model length units to m.
        Default 1e-3 for mm models.

    Returns
    -------
    dict
        Keys: mass_kg, Ixx, Iyy, Izz, Ixz (SI units: kg, kg·m²).
        Moments are about the CG in body axes.
    """
    total_mass = 0.0
    Ixx = Iyy = Izz = Ixz = 0.0

    # Conversion from model length to m
    mm_to_m = length_to_m

    for eid, elem in bdf_model.masses.items():
        if not hasattr(elem, 'mass'):
            continue
        m = elem.mass * mass_to_kg  # Convert from model units (Mg) to kg
        if m <= 0:
            continue

        # Get node position
        nid = elem.node_ids[0] if hasattr(elem, 'node_ids') else elem.nid
        if nid not in bdf_model.nodes:
            continue
        pos = bdf_model.nodes[nid].xyz_global

        # Distance from CG in meters
        dx = (pos[0] - cg_xyz[0]) * mm_to_m
        dy = (pos[1] - cg_xyz[1]) * mm_to_m
        dz = (pos[2] - cg_xyz[2]) * mm_to_m

        total_mass += m
        Ixx += m * (dy**2 + dz**2)
        Iyy += m * (dx**2 + dz**2)
        Izz += m * (dx**2 + dy**2)
        Ixz += m * (dx * dz)

    logger.info("  Inertia from %d CONM2 masses (%.1f kg):",
                len(bdf_model.masses), total_mass)
    logger.info("    Ixx = %.0f, Iyy = %.0f, Izz = %.0f, Ixz = %.0f kg·m²",
                Ixx, Iyy, Izz, Ixz)

    return {
        "mass_kg": total_mass,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Izz": Izz,
        "Ixz": Ixz,
    }


# ---------------------------------------------------------------------------
# Build complete derivative set
# ---------------------------------------------------------------------------

def build_derivative_set(
    bdf_model,
    config,
    wc,
    mach: float,
) -> AeroDerivativeSet:
    """Build complete derivative set combining VLM + empirical values.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model.
    config : AircraftConfig
        Aircraft configuration.
    wc : WeightCGCondition
        Weight/CG condition.
    mach : float
        Freestream Mach number.

    Returns
    -------
    AeroDerivativeSet
        Complete set ready for 6-DOF simulation.
    """
    aeros = getattr(bdf_model, 'aeros', None)
    if aeros is None:
        raise ValueError("No AEROS card in model")

    S_ref = aeros.refs
    b_ref = aeros.refb
    c_bar = aeros.refc

    # Compute VLM derivatives
    vlm = compute_vlm_derivatives(
        bdf_model, mach, ref_area_mm2=S_ref, cg_x_mm=wc.cg_x,
    )

    # Track which are VLM-computed vs empirical
    vlm_names = list(vlm.keys())
    empirical_names = [
        "CD0", "CDalpha", "Cmq", "Cmalpha_dot", "CLq",
        "Clp", "Cnr", "Clr", "Cnp", "CYp", "CYr",
    ]

    ds = AeroDerivativeSet(
        S_ref=S_ref,
        b_ref=b_ref,
        c_bar=c_bar,
        # VLM longitudinal
        CLalpha=vlm.get("CLalpha", 2 * math.pi),
        CLdelta_e=vlm.get("CLdelta_e", 0.4),
        Cmalpha=vlm.get("Cmalpha", -0.5),
        Cmdelta_e=vlm.get("Cmdelta_e", -1.1),
        # VLM lateral-directional
        CYbeta=vlm.get("CYbeta", -0.31),
        Clbeta=vlm.get("Clbeta", -0.089),
        Cnbeta=vlm.get("Cnbeta", 0.065),
        Cldelta_a=vlm.get("Cldelta_a", -0.17),
        CL_aileron_halfwing=vlm.get("CL_aileron_halfwing", 0.5),
        Cndelta_a=vlm.get("Cndelta_a", -0.02),
        CYdelta_r=vlm.get("CYdelta_r", 0.19),
        Cldelta_r=vlm.get("Cldelta_r", 0.01),
        Cndelta_r=vlm.get("Cndelta_r", -0.07),
        # Empirical (C172/Navion defaults)
        CD0=0.032,
        CDalpha=0.13,
        Cmq=-12.4,
        Cmalpha_dot=-5.2,
        CLq=3.9,
        Clp=-0.47,
        Cnr=-0.099,
        Clr=0.096,
        Cnp=-0.03,
        CYp=-0.037,
        CYr=0.21,
        # Tracking
        vlm_computed=vlm_names,
        empirical=empirical_names,
    )

    logger.info("  → %d VLM-computed, %d empirical derivatives",
                len(vlm_names), len(empirical_names))

    return ds
