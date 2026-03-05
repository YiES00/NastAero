"""V-n Diagram generation for FAA Part 23 certification.

Computes the maneuver and gust flight envelopes per §23.321-§23.341,
including all critical corner points for load case generation.

References
----------
- §23.321: Flight loads — General
- §23.333: Flight envelope
- §23.337: Limit maneuvering load factors
- §23.341: Gust load factors (Pratt formula)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .aircraft_config import (
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    part23_nz_max, part23_nz_min,
    eas_to_tas, eas_to_mach, dynamic_pressure_from_eas,
    RHO_0, G_MPS2, KG_PER_LB, FPS_TO_MPS,
)
from ..case_generator import isa_atmosphere


# ---------------------------------------------------------------------------
# V-n point
# ---------------------------------------------------------------------------

@dataclass
class VnPoint:
    """A point on the V-n diagram.

    Attributes
    ----------
    V_eas : float
        Equivalent airspeed (m/s).
    nz : float
        Load factor (g's).
    label : str
        Point identifier (e.g., "A+", "D-", "Gust_VC+").
    category : str
        Type: "maneuver", "gust", "flap", "stall".
    """
    V_eas: float = 0.0
    nz: float = 0.0
    label: str = ""
    category: str = "maneuver"


# ---------------------------------------------------------------------------
# Pratt gust formula
# ---------------------------------------------------------------------------

def pratt_gust_delta_nz(V_eas: float, Ude_fps: float,
                         wing_loading_pa: float, CLalpha: float,
                         mean_chord_m: float, weight_N: float,
                         wing_area_m2: float) -> float:
    """Compute gust-induced incremental load factor (Pratt formula).

    Δn = (ρ₀ × V × CLα × Kg × Ude) / (2 × W/S)

    where Kg = 0.88 × μg / (5.3 + μg)  (gust alleviation factor)
          μg = 2 × (W/S) / (ρ₀ × c̄ × CLα × g)

    Parameters
    ----------
    V_eas : float
        Equivalent airspeed (m/s).
    Ude_fps : float
        Derived gust velocity (ft/s).
    wing_loading_pa : float
        Wing loading W/S (Pa = N/m^2).
    CLalpha : float
        Lift curve slope (per radian).
    mean_chord_m : float
        Mean aerodynamic chord (m).
    weight_N : float
        Aircraft weight (N).
    wing_area_m2 : float
        Wing area (m^2).

    Returns
    -------
    float
        Incremental load factor Δn (positive).
    """
    if wing_loading_pa <= 0 or CLalpha <= 0 or mean_chord_m <= 0:
        return 0.0

    Ude_mps = Ude_fps * FPS_TO_MPS

    # Mass ratio (airplane mass parameter)
    mu_g = 2.0 * wing_loading_pa / (RHO_0 * mean_chord_m * CLalpha * G_MPS2)

    # Gust alleviation factor (Pratt)
    Kg = 0.88 * mu_g / (5.3 + mu_g)

    # Incremental load factor
    delta_n = (RHO_0 * V_eas * CLalpha * Kg * Ude_mps) / (2.0 * wing_loading_pa)

    return abs(delta_n)


# ---------------------------------------------------------------------------
# V-n Diagram
# ---------------------------------------------------------------------------

@dataclass
class VnDiagram:
    """Complete V-n diagram for a specific weight/altitude condition.

    Contains the maneuver envelope, gust envelope, and critical corner
    points used for load case generation.

    Attributes
    ----------
    corner_points : list of VnPoint
        Critical corner points of the combined envelope.
    maneuver_curve : list of (V_eas, nz)
        Maneuver envelope boundary.
    gust_curve : list of (V_eas, nz)
        Gust envelope boundary (positive and negative).
    weight_condition : WeightCGCondition
        Associated weight/CG condition.
    altitude_m : float
        Altitude for this diagram.
    nz_max : float
        Maximum positive load factor.
    nz_min : float
        Minimum negative load factor.
    """
    corner_points: List[VnPoint] = field(default_factory=list)
    maneuver_curve: List[Tuple[float, float]] = field(default_factory=list)
    gust_curve_pos: List[Tuple[float, float]] = field(default_factory=list)
    gust_curve_neg: List[Tuple[float, float]] = field(default_factory=list)
    weight_condition: Optional[WeightCGCondition] = None
    altitude_m: float = 0.0
    nz_max: float = 3.8
    nz_min: float = -1.52
    speeds: Optional[SpeedSchedule] = None

    def get_corner_labels(self) -> List[str]:
        """Return sorted list of corner point labels."""
        return sorted([p.label for p in self.corner_points])


def compute_vn_diagram(config: AircraftConfig,
                        weight_cond: WeightCGCondition,
                        altitude_m: float = 0.0,
                        n_curve_points: int = 100) -> VnDiagram:
    """Compute complete V-n diagram for given conditions.

    Parameters
    ----------
    config : AircraftConfig
        Aircraft configuration with speeds, aero data.
    weight_cond : WeightCGCondition
        Weight/CG condition.
    altitude_m : float
        Altitude in meters.
    n_curve_points : int
        Number of points for envelope curves.

    Returns
    -------
    VnDiagram
        Complete V-n diagram with corner points and curves.
    """
    speeds = config.speeds
    W = weight_cond.weight_N
    S = config.wing_area_m2
    WS = W / S if S > 0 else 0.0

    # Part 23 load factors
    nz_max = part23_nz_max(W)
    nz_min = part23_nz_min(nz_max)

    # Compute VA if not specified
    VA = speeds.VA
    if VA <= 0:
        VA = speeds.VS1 * math.sqrt(nz_max)

    VS1 = speeds.VS1
    VC = speeds.VC
    VD = speeds.VD
    VF = speeds.VF

    vn = VnDiagram(
        weight_condition=weight_cond,
        altitude_m=altitude_m,
        nz_max=nz_max,
        nz_min=nz_min,
        speeds=speeds,
    )

    # ---------------------------------------------------------------
    # 1. Maneuver envelope
    # ---------------------------------------------------------------
    corners = []

    # Point A+ : VA, nz_max (stall intersection with max nz)
    corners.append(VnPoint(VA, nz_max, "A+", "maneuver"))

    # Point A- : stall-negative at VA (or compute speed for nz_min)
    # Negative stall speed for nz_min: V_neg = VS1 * sqrt(|nz_min|)
    V_neg_stall = VS1 * math.sqrt(abs(nz_min)) if VS1 > 0 else 0.0
    if V_neg_stall <= VA:
        corners.append(VnPoint(V_neg_stall, nz_min, "A-", "maneuver"))
    else:
        corners.append(VnPoint(VA, nz_min, "A-", "maneuver"))

    # Point C+ : VC, nz_max
    corners.append(VnPoint(VC, nz_max, "C+", "maneuver"))

    # Point C- : VC, nz_min
    corners.append(VnPoint(VC, nz_min, "C-", "maneuver"))

    # Point D+ : VD, nz_max
    corners.append(VnPoint(VD, nz_max, "D+", "maneuver"))

    # Point D- : VD, nz_min (or 0 for some interpretations)
    corners.append(VnPoint(VD, nz_min, "D-", "maneuver"))

    # Build maneuver curve (stall line + flat top + dive descent)
    maneuver_curve = []
    # Positive stall line: n = (V/VS1)^2, from V=0 to VA
    for i in range(n_curve_points // 4):
        V = VS1 * 0.01 + (VA - VS1 * 0.01) * i / max(n_curve_points // 4 - 1, 1)
        n = min((V / VS1) ** 2, nz_max) if VS1 > 0 else nz_max
        maneuver_curve.append((V, n))
    # Flat top: VA to VD at nz_max
    for i in range(n_curve_points // 4):
        V = VA + (VD - VA) * i / max(n_curve_points // 4 - 1, 1)
        maneuver_curve.append((V, nz_max))
    # Close at VD, 0
    maneuver_curve.append((VD, 0.0))
    # Negative flat bottom: VD to negative stall speed
    for i in range(n_curve_points // 4):
        V = VD - (VD - V_neg_stall) * i / max(n_curve_points // 4 - 1, 1)
        maneuver_curve.append((V, nz_min))
    # Negative stall line back to origin
    for i in range(n_curve_points // 4):
        frac = i / max(n_curve_points // 4 - 1, 1)
        V = V_neg_stall * (1.0 - frac)
        n = -(V / VS1) ** 2 if VS1 > 0 else nz_min
        n = max(n, nz_min)
        maneuver_curve.append((V, n))

    vn.maneuver_curve = maneuver_curve

    # ---------------------------------------------------------------
    # 2. Gust envelope (Pratt formula)
    # ---------------------------------------------------------------
    CLalpha = config.CLalpha
    cbar = config.mean_chord_m
    Ude_VC = config.gust_Ude_VC_fps
    Ude_VD = config.gust_Ude_VD_fps

    # Gust Ude varies linearly between VB, VC, VD
    # At VB: Ude from §23.333(c) — linearly extrapolated from VC value
    # Simplified: use VC value at VB too
    Ude_VB = Ude_VC  # Conservative (actual may be slightly higher)

    # VB: §23.333(c) — intersection of gust line with stall curve, or VA
    # VB = max of (VA, speed where gust nz = stall nz)
    # For simplicity: VB = VA (conservative for most Part 23 aircraft)
    VB = VA if speeds.VB <= 0 else speeds.VB

    # Compute gust delta_nz at key speeds
    gust_speeds_ude = [
        (VB, Ude_VB),
        (VC, Ude_VC),
        (VD, Ude_VD),
    ]

    for V, Ude in gust_speeds_ude:
        if V <= 0 or WS <= 0:
            continue
        dn = pratt_gust_delta_nz(V, Ude, WS, CLalpha, cbar, W, S)

        nz_gust_pos = 1.0 + dn
        nz_gust_neg = 1.0 - dn

        if V == VB:
            corners.append(VnPoint(V, nz_gust_pos, "Gust_VB+", "gust"))
            corners.append(VnPoint(V, nz_gust_neg, "Gust_VB-", "gust"))
        elif V == VC:
            corners.append(VnPoint(V, nz_gust_pos, "Gust_VC+", "gust"))
            corners.append(VnPoint(V, nz_gust_neg, "Gust_VC-", "gust"))
        elif V == VD:
            corners.append(VnPoint(V, nz_gust_pos, "Gust_VD+", "gust"))
            corners.append(VnPoint(V, nz_gust_neg, "Gust_VD-", "gust"))

    # Build gust curves
    gust_curve_pos = []
    gust_curve_neg = []
    V_min = VS1 * 0.5 if VS1 > 0 else VD * 0.1
    for i in range(n_curve_points):
        V = V_min + (VD - V_min) * i / max(n_curve_points - 1, 1)
        if V <= 0 or WS <= 0:
            continue
        # Interpolate Ude
        if V <= VC:
            Ude = Ude_VC  # Constant from VB to VC
        else:
            # Linear from VC to VD
            frac = (V - VC) / (VD - VC) if VD > VC else 0.0
            Ude = Ude_VC + (Ude_VD - Ude_VC) * frac

        dn = pratt_gust_delta_nz(V, Ude, WS, CLalpha, cbar, W, S)
        gust_curve_pos.append((V, 1.0 + dn))
        gust_curve_neg.append((V, 1.0 - dn))

    vn.gust_curve_pos = gust_curve_pos
    vn.gust_curve_neg = gust_curve_neg

    # ---------------------------------------------------------------
    # 3. Flap corner (if VF specified)
    # ---------------------------------------------------------------
    if VF > 0:
        corners.append(VnPoint(VF, 2.0, "Flap+", "flap"))
        corners.append(VnPoint(VF, 0.0, "Flap0", "flap"))

    vn.corner_points = corners
    return vn


def compute_all_vn_diagrams(config: AircraftConfig
                             ) -> List[VnDiagram]:
    """Compute V-n diagrams for all weight/altitude combinations.

    Parameters
    ----------
    config : AircraftConfig
        Complete aircraft configuration.

    Returns
    -------
    list of VnDiagram
        One diagram per (weight_cg, altitude) combination.
    """
    diagrams = []
    for wc in config.weight_cg_conditions:
        for alt in config.altitudes_m:
            vn = compute_vn_diagram(config, wc, alt)
            diagrams.append(vn)
    return diagrams
