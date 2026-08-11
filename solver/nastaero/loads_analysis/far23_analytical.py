"""FAR 23 LOADS Analytical Solver.

Implements the DARCorporation FAR 23 LOADS methodology (Hal C. McMaster, 1996)
for analytical calculation of FAR 23 Subpart C flight and ground loads.

This module provides a standalone analytical loads calculation that can be
compared against NastAero's FEM-based SOL 144 aeroelastic trim results.

References:
    - FAR 23 LOADS Manual, Hal C. McMaster, Aero Science Software, 1996
    - 14 CFR Part 23 Subpart C - Structure
    - Pratt & Walker, "A Revised Gust-Load Formula", NACA Report 1206, 1954
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ── Constants ──
G_ACCEL = 9.80665       # m/s^2
G_ACCEL_FT = 32.174     # ft/s^2
RHO_SL_METRIC = 1.225   # kg/m^3
RHO_SL_IMP = 0.002378   # slug/ft^3
KTS_TO_FPS = 1.6878      # knots to ft/s
KTS_TO_MS = 0.51444      # knots to m/s
MS_TO_FPS = 3.28084      # m/s to ft/s
MS_TO_KTS = 1.94384      # m/s to knots
FPS_TO_MS = 0.3048       # ft/s to m/s
LB_TO_N = 4.44822        # lb to N
LB_TO_KG = 0.453592      # lb to kg
FT2_TO_M2 = 0.092903     # ft^2 to m^2
FT_TO_M = 0.3048         # ft to m


@dataclass
class FAR23Config:
    """Aircraft configuration for FAR 23 analytical loads.

    All inputs in metric (SI) units:
    - Weights in kg
    - Speeds in m/s (EAS)
    - Areas in m^2
    - Lengths in m
    """
    # Weight
    W_kg: float               # Max takeoff weight (kg)

    # Wing geometry
    S_m2: float               # Wing area (m^2)
    b_m: float                # Wing span (m)
    MAC_m: float              # Mean aerodynamic chord (m)
    CLalpha: float            # Lift curve slope (per radian)

    # Design speeds (EAS, m/s)
    VS1: float                # Stall speed clean (m/s)
    VC: float                 # Cruise speed (m/s)
    VD: float                 # Dive speed (m/s)
    VF: float = 0.0           # Flap speed (m/s), 0 = no flap analysis

    # Category
    category: str = "normal"  # "normal", "utility", "acrobatic"

    # Gust velocities (fps at sea level per FAR 23.333(c))
    Ude_VC_fps: float = 50.0  # Derived gust velocity at VC
    Ude_VD_fps: float = 25.0  # Derived gust velocity at VD

    # Geometry for moment balance (optional, for tail load calc)
    XCG_m: float = 0.0        # CG x-position (m, from datum)
    XW_m: float = 0.0         # Wing AC x-position (m)
    XT_m: float = 0.0         # Tail AC x-position (m)
    Cmac_wf: float = -0.05    # Pitching moment coeff about wing AC (wing+fuse)
    CD0: float = 0.03         # Zero-lift drag coefficient

    # Landing gear
    strut_efficiency: float = 0.7
    tire_efficiency: float = 0.47
    strut_stroke_m: float = 0.25
    tire_stroke_m: float = 0.05

    # Altitude
    altitude_m: float = 0.0   # Analysis altitude (m)

    @property
    def W_lb(self) -> float:
        return self.W_kg / LB_TO_KG

    @property
    def W_N(self) -> float:
        return self.W_kg * G_ACCEL

    @property
    def S_ft2(self) -> float:
        return self.S_m2 / FT2_TO_M2

    @property
    def WS_psf(self) -> float:
        """Wing loading in lb/ft^2."""
        return self.W_lb / self.S_ft2

    @property
    def WS_pa(self) -> float:
        """Wing loading in Pa (N/m^2)."""
        return self.W_N / self.S_m2

    @property
    def AR(self) -> float:
        """Aspect ratio."""
        return self.b_m ** 2 / self.S_m2


@dataclass
class VnPoint:
    """A point on the V-n diagram."""
    V_eas_ms: float          # EAS in m/s
    nz: float                # Load factor
    label: str               # Point label
    category: str            # "maneuver", "gust", "flap"

    @property
    def V_eas_kts(self) -> float:
        return self.V_eas_ms * MS_TO_KTS


@dataclass
class FAR23Results:
    """Results of FAR 23 analytical loads analysis."""
    # Design speeds
    VA: float                # Maneuvering speed (m/s)
    VB: float                # Design speed for max gust (m/s)

    # Load factors
    nz_max: float            # Max positive limit load factor
    nz_min: float            # Max negative limit load factor

    # V-n diagram points
    vn_points: List[VnPoint] = field(default_factory=list)

    # Gust load factors
    delta_n_gust_VC: float = 0.0
    delta_n_gust_VD: float = 0.0
    delta_n_gust_VB: float = 0.0
    Kg_VC: float = 0.0
    mu_g_VC: float = 0.0

    # Balanced tail loads at critical conditions
    tail_loads: Dict[str, float] = field(default_factory=dict)

    # Landing loads
    nz_landing: float = 0.0
    V_sink_fps: float = 0.0

    # Wing loads
    wing_root_shear_N: Dict[str, float] = field(default_factory=dict)
    wing_root_moment_Nm: Dict[str, float] = field(default_factory=dict)


def compute_far23_loads(cfg: FAR23Config) -> FAR23Results:
    """Run complete FAR 23 analytical loads analysis.

    Implements the DARCorporation FAR 23 LOADS methodology:
    1. Design speeds (§23.335)
    2. Maneuvering load factors (§23.337)
    3. Gust load factors (§23.341, Pratt formula)
    4. V-n diagram (§23.333)
    5. Balanced tail loads (§23.421-23.427)
    6. Landing loads (§23.473-23.485)
    """
    # ── 1. Design Speeds (§23.335) ──
    nz_max = _compute_nz_max(cfg)
    nz_min = _compute_nz_min(nz_max, cfg.category)

    VA = cfg.VS1 * math.sqrt(nz_max)
    if VA > cfg.VC:
        VA = cfg.VC  # VA cannot exceed VC

    # ── 2. Gust Load Factors (§23.341, Pratt formula) ──
    rho_alt = _density_at_altitude(cfg.altitude_m)
    dn_VC, Kg_VC, mu_g_VC = _pratt_gust_delta_n(
        cfg, cfg.VC, cfg.Ude_VC_fps, rho_alt)
    dn_VD, Kg_VD, mu_g_VD = _pratt_gust_delta_n(
        cfg, cfg.VD, cfg.Ude_VD_fps, rho_alt)

    # VB: design speed for max gust intensity
    # VB is the speed where gust nz intersects the stall curve
    # or VA, whichever is less (FAR 23.335(d))
    Ude_VB_fps = cfg.Ude_VC_fps  # Same gust at VB as at VC
    VB = _find_VB(cfg, nz_max, Ude_VB_fps, rho_alt)
    if VB > VA:
        VB = VA
    dn_VB, Kg_VB, mu_g_VB = _pratt_gust_delta_n(
        cfg, VB, Ude_VB_fps, rho_alt)

    # ── 3. V-n Diagram Points (§23.333) ──
    vn_points = _build_vn_points(cfg, nz_max, nz_min, VA, VB,
                                 dn_VC, dn_VD, dn_VB)

    # ── 4. Balanced Tail Loads (§23.421-23.427) ──
    tail_loads = _compute_tail_loads(cfg, nz_max, nz_min, VA)

    # ── 5. Landing Loads (§23.473-23.485) ──
    nz_landing, V_sink = _compute_landing_loads(cfg)

    # ── 6. Wing Root Loads (estimate) ──
    wing_shear, wing_moment = _compute_wing_root_loads(cfg, nz_max, nz_min)

    return FAR23Results(
        VA=VA, VB=VB,
        nz_max=nz_max, nz_min=nz_min,
        vn_points=vn_points,
        delta_n_gust_VC=dn_VC,
        delta_n_gust_VD=dn_VD,
        delta_n_gust_VB=dn_VB,
        Kg_VC=Kg_VC, mu_g_VC=mu_g_VC,
        tail_loads=tail_loads,
        nz_landing=nz_landing,
        V_sink_fps=V_sink,
        wing_root_shear_N=wing_shear,
        wing_root_moment_Nm=wing_moment,
    )


# ═══════════════════════════════════════════
# Internal computation functions
# ═══════════════════════════════════════════

def _compute_nz_max(cfg: FAR23Config) -> float:
    """Max positive limit load factor per FAR 23.337(a).

    Normal/Utility: n = min(3.8, 2.1 + 24000/(W_lb + 10000))
    Acrobatic: n = 6.0
    """
    W_lb = cfg.W_lb
    if cfg.category == "acrobatic":
        return 6.0

    # FAR 23.337(a): n = 2.1 + 24000/(W+10000), min 2.5
    # Note: the DARCorp manual uses N_min = 2.1 + 24000/(W+10000)
    # but standard FAR text is: n = 2.1 + 24000/(W+10000)
    n = 2.1 + 24000.0 / (W_lb + 10000.0)

    if cfg.category == "normal":
        return min(3.8, max(2.5, n))
    elif cfg.category == "utility":
        return min(4.4, max(2.5, n))
    return min(3.8, max(2.5, n))


def _compute_nz_min(nz_max: float, category: str) -> float:
    """Min negative limit load factor per FAR 23.337(b).

    Normal/Utility: n_min = -0.4 * n_max
    Acrobatic: n_min = -0.5 * n_max
    """
    if category == "acrobatic":
        return -0.5 * nz_max
    return -0.4 * nz_max


def _density_at_altitude(alt_m: float) -> float:
    """Air density at altitude (kg/m^3) using ISA model."""
    if alt_m <= 11000:
        T = 288.15 - 0.0065 * alt_m
        return RHO_SL_METRIC * (T / 288.15) ** 4.2559
    else:
        T11 = 216.65
        rho11 = RHO_SL_METRIC * (T11 / 288.15) ** 4.2559
        return rho11 * math.exp(-G_ACCEL * (alt_m - 11000) / (287.058 * T11))


def _pratt_gust_delta_n(
    cfg: FAR23Config, V_eas_ms: float, Ude_fps: float,
    rho_alt: float
) -> Tuple[float, float, float]:
    """Pratt quasi-static gust load factor increment.

    Per FAR 23.341 and NACA Report 1206:
        delta_n = (rho_0 * V_EAS * CLa * Kg * Ude) / (2 * W/S)

    In DARCorp formulation (imperial units):
        N = 1 + Kg * Ude * V * CLa / (498 * W/S)
        where V in ft/s (EAS), W/S in psf

    We use metric: delta_n = rho_0 * V * CLa * Kg * Ude_ms / (2 * W/S_pa)

    Returns: (delta_n, Kg, mu_g)
    """
    V_fps = V_eas_ms * MS_TO_FPS
    Ude_ms = Ude_fps * FPS_TO_MS
    WS_pa = cfg.WS_pa
    MAC_m = cfg.MAC_m
    CLa = cfg.CLalpha

    # Mass ratio (using density at altitude per FAR 23.341)
    mu_g = 2.0 * WS_pa / (rho_alt * MAC_m * CLa * G_ACCEL)

    # Gust alleviation factor
    Kg = 0.88 * mu_g / (5.3 + mu_g)

    # Gust load factor increment
    # delta_n = rho_0 * V_EAS * CLa * Kg * Ude / (2 * W/S)
    # Using EAS means we use rho_0 for dynamic pressure
    delta_n = RHO_SL_METRIC * V_eas_ms * CLa * Kg * Ude_ms / (2.0 * WS_pa)

    return delta_n, Kg, mu_g


def _find_VB(cfg: FAR23Config, nz_max: float, Ude_fps: float,
             rho_alt: float) -> float:
    """Find VB where gust envelope intersects positive stall curve.

    VB is the speed where: 1 + delta_n_gust(VB) = (VB/VS1)^2
    This is solved iteratively.
    """
    VS1 = cfg.VS1
    for i in range(50):
        # Try VB from VA downward
        VB_try = VS1 * (1.0 + i * 0.05) + VS1
        if VB_try > cfg.VC:
            VB_try = cfg.VC
        dn, _, _ = _pratt_gust_delta_n(cfg, VB_try, Ude_fps, rho_alt)
        n_gust = 1.0 + dn
        n_stall = (VB_try / VS1) ** 2
        if n_stall >= n_gust:
            return VB_try
    return cfg.VS1 * math.sqrt(nz_max)  # fallback to VA


def _build_vn_points(
    cfg: FAR23Config, nz_max: float, nz_min: float,
    VA: float, VB: float,
    dn_VC: float, dn_VD: float, dn_VB: float
) -> List[VnPoint]:
    """Build complete V-n diagram corner points."""
    pts = []

    # Maneuver envelope
    VS_neg = cfg.VS1 * math.sqrt(abs(nz_min))
    pts.append(VnPoint(VA, nz_max, "A+", "maneuver"))
    pts.append(VnPoint(VS_neg, nz_min, "G-", "maneuver"))
    pts.append(VnPoint(cfg.VC, nz_max, "C+", "maneuver"))
    pts.append(VnPoint(cfg.VC, nz_min, "C-", "maneuver"))
    pts.append(VnPoint(cfg.VD, nz_max, "D+", "maneuver"))
    pts.append(VnPoint(cfg.VD, 0.0, "D0", "maneuver"))
    pts.append(VnPoint(cfg.VD, nz_min, "D-", "maneuver"))

    # Gust envelope
    pts.append(VnPoint(VB, 1.0 + dn_VB, "Gust_VB+", "gust"))
    pts.append(VnPoint(VB, 1.0 - dn_VB, "Gust_VB-", "gust"))
    pts.append(VnPoint(cfg.VC, 1.0 + dn_VC, "Gust_VC+", "gust"))
    pts.append(VnPoint(cfg.VC, 1.0 - dn_VC, "Gust_VC-", "gust"))
    pts.append(VnPoint(cfg.VD, 1.0 + dn_VD, "Gust_VD+", "gust"))
    pts.append(VnPoint(cfg.VD, 1.0 - dn_VD, "Gust_VD-", "gust"))

    # Flap envelope (if VF specified)
    if cfg.VF > 0:
        pts.append(VnPoint(cfg.VF, 2.0, "Flap+", "flap"))
        pts.append(VnPoint(cfg.VF, 0.0, "Flap0", "flap"))

    return pts


def _compute_tail_loads(
    cfg: FAR23Config, nz_max: float, nz_min: float, VA: float
) -> Dict[str, float]:
    """Compute balancing tail loads per FAR 23.421.

    Tail load: LT = [M_acwf + L*(XCG-XW)] / (XT-XCG)
    where:
        M_acwf = Cmac * q * S * MAC  (wing+fuselage pitching moment)
        L = nz * W                    (total lift)
    """
    if cfg.XT_m == 0 or cfg.XW_m == 0:
        return {}

    tail_arm = cfg.XT_m - cfg.XCG_m
    if abs(tail_arm) < 0.01:
        return {}

    results = {}
    conditions = [
        ("A+ (nz_max, VA)", nz_max, VA),
        ("C+ (nz_max, VC)", nz_max, cfg.VC),
        ("D+ (nz_max, VD)", nz_max, cfg.VD),
        ("C- (nz_min, VC)", nz_min, cfg.VC),
        ("D- (nz_min, VD)", nz_min, cfg.VD),
        ("1g cruise", 1.0, cfg.VC),
    ]

    for label, nz, V in conditions:
        q = 0.5 * RHO_SL_METRIC * V ** 2  # dynamic pressure (EAS)
        L = nz * cfg.W_N                   # total lift (N)
        M_acwf = cfg.Cmac_wf * q * cfg.S_m2 * cfg.MAC_m  # pitching moment (Nm)

        # Moment balance about CG:
        # L acts at wing AC (XW), tail load acts at tail AC (XT)
        # Sum moments about CG = 0:
        # L * (XCG - XW) + M_acwf + LT * (XT - XCG) = 0
        LT = -(M_acwf + L * (cfg.XCG_m - cfg.XW_m)) / tail_arm
        results[label] = LT

    return results


def _compute_landing_loads(cfg: FAR23Config) -> Tuple[float, float]:
    """Compute landing load factor per FAR 23.473.

    DARCorp formula (energy balance):
        V_sink = 4.4 * (W/S)^0.25 fps, clamped [7, 10]
        nz = W*V^2/(2g) + W*(1-L)*(S_strut+S_tire)
             -------------------------------------------
             W * (eta_tire*S_tire + eta_strut*S_strut)

    Simplified to:
        nz = V_sink^2 / (2*g*(eta_strut*stroke_strut + eta_tire*stroke_tire))
             + 1.0  (for weight contribution)
    """
    WS_psf = cfg.WS_psf
    V_sink = 4.4 * WS_psf ** 0.25  # fps
    V_sink = max(7.0, min(10.0, V_sink))

    V_sink_ms = V_sink * FPS_TO_MS

    # Energy balance: nz = V^2/(2*g*d_eff) + 1
    d_eff = (cfg.strut_efficiency * cfg.strut_stroke_m +
             cfg.tire_efficiency * cfg.tire_stroke_m)
    nz = V_sink_ms ** 2 / (2.0 * G_ACCEL * d_eff) + 1.0

    # FAR 23.473(d): minimum landing load factor = 2.0
    nz = max(2.0, nz)

    return nz, V_sink


def _compute_wing_root_loads(
    cfg: FAR23Config, nz_max: float, nz_min: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Estimate wing root shear and bending moment.

    Simplified semi-span integration:
    - Assumes elliptic lift distribution
    - Total lift = nz * W / 2 (per semi-span)
    - Root shear = nz * W / 2 (approx, minus wing weight relief)
    - Root bending = nz * W * b / (2 * pi) * (4/(3*pi))
      (elliptic: centroid at 4b/(3*pi) from root)
    """
    b_semi = cfg.b_m / 2.0
    W_half = cfg.W_N / 2.0

    # Assume wing weight is ~12% of total weight (typical GA)
    W_wing_half = 0.12 * cfg.W_N / 2.0

    shear = {}
    moment = {}

    for label, nz in [("nz_max", nz_max), ("nz_min", nz_min), ("1g", 1.0)]:
        # Aerodynamic lift on semi-span
        L_semi = nz * W_half

        # Wing weight relief (inertia acts opposite to nz)
        inertia_relief = nz * W_wing_half

        # Net root shear (aero - inertia)
        Vz = L_semi - inertia_relief
        shear[label] = Vz

        # Elliptic distribution centroid at 4b/(3*pi) from root
        y_centroid_aero = 4.0 * b_semi / (3.0 * math.pi)
        # Wing inertia roughly uniform, centroid at b/4
        y_centroid_inertia = b_semi / 4.0

        Mx = L_semi * y_centroid_aero - inertia_relief * y_centroid_inertia
        moment[label] = Mx

    return shear, moment


# ═══════════════════════════════════════════
# Comparison with NastAero
# ═══════════════════════════════════════════

def compare_with_nastaero(
    far23: FAR23Results,
    nastaero_vn,
    nastaero_config
) -> Dict:
    """Compare FAR 23 analytical results with NastAero results.

    Args:
        far23: FAR23Results from compute_far23_loads
        nastaero_vn: NastAero VnDiagram object
        nastaero_config: NastAero AircraftConfig object

    Returns:
        Dict with comparison data
    """
    comparison = {
        "design_speeds": {},
        "load_factors": {},
        "vn_points": [],
        "gust_parameters": {},
    }

    # Compare design speeds
    comparison["design_speeds"] = {
        "VA": {"far23": far23.VA, "nastaero": nastaero_config.speeds.VA},
        "VC": {"far23": 0, "nastaero": nastaero_config.speeds.VC},
        "VD": {"far23": 0, "nastaero": nastaero_config.speeds.VD},
    }

    # Compare load factors
    comparison["load_factors"] = {
        "nz_max": {"far23": far23.nz_max, "nastaero": nastaero_vn.nz_max},
        "nz_min": {"far23": far23.nz_min, "nastaero": nastaero_vn.nz_min},
    }

    # Compare gust parameters
    comparison["gust_parameters"] = {
        "delta_n_VC": far23.delta_n_gust_VC,
        "delta_n_VD": far23.delta_n_gust_VD,
        "Kg_VC": far23.Kg_VC,
        "mu_g_VC": far23.mu_g_VC,
    }

    # Compare V-n corner points
    for pt in far23.vn_points:
        entry = {
            "label": pt.label,
            "V_eas_ms": pt.V_eas_ms,
            "nz": pt.nz,
            "category": pt.category,
        }
        # Find matching NastAero point
        for na_pt in nastaero_vn.corner_points:
            if _labels_match(pt.label, na_pt.label):
                entry["nastaero_V"] = na_pt.V_eas
                entry["nastaero_nz"] = na_pt.nz
                break
        comparison["vn_points"].append(entry)

    return comparison


def _labels_match(far23_label: str, nastaero_label: str) -> bool:
    """Check if V-n point labels match between systems."""
    mapping = {
        "A+": "A+", "C+": "C+", "D+": "D+",
        "C-": "C-", "D-": "D-", "G-": "A-",
        "Gust_VC+": "Gust_VC+", "Gust_VC-": "Gust_VC-",
        "Gust_VD+": "Gust_VD+", "Gust_VD-": "Gust_VD-",
        "Gust_VB+": "Gust_VB+", "Gust_VB-": "Gust_VB-",
    }
    return mapping.get(far23_label, far23_label) == nastaero_label
