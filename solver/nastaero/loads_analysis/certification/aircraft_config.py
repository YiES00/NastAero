"""Aircraft configuration for FAA Part 23 certification loads analysis.

Defines aircraft speed schedules, weight/CG conditions, control surface limits,
landing gear configuration, and the CONM2 mass adjustment mechanism for
achieving target weight/CG combinations.

References
----------
- 14 CFR Part 23: Airworthiness Standards — Normal Category Airplanes
- §23.337: Limit maneuvering load factors
- §23.333: Flight envelope speeds
- §23.473: Landing load factors
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..case_generator import isa_atmosphere
from ...rotor.rotor_config import VTOLConfig


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_0 = 1.225          # Sea level air density (kg/m^3)
G_MPS2 = 9.80665       # Gravitational acceleration (m/s^2)
KG_PER_LB = 0.453592   # kg per pound
M_PER_FT = 0.3048      # meters per foot
FPS_TO_MPS = 0.3048     # ft/s → m/s


# ---------------------------------------------------------------------------
# Speed conversions
# ---------------------------------------------------------------------------

def eas_to_tas(V_eas: float, altitude_m: float) -> float:
    """Convert Equivalent Airspeed to True Airspeed.

    Parameters
    ----------
    V_eas : float
        EAS in m/s.
    altitude_m : float
        Altitude in meters.

    Returns
    -------
    float
        TAS in m/s.
    """
    rho, _, _ = isa_atmosphere(altitude_m)
    return V_eas * math.sqrt(RHO_0 / rho)


def tas_to_eas(V_tas: float, altitude_m: float) -> float:
    """Convert True Airspeed to Equivalent Airspeed."""
    rho, _, _ = isa_atmosphere(altitude_m)
    return V_tas * math.sqrt(rho / RHO_0)


def eas_to_mach(V_eas: float, altitude_m: float) -> float:
    """Convert EAS to Mach number."""
    V_tas = eas_to_tas(V_eas, altitude_m)
    _, _, a = isa_atmosphere(altitude_m)
    return V_tas / a


def mach_to_eas(mach: float, altitude_m: float) -> float:
    """Convert Mach number to EAS."""
    _, _, a = isa_atmosphere(altitude_m)
    V_tas = mach * a
    return tas_to_eas(V_tas, altitude_m)


def dynamic_pressure_from_eas(V_eas: float) -> float:
    """Dynamic pressure from EAS: q = 0.5 * rho_0 * V_eas^2."""
    return 0.5 * RHO_0 * V_eas ** 2


# ---------------------------------------------------------------------------
# Altitude-dependent gust velocity — FAR 23.333(c)
# ---------------------------------------------------------------------------

# Altitude thresholds in meters
_ALT_20K_FT_M = 20000 * M_PER_FT   # 6,096 m
_ALT_50K_FT_M = 50000 * M_PER_FT   # 15,240 m


def gust_Ude_at_altitude(altitude_m: float, Ude_SL_fps: float) -> float:
    """Derived gust velocity adjusted for altitude per §23.333(c).

    FAR 23.333(c) prescribes:
      - Sea level to 20,000 ft: Ude constant (50 fps at VC, 25 fps at VD).
      - 20,000 ft to 50,000 ft: Ude linearly reduced to half the SL value
        (25 fps at VC, 12.5 fps at VD).

    Parameters
    ----------
    altitude_m : float
        Geometric altitude in meters.
    Ude_SL_fps : float
        Sea-level derived gust velocity (ft/s).

    Returns
    -------
    float
        Altitude-adjusted Ude in ft/s.
    """
    if altitude_m <= _ALT_20K_FT_M:
        return Ude_SL_fps
    elif altitude_m >= _ALT_50K_FT_M:
        return Ude_SL_fps * 0.5
    else:
        frac = (altitude_m - _ALT_20K_FT_M) / (_ALT_50K_FT_M - _ALT_20K_FT_M)
        return Ude_SL_fps * (1.0 - 0.5 * frac)


# ---------------------------------------------------------------------------
# Weight / CG condition
# ---------------------------------------------------------------------------

@dataclass
class WeightCGCondition:
    """A specific weight and CG configuration.

    Attributes
    ----------
    label : str
        Human-readable name (e.g., "MTOW Fwd CG").
    weight_N : float
        Total aircraft weight in Newtons.
    cg_x : float
        CG x-position in model coordinates.
    conm2_adjustments : dict
        CONM2 element ID → target mass (kg or model units).
        Used by CONM2Adjuster to achieve this weight/CG.
    """
    label: str = ""
    weight_N: float = 0.0
    cg_x: float = 0.0
    conm2_adjustments: Dict[int, float] = field(default_factory=dict)

    @property
    def weight_kg(self) -> float:
        """Weight in kg (assuming g=9.80665 m/s^2)."""
        return self.weight_N / G_MPS2

    @property
    def weight_lb(self) -> float:
        """Weight in pounds."""
        return self.weight_kg / KG_PER_LB


# ---------------------------------------------------------------------------
# Speed schedule
# ---------------------------------------------------------------------------

@dataclass
class SpeedSchedule:
    """Aircraft design speed schedule (all in EAS, m/s).

    These are the critical speeds from the flight envelope per §23.333-335.

    Attributes
    ----------
    VS1 : float
        Stall speed in clean configuration (1g, EAS m/s).
    VA : float
        Design maneuvering speed.
    VB : float
        Design speed for maximum gust intensity (optional).
    VC : float
        Design cruising speed.
    VD : float
        Design dive speed.
    VF : float
        Design flap speed (optional, for flap cases).
    """
    VS1: float = 0.0
    VA: float = 0.0
    VB: float = 0.0
    VC: float = 0.0
    VD: float = 0.0
    VF: float = 0.0

    def compute_VA(self, nz_max: float) -> float:
        """Compute VA = VS1 × √nz_max per §23.335(c)."""
        self.VA = self.VS1 * math.sqrt(nz_max)
        return self.VA

    def validate(self) -> List[str]:
        """Check speed schedule validity per Part 23."""
        issues = []
        if self.VS1 <= 0:
            issues.append("VS1 must be positive")
        if self.VA > 0 and self.VA < self.VS1:
            issues.append("VA must be >= VS1")
        if self.VC > 0 and self.VC < self.VA:
            issues.append("VC must be >= VA")
        if self.VD > 0 and self.VD < self.VC:
            issues.append("VD must be >= VC")
        if self.VD > 0 and self.VC > 0:
            if self.VD < 1.25 * self.VC:
                issues.append("VD should be >= 1.25 × VC per §23.335(b)")
        return issues


# ---------------------------------------------------------------------------
# Control surface limits
# ---------------------------------------------------------------------------

@dataclass
class ControlSurfaceLimits:
    """Maximum control surface deflections and speed schedule.

    Deflections decrease with speed per §23.349/§23.351.

    Attributes
    ----------
    aileron_max_deg : float
        Maximum aileron deflection at VA (degrees).
    rudder_max_deg : float
        Maximum rudder deflection at VA (degrees).
    elevator_max_deg : float
        Maximum elevator deflection (degrees).
    """
    aileron_max_deg: float = 20.0
    rudder_max_deg: float = 25.0
    elevator_max_deg: float = 25.0

    def aileron_at_speed(self, V_eas: float, VA: float, VC: float,
                          VD: float) -> float:
        """Aileron deflection schedule per §23.349.

        Returns deflection in radians.
        - VA: full deflection
        - VC: 2/3 of full
        - VD: 1/3 of full
        - Linear interpolation between.
        """
        delta_max = math.radians(self.aileron_max_deg)
        if V_eas <= VA:
            return delta_max
        elif V_eas <= VC:
            # Linear from 1.0 at VA to 2/3 at VC
            frac = (V_eas - VA) / (VC - VA) if VC > VA else 0.0
            return delta_max * (1.0 - frac * 1 / 3)
        elif V_eas <= VD:
            # Linear from 2/3 at VC to 1/3 at VD
            frac = (V_eas - VC) / (VD - VC) if VD > VC else 0.0
            return delta_max * (2 / 3 - frac * 1 / 3)
        else:
            return delta_max * 1 / 3

    def rudder_at_speed(self, V_eas: float, VA: float, VD: float) -> float:
        """Rudder deflection schedule per §23.351.

        Returns deflection in radians.
        - VA: full deflection
        - VD: linearly reduced (proportional to VA/V)
        """
        delta_max = math.radians(self.rudder_max_deg)
        if V_eas <= VA:
            return delta_max
        else:
            # Reduce proportionally
            return delta_max * (VA / V_eas)


# ---------------------------------------------------------------------------
# Landing gear configuration
# ---------------------------------------------------------------------------

@dataclass
class LandingGearConfig:
    """Landing gear geometry and properties for ground loads.

    Attributes
    ----------
    main_gear_node_ids : list of int
        Structural node IDs at main gear attach points.
    nose_gear_node_ids : list of int
        Structural node IDs at nose gear attach points.
    main_gear_x : float
        X-coordinate of main gear (model units).
    nose_gear_x : float
        X-coordinate of nose gear (model units).
    strut_efficiency : float
        Landing gear energy absorption efficiency (0.5-0.8 typical).
    stroke : float
        Maximum gear stroke (model units).
    sink_rate_fps : float
        Design sink rate at landing (ft/s). §23.473: typically 10 fps.
    """
    main_gear_node_ids: List[int] = field(default_factory=list)
    nose_gear_node_ids: List[int] = field(default_factory=list)
    main_gear_x: float = 0.0
    nose_gear_x: float = 0.0
    strut_efficiency: float = 0.7
    stroke: float = 0.0
    sink_rate_fps: float = 10.0

    def compute_nz_landing(self, weight_N: float, lift_factor: float = 2 / 3) -> float:
        """Compute landing load factor per §23.473.

        Simplified energy method:
            nz = 1 + (sink_rate^2) / (2 * g * η * d)
        where η=strut_efficiency, d=stroke.

        For conservatism, uses the simplified §23.473 formula.

        Parameters
        ----------
        weight_N : float
            Landing weight in Newtons.
        lift_factor : float
            Fraction of weight carried by lift (default 2/3 per §23.473).

        Returns
        -------
        float
            Landing load factor at CG.
        """
        if self.stroke <= 0:
            return 3.0  # Conservative default

        sink_rate_mps = self.sink_rate_fps * FPS_TO_MPS
        g = G_MPS2

        # Energy method: KE = W × nz × η × d
        # 0.5 × m × Vs² = (W/g) × nz_gear × η × d (gear only)
        # nz_cg = nz_gear × (1 - lift_factor) + lift_factor
        nz_gear = sink_rate_mps ** 2 / (2 * g * self.strut_efficiency * self.stroke)
        nz_cg = 1.0 + nz_gear * (1.0 - lift_factor)

        return max(nz_cg, 2.0)  # Minimum per §23.473(d)


# ---------------------------------------------------------------------------
# Part 23 load factor computation
# ---------------------------------------------------------------------------

def part23_nz_max(weight_N: float) -> float:
    """Compute maximum positive load factor per §23.337(a).

    nz_max = min(3.8, 2.1 + 24000 / W_lb)

    Parameters
    ----------
    weight_N : float
        Aircraft weight in Newtons.

    Returns
    -------
    float
        Maximum positive maneuvering load factor.
    """
    W_lb = weight_N / (G_MPS2 * KG_PER_LB)
    if W_lb <= 0:
        return 3.8
    return min(3.8, 2.1 + 24000.0 / W_lb)


def part23_nz_min(nz_max: float) -> float:
    """Compute minimum negative load factor per §23.337(b).

    nz_min = -0.4 × nz_max (for normal category)

    Parameters
    ----------
    nz_max : float
        Maximum positive load factor.

    Returns
    -------
    float
        Minimum negative maneuvering load factor.
    """
    return -0.4 * nz_max


# ---------------------------------------------------------------------------
# Aircraft configuration
# ---------------------------------------------------------------------------

@dataclass
class AircraftConfig:
    """Complete aircraft configuration for Part 23 certification loads.

    Attributes
    ----------
    speeds : SpeedSchedule
        Design speed schedule (EAS, m/s).
    weight_cg_conditions : list of WeightCGCondition
        Weight/CG combinations to analyze.
    altitudes_m : list of float
        Analysis altitudes in meters.
    wing_area_m2 : float
        Reference wing area in m^2.
    wing_loading_pa : float
        Wing loading W/S in Pa (computed from weight and area).
    CLalpha : float
        Lift curve slope per radian (for gust loads).
    mean_chord_m : float
        Wing mean aerodynamic chord in meters.
    ctrl_limits : ControlSurfaceLimits
        Control surface deflection limits.
    landing_gear : LandingGearConfig
        Landing gear properties.
    gust_Ude_VC_fps : float
        Derived gust velocity at VC in ft/s (default 50 per §23.333).
    gust_Ude_VD_fps : float
        Derived gust velocity at VD in ft/s (default 25 per §23.333).
    """
    speeds: SpeedSchedule = field(default_factory=SpeedSchedule)
    weight_cg_conditions: List[WeightCGCondition] = field(default_factory=list)
    altitudes_m: List[float] = field(default_factory=lambda: [0.0])
    wing_area_m2: float = 0.0
    CLalpha: float = 2 * math.pi  # Default: thin airfoil
    mean_chord_m: float = 0.0
    ctrl_limits: ControlSurfaceLimits = field(default_factory=ControlSurfaceLimits)
    landing_gear: LandingGearConfig = field(default_factory=LandingGearConfig)
    gust_Ude_VC_fps: float = 50.0
    gust_Ude_VD_fps: float = 25.0
    vtol_config: Optional[VTOLConfig] = None
    nz_max_override: Optional[float] = None

    def nz_max(self, weight_N: float) -> float:
        """Part 23 max load factor for given weight.

        If nz_max_override is set (e.g., 2.5 for commuter/UAM), uses that
        instead of the FAR §23.337 formula.
        """
        if self.nz_max_override is not None:
            return self.nz_max_override
        return part23_nz_max(weight_N)

    def nz_min(self, weight_N: float) -> float:
        """Part 23 min load factor for given weight."""
        return part23_nz_min(self.nz_max(weight_N))

    def wing_loading(self, weight_N: float) -> float:
        """Wing loading W/S in Pa."""
        if self.wing_area_m2 > 0:
            return weight_N / self.wing_area_m2
        return 0.0

    @classmethod
    def from_dict(cls, d: dict) -> AircraftConfig:
        """Create from dictionary (e.g., parsed YAML).

        Expected keys match attribute names. Nested dicts for
        speeds, weight_cg_conditions, ctrl_limits, landing_gear.
        """
        cfg = cls()

        # Speeds
        if 'speeds' in d:
            s = d['speeds']
            cfg.speeds = SpeedSchedule(
                VS1=s.get('VS1', 0.0),
                VA=s.get('VA', 0.0),
                VB=s.get('VB', 0.0),
                VC=s.get('VC', 0.0),
                VD=s.get('VD', 0.0),
                VF=s.get('VF', 0.0),
            )

        # Weight/CG
        if 'weight_cg' in d:
            for wc in d['weight_cg']:
                cfg.weight_cg_conditions.append(WeightCGCondition(
                    label=wc.get('label', ''),
                    weight_N=wc.get('weight_N', 0.0),
                    cg_x=wc.get('cg_x', 0.0),
                    conm2_adjustments=wc.get('conm2_adjustments', {}),
                ))

        # Altitudes
        if 'altitudes_m' in d:
            cfg.altitudes_m = list(d['altitudes_m'])

        # Scalars
        for attr in ('wing_area_m2', 'CLalpha', 'mean_chord_m',
                      'gust_Ude_VC_fps', 'gust_Ude_VD_fps'):
            if attr in d:
                setattr(cfg, attr, d[attr])

        # Load factor override
        if 'nz_max' in d:
            cfg.nz_max_override = float(d['nz_max'])

        # Control limits
        if 'ctrl_limits' in d:
            cl = d['ctrl_limits']
            cfg.ctrl_limits = ControlSurfaceLimits(
                aileron_max_deg=cl.get('aileron_max_deg', 20.0),
                rudder_max_deg=cl.get('rudder_max_deg', 25.0),
                elevator_max_deg=cl.get('elevator_max_deg', 25.0),
            )

        # Landing gear
        if 'landing_gear' in d:
            lg = d['landing_gear']
            cfg.landing_gear = LandingGearConfig(
                main_gear_node_ids=lg.get('main_gear_node_ids', []),
                nose_gear_node_ids=lg.get('nose_gear_node_ids', []),
                main_gear_x=lg.get('main_gear_x', 0.0),
                nose_gear_x=lg.get('nose_gear_x', 0.0),
                strut_efficiency=lg.get('strut_efficiency', 0.7),
                stroke=lg.get('stroke', 0.0),
                sink_rate_fps=lg.get('sink_rate_fps', 10.0),
            )

        # VTOL configuration
        if 'vtol' in d:
            cfg.vtol_config = VTOLConfig.from_dict(d['vtol'])

        return cfg

    @classmethod
    def from_model_defaults(cls, bdf_model) -> AircraftConfig:
        """Create a default config by inspecting the BDF model.

        Extracts total mass from CONM2 cards and estimates reasonable
        speed/area defaults. Intended as a starting point for
        ``--cert-loads auto`` mode.
        """
        import numpy as np

        cfg = cls()

        # Total weight from CONM2s
        total_mass = 0.0
        xs = []
        for eid, elem in bdf_model.elements.items():
            if hasattr(elem, 'mass') and hasattr(elem, 'nid'):
                total_mass += elem.mass
                nid = elem.nid
                if nid in bdf_model.nodes:
                    xs.append(bdf_model.nodes[nid].xyz[0])

        weight_N = total_mass * 9.80665
        cg_x = float(np.mean(xs)) if xs else 0.0

        cfg.weight_cg_conditions = [
            WeightCGCondition(label="Auto MTOW", weight_N=weight_N,
                              cg_x=cg_x),
        ]

        # Estimate wing area from CAERO panels if available
        if hasattr(bdf_model, 'caeros') and bdf_model.caeros:
            total_area = 0.0
            for cid, caero in bdf_model.caeros.items():
                if hasattr(caero, 'chord_root') and hasattr(caero, 'span'):
                    total_area += caero.chord_root * caero.span
            if total_area > 0:
                cfg.wing_area_m2 = total_area * 1e-6  # mm² → m²

        return cfg


# ---------------------------------------------------------------------------
# CONM2 mass adjuster
# ---------------------------------------------------------------------------

class CONM2Adjuster:
    """Adjust CONM2 element masses to achieve target weight and CG.

    Given a set of "adjustable" CONM2 elements (fuel tanks, payload bays),
    computes the mass distribution that achieves the target total weight
    and CG position.

    The algorithm solves a constrained linear system:
        Σ m_i = W_target / g           (total mass constraint)
        Σ m_i × x_i = CG_x × W_target / g  (CG constraint)
        m_i >= 0                        (non-negative masses)
    """

    @staticmethod
    def identify_adjustable_conm2s(model, tags: Optional[List[int]] = None
                                    ) -> List[Tuple[int, float]]:
        """Identify CONM2 elements that can be adjusted.

        Parameters
        ----------
        model : BDFModel
            The BDF model.
        tags : list of int, optional
            Explicit CONM2 element IDs to use. If None, all CONM2s
            with mass > 0 are considered adjustable.

        Returns
        -------
        list of (eid, x_position)
            Adjustable CONM2 IDs and their x-coordinates.
        """
        adjustable = []
        for eid, conm2 in model.conm2s.items():
            if tags is not None and eid not in tags:
                continue
            # Get x-position of the CONM2
            node_id = conm2.node_id
            if node_id in model.nodes:
                x_pos = model.nodes[node_id].xyz_global[0]
                # Include offset if any
                x_pos += conm2.offset[0] if hasattr(conm2, 'offset') else 0.0
                adjustable.append((eid, x_pos))

        return adjustable

    @staticmethod
    def compute_adjustments(target_weight_N: float,
                             target_cg_x: float,
                             model,
                             adjustable_eids: List[int],
                             g: float = 9810.0,
                             ) -> Dict[int, float]:
        """Compute CONM2 mass adjustments to achieve target W/CG.

        For 2 adjustable CONM2s, solves exactly:
            m1 + m2 = M_target
            m1*x1 + m2*x2 = M_target * CG_x

        For N > 2, uses least-squares with non-negativity.

        Parameters
        ----------
        target_weight_N : float
            Target total weight in Newtons (model units: N or force).
        target_cg_x : float
            Target CG x-position.
        model : BDFModel
            The BDF model (for node positions).
        adjustable_eids : list of int
            CONM2 element IDs to adjust.
        g : float
            Gravitational acceleration in model units.

        Returns
        -------
        dict of {eid: mass}
            Target mass for each adjustable CONM2.
        """
        # Compute fixed (non-adjustable) mass contribution
        fixed_mass = 0.0
        fixed_moment = 0.0

        for eid, conm2 in model.conm2s.items():
            if eid in adjustable_eids:
                continue
            node_id = conm2.node_id
            if node_id in model.nodes:
                x_pos = model.nodes[node_id].xyz_global[0]
                if hasattr(conm2, 'offset'):
                    x_pos += conm2.offset[0]
                fixed_mass += conm2.mass
                fixed_moment += conm2.mass * x_pos

        # Also add element masses (non-CONM2)
        # For simplicity, assume structural mass is fixed
        # A more complete implementation would lump element masses too

        target_mass = target_weight_N / g
        adjustable_mass = target_mass - fixed_mass
        adjustable_moment = target_mass * target_cg_x - fixed_moment

        if adjustable_mass < 0:
            raise ValueError(
                f"Target weight ({target_weight_N:.0f} N) is less than "
                f"fixed structural mass ({fixed_mass * g:.0f} N)")

        # Get positions of adjustable CONM2s
        positions = []
        eids = []
        for eid in adjustable_eids:
            if eid not in model.conm2s:
                continue
            conm2 = model.conm2s[eid]
            node_id = conm2.node_id
            if node_id in model.nodes:
                x_pos = model.nodes[node_id].xyz_global[0]
                if hasattr(conm2, 'offset'):
                    x_pos += conm2.offset[0]
                positions.append(x_pos)
                eids.append(eid)

        n = len(eids)
        if n == 0:
            return {}

        if n == 1:
            # Only one CONM2: assign all adjustable mass (CG not controllable)
            return {eids[0]: adjustable_mass}

        if n == 2:
            # Exact solution for 2 CONM2s
            x1, x2 = positions
            if abs(x2 - x1) < 1e-10:
                # Same position, split equally
                return {eids[0]: adjustable_mass / 2,
                        eids[1]: adjustable_mass / 2}
            m2 = (adjustable_moment - adjustable_mass * x1) / (x2 - x1)
            m1 = adjustable_mass - m2
            # Clamp to non-negative
            m1 = max(m1, 0.0)
            m2 = max(m2, 0.0)
            # Rescale to match total mass
            total = m1 + m2
            if total > 0:
                scale = adjustable_mass / total
                m1 *= scale
                m2 *= scale
            return {eids[0]: m1, eids[1]: m2}

        # N > 2: Least-squares with constraints
        # minimize ||m - m_uniform||^2
        # subject to: sum(m) = M_adj, sum(m*x) = moment_adj, m >= 0
        A = np.zeros((2, n))
        A[0, :] = 1.0  # mass constraint
        A[1, :] = np.array(positions)  # moment constraint
        b = np.array([adjustable_mass, adjustable_moment])

        # Start with uniform distribution
        m_init = np.full(n, adjustable_mass / n)

        # Simple iterative projection
        m = m_init.copy()
        for _ in range(100):
            # Project onto constraints
            residual = b - A @ m
            # Pseudo-inverse correction
            A_pinv = np.linalg.pinv(A)
            m += A_pinv @ residual
            # Clamp non-negative
            m = np.maximum(m, 0.0)

        return {eid: mass for eid, mass in zip(eids, m)}

    @staticmethod
    def apply_adjustments(model, adjustments: Dict[int, float]) -> None:
        """Apply mass adjustments to CONM2 elements in-place.

        Parameters
        ----------
        model : BDFModel
            The BDF model to modify.
        adjustments : dict
            CONM2 EID → new mass value.
        """
        for eid, new_mass in adjustments.items():
            if eid in model.conm2s:
                model.conm2s[eid].mass = new_mass
