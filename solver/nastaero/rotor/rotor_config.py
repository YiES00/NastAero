"""Rotor and VTOL configuration definitions.

Defines individual rotor placement, orientation, and operating parameters
for Lift+Cruise VTOL configurations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np

from .blade import BladeDef
from .airfoil import RotorAirfoil


class RotorType(Enum):
    """Rotor function in VTOL configuration."""
    LIFT = "lift"       # Vertical lift rotor (shaft ~ vertical)
    CRUISE = "cruise"   # Forward thrust (shaft ~ horizontal, pusher/tractor)
    TILT = "tilt"       # Tiltrotor (variable shaft angle)


class RotationDir(Enum):
    """Rotor rotation direction (viewed from above)."""
    CW = "CW"    # Clockwise
    CCW = "CCW"  # Counter-clockwise


@dataclass
class RotorDef:
    """Single rotor definition and placement.

    Attributes
    ----------
    rotor_id : int
        Unique identifier for this rotor.
    label : str
        Human-readable name (e.g., "Lift Rotor L1").
    rotor_type : RotorType
        Function of this rotor (lift/cruise/tilt).
    hub_position : np.ndarray
        Hub center position in model coordinates [x, y, z] (mm).
    shaft_axis : np.ndarray
        Unit vector along shaft axis. For lift rotors: [0, 0, 1] (up).
        For pusher: [-1, 0, 0] (aft).
    blade : BladeDef
        Blade geometry and airfoil.
    n_blades : int
        Number of blades.
    rotation_dir : RotationDir
        Rotation direction.
    rpm_hover : float
        Design RPM for hover condition.
    rpm_cruise : float
        Design RPM for cruise (may differ from hover, or 0 if stopped).
    hub_node_id : int
        Structural node ID at hub attachment point.
        Rotor forces are applied at this node.
    mass_kg : float
        Rotor assembly mass (motor + blades + hub) in kg.
    can_fail : bool
        Whether this rotor is included in OEI analysis.
    """
    rotor_id: int = 0
    label: str = ""
    rotor_type: RotorType = RotorType.LIFT
    hub_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shaft_axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    blade: BladeDef = field(default_factory=BladeDef)
    n_blades: int = 4
    rotation_dir: RotationDir = RotationDir.CW
    rpm_hover: float = 3000.0
    rpm_cruise: float = 0.0
    hub_node_id: int = 0
    mass_kg: float = 15.0
    can_fail: bool = True
    tilt_angle_deg: float = 0.0

    @property
    def is_lift_rotor(self) -> bool:
        return self.rotor_type == RotorType.LIFT

    @property
    def is_cruise_rotor(self) -> bool:
        return self.rotor_type == RotorType.CRUISE

    @property
    def is_tilt_rotor(self) -> bool:
        return self.rotor_type == RotorType.TILT

    @property
    def is_hover_capable(self) -> bool:
        """Whether this rotor produces vertical thrust in hover."""
        return self.rotor_type in (RotorType.LIFT, RotorType.TILT)

    @property
    def effective_shaft_axis(self) -> np.ndarray:
        """Shaft axis accounting for tilt angle.

        For TILT rotors, tilt rotates in XZ plane:
          tilt=0deg  -> [0, 0, 1] (vertical, hover)
          tilt=90deg -> [-1, 0, 0] (horizontal, cruise, aft-facing)

        For non-TILT rotors, returns the fixed shaft_axis.
        """
        if self.rotor_type != RotorType.TILT or self.tilt_angle_deg == 0.0:
            return self.shaft_axis / np.linalg.norm(self.shaft_axis)

        alpha = np.radians(self.tilt_angle_deg)
        return np.array([-np.sin(alpha), 0.0, np.cos(alpha)])


@dataclass
class VTOLConfig:
    """Complete VTOL rotor configuration.

    Attributes
    ----------
    rotors : list of RotorDef
        All rotors in the configuration.
    config_type : str
        Configuration description (e.g., "Lift+Cruise").
    v_mca : float
        Minimum controllable airspeed in VTOL mode (m/s EAS).
        Transition corridor boundary.
    v_transition_end : float
        Speed at which transition to wing-borne flight is complete (m/s EAS).
    hover_ceiling_m : float
        Maximum hover altitude (m).
    """
    rotors: List[RotorDef] = field(default_factory=list)
    config_type: str = "Lift+Cruise"
    v_mca: float = 15.0        # ~30 knots
    v_transition_end: float = 35.0  # ~68 knots
    hover_ceiling_m: float = 3000.0

    @property
    def lift_rotors(self) -> List[RotorDef]:
        return [r for r in self.rotors if r.is_lift_rotor]

    @property
    def cruise_rotors(self) -> List[RotorDef]:
        return [r for r in self.rotors if r.is_cruise_rotor]

    @property
    def tilt_rotors(self) -> List[RotorDef]:
        return [r for r in self.rotors if r.is_tilt_rotor]

    @property
    def hover_rotors(self) -> List[RotorDef]:
        """All rotors that produce vertical thrust in hover (LIFT + TILT)."""
        return [r for r in self.rotors if r.is_hover_capable]

    @property
    def n_lift_rotors(self) -> int:
        return len(self.lift_rotors)

    @property
    def n_hover_rotors(self) -> int:
        return len(self.hover_rotors)

    @property
    def total_rotor_mass_kg(self) -> float:
        return sum(r.mass_kg for r in self.rotors)

    def get_rotor(self, rotor_id: int) -> Optional[RotorDef]:
        for r in self.rotors:
            if r.rotor_id == rotor_id:
                return r
        return None

    @classmethod
    def kc100_lift_cruise(cls) -> VTOLConfig:
        """Create KC-100 Lift+Cruise default configuration.

        6 lift rotors on wings (3 per side) + 1 pusher at tail.

        Layout (top view, Y-axis = spanwise):
            Left wing:  L3 (outboard) -- L2 (mid) -- L1 (inboard)
            Right wing: R1 (inboard) -- R2 (mid) -- R3 (outboard)
            Tail: Pusher (P1)

        Wing span positions correspond to KC-100 wing stations.
        """
        # Common lift rotor blade (BEMT uses SI: meters)
        lift_blade = BladeDef(
            radius=0.6,     # 0.6 m
            root_cutout=0.15,
            n_elements=20,
            mean_chord=0.05, # 50 mm = 0.05 m
            twist_root=np.radians(12.0),
            twist_tip=np.radians(3.0),
            airfoil=RotorAirfoil.naca0012(),
        )

        # Pusher blade (slightly smaller)
        pusher_blade = BladeDef(
            radius=0.5,     # 0.5 m
            root_cutout=0.15,
            n_elements=20,
            mean_chord=0.045, # 45 mm = 0.045 m
            twist_root=np.radians(15.0),
            twist_tip=np.radians(5.0),
            airfoil=RotorAirfoil.naca0012(),
        )

        # Wing span stations for rotor placement (Y positions in mm)
        # KC-100 wing: Y = 465mm (root) to 5617mm (tip)
        y_stations = [1500.0, 3000.0, 4500.0]  # Inboard, mid, outboard
        x_wing_le = 4500.0  # Approximate wing LE x-position
        z_wing = 500.0       # Wing Z-position

        rotors = []
        rotor_id = 1

        # Left wing lift rotors (negative Y)
        for i, y in enumerate(y_stations):
            rotors.append(RotorDef(
                rotor_id=rotor_id,
                label=f"Lift Rotor L{i + 1}",
                rotor_type=RotorType.LIFT,
                hub_position=np.array([x_wing_le, -y, z_wing + 200.0]),
                shaft_axis=np.array([0., 0., 1.]),
                blade=lift_blade,
                n_blades=4,
                rotation_dir=RotationDir.CW if i % 2 == 0 else RotationDir.CCW,
                rpm_hover=3000.0,
                rpm_cruise=0.0,  # Stopped in cruise
                hub_node_id=990000 + rotor_id,  # Will be created
                mass_kg=15.0,
            ))
            rotor_id += 1

        # Right wing lift rotors (positive Y)
        for i, y in enumerate(y_stations):
            rotors.append(RotorDef(
                rotor_id=rotor_id,
                label=f"Lift Rotor R{i + 1}",
                rotor_type=RotorType.LIFT,
                hub_position=np.array([x_wing_le, y, z_wing + 200.0]),
                shaft_axis=np.array([0., 0., 1.]),
                blade=lift_blade,
                n_blades=4,
                rotation_dir=RotationDir.CCW if i % 2 == 0 else RotationDir.CW,
                rpm_hover=3000.0,
                rpm_cruise=0.0,
                hub_node_id=990000 + rotor_id,
                mass_kg=15.0,
            ))
            rotor_id += 1

        # Pusher propeller at tail
        rotors.append(RotorDef(
            rotor_id=rotor_id,
            label="Pusher P1",
            rotor_type=RotorType.CRUISE,
            hub_position=np.array([11000.0, 0.0, 1500.0]),  # Tail position
            shaft_axis=np.array([-1., 0., 0.]),  # Aft-facing
            blade=pusher_blade,
            n_blades=3,
            rotation_dir=RotationDir.CW,
            rpm_hover=0.0,    # Off in hover
            rpm_cruise=2500.0,
            hub_node_id=990000 + rotor_id,
            mass_kg=12.0,
            can_fail=True,
        ))

        return cls(
            rotors=rotors,
            config_type="Lift+Cruise",
            v_mca=15.0,
            v_transition_end=35.0,
            hover_ceiling_m=3000.0,
        )

    @classmethod
    def kc100_tilt_rotor_12(cls) -> VTOLConfig:
        """Create KC-100 12-Tilt-Rotor configuration.

        12 tilt rotors (6 forward + 6 aft of CG), no pusher.
        All rotors tilt from vertical (hover) to horizontal (cruise).

        Layout (top view):
            Forward (X=2800):
              FL3 (Y=-4500) -- FL2 (Y=-3000) -- FL1 (Y=-1500)
              FR1 (Y=+1500) -- FR2 (Y=+3000) -- FR3 (Y=+4500)

            Aft (X=5000):
              RL3 (Y=-4500) -- RL2 (Y=-3000) -- RL1 (Y=-1500)
              RR1 (Y=+1500) -- RR2 (Y=+3000) -- RR3 (Y=+4500)

        CG at X=3882mm is bracketed:
          Forward arm = 1082mm, Aft arm = 1118mm

        Design rationale:
        - Pitch trim via fore/aft differential thrust
        - Disk loading reduced 50% vs 6-rotor (931 vs 1863 N/m^2)
        - OEI survival 91.7% vs 83.3%
        - Cruise thrust by tilting all rotors (no separate pusher)
        - CT/sigma ~0.20 (vs 0.40 for 6-rotor) — comfortable margin
        """
        # Tilt rotor blade: same as L+C lift blade for proven performance
        # With 12 rotors, each only needs ~1053 N (half of 6-rotor),
        # so CT/sigma drops from ~0.40 to ~0.20 — well below stall
        tilt_blade = BladeDef(
            radius=0.6,      # 0.6 m (same as L+C lift blade)
            root_cutout=0.15,
            n_elements=20,
            mean_chord=0.05,  # 50 mm = 0.05 m
            twist_root=np.radians(12.0),
            twist_tip=np.radians(3.0),
            airfoil=RotorAirfoil.naca0012(),
        )

        # Span stations (same as L+C for wing structure compatibility)
        y_stations = [1500.0, 3000.0, 4500.0]

        # X positions: bracket the CG (X=3882mm)
        x_forward = 2800.0   # 1082mm forward of CG
        x_aft = 5000.0       # 1118mm aft of CG

        # Z positions: above wing for rotor clearance
        z_forward = 900.0    # Forward booms/pylons
        z_aft = 700.0        # Wing-mounted aft

        rotors = []
        rotor_id = 1

        # ---- Forward rotors (6): 3 left + 3 right ----
        # Left forward (negative Y)
        for i, y in enumerate(y_stations):
            rotors.append(RotorDef(
                rotor_id=rotor_id,
                label=f"Tilt Rotor FL{i + 1}",
                rotor_type=RotorType.TILT,
                hub_position=np.array([x_forward, -y, z_forward]),
                shaft_axis=np.array([0., 0., 1.]),  # Hover default
                blade=tilt_blade,
                n_blades=4,
                rotation_dir=RotationDir.CW if i % 2 == 0 else RotationDir.CCW,
                rpm_hover=3000.0,
                rpm_cruise=2500.0,  # Reduced RPM for cruise
                hub_node_id=990000 + rotor_id,
                mass_kg=14.0,  # R=0.6m rotor + tilt actuator
                tilt_angle_deg=0.0,  # Set per flight phase
            ))
            rotor_id += 1

        # Right forward (positive Y)
        for i, y in enumerate(y_stations):
            rotors.append(RotorDef(
                rotor_id=rotor_id,
                label=f"Tilt Rotor FR{i + 1}",
                rotor_type=RotorType.TILT,
                hub_position=np.array([x_forward, y, z_forward]),
                shaft_axis=np.array([0., 0., 1.]),
                blade=tilt_blade,
                n_blades=4,
                rotation_dir=RotationDir.CCW if i % 2 == 0 else RotationDir.CW,
                rpm_hover=3000.0,
                rpm_cruise=2500.0,
                hub_node_id=990000 + rotor_id,
                mass_kg=14.0,  # R=0.6m rotor + tilt actuator
                tilt_angle_deg=0.0,
            ))
            rotor_id += 1

        # ---- Aft rotors (6): 3 left + 3 right ----
        # Left aft (negative Y)
        for i, y in enumerate(y_stations):
            rotors.append(RotorDef(
                rotor_id=rotor_id,
                label=f"Tilt Rotor RL{i + 1}",
                rotor_type=RotorType.TILT,
                hub_position=np.array([x_aft, -y, z_aft]),
                shaft_axis=np.array([0., 0., 1.]),
                blade=tilt_blade,
                n_blades=4,
                rotation_dir=RotationDir.CCW if i % 2 == 0 else RotationDir.CW,
                rpm_hover=3000.0,
                rpm_cruise=2500.0,
                hub_node_id=990000 + rotor_id,
                mass_kg=14.0,  # R=0.6m rotor + tilt actuator
                tilt_angle_deg=0.0,
            ))
            rotor_id += 1

        # Right aft (positive Y)
        for i, y in enumerate(y_stations):
            rotors.append(RotorDef(
                rotor_id=rotor_id,
                label=f"Tilt Rotor RR{i + 1}",
                rotor_type=RotorType.TILT,
                hub_position=np.array([x_aft, y, z_aft]),
                shaft_axis=np.array([0., 0., 1.]),
                blade=tilt_blade,
                n_blades=4,
                rotation_dir=RotationDir.CW if i % 2 == 0 else RotationDir.CCW,
                rpm_hover=3000.0,
                rpm_cruise=2500.0,
                hub_node_id=990000 + rotor_id,
                mass_kg=14.0,  # R=0.6m rotor + tilt actuator
                tilt_angle_deg=0.0,
            ))
            rotor_id += 1

        return cls(
            rotors=rotors,
            config_type="Tilt-Rotor-12",
            v_mca=15.0,
            v_transition_end=35.0,
            hover_ceiling_m=3000.0,
        )

    @classmethod
    def from_dict(cls, d: dict) -> VTOLConfig:
        """Create from dictionary (e.g., parsed YAML config).

        Parameters
        ----------
        d : dict
            Configuration dictionary with 'rotors' list and optional
            'config_type', 'v_mca', 'v_transition_end' keys.
        """
        cfg = cls()
        cfg.config_type = d.get('config_type', 'Lift+Cruise')
        cfg.v_mca = d.get('v_mca', 15.0)
        cfg.v_transition_end = d.get('v_transition_end', 35.0)
        cfg.hover_ceiling_m = d.get('hover_ceiling_m', 3000.0)

        for rd in d.get('rotors', []):
            blade = BladeDef(
                radius=rd.get('radius', 600.0),
                root_cutout=rd.get('root_cutout', 0.15),
                mean_chord=rd.get('mean_chord', 50.0),
                twist_root=np.radians(rd.get('twist_root_deg', 12.0)),
                twist_tip=np.radians(rd.get('twist_tip_deg', 3.0)),
            )
            pos = rd.get('hub_position', [0, 0, 0])
            axis = rd.get('shaft_axis', [0, 0, 1])
            cfg.rotors.append(RotorDef(
                rotor_id=rd.get('rotor_id', 0),
                label=rd.get('label', ''),
                rotor_type=RotorType(rd.get('rotor_type', 'lift')),
                hub_position=np.array(pos, dtype=float),
                shaft_axis=np.array(axis, dtype=float),
                blade=blade,
                n_blades=rd.get('n_blades', 4),
                rotation_dir=RotationDir(rd.get('rotation_dir', 'CW')),
                rpm_hover=rd.get('rpm_hover', 3000.0),
                rpm_cruise=rd.get('rpm_cruise', 0.0),
                hub_node_id=rd.get('hub_node_id', 0),
                mass_kg=rd.get('mass_kg', 15.0),
                can_fail=rd.get('can_fail', True),
                tilt_angle_deg=rd.get('tilt_angle_deg', 0.0),
            ))

        return cfg
