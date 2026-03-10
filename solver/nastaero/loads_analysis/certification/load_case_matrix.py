"""Complete certification load case matrix per FAA Part 23.

Generates all structural load cases required for certification of normal
category airplanes under 14 CFR Part 23. Combines V-n diagram corner
points with aircraft configuration to produce:

- Symmetric maneuver cases (§23.321/333/337)
- Gust load cases (§23.341, Pratt quasi-static)
- Rolling maneuver cases (§23.349)
- Yaw maneuver cases (§23.351)
- Checked maneuver cases (§23.331(c))
- Flap load cases (§23.345)
- Landing cases (§23.471-§23.511)
- Ground handling cases (§23.491-§23.497)

Each case is a CertLoadCase (extending TrimCondition) or LandingCondition,
ready for batch trim execution or static analysis.

References
----------
- 14 CFR Part 23: Airworthiness Standards — Normal Category Airplanes
"""
from __future__ import annotations

import math
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..case_generator import TrimCondition, isa_atmosphere
from .aircraft_config import (
    AircraftConfig, WeightCGCondition, SpeedSchedule,
    ControlSurfaceLimits, part23_nz_max, part23_nz_min,
    eas_to_tas, eas_to_mach, dynamic_pressure_from_eas,
)
from .vn_diagram import VnDiagram, VnPoint, compute_vn_diagram
from .landing_loads import (
    LandingCondition, generate_all_landing_conditions,
)


# ---------------------------------------------------------------------------
# CertLoadCase — extends TrimCondition with certification metadata
# ---------------------------------------------------------------------------

@dataclass
class CertLoadCase:
    """A single certification load case.

    Wraps a TrimCondition with additional metadata for tracking
    the regulatory origin and physical condition of each case.

    Attributes
    ----------
    trim_condition : TrimCondition
        The underlying trim analysis condition.
    category : str
        Case category: "symmetric", "gust", "rolling", "yaw",
        "checked", "flap", "landing", "ground".
    far_section : str
        FAR section reference (e.g., "§23.337").
    weight_cg : WeightCGCondition
        Weight/CG condition.
    altitude_m : float
        Flight altitude (meters).
    vn_point : VnPoint or None
        Associated V-n diagram corner point.
    config_label : str
        Configuration description (e.g., "Clean", "Flaps 30°").
    solve_type : str
        Analysis method: "trim" for SOL 144, "static" for SOL 101 or direct.
    """
    trim_condition: Optional[TrimCondition] = None
    category: str = "symmetric"
    far_section: str = ""
    weight_cg: Optional[WeightCGCondition] = None
    altitude_m: float = 0.0
    vn_point: Optional[VnPoint] = None
    config_label: str = "Clean"
    solve_type: str = "trim"
    flight_state: Optional[Dict[str, float]] = None
    rotor_forces: Optional[Dict[int, "np.ndarray"]] = None

    @property
    def case_id(self) -> int:
        """Case ID from underlying TrimCondition."""
        return self.trim_condition.case_id if self.trim_condition else 0

    @property
    def label(self) -> str:
        """Case label from underlying TrimCondition."""
        return self.trim_condition.label if self.trim_condition else ""


# ---------------------------------------------------------------------------
# Load case matrix
# ---------------------------------------------------------------------------

class LoadCaseMatrix:
    """Generator for complete Part 23 certification load case matrix.

    Given an AircraftConfig, generates ALL required load cases for
    certification, covering flight, maneuver, gust, and ground conditions.

    Parameters
    ----------
    config : AircraftConfig
        Complete aircraft configuration.

    Attributes
    ----------
    flight_cases : list of CertLoadCase
        All flight load cases (trim solver required).
    landing_cases : list of LandingCondition
        All landing/ground cases (static analysis).
    """

    def __init__(self, config: AircraftConfig):
        self.config = config
        self.flight_cases: List[CertLoadCase] = []
        self.landing_cases: List[LandingCondition] = []
        self._next_id = 1

    def _next_case_id(self) -> int:
        """Get next unique case ID."""
        cid = self._next_id
        self._next_id += 1
        return cid

    # ---------------------------------------------------------------
    # Main generator
    # ---------------------------------------------------------------

    def generate_all(self) -> None:
        """Generate all certification load cases.

        Populates self.flight_cases and self.landing_cases.
        """
        self.flight_cases = []
        self.landing_cases = []
        self._next_id = 1

        for wc in self.config.weight_cg_conditions:
            for alt_m in self.config.altitudes_m:
                # Compute V-n diagram for this condition
                vn = compute_vn_diagram(self.config, wc, alt_m)

                # Flight envelope cases
                self._add_symmetric_maneuver_cases(vn, wc, alt_m)
                self._add_gust_cases(vn, wc, alt_m)
                self._add_rolling_cases(vn, wc, alt_m)
                self._add_yaw_cases(vn, wc, alt_m)
                self._add_checked_maneuver_cases(vn, wc, alt_m)
                self._add_flap_cases(vn, wc, alt_m)

        # Landing cases (not altitude-dependent for flight conditions)
        self.landing_cases = generate_all_landing_conditions(
            self.config, case_id_start=self._next_id)
        self._next_id += len(self.landing_cases)

    # ---------------------------------------------------------------
    # Symmetric maneuver cases — §23.321/333/337
    # ---------------------------------------------------------------

    def _add_symmetric_maneuver_cases(self, vn: VnDiagram,
                                        wc: WeightCGCondition,
                                        alt_m: float) -> None:
        """Add symmetric maneuver cases from V-n diagram corners.

        Each maneuver corner point (A+, A-, C+, C-, D+, D-) generates
        a trim case at the corresponding speed and load factor.
        """
        for pt in vn.corner_points:
            if pt.category != "maneuver":
                continue

            tc = self._make_symmetric_trim(pt, wc, alt_m)
            self.flight_cases.append(CertLoadCase(
                trim_condition=tc,
                category="symmetric",
                far_section="§23.337",
                weight_cg=wc,
                altitude_m=alt_m,
                vn_point=pt,
            ))

    # ---------------------------------------------------------------
    # Gust cases — §23.341
    # ---------------------------------------------------------------

    def _add_gust_cases(self, vn: VnDiagram,
                          wc: WeightCGCondition,
                          alt_m: float) -> None:
        """Add gust load cases from Pratt formula.

        Each gust V-n corner point generates a symmetric trim case
        at nz = 1 + Δn_gust (or 1 - Δn_gust).
        """
        for pt in vn.corner_points:
            if pt.category != "gust":
                continue

            tc = self._make_symmetric_trim(pt, wc, alt_m)
            self.flight_cases.append(CertLoadCase(
                trim_condition=tc,
                category="gust",
                far_section="§23.341",
                weight_cg=wc,
                altitude_m=alt_m,
                vn_point=pt,
            ))

    # ---------------------------------------------------------------
    # Rolling conditions — §23.349
    # ---------------------------------------------------------------

    def _add_rolling_cases(self, vn: VnDiagram,
                             wc: WeightCGCondition,
                             alt_m: float) -> None:
        """Add rolling maneuver cases per §23.349.

        Conservative quasi-static approach:
        - At VA: full aileron deflection (left/right)
        - At VC: 2/3 aileron
        - At VD: 1/3 aileron
        All solved as symmetric trim with fixed ARON deflection.
        """
        speeds = vn.speeds
        if speeds is None:
            return

        VA = speeds.VA if speeds.VA > 0 else speeds.VS1 * math.sqrt(vn.nz_max)
        VC = speeds.VC
        VD = speeds.VD
        ctrl = self.config.ctrl_limits

        roll_speeds = []
        if VA > 0:
            roll_speeds.append(("VA", VA))
        if VC > 0:
            roll_speeds.append(("VC", VC))
        if VD > 0:
            roll_speeds.append(("VD", VD))

        for speed_label, V_eas in roll_speeds:
            # Get aileron deflection at this speed
            delta_ail = ctrl.aileron_at_speed(V_eas, VA, VC, VD)

            for sign, sign_label in [(1.0, "Right"), (-1.0, "Left")]:
                aron_val = sign * delta_ail

                # §23.349: Rolling at nz=1.0
                mach = eas_to_mach(V_eas, alt_m)
                q = dynamic_pressure_from_eas(V_eas)

                tc = TrimCondition(
                    case_id=self._next_case_id(),
                    mach=mach, q=q, nz=1.0,
                    fixed_vars={
                        "ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                        "ARON": aron_val, "RUD": 0.0,
                    },
                    free_vars=["ANGLEA", "ELEV"],
                    label=(f"Roll {sign_label} {speed_label} "
                           f"ARON={math.degrees(aron_val):.1f}° "
                           f"{wc.label}"),
                    altitude_m=alt_m,
                )

                self.flight_cases.append(CertLoadCase(
                    trim_condition=tc,
                    category="rolling",
                    far_section="§23.349",
                    weight_cg=wc,
                    altitude_m=alt_m,
                    vn_point=VnPoint(V_eas, 1.0, f"Roll_{speed_label}",
                                     "rolling"),
                ))

    # ---------------------------------------------------------------
    # Yaw conditions — §23.351
    # ---------------------------------------------------------------

    def _add_yaw_cases(self, vn: VnDiagram,
                         wc: WeightCGCondition,
                         alt_m: float) -> None:
        """Add yaw maneuver cases per §23.351.

        Conservative quasi-static approach:
        - At VA: full rudder (left/right)
        - At higher speeds: reduced rudder
        Two snapshots per condition:
          1. Pure rudder, zero sideslip
          2. Rudder + sideslip (overswing approximation)
        """
        speeds = vn.speeds
        if speeds is None:
            return

        VA = speeds.VA if speeds.VA > 0 else speeds.VS1 * math.sqrt(vn.nz_max)
        VD = speeds.VD
        ctrl = self.config.ctrl_limits

        yaw_speeds = []
        if VA > 0:
            yaw_speeds.append(("VA", VA))
        if speeds.VC > 0:
            yaw_speeds.append(("VC", speeds.VC))
        if VD > 0:
            yaw_speeds.append(("VD", VD))

        for speed_label, V_eas in yaw_speeds:
            delta_rud = ctrl.rudder_at_speed(V_eas, VA, VD)

            for sign, sign_label in [(1.0, "Right"), (-1.0, "Left")]:
                rud_val = sign * delta_rud

                mach = eas_to_mach(V_eas, alt_m)
                q = dynamic_pressure_from_eas(V_eas)

                # Snapshot 1: Pure rudder, zero sideslip
                tc1 = TrimCondition(
                    case_id=self._next_case_id(),
                    mach=mach, q=q, nz=1.0,
                    fixed_vars={
                        "ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                        "ARON": 0.0, "RUD": rud_val,
                    },
                    free_vars=["ANGLEA", "ELEV"],
                    label=(f"Yaw {sign_label} {speed_label} "
                           f"RUD={math.degrees(rud_val):.1f}° "
                           f"β=0 {wc.label}"),
                    altitude_m=alt_m,
                )

                self.flight_cases.append(CertLoadCase(
                    trim_condition=tc1,
                    category="yaw",
                    far_section="§23.351",
                    weight_cg=wc,
                    altitude_m=alt_m,
                    vn_point=VnPoint(V_eas, 1.0, f"Yaw_{speed_label}",
                                     "yaw"),
                ))

                # Snapshot 2: Rudder + overswing sideslip
                # β_overswing ≈ 1.5 × steady sideslip (simplified)
                # Use SIDES = β_overswing as a fixed variable
                # Note: SIDES normalwash is not fully implemented in SOL 144,
                # so sideslip aero effect is approximate.
                beta_approx = math.radians(5.0) * sign  # ~5° overswing
                tc2 = TrimCondition(
                    case_id=self._next_case_id(),
                    mach=mach, q=q, nz=1.0,
                    fixed_vars={
                        "ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                        "ARON": 0.0, "RUD": rud_val,
                        "SIDES": beta_approx,
                    },
                    free_vars=["ANGLEA", "ELEV"],
                    label=(f"Yaw {sign_label} {speed_label} "
                           f"RUD={math.degrees(rud_val):.1f}° "
                           f"β=overswing {wc.label}"),
                    altitude_m=alt_m,
                )

                self.flight_cases.append(CertLoadCase(
                    trim_condition=tc2,
                    category="yaw",
                    far_section="§23.351",
                    weight_cg=wc,
                    altitude_m=alt_m,
                    vn_point=VnPoint(V_eas, 1.0,
                                     f"Yaw_{speed_label}_overswing",
                                     "yaw"),
                ))

    # ---------------------------------------------------------------
    # Checked maneuver — §23.331(c)
    # ---------------------------------------------------------------

    def _add_checked_maneuver_cases(self, vn: VnDiagram,
                                      wc: WeightCGCondition,
                                      alt_m: float) -> None:
        """Add checked maneuver cases per §23.331(c).

        Pull-up → push-over and push-over → pull-up sequences at VA and VC.
        Each nz point generates an independent symmetric trim case.
        Intermediate nz values (0g transition) included.
        """
        speeds = vn.speeds
        if speeds is None:
            return

        VA = speeds.VA if speeds.VA > 0 else speeds.VS1 * math.sqrt(vn.nz_max)
        VC = speeds.VC

        nz_max = vn.nz_max
        nz_min = vn.nz_min

        # Checked maneuver nz sequence:
        # Pull-up then push-over: nz_max → 0g → nz_min
        # Push-over then pull-up: nz_min → 0g → nz_max
        # Intermediate: 0g transition point
        nz_values = [nz_max, 0.0, nz_min]

        for speed_label, V_eas in [("VA", VA), ("VC", VC)]:
            if V_eas <= 0:
                continue
            mach = eas_to_mach(V_eas, alt_m)
            q = dynamic_pressure_from_eas(V_eas)

            for nz in nz_values:
                tc = TrimCondition(
                    case_id=self._next_case_id(),
                    mach=mach, q=q, nz=nz,
                    fixed_vars={
                        "ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                    },
                    free_vars=["ANGLEA", "ELEV"],
                    label=(f"Checked {speed_label} nz={nz:.2f} "
                           f"{wc.label}"),
                    altitude_m=alt_m,
                )
                self.flight_cases.append(CertLoadCase(
                    trim_condition=tc,
                    category="checked",
                    far_section="§23.331(c)",
                    weight_cg=wc,
                    altitude_m=alt_m,
                    vn_point=VnPoint(V_eas, nz,
                                     f"Checked_{speed_label}_{nz:.1f}",
                                     "checked"),
                ))

    # ---------------------------------------------------------------
    # Flap cases — §23.345
    # ---------------------------------------------------------------

    def _add_flap_cases(self, vn: VnDiagram,
                          wc: WeightCGCondition,
                          alt_m: float) -> None:
        """Add flap load cases per §23.345.

        At VF with flaps deployed, nz = 0 to 2.0.
        """
        speeds = vn.speeds
        if speeds is None or speeds.VF <= 0:
            return

        VF = speeds.VF
        mach = eas_to_mach(VF, alt_m)
        q = dynamic_pressure_from_eas(VF)

        for nz in [2.0, 1.0, 0.0]:
            tc = TrimCondition(
                case_id=self._next_case_id(),
                mach=mach, q=q, nz=nz,
                fixed_vars={
                    "ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                },
                free_vars=["ANGLEA", "ELEV"],
                label=f"Flap VF nz={nz:.1f} {wc.label}",
                altitude_m=alt_m,
            )
            self.flight_cases.append(CertLoadCase(
                trim_condition=tc,
                category="flap",
                far_section="§23.345",
                weight_cg=wc,
                altitude_m=alt_m,
                vn_point=VnPoint(VF, nz, f"Flap_{nz:.0f}", "flap"),
                config_label="Flap deployed",
            ))

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _make_symmetric_trim(self, pt: VnPoint,
                               wc: WeightCGCondition,
                               alt_m: float) -> TrimCondition:
        """Create a symmetric trim condition from a V-n point.

        Parameters
        ----------
        pt : VnPoint
            V-n diagram point with V_eas and nz.
        wc : WeightCGCondition
            Weight/CG condition.
        alt_m : float
            Altitude in meters.

        Returns
        -------
        TrimCondition
        """
        V_eas = pt.V_eas
        mach = eas_to_mach(V_eas, alt_m)
        q = dynamic_pressure_from_eas(V_eas)

        return TrimCondition(
            case_id=self._next_case_id(),
            mach=mach, q=q, nz=pt.nz,
            fixed_vars={
                "ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
            },
            free_vars=["ANGLEA", "ELEV"],
            label=f"{pt.label} V={V_eas:.1f}m/s nz={pt.nz:.2f} {wc.label}",
            altitude_m=alt_m,
        )

    # ---------------------------------------------------------------
    # Summary & I/O
    # ---------------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """Return case count by category.

        Returns
        -------
        dict of {category: count}
        """
        counts: Dict[str, int] = {}
        for c in self.flight_cases:
            cat = c.category
            counts[cat] = counts.get(cat, 0) + 1

        # Landing cases
        landing_count = len(self.landing_cases)
        if landing_count > 0:
            counts["landing"] = landing_count

        return counts

    @property
    def total_cases(self) -> int:
        """Total number of cases (flight + landing)."""
        return len(self.flight_cases) + len(self.landing_cases)

    def all_case_ids(self) -> List[int]:
        """Return all unique case IDs."""
        ids = [c.case_id for c in self.flight_cases]
        ids.extend([c.case_id for c in self.landing_cases])
        return ids

    def cases_by_category(self, category: str) -> List[CertLoadCase]:
        """Return flight cases filtered by category."""
        return [c for c in self.flight_cases if c.category == category]

    def far_sections_covered(self) -> List[str]:
        """Return sorted list of unique FAR sections covered."""
        sections = set()
        for c in self.flight_cases:
            if c.far_section:
                sections.add(c.far_section)
        for c in self.landing_cases:
            if c.far_section:
                sections.add(c.far_section)
        return sorted(sections)

    def to_csv(self, filepath: str) -> None:
        """Export flight cases to CSV.

        Parameters
        ----------
        filepath : str
            Output CSV file path.
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'case_id', 'category', 'far_section',
                'mach', 'q', 'nz', 'altitude_m',
                'V_eas', 'weight_label', 'label', 'solve_type',
            ])
            for c in self.flight_cases:
                tc = c.trim_condition
                V_eas = c.vn_point.V_eas if c.vn_point else 0.0
                wc_label = c.weight_cg.label if c.weight_cg else ""
                writer.writerow([
                    tc.case_id, c.category, c.far_section,
                    f"{tc.mach:.6f}", f"{tc.q:.2f}", f"{tc.nz:.4f}",
                    f"{c.altitude_m:.1f}",
                    f"{V_eas:.2f}", wc_label,
                    tc.label, c.solve_type,
                ])

    def merge_vtol_cases(self, vtol_cases: List[CertLoadCase]) -> None:
        """Merge VTOL load cases into the flight case list.

        Parameters
        ----------
        vtol_cases : list of CertLoadCase
            VTOL-specific load cases (hover, OEI, transition, etc.).
        """
        self.flight_cases.extend(vtol_cases)

    @classmethod
    def from_csv(cls, filepath: str, config: AircraftConfig
                  ) -> LoadCaseMatrix:
        """Import flight cases from CSV.

        Parameters
        ----------
        filepath : str
            Input CSV file path.
        config : AircraftConfig
            Aircraft configuration (for weight_cg lookup).

        Returns
        -------
        LoadCaseMatrix
        """
        matrix = cls(config)

        # Build weight_cg lookup
        wc_lookup = {wc.label: wc for wc in config.weight_cg_conditions}

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wc = wc_lookup.get(row['weight_label'])
                tc = TrimCondition(
                    case_id=int(row['case_id']),
                    mach=float(row['mach']),
                    q=float(row['q']),
                    nz=float(row['nz']),
                    label=row['label'],
                    altitude_m=float(row['altitude_m']),
                )
                matrix.flight_cases.append(CertLoadCase(
                    trim_condition=tc,
                    category=row['category'],
                    far_section=row['far_section'],
                    weight_cg=wc,
                    altitude_m=float(row['altitude_m']),
                    solve_type=row.get('solve_type', 'trim'),
                ))

        return matrix
