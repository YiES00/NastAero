"""VTOL flight conditions and load factor definitions.

Defines the VTOL-specific flight envelope including hover, transition,
OEI (One Engine Inoperative), and VTOL landing conditions per
EASA SC-VTOL and FAA Part 23 MOC for powered lift.

References
----------
- EASA SC-VTOL-01: Special Condition for VTOL Aircraft
- FAA Part 23 Means of Compliance for Powered Lift
- K-UAM Airworthiness Standards (한국형 도심항공교통)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class VTOLFlightPhase(Enum):
    """VTOL flight phase categories."""
    HOVER = "hover"
    TRANSITION = "transition"
    CRUISE = "cruise"
    VTOL_LANDING = "vtol_landing"
    OEI = "oei"
    ROTOR_JAM = "rotor_jam"


@dataclass
class VTOLCondition:
    """A single VTOL flight condition.

    Attributes
    ----------
    label : str
        Descriptive label.
    phase : VTOLFlightPhase
        Flight phase category.
    V_eas : float
        Equivalent airspeed (m/s). 0 for hover.
    nz : float
        Load factor at CG. Includes rotor thrust component.
    altitude_m : float
        Altitude (m).
    thrust_fraction : float
        Fraction of total weight supported by rotors (1.0 = full hover).
    failed_rotor_id : int or None
        ID of failed rotor for OEI/jam conditions. None if all operative.
    rotor_rpm_factor : float
        RPM multiplier relative to design RPM (1.0 = nominal).
    sink_rate_mps : float
        Vertical sink rate for VTOL landing (m/s, positive = down).
    far_section : str
        Applicable regulation section reference.
    """
    label: str = ""
    phase: VTOLFlightPhase = VTOLFlightPhase.HOVER
    V_eas: float = 0.0
    nz: float = 1.0
    altitude_m: float = 0.0
    thrust_fraction: float = 1.0
    failed_rotor_id: Optional[int] = None
    rotor_rpm_factor: float = 1.0
    sink_rate_mps: float = 0.0
    far_section: str = "SC-VTOL"


def generate_hover_conditions(altitudes_m: List[float],
                               ) -> List[VTOLCondition]:
    """Generate hover load conditions.

    Includes:
    - Hover at 1.0g (steady hover)
    - Hover at 1.15g (maneuvering margin per SC-VTOL)
    - Hover with vertical gust (effective nz = 1.0 ± Δnz_gust)

    Parameters
    ----------
    altitudes_m : list of float
        Analysis altitudes.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    for alt in altitudes_m:
        # Steady hover
        conditions.append(VTOLCondition(
            label=f"Hover 1.0g h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.0,
            altitude_m=alt,
            thrust_fraction=1.0,
            far_section="SC-VTOL.2135",
        ))
        # Maneuvering hover (15% thrust margin)
        conditions.append(VTOLCondition(
            label=f"Hover 1.15g h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.15,
            altitude_m=alt,
            thrust_fraction=1.15,
            far_section="SC-VTOL.2135",
        ))
        # Vertical gust in hover (positive)
        conditions.append(VTOLCondition(
            label=f"Hover gust+ h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.3,
            altitude_m=alt,
            thrust_fraction=1.3,
            far_section="SC-VTOL.2135",
        ))
        # Vertical gust in hover (negative)
        conditions.append(VTOLCondition(
            label=f"Hover gust- h={alt:.0f}m",
            phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=0.7,
            altitude_m=alt,
            thrust_fraction=0.7,
            far_section="SC-VTOL.2135",
        ))
    return conditions


def generate_oei_conditions(n_lift_rotors: int,
                             rotor_ids: List[int],
                             altitudes_m: List[float],
                             ) -> List[VTOLCondition]:
    """Generate One Engine Inoperative (OEI) conditions.

    Each lift rotor is failed in turn. Remaining rotors increase
    thrust to maintain hover (if possible).

    Parameters
    ----------
    n_lift_rotors : int
        Total number of lift rotors.
    rotor_ids : list of int
        IDs of rotors that can fail.
    altitudes_m : list of float
        Analysis altitudes.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    # OEI thrust factor: remaining rotors share full weight
    # rpm_factor > 1.0 to compensate (limited by motor capability)
    rpm_factor = min(1.15, (n_lift_rotors / (n_lift_rotors - 1)) ** 0.5)

    for alt in altitudes_m:
        for rid in rotor_ids:
            # OEI hover
            conditions.append(VTOLCondition(
                label=f"OEI Hover R{rid} fail h={alt:.0f}m",
                phase=VTOLFlightPhase.OEI,
                V_eas=0.0, nz=1.0,
                altitude_m=alt,
                thrust_fraction=1.0,
                failed_rotor_id=rid,
                rotor_rpm_factor=rpm_factor,
                far_section="SC-VTOL.2140",
            ))
    return conditions


def generate_transition_conditions(v_mca: float,
                                    v_transition_end: float,
                                    altitudes_m: List[float],
                                    n_speed_steps: int = 4,
                                    ) -> List[VTOLCondition]:
    """Generate transition corridor conditions.

    In transition, both rotors and wings provide lift. The thrust
    fraction decreases linearly from 1.0 at V=0 to 0.0 at
    v_transition_end.

    Parameters
    ----------
    v_mca : float
        Minimum controllable airspeed in VTOL mode (m/s EAS).
    v_transition_end : float
        Speed at which transition is complete (m/s EAS).
    altitudes_m : list of float
        Analysis altitudes.
    n_speed_steps : int
        Number of speed points in transition corridor.

    Returns
    -------
    list of VTOLCondition
    """
    import numpy as np

    conditions = []
    speeds = np.linspace(v_mca, v_transition_end, n_speed_steps)

    for alt in altitudes_m:
        for V in speeds:
            # Thrust fraction decreases linearly in transition
            if v_transition_end > 0:
                tf = max(0.0, 1.0 - V / v_transition_end)
            else:
                tf = 1.0

            # nz = 1.0 (level flight) and nz_max for maneuvering
            for nz in [1.0, 1.5]:
                conditions.append(VTOLCondition(
                    label=(f"Transition V={V:.1f}m/s nz={nz:.1f} "
                           f"h={alt:.0f}m"),
                    phase=VTOLFlightPhase.TRANSITION,
                    V_eas=V, nz=nz,
                    altitude_m=alt,
                    thrust_fraction=tf * nz,
                    far_section="SC-VTOL.2135",
                ))
    return conditions


def generate_vtol_landing_conditions(altitudes_m: List[float],
                                      ) -> List[VTOLCondition]:
    """Generate VTOL landing conditions.

    Vertical descent landings at specified sink rates per SC-VTOL.

    Parameters
    ----------
    altitudes_m : list of float
        Only ground level (alt=0) is relevant for landing.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    # VTOL landing sink rates and load factors
    # SC-VTOL requires 2.0g for normal VTOL landing
    sink_rates = [
        (1.5, 2.0, "Normal VTOL landing"),
        (3.0, 2.5, "Emergency VTOL landing"),
    ]

    for sink_rate_mps, nz, desc in sink_rates:
        conditions.append(VTOLCondition(
            label=f"{desc} Vs={sink_rate_mps:.1f}m/s",
            phase=VTOLFlightPhase.VTOL_LANDING,
            V_eas=0.0, nz=nz,
            altitude_m=0.0,
            thrust_fraction=0.5,  # Partial thrust during touchdown
            sink_rate_mps=sink_rate_mps,
            far_section="SC-VTOL.2480",
        ))
    return conditions


def generate_rotor_jam_conditions(rotor_ids: List[int],
                                   altitudes_m: List[float],
                                   ) -> List[VTOLCondition]:
    """Generate rotor jam/seizure conditions.

    Sudden stop of a single rotor — produces asymmetric torque and
    yaw moment. Treated as a dynamic case.

    Parameters
    ----------
    rotor_ids : list of int
        Rotor IDs that can jam.
    altitudes_m : list of float
        Analysis altitudes.

    Returns
    -------
    list of VTOLCondition
    """
    conditions = []
    for alt in altitudes_m:
        for rid in rotor_ids:
            conditions.append(VTOLCondition(
                label=f"Rotor Jam R{rid} h={alt:.0f}m",
                phase=VTOLFlightPhase.ROTOR_JAM,
                V_eas=0.0, nz=1.0,
                altitude_m=alt,
                thrust_fraction=1.0,
                failed_rotor_id=rid,
                far_section="SC-VTOL.2150",
            ))
    return conditions
