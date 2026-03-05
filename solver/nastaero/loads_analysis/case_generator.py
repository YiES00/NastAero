"""Trim case generation for loads analysis campaigns.

This module provides tools for generating large sets of trim conditions
for systematic loads analysis. It supports:

- Speed sweeps (Mach number / EAS / TAS)
- Load factor sweeps (V-n diagram corners)
- ISA atmospheric model for dynamic pressure computation
- CSV import/export for case management
- TrimCondition → TRIM card conversion (for solver integration)
- Batch trim case execution via solve_trim_cases()

Certification load case generation is in:
  nastaero.loads_analysis.certification.load_case_matrix
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrimCondition:
    """Single trim analysis condition.

    Attributes
    ----------
    case_id : int
        Unique case identifier.
    mach : float
        Flight Mach number.
    q : float
        Dynamic pressure (Pa or consistent units).
    nz : float
        Load factor (g's). Default 1.0 for 1g level flight.
    fixed_vars : dict
        Variables held fixed: {label: value}.
    free_vars : list
        Variables to be solved by trim: [label, ...].
    label : str
        Human-readable case description.
    altitude_m : float
        Flight altitude in meters.
    velocity : float
        True airspeed (m/s or consistent units).
    """
    case_id: int = 0
    mach: float = 0.0
    q: float = 0.0
    nz: float = 1.0
    fixed_vars: Dict[str, float] = field(default_factory=dict)
    free_vars: List[str] = field(default_factory=list)
    label: str = ""
    altitude_m: float = 0.0
    velocity: float = 0.0


def isa_atmosphere(altitude_m: float) -> Tuple[float, float, float]:
    """International Standard Atmosphere (ISA) model.

    Parameters
    ----------
    altitude_m : float
        Geometric altitude in meters.

    Returns
    -------
    rho : float
        Air density (kg/m^3).
    T : float
        Temperature (K).
    a : float
        Speed of sound (m/s).
    """
    # Constants
    g0 = 9.80665     # m/s^2
    R = 287.05287     # J/(kg·K)
    gamma = 1.4

    # Troposphere (0 - 11000 m)
    T0 = 288.15       # K
    p0 = 101325.0      # Pa
    rho0 = 1.225       # kg/m^3
    lapse = -0.0065    # K/m

    if altitude_m <= 11000.0:
        T = T0 + lapse * altitude_m
        p = p0 * (T / T0) ** (-g0 / (lapse * R))
    else:
        # Tropopause (11000 - 20000 m) — isothermal
        T11 = T0 + lapse * 11000.0
        p11 = p0 * (T11 / T0) ** (-g0 / (lapse * R))
        T = T11
        p = p11 * math.exp(-g0 * (altitude_m - 11000.0) / (R * T11))

    rho = p / (R * T)
    a = math.sqrt(gamma * R * T)
    return rho, T, a


class CaseGenerator:
    """Generate sets of TrimConditions for loads analysis.

    Examples
    --------
    >>> cases = CaseGenerator.level_flight_sweep(
    ...     machs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ...     altitude_m=0.0)
    >>> len(cases)
    7
    >>> cases[0].mach
    0.1
    """

    @staticmethod
    def level_flight_sweep(machs: List[float],
                            altitude_m: float = 0.0,
                            nz: float = 1.0,
                            free_vars: Optional[List[str]] = None,
                            fixed_defaults: Optional[Dict[str, float]] = None,
                            ) -> List[TrimCondition]:
        """Generate 1g level flight speed sweep cases.

        Parameters
        ----------
        machs : list of float
            Mach numbers to analyze.
        altitude_m : float
            Flight altitude in meters (default: sea level).
        nz : float
            Load factor (default: 1.0).
        free_vars : list of str, optional
            Trim variables to solve (default: ["ANGLEA", "ELEV"]).
        fixed_defaults : dict, optional
            Default fixed variable values (default: all zero).

        Returns
        -------
        list of TrimCondition
        """
        if free_vars is None:
            free_vars = ["ANGLEA", "ELEV"]
        if fixed_defaults is None:
            fixed_defaults = {
                "ROLL": 0.0, "YAW": 0.0,
                "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
                "ARON": 0.0, "RUD": 0.0,
            }

        rho, T, a = isa_atmosphere(altitude_m)
        cases = []

        for i, mach in enumerate(machs):
            V = mach * a  # True airspeed
            q = 0.5 * rho * V ** 2  # Dynamic pressure

            tc = TrimCondition(
                case_id=i + 1,
                mach=mach,
                q=q,
                nz=nz,
                fixed_vars=dict(fixed_defaults),
                free_vars=list(free_vars),
                label=f"M={mach:.2f} H={altitude_m:.0f}m nz={nz:.1f}g",
                altitude_m=altitude_m,
                velocity=V,
            )
            cases.append(tc)

        return cases

    @staticmethod
    def vn_diagram_cases(machs: List[float],
                          nz_values: List[float],
                          altitude_m: float = 0.0,
                          free_vars: Optional[List[str]] = None,
                          ) -> List[TrimCondition]:
        """Generate V-n diagram corner cases.

        Creates cases for each combination of Mach and load factor.

        Parameters
        ----------
        machs : list of float
        nz_values : list of float
            Load factors (e.g., [-1.0, 1.0, 2.5, -1.0]).
        altitude_m : float
        free_vars : list of str, optional

        Returns
        -------
        list of TrimCondition
        """
        if free_vars is None:
            free_vars = ["ANGLEA", "ELEV"]

        fixed_defaults = {
            "ROLL": 0.0, "YAW": 0.0,
            "URDD2": 0.0, "URDD4": 0.0, "URDD6": 0.0,
            "ARON": 0.0, "RUD": 0.0,
        }

        rho, T, a = isa_atmosphere(altitude_m)
        cases = []
        case_id = 1

        for mach in machs:
            V = mach * a
            q = 0.5 * rho * V ** 2

            for nz in nz_values:
                tc = TrimCondition(
                    case_id=case_id,
                    mach=mach,
                    q=q,
                    nz=nz,
                    fixed_vars=dict(fixed_defaults),
                    free_vars=list(free_vars),
                    label=f"V-n M={mach:.2f} nz={nz:+.1f}g",
                    altitude_m=altitude_m,
                    velocity=V,
                )
                cases.append(tc)
                case_id += 1

        return cases

    @staticmethod
    def from_csv(filepath: str) -> List[TrimCondition]:
        """Load trim conditions from CSV file.

        CSV format:
            case_id, mach, altitude_m, nz, free_vars, label
            1, 0.3, 0.0, 1.0, "ANGLEA;ELEV", "Level flight M=0.3"

        Parameters
        ----------
        filepath : str
            Path to CSV file.

        Returns
        -------
        list of TrimCondition
        """
        import csv
        cases = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = int(row.get('case_id', 0))
                mach = float(row.get('mach', 0.0))
                alt = float(row.get('altitude_m', 0.0))
                nz = float(row.get('nz', 1.0))
                free_str = row.get('free_vars', 'ANGLEA;ELEV')
                free_vars = [v.strip() for v in free_str.split(';') if v.strip()]
                label = row.get('label', f'Case {case_id}')

                rho, T, a = isa_atmosphere(alt)
                V = mach * a
                q = 0.5 * rho * V ** 2

                tc = TrimCondition(
                    case_id=case_id, mach=mach, q=q, nz=nz,
                    free_vars=free_vars, label=label,
                    altitude_m=alt, velocity=V,
                )
                cases.append(tc)

        return cases

    @staticmethod
    def to_csv(cases: List[TrimCondition], filepath: str) -> None:
        """Export trim conditions to CSV file.

        Parameters
        ----------
        cases : list of TrimCondition
        filepath : str
        """
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['case_id', 'mach', 'altitude_m', 'nz',
                            'q', 'velocity', 'free_vars', 'label'])
            for tc in cases:
                writer.writerow([
                    tc.case_id, tc.mach, tc.altitude_m, tc.nz,
                    tc.q, tc.velocity,
                    ';'.join(tc.free_vars), tc.label
                ])


# ---------------------------------------------------------------------------
# TrimCondition → TRIM card bridge
# ---------------------------------------------------------------------------

def trim_condition_to_trim_card(tc: TrimCondition, tid: int = None):
    """Convert a TrimCondition to a TRIM card dataclass.

    Creates a TRIM card with:
    - All fixed_vars as label-value pairs
    - URDD3 = tc.nz (load factor) if not already in fixed_vars
    - Free vars are NOT listed (solver determines them from AESTAT labels)

    Parameters
    ----------
    tc : TrimCondition
        The trim condition to convert.
    tid : int, optional
        Override TRIM card ID. Default: tc.case_id.

    Returns
    -------
    TRIM card dataclass (from nastaero.bdf.cards.aero)
    """
    from ..bdf.cards.aero import TRIM

    trim = TRIM()
    trim.tid = tid if tid is not None else tc.case_id
    trim.mach = tc.mach
    trim.q = tc.q
    trim.aeqr = 1.0

    # Build variables list: all fixed vars + URDD3 for load factor
    variables = []
    for label, value in tc.fixed_vars.items():
        variables.append((label.upper(), value))

    # Add URDD3 = nz if not already specified
    fixed_labels_upper = {k.upper() for k in tc.fixed_vars}
    if "URDD3" not in fixed_labels_upper:
        variables.append(("URDD3", tc.nz))

    trim.variables = variables
    return trim


def trim_conditions_to_model(base_model, cases: List[TrimCondition]):
    """Inject TrimConditions into a BDF model as TRIM cards and subcases.

    Modifies the model in-place by adding TRIM cards and subcases.
    Existing subcases and TRIM cards are preserved; new ones are appended
    with IDs starting after the current maximum.

    Parameters
    ----------
    base_model : BDFModel
        The base BDF model (with structural/aero data).
    cases : list of TrimCondition
        Trim conditions to inject.

    Returns
    -------
    list of tuple (TRIM, subcase_id)
        The created TRIM cards and their subcase IDs.
    """
    from ..bdf.model import Subcase

    # Determine starting IDs
    existing_trim_ids = set(base_model.trims.keys()) if base_model.trims else set()
    existing_sc_ids = {sc.id for sc in base_model.subcases} if base_model.subcases else set()

    max_trim_id = max(existing_trim_ids) if existing_trim_ids else 0
    max_sc_id = max(existing_sc_ids) if existing_sc_ids else 0

    created = []
    for i, tc in enumerate(cases):
        trim_id = max_trim_id + i + 1
        sc_id = max_sc_id + i + 1

        # Create TRIM card
        trim_card = trim_condition_to_trim_card(tc, tid=trim_id)
        base_model.trims[trim_id] = trim_card

        # Create subcase
        subcase = Subcase()
        subcase.id = sc_id
        subcase.trim_id = trim_id
        subcase.label = tc.label
        if not base_model.subcases:
            base_model.subcases = []
        base_model.subcases.append(subcase)

        created.append((trim_card, sc_id))

    return created


def solve_trim_cases(base_model, cases: List[TrimCondition],
                     n_workers: int = 0, blas_threads: int = 1):
    """Solve multiple TrimConditions using the SOL 144 solver.

    This is the high-level API for running certification trim cases.
    It converts TrimConditions to TRIM cards, injects them into the model,
    and runs the solver.

    Parameters
    ----------
    base_model : BDFModel
        The base BDF model. Will be modified in-place (TRIM cards/subcases added).
    cases : list of TrimCondition
        Trim conditions to solve.
    n_workers : int
        Parallel workers: 0=sequential, -1=auto, >0=explicit.
    blas_threads : int
        BLAS threads per worker.

    Returns
    -------
    ResultData
        Solver results with one SubcaseResult per TrimCondition.
    """
    from ..solvers.sol144 import solve_trim

    # Inject cases into model
    trim_conditions_to_model(base_model, cases)

    # Solve
    return solve_trim(base_model, n_workers=n_workers, blas_threads=blas_threads)
