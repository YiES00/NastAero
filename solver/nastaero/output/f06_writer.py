"""F06-format output writer."""
from __future__ import annotations
import datetime
from pathlib import Path
from typing import TextIO
import numpy as np
from .result_data import ResultData, SubcaseResult
from ..bdf.model import BDFModel


PAGE_WIDTH = 132

def write_f06(results: ResultData, bdf_model: BDFModel, filepath: str) -> None:
    with open(filepath, "w") as f:
        _write_header(f, results.title)
        for sc in results.subcases:
            if sc.eigenvalues is not None:
                _write_eigenvalue_summary(f, sc)
                _write_mode_shapes(f, sc)
            if sc.trim_variables:
                _write_trim_results(f, sc)
            if sc.displacements:
                _write_displacements(f, sc)
            if sc.spc_forces:
                _write_spc_forces(f, sc)
            if sc.aero_pressures is not None:
                _write_aero_pressures(f, sc)
            if sc.nodal_aero_forces is not None:
                _write_nodal_forces(f, sc, 'AERODYNAMIC', sc.nodal_aero_forces)
            if sc.nodal_inertial_forces is not None:
                _write_nodal_forces(f, sc, 'INERTIAL', sc.nodal_inertial_forces)
            if sc.nodal_combined_forces is not None:
                _write_nodal_forces(f, sc, 'COMBINED (AERO+INERTIAL)',
                                    sc.nodal_combined_forces)
            if sc.trim_balance is not None:
                _write_trim_balance(f, sc)
        _write_footer(f)


def _write_header(f: TextIO, title: str) -> None:
    ts = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    f.write("1" + " " * 50 + "N A S T A E R O" + " " * 50 + "\n")
    f.write("0" + " " * 50 + title + "\n")
    f.write(" " * 51 + ts + "\n")
    f.write("\n")


def _write_displacements(f: TextIO, sc: SubcaseResult) -> None:
    f.write("1" + " " * 20 + f"SUBCASE {sc.subcase_id}\n\n")
    f.write("                               D I S P L A C E M E N T   V E C T O R\n\n")
    f.write("      POINT ID.   TYPE          T1             T2             T3"
            "             R1             R2             R3\n")
    for nid in sorted(sc.displacements.keys()):
        d = sc.displacements[nid]
        f.write(f"  {nid:>12d}      G  ")
        for i in range(6):
            f.write(f"  {d[i]:>13.6E}")
        f.write("\n")
    f.write("\n")


def _write_spc_forces(f: TextIO, sc: SubcaseResult) -> None:
    has_spc = any(np.any(np.abs(v) > 1e-20) for v in sc.spc_forces.values())
    if not has_spc:
        return
    f.write("                      F O R C E S   O F   S I N G L E - P O I N T   C O N S T R A I N T\n\n")
    f.write("      POINT ID.   TYPE          T1             T2             T3"
            "             R1             R2             R3\n")
    for nid in sorted(sc.spc_forces.keys()):
        v = sc.spc_forces[nid]
        if np.max(np.abs(v)) < 1e-20:
            continue
        f.write(f"  {nid:>12d}      G  ")
        for i in range(6):
            f.write(f"  {v[i]:>13.6E}")
        f.write("\n")
    f.write("\n")


def _write_eigenvalue_summary(f: TextIO, sc: SubcaseResult) -> None:
    f.write("1" + " " * 20 + f"SUBCASE {sc.subcase_id}\n\n")
    f.write("                                        R E A L   E I G E N V A L U E S\n\n")
    f.write("   MODE    EXTRACTION      EIGENVALUE            RADIANS             CYCLES"
            "            GENERALIZED         GENERALIZED\n")
    f.write("    NO.       ORDER                                                  "
            "              MASS            STIFFNESS\n")
    n_modes = len(sc.eigenvalues)
    for j in range(n_modes):
        omega2 = sc.eigenvalues[j]
        omega = np.sqrt(abs(omega2))
        freq = sc.frequencies[j]
        f.write(f"  {j+1:>5d}     {j+1:>5d}      {omega2:>16.6E}    {omega:>16.6E}    {freq:>16.6E}")
        f.write(f"    {'':>16s}    {'':>16s}\n")
    f.write("\n")


def _write_mode_shapes(f: TextIO, sc: SubcaseResult) -> None:
    for j, mode_disp in enumerate(sc.mode_shapes):
        f.write(f"\n      EIGENVALUE = {sc.eigenvalues[j]:>16.6E}\n")
        f.write(f"      CYCLES     = {sc.frequencies[j]:>16.6E}\n\n")
        f.write(f"                          R E A L   E I G E N V E C T O R   N O .  {j+1:>5d}\n\n")
        f.write("      POINT ID.   TYPE          T1             T2             T3"
                "             R1             R2             R3\n")
        for nid in sorted(mode_disp.keys()):
            d = mode_disp[nid]
            f.write(f"  {nid:>12d}      G  ")
            for i in range(6):
                f.write(f"  {d[i]:>13.6E}")
            f.write("\n")
    f.write("\n")


def _write_trim_results(f: TextIO, sc: SubcaseResult) -> None:
    f.write("1" + " " * 20 + f"SUBCASE {sc.subcase_id}\n\n")
    f.write("                           S T A T I C   A E R O E L A S T I C   T R I M   V A R I A B L E S\n\n")
    f.write("      VARIABLE         VALUE            UNITS\n")
    f.write("      " + "-" * 50 + "\n")
    for label, val in sc.trim_variables.items():
        if label == "ANGLEA":
            f.write(f"      {label:<16s}  {val:>13.6E}    rad  ({np.degrees(val):.4f} deg)\n")
        elif label == "SIDES":
            f.write(f"      {label:<16s}  {val:>13.6E}    rad\n")
        else:
            f.write(f"      {label:<16s}  {val:>13.6E}\n")
    f.write("\n")

    if sc.aero_forces is not None:
        total_fz = np.sum(sc.aero_forces[:, 2])
        f.write(f"      TOTAL AERO Fz = {total_fz:>13.6E} N\n\n")


def _write_aero_pressures(f: TextIO, sc: SubcaseResult) -> None:
    f.write("                    A E R O D Y N A M I C   P R E S S U R E   C O E F F I C I E N T S\n\n")
    f.write("      BOX ID       DELTA CP          FX              FY              FZ\n")
    n = len(sc.aero_pressures)
    for i in range(n):
        box_id = sc.aero_boxes[i].box_id if sc.aero_boxes else i
        cp = sc.aero_pressures[i]
        fx, fy, fz = sc.aero_forces[i] if sc.aero_forces is not None else (0, 0, 0)
        f.write(f"  {box_id:>10d}  {cp:>13.6E}  {fx:>13.6E}  {fy:>13.6E}  {fz:>13.6E}\n")
    f.write("\n")


def _write_nodal_forces(f: TextIO, sc: SubcaseResult, label: str,
                         forces: dict) -> None:
    """Write nodal force table for a specific load type."""
    f.write("1" + " " * 20 + f"SUBCASE {sc.subcase_id}\n\n")
    f.write(f"              {label}   N O D A L   F O R C E S\n\n")
    f.write("      POINT ID.   TYPE          FX             FY             FZ"
            "             MX             MY             MZ\n")

    total = np.zeros(6)
    n_written = 0
    for nid in sorted(forces.keys()):
        fv = forces[nid]
        mag = np.linalg.norm(fv[:3])
        if mag < 1e-20:
            continue
        f.write(f"  {nid:>12d}      G  ")
        for i in range(6):
            f.write(f"  {fv[i]:>13.6E}")
        f.write("\n")
        total += fv
        n_written += 1

    f.write("\n")
    f.write(f"      TOTAL ({n_written} nodes):  ")
    for i in range(6):
        f.write(f"  {total[i]:>13.6E}")
    f.write("\n\n")


def _write_trim_balance(f: TextIO, sc: SubcaseResult) -> None:
    """Write trim equilibrium balance check."""
    f.write("1" + " " * 20 + f"SUBCASE {sc.subcase_id}\n\n")
    f.write("              T R I M   E Q U I L I B R I U M   B A L A N C E   C H E C K\n\n")
    f.write("      COMPONENT       RESULTANT            STATUS\n")
    f.write("      " + "-" * 60 + "\n")

    b = sc.trim_balance
    for comp in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']:
        val = b[comp]
        # Determine trim status
        if abs(val) < 1.0:
            status = "OK (< 1.0)"
        elif abs(val) < 100.0:
            status = "MARGINAL"
        else:
            status = "CHECK"
        f.write(f"      {comp:<16s}  {val:>16.6E}    {status}\n")
    f.write("\n")


def _write_footer(f: TextIO) -> None:
    f.write("\n" + "1" + " " * 50 + "* * * E N D   O F   J O B * * *\n")
