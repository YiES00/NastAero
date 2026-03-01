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
            if sc.displacements:
                _write_displacements(f, sc)
            if sc.spc_forces:
                _write_spc_forces(f, sc)
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


def _write_footer(f: TextIO) -> None:
    f.write("\n" + "1" + " " * 50 + "* * * E N D   O F   J O B * * *\n")
