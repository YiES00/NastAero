#!/usr/bin/env python3
"""Diagnostic: run static vs dynamic case to compare force magnitudes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from ascent_load.bdf.parser import parse_bdf
from ascent_load.solvers.sol144 import _build_shared_data, _solve_trim_subcase_from_shared
from ascent_load.loads_analysis.case_generator import TrimCondition, trim_condition_to_trim_card

BDF_PATH = "tests/validation/GACOMP/p400r3-free-trim.bdf"
print("Parsing BDF...")
bdf = parse_bdf(BDF_PATH)

print("Building shared data...")
shared = _build_shared_data(bdf)
print(f"  total_weight = {shared.total_weight:.2f} N")
print(f"  g = {shared.g:.2f} mm/s²")
total_mass_kg = shared.total_weight / shared.g * 1000  # Mg → kg
print(f"  total_mass = {total_mass_kg:.1f} kg")

# q_scale for mm model
q_scale = 1e-6  # Pa → N/mm²

def run_case(label, tc, subcase_id=1):
    """Run a single trim case and report diagnostics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Scale q for mm model
    tc.q *= q_scale
    trim_card = trim_condition_to_trim_card(tc, tid=subcase_id)

    # Add to model temporarily
    bdf.trims[subcase_id] = trim_card

    sc = _solve_trim_subcase_from_shared(shared, trim_card, subcase_id)

    # Clean up
    del bdf.trims[subcase_id]

    if sc.nodal_combined_forces is None:
        print("  FAILED - no nodal forces!")
        return

    # Total forces
    total_aero = np.zeros(3)
    total_inertial = np.zeros(3)
    total_combined = np.zeros(3)
    for nid in sc.nodal_combined_forces:
        total_combined += sc.nodal_combined_forces[nid][:3]
        if sc.nodal_aero_forces:
            total_aero += sc.nodal_aero_forces.get(nid, np.zeros(6))[:3]
        if sc.nodal_inertial_forces:
            total_inertial += sc.nodal_inertial_forces.get(nid, np.zeros(6))[:3]

    print(f"  Total aero Fz:     {total_aero[2]:>12.1f} N  (expect: nz*W = {tc.nz * shared.total_weight / q_scale * q_scale:.1f})")
    print(f"  Total inertial Fz: {total_inertial[2]:>12.1f} N")
    print(f"  Total combined Fz: {total_combined[2]:>12.1f} N  (should be ~0)")
    print(f"  Total aero Fy:     {total_aero[1]:>12.1f} N")
    print(f"  Total inertial Fy: {total_inertial[1]:>12.1f} N")

    # Panel-level totals
    if sc.aero_forces is not None:
        panel_fz = float(np.sum(sc.aero_forces[:, 2]))
        panel_fy = float(np.sum(sc.aero_forces[:, 1]))
        print(f"  Panel total Fz:    {panel_fz:>12.1f} N")
        print(f"  Panel total Fy:    {panel_fy:>12.1f} N")
        print(f"  Nodal/Panel ratio: {total_aero[2]/panel_fz:.6f}" if abs(panel_fz) > 1 else "")

    # Wing root shear (right wing, y > 700mm)
    wing_aero_z = 0.0
    wing_inertial_z = 0.0
    n_wing = 0
    for nid in sc.nodal_combined_forces:
        node = bdf.nodes.get(nid)
        if node is not None and node.xyz_global[1] > 700:
            if sc.nodal_aero_forces:
                wing_aero_z += sc.nodal_aero_forces.get(nid, np.zeros(6))[2]
            if sc.nodal_inertial_forces:
                wing_inertial_z += sc.nodal_inertial_forces.get(nid, np.zeros(6))[2]
            n_wing += 1

    wing_combined_z = wing_aero_z + wing_inertial_z
    print(f"\n  Right wing (y>700mm): {n_wing} nodes")
    print(f"  Wing aero Fz:     {wing_aero_z:>12.1f} N")
    print(f"  Wing inertial Fz: {wing_inertial_z:>12.1f} N")
    print(f"  Wing root shear:  {wing_combined_z:>12.1f} N")

    # Trim variables
    print(f"\n  Trim variables:")
    for k, v in sc.trim_variables.items():
        if k == "ANGLEA":
            print(f"    {k} = {v:.6f} rad ({np.degrees(v):.2f} deg)")
        else:
            print(f"    {k} = {v:.6f}")

# Case 1: Static 1g (free alpha + elevator)
tc1 = TrimCondition(
    case_id=1, mach=0.17, q=2205.0, nz=1.0,
    fixed_vars={"SIDES": 0.0, "YAW": 0.0, "ROLL": 0.0, "AILERON": 0.0, "RUDDER": 0.0},
    free_vars=["ANGLEA", "ELEV"],
)
run_case("Static 1g (ANGLEA+ELEV free)", tc1, 1)

# Case 2: Static nz=3.80 (free alpha + elevator)
tc2 = TrimCondition(
    case_id=2, mach=0.17, q=2205.0, nz=3.80,
    fixed_vars={"SIDES": 0.0, "YAW": 0.0, "ROLL": 0.0, "AILERON": 0.0, "RUDDER": 0.0},
    free_vars=["ANGLEA", "ELEV"],
)
run_case("Static nz=3.80 (ANGLEA+ELEV free)", tc2, 2)

# Case 3: Dynamic nz=3.80 (all fixed, like 6-DOF sim output)
# Use alpha value that would give CL=3.80W/(qS)
# S = refs area in mm², q in N/mm²
# CL = nz * W / (q * S) = 3.80 * 12646 / (0.002205 * 17090000) ≈ 1.275
# alpha ≈ CL / CLalpha = 1.275 / 5.301 = 0.2406 rad
tc3 = TrimCondition(
    case_id=3, mach=0.17, q=2205.0, nz=3.80,
    fixed_vars={
        "ANGLEA": 0.24, "ELEV": -0.05, "SIDES": 0.0,
        "YAW": 0.0, "ROLL": 0.0, "AILERON": 0.0, "RUDDER": 0.0,
    },
    free_vars=[],
)
run_case("Dynamic nz=3.80 (ALL fixed, α=0.24)", tc3, 3)

# Case 4: Dynamic nz=0.78 (mimic case 10012 - low nz but high alpha?)
tc4 = TrimCondition(
    case_id=4, mach=0.23, q=3650.0, nz=0.78,
    fixed_vars={
        "ANGLEA": 0.24, "ELEV": -0.10, "SIDES": 0.0,
        "YAW": 0.0, "ROLL": 0.0, "AILERON": 0.0, "RUDDER": 0.0,
    },
    free_vars=[],
)
run_case("Dynamic nz=0.78, α=0.24 (like case 10012?)", tc4, 4)

print("\n\nDone!")
