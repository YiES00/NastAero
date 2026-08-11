#!/usr/bin/env python3
"""FAR 23 LOADS vs NastAero: Actual Structural Loads Comparison for GACOMP.

Runs NastAero SOL 144 trim on the GACOMP BDF for representative V-n
conditions, extracts VMT (shear, bending, torsion) at component roots,
and compares against FAR 23 LOADS analytical estimates.
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np

# ── Constants ──
G = 9.80665          # m/s^2
RHO0 = 1.225         # kg/m^3 (sea level)
FPS_TO_MS = 0.3048

# ── GACOMP Parameters ──
W_KG = 1288.9        # MTOW (kg)
W_N = W_KG * G       # Weight (N)  = 12640 N
W_LB = W_KG / 0.4536 # = 2842 lb
S_M2 = 17.0          # Wing area (m^2)
B_M = 11.233         # Wing span (m)
MAC_M = 1.6          # MAC (m)
CLA = 5.5            # CLα (/rad)
NZ_MAX = 3.8
NZ_MIN = -1.52

# Speeds (m/s EAS)
VS1, VA, VC, VD, VF = 33.0, 62.0, 80.0, 100.0, 40.0

# Geometry for moment balance
XCG = 3882.0         # CG x-position (mm)
XW = 3500.0          # Wing AC ~25%MAC (mm)
XT = 8500.0          # Tail AC (mm)
CMAC_WF = -0.05      # Cm_ac wing+fuse
TAIL_ARM = XT - XCG  # mm

# HTP
S_HTP = 3.5          # HTP area (m^2, estimated)
CLa_HTP = 4.2        # HTP lift curve slope (/rad)
AR_W = B_M ** 2 / S_M2

# Aileron
S_AIL = 0.8          # Aileron area (m^2, each side)
AIL_MAX_DEG = 20.0   # Max aileron deflection

# Flap
S_FLAP = 1.5         # Flap area (m^2, each side)

# Landing gear
STRUT_EFF = 0.7; TIRE_EFF = 0.47
STRUT_STROKE = 0.25; TIRE_STROKE = 0.05  # meters


# ═══════════════════════════════════════════
# FAR 23 LOADS Analytical Component Loads
# ═══════════════════════════════════════════

def far23_balanced_tail_load(nz, V_eas):
    """Compute balanced horizontal tail load per FAR 23.421.

    LT = -[M_acwf + L*(XCG-XW)] / (XT-XCG)

    M_acwf = Cmac * q * S * MAC (in N-mm)
    L = nz * W (in N)
    """
    q = 0.5 * RHO0 * V_eas ** 2  # Pa = N/m^2
    M_acwf = CMAC_WF * q * S_M2 * MAC_M * 1000  # N-mm (MAC in m -> mm)
    L = nz * W_N  # N

    LT = -(M_acwf + L * (XCG - XW)) / TAIL_ARM  # N
    return LT


def far23_downwash(CL_wing):
    """Downwash angle at tail (degrees).
    E = 114.6 * CLW / (pi * ARW)
    """
    return 114.6 * CL_wing / (math.pi * AR_W)


def far23_gust_tail_increment(V_eas):
    """Gust increment on tail load per FAR 23.425(d).
    dLT = Kg * Ude * V * aHT * ST * (1 - de/da) / 498  [in imperial]
    """
    V_fps = V_eas * 3.28084
    Ude = 50.0 if V_eas <= VC else 25.0
    WS_psf = W_LB / (S_M2 / 0.0929)

    mu_g = 2 * (W_N / S_M2) / (RHO0 * MAC_M * CLA * G)
    Kg = 0.88 * mu_g / (5.3 + mu_g)

    # Downwash derivative de/da
    de_da = 114.6 * CLA / (57.3 * math.pi * AR_W)

    # Tail gust load (imperial then convert)
    S_HTP_ft2 = S_HTP / 0.0929
    dLT_lb = Kg * Ude * V_fps * CLa_HTP * S_HTP_ft2 * (1 - de_da) / 498
    dLT_N = dLT_lb * 4.4482
    return dLT_N


def far23_wing_loads(nz, V_eas, LT_N):
    """Compute wing root loads (semi-span).

    Wing lift = nz*W - LT (total lift minus tail)
    Assumes elliptic spanwise distribution.

    Returns: (shear_N, bending_Nmm, torsion_Nmm) at root
    """
    L_wing = nz * W_N - LT_N  # Total wing lift (N)
    L_semi = L_wing / 2.0     # Per semi-span

    # Wing weight relief (assume 12% of total weight in each semi-span)
    W_wing_semi = 0.12 * W_N / 2.0
    inertia_relief = nz * W_wing_semi

    # Fuel weight relief (assume 8% of total in each semi-span wing tank)
    W_fuel_semi = 0.08 * W_N / 2.0
    fuel_relief = nz * W_fuel_semi

    # Net root shear
    V_root = L_semi - inertia_relief - fuel_relief

    # Bending moment (elliptic centroid at 4b/(3*pi) from root)
    b_semi = B_M / 2.0
    y_aero = 4.0 * b_semi / (3.0 * math.pi)  # Aero load centroid
    y_wing = b_semi * 0.4                      # Wing CG ~40% span
    y_fuel = b_semi * 0.35                     # Fuel CG ~35% span

    M_root = (L_semi * y_aero
              - inertia_relief * y_wing
              - fuel_relief * y_fuel) * 1000  # N-mm

    # Torsion (pitching moment about elastic axis at 40% chord)
    q = 0.5 * RHO0 * V_eas ** 2
    Cm_wing = -0.08  # Typical airfoil Cm at 25% chord
    # Moment arm from AC (25%) to EA (40%) = 0.15 * MAC
    arm_ac_ea = 0.15 * MAC_M * 1000  # mm
    T_aero = Cm_wing * q * S_M2 / 2 * MAC_M * 1000  # N-mm per semi-span
    T_offset = L_semi * arm_ac_ea  # Lift offset from AC to EA
    T_root = T_aero + T_offset  # N-mm

    return V_root, M_root, T_root


def far23_fuselage_loads(nz, LT_N):
    """Estimate fuselage loads at wing front spar (simplified beam).

    Aft fuselage carries tail load to wing attachment.
    Forward fuselage carries nose loads.
    """
    # Aft fuselage bending at wing rear spar
    aft_length = XT - XCG  # mm, tail arm
    M_aft = LT_N * aft_length  # N-mm, bending at wing attachment

    # Forward fuselage: nose equipment ~5% W at ~1000mm forward
    W_nose = 0.05 * W_N
    fwd_arm = XCG - 1500  # mm from CG to nose equipment
    M_fwd = nz * W_nose * fwd_arm  # N-mm

    # Fuselage shear at CG
    V_fuse = nz * W_N * 0.15  # ~15% of total inertia (fuselage mass fraction)

    return V_fuse, M_aft, M_fwd


def far23_aileron_loads(V_eas):
    """Aileron design load per FAR 23.455/CAM 3.222.

    L_ail = CL_ail * q * S_ail
    Deflections reduced at higher speeds per §23.349.
    """
    # Speed-dependent deflection
    if V_eas <= VA:
        defl = AIL_MAX_DEG
    elif V_eas <= VC:
        defl = AIL_MAX_DEG * 2 / 3
    else:
        defl = AIL_MAX_DEG * 1 / 3

    q = 0.5 * RHO0 * V_eas ** 2
    # Aileron effectiveness ~0.04 per degree (typical)
    CL_ail = 0.04 * defl
    L_ail = CL_ail * q * S_AIL  # N per side
    return L_ail, defl


def far23_flap_loads(V_eas, nz_flap=2.0):
    """Flap design load per FAR 23.345.

    At VF with nz=2.0 (FAR 23.345(a)(2)).
    """
    q = 0.5 * RHO0 * V_eas ** 2
    CL_flap = nz_flap * W_N / (q * S_M2)  # CL at nz=2
    # Flap contributes ~30% of total CL
    CL_flap_portion = CL_flap * 0.30
    L_flap = CL_flap_portion * q * S_FLAP  # N per side
    return L_flap


def far23_landing_loads():
    """Landing loads per FAR 23.473-23.499.

    Returns dict with load conditions.
    """
    WS_psf = W_LB / (S_M2 / 0.0929)
    V_sink = 4.4 * WS_psf ** 0.25  # fps
    V_sink = max(7.0, min(10.0, V_sink))
    V_sink_ms = V_sink * FPS_TO_MS

    d_eff = STRUT_EFF * STRUT_STROKE + TIRE_EFF * TIRE_STROKE
    nz_air = V_sink_ms ** 2 / (2 * G * d_eff) + 1.0
    nz_air = max(2.67, nz_air)  # FAR 23.473(g) minimum

    # Wing lift during landing = 2/3 W per FAR 23.473(e)
    L_wing = 2.0 / 3.0 * W_N
    nz_gear = nz_air - 2.0 / 3.0  # Ground reaction factor

    # Main gear proportion (assume 90% on mains, 10% on nose)
    F_main_total = nz_gear * W_N * 0.90  # N total on both mains
    F_main_each = F_main_total / 2       # Per strut
    F_nose = nz_gear * W_N * 0.10        # Nose gear

    results = {
        "V_sink_fps": V_sink,
        "nz_airplane": nz_air,
        "nz_gear": nz_gear,
        # Level landing (§23.479)
        "level_main_V_N": F_main_each,
        "level_main_D_N": 0.25 * F_main_each,  # Spin-up drag
        "level_nose_V_N": F_nose,
        # Side load (§23.485)
        "side_nz": 1.33,
        "side_V_N": 1.33 * W_N * 0.90 / 2,
        "side_S_N": 0.83 * W_N / 2,  # Side force
        # One-wheel (§23.483)
        "onewheel_V_N": 0.75 * F_main_each * 2,  # One side only
        # Braked roll (§23.493)
        "brake_V_N": 1.33 * W_N * 0.90 / 2,
        "brake_D_N": 0.8 * 1.33 * W_N * 0.90 / 2,
    }
    return results


# ═══════════════════════════════════════════
# NastAero SOL 144 Analysis
# ═══════════════════════════════════════════

def run_nastaero_trim(cases):
    """Run NastAero SOL 144 for multiple trim cases on GACOMP.

    Args:
        cases: list of (label, nz, V_eas_ms) tuples

    Returns:
        dict of case_label -> {trim_vars, aero_forces, inertia_forces, vmt}
    """
    from nastaero.bdf.parser import BDFParser
    from nastaero.solvers.sol144 import solve_trim
    from nastaero.loads_analysis.trim_loads import (
        compute_nodal_aero_forces_fast,
        compute_nodal_inertial_forces,
        compute_nodal_combined_forces,
    )
    from nastaero.loads_analysis.vmt import compute_vmt_all
    from nastaero.loads_analysis.component_id import identify_components

    bdf_path = os.path.join(
        os.path.dirname(__file__),
        "validation", "GACOMP", "p400r3-free-trim.bdf"
    )

    print(f"\n  Parsing GACOMP BDF model...")
    t0 = time.time()
    parser = BDFParser()
    model = parser.parse(bdf_path)
    print(f"  Parsed in {time.time()-t0:.1f}s: {len(model.nodes)} nodes, "
          f"{len(model.elements)} elements")

    # Identify structural components
    components = identify_components(model)
    comp_names = [c.name for c in components]
    print(f"  Components: {comp_names}")

    results = {}

    for label, nz, V_eas in cases:
        print(f"\n  ── Case: {label} (nz={nz:+.2f}, V={V_eas:.0f} m/s) ──")

        # Compute Mach and dynamic pressure
        mach = V_eas / 340.3  # Approximate
        q_pa = 0.5 * RHO0 * V_eas ** 2

        # Build trim condition - modify the existing subcase
        # Use subcase 1 as template, modify mach/q/nz
        model_copy = parser.parse(bdf_path)

        # Override trim parameters in the first subcase
        if model_copy.trims:
            trim_key = list(model_copy.trims.keys())[0]
            trim_card = model_copy.trims[trim_key]
            trim_card.mach = mach
            trim_card.q = q_pa * 1e-6  # Convert Pa to MPa (N/mm^2)
            # Set nz as fixed trim variable
            trim_card.fixed_variables = {"URDD3": nz * G * 1000}  # mm/s^2
            trim_card.free_variables = ["ANGLEA", "ELEV"]

        try:
            t1 = time.time()
            result = solve_trim(model_copy)
            dt = time.time() - t1
            print(f"    Solved in {dt:.1f}s")

            if not result.subcases:
                print(f"    WARNING: No subcases in result")
                continue

            sc = result.subcases[0]

            # Extract trim variables
            trim_vars = sc.trim_variables if hasattr(sc, 'trim_variables') else {}
            print(f"    Trim vars: {trim_vars}")

            # Compute nodal forces
            aero_f = {}
            inertia_f = {}
            combined_f = {}

            if hasattr(sc, 'nodal_aero_forces') and sc.nodal_aero_forces:
                aero_f = sc.nodal_aero_forces
            if hasattr(sc, 'nodal_inertial_forces') and sc.nodal_inertial_forces:
                inertia_f = sc.nodal_inertial_forces
            if hasattr(sc, 'nodal_combined_forces') and sc.nodal_combined_forces:
                combined_f = sc.nodal_combined_forces

            # Total forces
            F_aero_total = np.zeros(6)
            for nid, f in aero_f.items():
                F_aero_total += f
            F_inertia_total = np.zeros(6)
            for nid, f in inertia_f.items():
                F_inertia_total += f

            print(f"    Total aero:    Fz={F_aero_total[2]:>12.1f} N")
            print(f"    Total inertia: Fz={F_inertia_total[2]:>12.1f} N")

            # Compute VMT
            if combined_f:
                vmt_all = compute_vmt_all(
                    model_copy, combined_f, components, n_stations=30)
                vmt_data = {}
                for comp_name, curve in vmt_all.items():
                    vmt_data[comp_name] = {
                        "stations": curve.stations,
                        "shear": curve.shear,
                        "bending": curve.bending_moment,
                        "torsion": curve.torsion,
                    }
                    # Print root values
                    root_V = curve.shear[0]
                    root_M = curve.bending_moment[0]
                    root_T = curve.torsion[0]
                    print(f"    {comp_name}: V_root={root_V:>10.1f} N, "
                          f"M_root={root_M:>12.1f} N-mm, "
                          f"T_root={root_T:>12.1f} N-mm")
            else:
                vmt_data = {}

            results[label] = {
                "nz": nz,
                "V_eas": V_eas,
                "trim_vars": trim_vars,
                "F_aero_total": F_aero_total,
                "F_inertia_total": F_inertia_total,
                "vmt": vmt_data,
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    return results, components


def print_comparison(na_results, components):
    """Print formatted comparison of FAR 23 vs NastAero loads."""
    print("\n" + "=" * 90)
    print("FAR 23 LOADS (DARCorp) vs NastAero — Structural Loads Comparison — GACOMP")
    print("=" * 90)

    print(f"\n■ Aircraft: 제작사 GACOMP | W={W_KG:.1f}kg={W_N:.0f}N | "
          f"S={S_M2}m² | b={B_M:.3f}m")

    # ── Representative Cases ──
    cases_def = [
        ("A+ (VA, 3.8g)", NZ_MAX, VA),
        ("C+ (VC, 3.8g)", NZ_MAX, VC),
        ("D+ (VD, 3.8g)", NZ_MAX, VD),
        ("C- (VC, -1.52g)", NZ_MIN, VC),
        ("1g cruise", 1.0, VC),
    ]

    # ── Wing Root Loads ──
    print(f"\n{'─'*90}")
    print(f"■ Wing Root Loads (Right Wing Semi-span)")
    print(f"{'─'*90}")
    print(f"  {'Case':<22} │ {'FAR23':^32} │ {'NastAero':^32}")
    print(f"  {'':22} │ {'V(N)':>10} {'M(N-mm)':>12} {'T(N-mm)':>10}"
          f" │ {'V(N)':>10} {'M(N-mm)':>12} {'T(N-mm)':>10}")
    print(f"  {'─'*87}")

    for label, nz, V_eas in cases_def:
        # FAR 23 analytical
        LT = far23_balanced_tail_load(nz, V_eas)
        V_f, M_f, T_f = far23_wing_loads(nz, V_eas, LT)

        # NastAero
        na_key = label
        V_n = M_n = T_n = 0
        if na_key in na_results:
            vmt = na_results[na_key].get("vmt", {})
            for comp_name, data in vmt.items():
                if "Right" in comp_name and "Wing" in comp_name:
                    V_n = data["shear"][0]
                    M_n = data["bending"][0]
                    T_n = data["torsion"][0]
                    break

        print(f"  {label:<22} │ {V_f:>10.0f} {M_f:>12.0f} {T_f:>10.0f}"
              f" │ {V_n:>10.0f} {M_n:>12.0f} {T_n:>10.0f}")

    # ── Tail Loads ──
    print(f"\n{'─'*90}")
    print(f"■ Horizontal Tail Loads (Balanced)")
    print(f"{'─'*90}")
    print(f"  {'Case':<22} │ {'FAR23 LT(N)':>12} {'LT(kgf)':>10}"
          f" │ {'NastAero':>12} {'Ratio':>8}")
    print(f"  {'─'*65}")

    for label, nz, V_eas in cases_def:
        LT_f = far23_balanced_tail_load(nz, V_eas)
        LT_n = 0
        if label in na_results:
            vmt = na_results[label].get("vmt", {})
            for comp_name, data in vmt.items():
                if "HTP" in comp_name or "Horizontal" in comp_name:
                    LT_n = data["shear"][0]
                    break
        ratio = LT_n / LT_f if abs(LT_f) > 1 else 0
        print(f"  {label:<22} │ {LT_f:>12.1f} {LT_f/G:>10.1f}"
              f" │ {LT_n:>12.1f} {ratio:>8.2f}")

    # ── Aileron Loads ──
    print(f"\n{'─'*90}")
    print(f"■ Aileron Loads")
    print(f"{'─'*90}")
    for V_name, V_eas in [("VA", VA), ("VC", VC), ("VD", VD)]:
        L_ail, defl = far23_aileron_loads(V_eas)
        print(f"  {V_name} ({V_eas:.0f} m/s): defl={defl:.1f}°, "
              f"L_ail={L_ail:.1f} N ({L_ail/G:.1f} kgf)")

    # ── Flap Loads ──
    print(f"\n{'─'*90}")
    print(f"■ Flap Loads (at VF={VF:.0f} m/s)")
    print(f"{'─'*90}")
    L_flap = far23_flap_loads(VF, nz_flap=2.0)
    print(f"  nz=2.0: L_flap={L_flap:.1f} N ({L_flap/G:.1f} kgf) per side")

    # ── Landing Loads ──
    print(f"\n{'─'*90}")
    print(f"■ Landing Loads (§23.473-23.499)")
    print(f"{'─'*90}")
    ldg = far23_landing_loads()
    print(f"  Sink rate: {ldg['V_sink_fps']:.2f} fps")
    print(f"  Airplane nz: {ldg['nz_airplane']:.4f}")
    print(f"  Gear nz: {ldg['nz_gear']:.4f}")
    print(f"\n  {'Condition':<25} {'V_vert(N)':>10} {'V_drag(N)':>10} "
          f"{'V_side(N)':>10} {'V_vert(kgf)':>12}")
    print(f"  {'─'*70}")
    print(f"  {'Level (main, each)':<25} {ldg['level_main_V_N']:>10.0f} "
          f"{ldg['level_main_D_N']:>10.0f} {'—':>10} "
          f"{ldg['level_main_V_N']/G:>12.0f}")
    print(f"  {'Level (nose)':<25} {ldg['level_nose_V_N']:>10.0f} "
          f"{'—':>10} {'—':>10} {ldg['level_nose_V_N']/G:>12.0f}")
    print(f"  {'Side load (§23.485)':<25} {ldg['side_V_N']:>10.0f} "
          f"{'—':>10} {ldg['side_S_N']:>10.0f} {ldg['side_V_N']/G:>12.0f}")
    print(f"  {'One-wheel (§23.483)':<25} {ldg['onewheel_V_N']:>10.0f} "
          f"{'—':>10} {'—':>10} {ldg['onewheel_V_N']/G:>12.0f}")
    print(f"  {'Braked roll (§23.493)':<25} {ldg['brake_V_N']:>10.0f} "
          f"{ldg['brake_D_N']:>10.0f} {'—':>10} {ldg['brake_V_N']/G:>12.0f}")

    # ── Critical Design Loads Summary ──
    print(f"\n{'─'*90}")
    print(f"■ Critical Design Loads Summary (FAR 23 LOADS)")
    print(f"{'─'*90}")

    # Find critical wing loads
    max_V = max_M = max_T = 0
    max_V_case = max_M_case = max_T_case = ""
    for label, nz, V_eas in cases_def:
        LT = far23_balanced_tail_load(nz, V_eas)
        V_f, M_f, T_f = far23_wing_loads(nz, V_eas, LT)
        if abs(V_f) > abs(max_V): max_V = V_f; max_V_case = label
        if abs(M_f) > abs(max_M): max_M = M_f; max_M_case = label
        if abs(T_f) > abs(max_T): max_T = T_f; max_T_case = label

    print(f"  Wing Root:")
    print(f"    Max Shear:   {max_V:>12.0f} N ({max_V/G:>8.0f} kgf) ← {max_V_case}")
    print(f"    Max Bending: {max_M:>12.0f} N-mm ({max_M/G/1000:>8.1f} kgf-m) ← {max_M_case}")
    print(f"    Max Torsion: {max_T:>12.0f} N-mm ({max_T/G/1000:>8.1f} kgf-m) ← {max_T_case}")

    # Critical tail
    max_LT = 0; max_LT_case = ""
    for label, nz, V_eas in cases_def:
        LT = far23_balanced_tail_load(nz, V_eas)
        if abs(LT) > abs(max_LT): max_LT = LT; max_LT_case = label
    print(f"  Tail:")
    print(f"    Max Tail Load: {max_LT:>10.0f} N ({max_LT/G:>8.0f} kgf) ← {max_LT_case}")

    # Critical landing
    print(f"  Landing:")
    print(f"    Max Main Gear: {ldg['level_main_V_N']:>10.0f} N "
          f"({ldg['level_main_V_N']/G:>8.0f} kgf) ← Level landing")
    print(f"    Max Brake:     {ldg['brake_D_N']:>10.0f} N "
          f"({ldg['brake_D_N']/G:>8.0f} kgf) ← Braked roll")

    # ── NastAero Critical ──
    if na_results:
        print(f"\n{'─'*90}")
        print(f"■ Critical Design Loads Summary (NastAero SOL 144)")
        print(f"{'─'*90}")
        for comp_name_key in ["Right Wing", "Right HTP", "VTP"]:
            max_V_na = max_M_na = max_T_na = 0
            max_V_case_na = max_M_case_na = max_T_case_na = ""
            for case_label, data in na_results.items():
                vmt = data.get("vmt", {})
                for cn, vd in vmt.items():
                    if comp_name_key in cn:
                        if abs(vd["shear"][0]) > abs(max_V_na):
                            max_V_na = vd["shear"][0]; max_V_case_na = case_label
                        if abs(vd["bending"][0]) > abs(max_M_na):
                            max_M_na = vd["bending"][0]; max_M_case_na = case_label
                        if abs(vd["torsion"][0]) > abs(max_T_na):
                            max_T_na = vd["torsion"][0]; max_T_case_na = case_label

            if max_V_na != 0:
                print(f"  {comp_name_key}:")
                print(f"    Max Shear:   {max_V_na:>12.0f} N ← {max_V_case_na}")
                print(f"    Max Bending: {max_M_na:>12.0f} N-mm ← {max_M_case_na}")
                print(f"    Max Torsion: {max_T_na:>12.0f} N-mm ← {max_T_case_na}")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    # Define representative V-n cases
    cases = [
        ("A+ (VA, 3.8g)", NZ_MAX, VA),
        ("C+ (VC, 3.8g)", NZ_MAX, VC),
        ("D+ (VD, 3.8g)", NZ_MAX, VD),
        ("C- (VC, -1.52g)", NZ_MIN, VC),
        ("1g cruise", 1.0, VC),
    ]

    print("=" * 90)
    print("Running NastAero SOL 144 trim on GACOMP for representative V-n cases...")
    print("=" * 90)

    try:
        na_results, components = run_nastaero_trim(cases)
    except Exception as e:
        print(f"\nNastAero solve failed: {e}")
        import traceback; traceback.print_exc()
        na_results = {}
        components = []

    print_comparison(na_results, components)
