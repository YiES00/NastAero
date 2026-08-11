#!/usr/bin/env python3
"""Detailed 1g wing root load decomposition: FAR 23 vs NastAero.

Investigates the 0.59 ratio discrepancy by decomposing loads into
aerodynamic and inertial contributions, comparing elastic axis positions,
and examining the actual mass and lift distributions.
"""
import os, math, time
import numpy as np

G = 9.80665; RHO0 = 1.225
W_KG = 1288.9; W_N = W_KG * G
S_M2 = 17.0; B_M = 11.233; MAC_M = 1.6; CLA = 5.5
XCG = 3882.0; XW = 3500.0; XT = 8500.0; CMAC_WF = -0.05


def run():
    from nastaero.bdf.parser import BDFParser
    from nastaero.solvers.sol144 import solve_trim
    from nastaero.loads_analysis.vmt import compute_vmt, _compute_elastic_axis
    from nastaero.loads_analysis.component_id import identify_components

    bdf_path = os.path.join(os.path.dirname(__file__),
        "validation", "GACOMP", "p400r3-free-trim.bdf")

    print("Parsing GACOMP BDF...")
    parser = BDFParser(); model = parser.parse(bdf_path)
    print(f"  {len(model.nodes)} nodes, {len(model.elements)} elements")

    print("Solving SOL 144 (M=0.2, 1g)...")
    t0 = time.time()
    result = solve_trim(model)
    print(f"  Solved in {time.time()-t0:.1f}s")

    # Use subcase 2 (M=0.2, ~68 m/s, closest to typical GA cruise)
    sc = result.subcases[1]
    trim_vars = sc.trim_variables
    mach = list(model.trims.values())[1].mach
    V_eas = mach * 340.3
    q_pa = 0.5 * RHO0 * V_eas**2

    alpha_rad = trim_vars.get('ANGLEA', 0)
    elev_rad = trim_vars.get('ELEV', 0)
    nz = 1.0

    print(f"\n  Subcase 2: M={mach:.3f}, V≈{V_eas:.1f} m/s, q={q_pa:.1f} Pa")
    print(f"  Alpha={math.degrees(alpha_rad):.2f}°, Elev={math.degrees(elev_rad):.2f}°")

    # ── Component Identification ──
    comp_set = identify_components(model)
    rw_comp = comp_set.get("Right Wing")
    if not rw_comp:
        print("ERROR: Right Wing not found"); return

    rw_nids = set(rw_comp.node_ids)
    print(f"\n  Right Wing: {len(rw_nids)} nodes")

    # ── Separate Aero and Inertia on Right Wing ──
    aero_f = sc.nodal_aero_forces or {}
    inertia_f = sc.nodal_inertial_forces or {}
    combined_f = sc.nodal_combined_forces or {}

    # Build combined if not available
    if not combined_f:
        all_nids = set(aero_f.keys()) | set(inertia_f.keys())
        for nid in all_nids:
            combined_f[nid] = aero_f.get(nid, np.zeros(6)) + inertia_f.get(nid, np.zeros(6))

    # Sum forces by component
    F_aero_rw = np.zeros(6)
    F_inertia_rw = np.zeros(6)
    F_combined_rw = np.zeros(6)
    n_aero_rw = n_inertia_rw = 0

    for nid in rw_nids:
        if nid in aero_f:
            F_aero_rw += aero_f[nid]
            n_aero_rw += 1
        if nid in inertia_f:
            F_inertia_rw += inertia_f[nid]
            n_inertia_rw += 1
        if nid in combined_f:
            F_combined_rw += combined_f[nid]

    # Total aircraft forces
    F_aero_total = np.zeros(6)
    F_inertia_total = np.zeros(6)
    for f in aero_f.values(): F_aero_total += f
    for f in inertia_f.values(): F_inertia_total += f

    print("\n" + "=" * 80)
    print("  1g WING ROOT LOAD DECOMPOSITION — Right Wing")
    print("=" * 80)

    # ── 1. Total Aircraft Balance ──
    print(f"\n{'─'*80}")
    print(f"  1. TOTAL AIRCRAFT FORCE BALANCE")
    print(f"{'─'*80}")
    print(f"  Total Aero:    Fx={F_aero_total[0]:>10.1f}  Fy={F_aero_total[1]:>10.1f}"
          f"  Fz={F_aero_total[2]:>10.1f} N")
    print(f"  Total Inertia: Fx={F_inertia_total[0]:>10.1f}  Fy={F_inertia_total[1]:>10.1f}"
          f"  Fz={F_inertia_total[2]:>10.1f} N")
    print(f"  Residual:      Fx={(F_aero_total+F_inertia_total)[0]:>10.1f}"
          f"  Fy={(F_aero_total+F_inertia_total)[1]:>10.1f}"
          f"  Fz={(F_aero_total+F_inertia_total)[2]:>10.1f} N")
    print(f"\n  Weight = {W_N:.1f} N, Aero lift = {F_aero_total[2]:.1f} N")
    print(f"  Lift/Weight = {F_aero_total[2]/W_N:.4f} "
          f"({'OK' if abs(F_aero_total[2]/W_N - 1.0) < 0.2 else 'MISMATCH'})")

    # ── 2. Right Wing Forces ──
    print(f"\n{'─'*80}")
    print(f"  2. RIGHT WING FORCE BREAKDOWN")
    print(f"{'─'*80}")
    print(f"  Aero nodes: {n_aero_rw}, Inertia nodes: {n_inertia_rw}")
    print(f"\n  {'Component':<20} {'Fx(N)':>10} {'Fy(N)':>10} {'Fz(N)':>10}"
          f" │ {'Mx(N-mm)':>12} {'My(N-mm)':>12} {'Mz(N-mm)':>12}")
    print(f"  {'─'*88}")
    print(f"  {'Aero (wing)':<20} {F_aero_rw[0]:>10.1f} {F_aero_rw[1]:>10.1f}"
          f" {F_aero_rw[2]:>10.1f} │ {F_aero_rw[3]:>12.1f} {F_aero_rw[4]:>12.1f}"
          f" {F_aero_rw[5]:>12.1f}")
    print(f"  {'Inertia (wing)':<20} {F_inertia_rw[0]:>10.1f} {F_inertia_rw[1]:>10.1f}"
          f" {F_inertia_rw[2]:>10.1f} │ {F_inertia_rw[3]:>12.1f}"
          f" {F_inertia_rw[4]:>12.1f} {F_inertia_rw[5]:>12.1f}")
    print(f"  {'Combined (wing)':<20} {F_combined_rw[0]:>10.1f} {F_combined_rw[1]:>10.1f}"
          f" {F_combined_rw[2]:>10.1f} │ {F_combined_rw[3]:>12.1f}"
          f" {F_combined_rw[4]:>12.1f} {F_combined_rw[5]:>12.1f}")

    # Wing weight fraction
    W_wing_N = abs(F_inertia_rw[2])
    W_wing_frac = W_wing_N / W_N
    print(f"\n  Right wing inertia Fz = {F_inertia_rw[2]:.1f} N")
    print(f"  Wing weight fraction = {W_wing_frac:.4f} ({W_wing_frac*100:.1f}% of total)")
    print(f"  Wing aero Fz = {F_aero_rw[2]:.1f} N")
    print(f"  Wing aero / Total aero = {F_aero_rw[2]/F_aero_total[2]:.4f}")

    # ── 3. VMT Decomposition ──
    print(f"\n{'─'*80}")
    print(f"  3. VMT DECOMPOSITION (Aero vs Inertia vs Combined)")
    print(f"{'─'*80}")

    # Compute VMT separately for aero, inertia, combined
    curve_combined = compute_vmt(model, combined_f, rw_comp, n_stations=30)
    curve_aero = compute_vmt(model, aero_f, rw_comp, n_stations=30,
                             load_type='aero')
    curve_inertia = compute_vmt(model, inertia_f, rw_comp, n_stations=30,
                                load_type='inertia')

    print(f"\n  {'Source':<12} │ {'V_root (N)':>12} {'M_root (N-mm)':>14} {'T_root (N-mm)':>14}")
    print(f"  {'─'*56}")
    print(f"  {'Aero':<12} │ {curve_aero.shear[0]:>12.1f} "
          f"{curve_aero.bending_moment[0]:>14.1f} {curve_aero.torsion[0]:>14.1f}")
    print(f"  {'Inertia':<12} │ {curve_inertia.shear[0]:>12.1f} "
          f"{curve_inertia.bending_moment[0]:>14.1f} {curve_inertia.torsion[0]:>14.1f}")
    print(f"  {'Combined':<12} │ {curve_combined.shear[0]:>12.1f} "
          f"{curve_combined.bending_moment[0]:>14.1f} {curve_combined.torsion[0]:>14.1f}")
    print(f"  {'A+I check':<12} │ "
          f"{curve_aero.shear[0]+curve_inertia.shear[0]:>12.1f} "
          f"{curve_aero.bending_moment[0]+curve_inertia.bending_moment[0]:>14.1f} "
          f"{curve_aero.torsion[0]+curve_inertia.torsion[0]:>14.1f}")

    # ── 4. Elastic Axis Comparison ──
    print(f"\n{'─'*80}")
    print(f"  4. ELASTIC AXIS (EA) POSITION COMPARISON")
    print(f"{'─'*80}")

    # NastAero EA: compute from component node chord extent
    all_xyz_rw = np.array([model.nodes[nid].xyz_global for nid in rw_comp.node_ids
                           if nid in model.nodes], dtype=np.float64)
    all_y_rw = all_xyz_rw[:, 1]  # span axis = Y
    all_x_rw = all_xyz_rw[:, 0]  # chord axis = X

    # Show chord extent at several span stations
    stations_check = np.linspace(all_y_rw.min(), all_y_rw.max(), 8)
    half_bin = (all_y_rw.max() - all_y_rw.min()) / 20

    print(f"\n  NastAero elastic axis (40% chord from LE):")
    print(f"  {'Y_span(mm)':>12} {'X_LE(mm)':>10} {'X_TE(mm)':>10} {'Chord(mm)':>10}"
          f" {'EA_40%(mm)':>12} {'EA_25%(mm)':>12}")
    print(f"  {'─'*68}")
    for s in stations_check:
        nearby = np.abs(all_y_rw - s) < half_bin
        if np.any(nearby):
            x_le = all_x_rw[nearby].min()
            x_te = all_x_rw[nearby].max()
            chord = x_te - x_le
            ea_40 = x_le + 0.40 * chord
            ea_25 = x_le + 0.25 * chord
            print(f"  {s:>12.1f} {x_le:>10.1f} {x_te:>10.1f} {chord:>10.1f}"
                  f" {ea_40:>12.1f} {ea_25:>12.1f}")

    # ── 5. FAR 23 Analytical Breakdown ──
    print(f"\n{'─'*80}")
    print(f"  5. FAR 23 ANALYTICAL METHOD — STEP BY STEP (1g, VC=80 m/s)")
    print(f"{'─'*80}")

    V_eas = 80.0  # m/s (VC)
    q = 0.5 * RHO0 * V_eas**2
    nz = 1.0

    # Step 1: Total lift
    L_total = nz * W_N
    print(f"\n  Step 1: Total lift L = nz × W = {nz} × {W_N:.1f} = {L_total:.1f} N")

    # Step 2: Tail load
    M_acwf = CMAC_WF * q * S_M2 * MAC_M * 1000  # N-mm
    LT = -(M_acwf + L_total * (XCG - XW)) / (XT - XCG)
    print(f"\n  Step 2: Tail load")
    print(f"    M_acwf = Cm × q × S × MAC = {CMAC_WF} × {q:.1f} × {S_M2} × {MAC_M}")
    print(f"           = {M_acwf:.0f} N-mm = {M_acwf/1000:.1f} N-m")
    print(f"    LT = -[M_acwf + L×(XCG-XW)] / (XT-XCG)")
    print(f"       = -[{M_acwf:.0f} + {L_total:.0f}×({XCG:.0f}-{XW:.0f})] / {XT-XCG:.0f}")
    print(f"       = {LT:.1f} N ({LT/G:.1f} kgf)")

    # Step 3: Wing lift
    L_wing = L_total - LT
    L_semi = L_wing / 2.0
    print(f"\n  Step 3: Wing lift = L - LT = {L_total:.0f} - ({LT:.0f}) = {L_wing:.0f} N")
    print(f"    Per semi-span = {L_semi:.0f} N")

    # Step 4: Wing weight relief
    W_wing_assumed = 0.20 * W_N  # FAR 23 assumed 20% (wing+fuel)
    W_wing_semi_assumed = W_wing_assumed / 2.0
    inertia_relief = nz * W_wing_semi_assumed
    print(f"\n  Step 4: Inertia relief (FAR 23 assumption)")
    print(f"    Wing+fuel weight = {W_wing_assumed/W_N*100:.0f}% of W = {W_wing_assumed:.0f} N")
    print(f"    Per semi-span = {W_wing_semi_assumed:.0f} N")
    print(f"    NastAero actual R.Wing inertia Fz = {abs(F_inertia_rw[2]):.0f} N"
          f" = {abs(F_inertia_rw[2])/W_N*100:.1f}% of W")

    # Step 5: Net root shear
    V_root_f23 = L_semi - inertia_relief
    print(f"\n  Step 5: Root shear = Aero - Inertia = {L_semi:.0f} - {inertia_relief:.0f}"
          f" = {V_root_f23:.0f} N")
    print(f"    NastAero combined = {curve_combined.shear[0]:.0f} N")
    print(f"    Ratio = {curve_combined.shear[0] / V_root_f23:.3f}")

    # Step 6: Root bending
    b_semi = B_M / 2.0
    y_aero = 4.0 * b_semi / (3.0 * math.pi)
    y_mass = b_semi * 0.38
    M_root_f23 = (L_semi * y_aero - inertia_relief * y_mass) * 1000
    print(f"\n  Step 6: Root bending")
    print(f"    Aero centroid (elliptic) = {y_aero:.1f} mm from root")
    print(f"    Mass centroid (assumed 38% span) = {y_mass:.1f} mm from root")
    print(f"    M = L×y_aero - I×y_mass = {L_semi:.0f}×{y_aero:.0f}"
          f" - {inertia_relief:.0f}×{y_mass:.0f}")
    print(f"      = {M_root_f23:.0f} N-mm = {M_root_f23/1000:.0f} N-m")
    print(f"    NastAero combined = {curve_combined.bending_moment[0]:.0f} N-mm"
          f" = {curve_combined.bending_moment[0]/1000:.0f} N-m")
    print(f"    Ratio = {curve_combined.bending_moment[0] / M_root_f23:.3f}")

    # ── 6. Root Cause Analysis ──
    print(f"\n{'─'*80}")
    print(f"  6. ROOT CAUSE ANALYSIS")
    print(f"{'─'*80}")

    # The key difference: actual wing aero lift vs assumed
    print(f"\n  [A] Wing Aerodynamic Lift:")
    print(f"    FAR 23 assumed wing aero (semi-span): {L_semi:.0f} N")
    print(f"    NastAero actual wing aero Fz (R.Wing): {F_aero_rw[2]:.0f} N")
    print(f"    → FAR 23 overestimates wing lift by "
          f"{L_semi/F_aero_rw[2]:.2f}x" if F_aero_rw[2] > 0 else "")

    print(f"\n  [B] Wing Inertia Relief:")
    print(f"    FAR 23 assumed (20% of W): {W_wing_semi_assumed:.0f} N per semi-span")
    print(f"    NastAero actual R.Wing inertia |Fz|: {abs(F_inertia_rw[2]):.0f} N")
    print(f"    → FAR 23 {'under' if W_wing_semi_assumed < abs(F_inertia_rw[2]) else 'over'}estimates"
          f" inertia by {abs(F_inertia_rw[2])/W_wing_semi_assumed:.2f}x")

    print(f"\n  [C] Total aircraft aero Fz vs Weight:")
    print(f"    Aero total Fz = {F_aero_total[2]:.0f} N")
    print(f"    Weight = {W_N:.0f} N")
    print(f"    Ratio = {F_aero_total[2]/W_N:.4f}")

    # ── 7. Corrected FAR 23 with NastAero parameters ──
    print(f"\n{'─'*80}")
    print(f"  7. CORRECTED FAR 23 (using NastAero actual parameters)")
    print(f"{'─'*80}")

    # Use actual wing aero and inertia from NastAero
    V_corrected = F_aero_rw[2] + F_inertia_rw[2]  # Net Fz on wing
    print(f"    Using NastAero wing aero Fz = {F_aero_rw[2]:.0f} N")
    print(f"    Using NastAero wing inertia Fz = {F_inertia_rw[2]:.0f} N")
    print(f"    Net Fz = {V_corrected:.0f} N (this ≈ VMT root shear)")
    print(f"    NastAero VMT root shear = {curve_combined.shear[0]:.0f} N")
    print(f"    Agreement: {V_corrected/curve_combined.shear[0]:.3f}"
          if abs(curve_combined.shear[0]) > 1 else "")

    # ── 8. Spanwise distribution comparison ──
    print(f"\n{'─'*80}")
    print(f"  8. SPANWISE SHEAR DISTRIBUTION (NastAero)")
    print(f"{'─'*80}")
    print(f"  {'Y(mm)':>8} │ {'V_comb(N)':>10} {'V_aero(N)':>10} {'V_iner(N)':>10}"
          f" │ {'M_comb(N-m)':>12}")
    print(f"  {'─'*58}")
    for i in range(0, len(curve_combined.stations), 3):
        y = curve_combined.stations[i]
        vc = curve_combined.shear[i]
        va = curve_aero.shear[i]
        vi = curve_inertia.shear[i]
        mc = curve_combined.bending_moment[i] / 1000
        print(f"  {y:>8.0f} │ {vc:>10.0f} {va:>10.0f} {vi:>10.0f} │ {mc:>12.0f}")

    # ── 9. Torsion EA sensitivity ──
    print(f"\n{'─'*80}")
    print(f"  9. TORSION SENSITIVITY TO ELASTIC AXIS POSITION")
    print(f"{'─'*80}")
    for ea_frac in [0.25, 0.30, 0.35, 0.40, 0.50]:
        curve_t = compute_vmt(model, combined_f, rw_comp, n_stations=30,
                              elastic_axis_frac=ea_frac)
        print(f"    EA={ea_frac*100:.0f}% chord: T_root = {curve_t.torsion[0]:>12.0f} N-mm"
              f" = {curve_t.torsion[0]/1000:>8.0f} N-m")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    run()
