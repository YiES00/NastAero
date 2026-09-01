#!/usr/bin/env python3
"""FAR 23 LOADS vs ASCENT-Load: Structural Loads Comparison for GACOMP.

Runs the GACOMP BDF through SOL 144 (existing subcases), extracts VMT,
and compares with FAR 23 LOADS analytical method.
"""
import os, sys, math, time
import numpy as np

G = 9.80665; RHO0 = 1.225; FPS_TO_MS = 0.3048
W_KG = 1288.9; W_N = W_KG * G; W_LB = W_KG / 0.4536
S_M2 = 17.0; B_M = 11.233; MAC_M = 1.6; CLA = 5.5
NZ_MAX = 3.8; NZ_MIN = -1.52
VS1, VA, VC, VD, VF = 33.0, 62.0, 80.0, 100.0, 40.0
XCG = 3882.0; XW = 3500.0; XT = 8500.0; CMAC_WF = -0.05
TAIL_ARM = XT - XCG
S_HTP = 3.5; S_AIL = 0.8; S_FLAP = 1.5; AR_W = B_M**2 / S_M2
AIL_MAX_DEG = 20.0

# ═══════════════════════════════════════
# FAR 23 LOADS Analytical
# ═══════════════════════════════════════

def far23_tail_load(nz, V_eas):
    q = 0.5 * RHO0 * V_eas**2
    M_acwf = CMAC_WF * q * S_M2 * MAC_M * 1000
    L = nz * W_N
    return -(M_acwf + L * (XCG - XW)) / TAIL_ARM

def far23_wing_root(nz, V_eas, LT):
    L_wing = nz * W_N - LT
    L_semi = L_wing / 2.0
    W_wing_semi = 0.12 * W_N / 2.0
    W_fuel_semi = 0.08 * W_N / 2.0
    inertia_r = nz * (W_wing_semi + W_fuel_semi)
    V_root = L_semi - inertia_r
    b_semi = B_M / 2.0
    y_aero = 4.0 * b_semi / (3.0 * math.pi)
    y_mass = b_semi * 0.38
    M_root = (L_semi * y_aero - inertia_r * y_mass) * 1000  # N-mm
    q = 0.5 * RHO0 * V_eas**2
    T_aero = -0.08 * q * S_M2/2 * MAC_M * 1000
    T_offset = L_semi * 0.15 * MAC_M * 1000
    T_root = T_aero + T_offset
    return V_root, M_root, T_root

def far23_fuselage(nz, LT):
    M_aft = LT * TAIL_ARM
    fwd_arm = XCG - 1500
    M_fwd = nz * 0.05 * W_N * fwd_arm
    V_fuse = nz * W_N * 0.15
    return V_fuse, M_aft, M_fwd

def far23_aileron(V_eas):
    if V_eas <= VA: d = AIL_MAX_DEG
    elif V_eas <= VC: d = AIL_MAX_DEG * 2/3
    else: d = AIL_MAX_DEG * 1/3
    q = 0.5 * RHO0 * V_eas**2
    return 0.04 * d * q * S_AIL, d

def far23_landing():
    WS_psf = W_LB / (S_M2 / 0.0929)
    V_sink = max(7.0, min(10.0, 4.4 * WS_psf**0.25))
    V_ms = V_sink * FPS_TO_MS
    d_eff = 0.7*0.25 + 0.47*0.05
    nz_air = max(2.67, V_ms**2/(2*G*d_eff) + 1.0)
    nz_gear = nz_air - 2/3
    main_each = nz_gear * W_N * 0.90 / 2
    nose = nz_gear * W_N * 0.10
    return {"nz": nz_air, "nz_gear": nz_gear, "V_sink": V_sink,
            "main_V": main_each, "main_D": 0.25*main_each, "nose_V": nose,
            "side_V": 1.33*W_N*0.90/2, "side_S": 0.83*W_N/2,
            "onewheel": 0.75*main_each*2,
            "brake_V": 1.33*W_N*0.90/2, "brake_D": 0.8*1.33*W_N*0.90/2}


# ═══════════════════════════════════════
# ASCENT-Load Solver
# ═══════════════════════════════════════

def run_ascent_load():
    """Run GACOMP through SOL 144 and extract VMT."""
    from ascent_load.bdf.parser import BDFParser
    from ascent_load.solvers.sol144 import solve_trim
    from ascent_load.loads_analysis.vmt import compute_vmt_all
    from ascent_load.loads_analysis.component_id import identify_components

    bdf_path = os.path.join(os.path.dirname(__file__),
        "validation", "GACOMP", "p400r3-free-trim.bdf")

    print("  Parsing GACOMP BDF...")
    t0 = time.time()
    parser = BDFParser(); model = parser.parse(bdf_path)
    print(f"  Parsed: {len(model.nodes)} nodes, {len(model.elements)} elements "
          f"({time.time()-t0:.1f}s)")

    # Show subcases
    print(f"  Subcases: {len(model.trims)} trim conditions")
    for tid, trim in model.trims.items():
        vars_str = ", ".join(f"{v[0]}={v[1]}" for v in trim.variables[:3])
        print(f"    TRIM {tid}: M={trim.mach:.3f}, q={trim.q:.6f} [{vars_str}...]")

    print("  Solving SOL 144...")
    t1 = time.time()
    result = solve_trim(model)
    dt = time.time() - t1
    print(f"  Solved in {dt:.1f}s, {len(result.subcases)} subcases")

    # Identify components
    comp_set = identify_components(model)
    components = comp_set.components
    comp_names = [c.name for c in components]
    print(f"  Components: {comp_names}")

    # Extract loads per subcase
    na_data = []
    for i, sc in enumerate(result.subcases):
        trim_vars = sc.trim_variables if hasattr(sc, 'trim_variables') else {}
        nz = trim_vars.get('URDD3', 0) / 9810.0 if 'URDD3' in trim_vars else 1.0

        # Get nodal forces
        combined = {}
        if hasattr(sc, 'nodal_combined_forces') and sc.nodal_combined_forces:
            combined = sc.nodal_combined_forces
        elif hasattr(sc, 'nodal_aero_forces') and sc.nodal_aero_forces:
            # Build combined from aero + inertial
            aero = sc.nodal_aero_forces
            inertia = sc.nodal_inertial_forces if hasattr(sc, 'nodal_inertial_forces') else {}
            all_nids = set(aero.keys()) | set(inertia.keys())
            for nid in all_nids:
                combined[nid] = aero.get(nid, np.zeros(6)) + inertia.get(nid, np.zeros(6))

        # Total forces
        F_aero_z = sum(f[2] for f in (sc.nodal_aero_forces or {}).values()) if hasattr(sc, 'nodal_aero_forces') and sc.nodal_aero_forces else 0
        F_iner_z = sum(f[2] for f in (sc.nodal_inertial_forces or {}).values()) if hasattr(sc, 'nodal_inertial_forces') and sc.nodal_inertial_forces else 0

        # Mach/speed from trim card
        tid_list = list(model.trims.keys())
        V_eas = 0
        if i < len(tid_list):
            trim = model.trims[tid_list[i]]
            mach = trim.mach
            V_eas = mach * 340.3  # Approximate EAS at sea level

        label = f"SC{i+1} M={mach:.3f} nz={nz:.2f}"
        print(f"\n  {label}:")
        print(f"    Aero Fz={F_aero_z:.1f} N, Inertia Fz={F_iner_z:.1f} N, "
              f"nz={nz:.3f}, V≈{V_eas:.0f} m/s")
        print(f"    Trim: {trim_vars}")

        # VMT
        vmt_data = {}
        if combined:
            vmt_result = compute_vmt_all(model, combined, comp_set, n_stations=30)
            for curve in vmt_result.curves:
                cn = curve.component_name
                vmt_data[cn] = {
                    "shear": curve.shear, "bending": curve.bending_moment,
                    "torsion": curve.torsion, "stations": curve.stations}
                print(f"    {cn}: V_root={curve.shear[0]:>10.1f} N, "
                      f"M_root={curve.bending_moment[0]:>12.1f} N-mm, "
                      f"T_root={curve.torsion[0]:>12.1f} N-mm")
        else:
            print(f"    (no combined forces available)")

        na_data.append({"label": label, "nz": nz, "V_eas": V_eas,
                        "trim": trim_vars, "vmt": vmt_data,
                        "F_aero_z": F_aero_z, "F_iner_z": F_iner_z})

    return na_data, comp_names


# ═══════════════════════════════════════
# Comparison Output
# ═══════════════════════════════════════

def print_all(na_data, comp_names):
    print("\n" + "=" * 95)
    print("  FAR 23 LOADS (DARCorp) vs ASCENT-Load SOL 144 — GACOMP Structural Loads")
    print("=" * 95)

    # Representative cases for FAR 23
    far23_cases = [
        ("A+ (VA=62, 3.8g)", 3.8, VA),
        ("C+ (VC=80, 3.8g)", 3.8, VC),
        ("D+ (VD=100, 3.8g)", 3.8, VD),
        ("C- (VC=80, -1.52g)", -1.52, VC),
        ("Gust VC+ (4.53g)", 4.53, VC),
        ("1g cruise (VC=80)", 1.0, VC),
    ]

    # ── Wing Root ──
    print(f"\n{'─'*95}")
    print(f"  ■ WING ROOT LOADS (Right Semi-span) — FAR 23 LOADS Analytical")
    print(f"{'─'*95}")
    print(f"  {'Case':<25} {'nz':>5} {'V(m/s)':>7} │ {'Shear(N)':>10} "
          f"{'Bend(N-m)':>12} {'Tors(N-m)':>12} │ {'Tail(N)':>10}")
    print(f"  {'─'*92}")
    for label, nz, V in far23_cases:
        LT = far23_tail_load(nz, V)
        Vr, Mr, Tr = far23_wing_root(nz, V, LT)
        print(f"  {label:<25} {nz:>5.2f} {V:>7.0f} │ {Vr:>10.0f} "
              f"{Mr/1000:>12.0f} {Tr/1000:>12.0f} │ {LT:>10.0f}")

    # ── ASCENT-Load Results ──
    if na_data:
        print(f"\n{'─'*95}")
        print(f"  ■ WING ROOT LOADS — ASCENT-Load SOL 144 (from BDF subcases)")
        print(f"{'─'*95}")
        rw_key = None
        for cn in comp_names:
            if "Right" in cn and "Wing" in cn: rw_key = cn; break

        if rw_key:
            print(f"  {'Subcase':<30} {'nz':>5} {'V(m/s)':>7} │ {'Shear(N)':>10} "
                  f"{'Bend(N-m)':>12} {'Tors(N-m)':>12}")
            print(f"  {'─'*80}")
            for d in na_data:
                vmt = d["vmt"].get(rw_key, {})
                if vmt:
                    V = vmt["shear"][0]
                    M = vmt["bending"][0]
                    T = vmt["torsion"][0]
                    print(f"  {d['label']:<30} {d['nz']:>5.2f} {d['V_eas']:>7.0f} │ "
                          f"{V:>10.0f} {M/1000:>12.0f} {T/1000:>12.0f}")

        # ── HTP ──
        ht_key = None
        for cn in comp_names:
            if "HTP" in cn or "Horizontal" in cn:
                if "Right" in cn or "R " in cn: ht_key = cn; break
        if not ht_key:
            for cn in comp_names:
                if "HTP" in cn or "Horizontal" in cn: ht_key = cn; break

        if ht_key:
            print(f"\n{'─'*95}")
            print(f"  ■ HORIZONTAL TAIL LOADS — ASCENT-Load")
            print(f"{'─'*95}")
            print(f"  {'Subcase':<30} {'nz':>5} │ {'Shear(N)':>10} "
                  f"{'Bend(N-m)':>12} {'Tors(N-m)':>12}")
            print(f"  {'─'*72}")
            for d in na_data:
                vmt = d["vmt"].get(ht_key, {})
                if vmt:
                    V = vmt["shear"][0]
                    M = vmt["bending"][0]
                    T = vmt["torsion"][0]
                    print(f"  {d['label']:<30} {d['nz']:>5.2f} │ "
                          f"{V:>10.0f} {M/1000:>12.0f} {T/1000:>12.0f}")

        # ── VTP ──
        vt_key = None
        for cn in comp_names:
            if "VTP" in cn or "Vertical" in cn: vt_key = cn; break
        if vt_key:
            print(f"\n{'─'*95}")
            print(f"  ■ VERTICAL TAIL LOADS — ASCENT-Load")
            print(f"{'─'*95}")
            for d in na_data:
                vmt = d["vmt"].get(vt_key, {})
                if vmt:
                    V = vmt["shear"][0]; M = vmt["bending"][0]; T = vmt["torsion"][0]
                    print(f"  {d['label']:<30}: V={V:>8.0f} N, M={M/1000:>10.0f} N-m")

    # ── Scaling comparison ──
    if na_data:
        # Find 1g case in ASCENT-Load
        base = None
        for d in na_data:
            if abs(d["nz"] - 1.0) < 0.1: base = d; break

        if base and rw_key and rw_key in base["vmt"]:
            print(f"\n{'─'*95}")
            print(f"  ■ SCALED COMPARISON: ASCENT-Load 1g → nz×scaling vs FAR 23")
            print(f"{'─'*95}")
            V_1g = base["vmt"][rw_key]["shear"][0]
            M_1g = base["vmt"][rw_key]["bending"][0]

            print(f"  ASCENT-Load 1g wing root: V={V_1g:.0f} N, M={M_1g/1000:.0f} N-m")
            print(f"\n  {'Condition':<20} {'nz':>5} │ {'FAR23 V(N)':>12} {'NA×nz V(N)':>12} "
                  f"{'Ratio':>7} │ {'FAR23 M(N-m)':>14} {'NA×nz M(N-m)':>14} {'Ratio':>7}")
            print(f"  {'─'*95}")
            for label, nz, V in [("3.8g", 3.8, VC), ("-1.52g", -1.52, VC), ("4.53g gust", 4.53, VC)]:
                LT = far23_tail_load(nz, V)
                Vf, Mf, Tf = far23_wing_root(nz, V, LT)
                V_sc = V_1g * nz
                M_sc = M_1g * nz
                rV = V_sc / Vf if abs(Vf) > 1 else 0
                rM = M_sc / (Mf/1000) if abs(Mf) > 1000 else 0
                print(f"  {label:<20} {nz:>5.2f} │ {Vf:>12.0f} {V_sc:>12.0f} "
                      f"{rV:>7.2f} │ {Mf/1000:>14.0f} {M_sc/1000:>14.0f} {rM:>7.2f}")

    # ── Aileron ──
    print(f"\n{'─'*95}")
    print(f"  ■ AILERON LOADS — FAR 23 LOADS")
    print(f"{'─'*95}")
    for name, V in [("VA", VA), ("VC", VC), ("VD", VD)]:
        L, d = far23_aileron(V)
        print(f"  {name} ({V:.0f} m/s): δ={d:.1f}°, L={L:.0f} N ({L/G:.0f} kgf)")

    # ── Landing ──
    print(f"\n{'─'*95}")
    print(f"  ■ LANDING LOADS — FAR 23 LOADS (§23.473-23.499)")
    print(f"{'─'*95}")
    ldg = far23_landing()
    print(f"  V_sink={ldg['V_sink']:.2f} fps, nz_air={ldg['nz']:.3f}, nz_gear={ldg['nz_gear']:.3f}")
    print(f"\n  {'Condition':<24} {'Vert(N)':>8} {'Drag(N)':>8} {'Side(N)':>8} │ {'Vert(kgf)':>10}")
    print(f"  {'─'*62}")
    rows = [
        ("Level main (each)", ldg["main_V"], ldg["main_D"], 0),
        ("Level nose", ldg["nose_V"], 0, 0),
        ("Side (§23.485)", ldg["side_V"], 0, ldg["side_S"]),
        ("One-wheel (§23.483)", ldg["onewheel"], 0, 0),
        ("Braked roll (§23.493)", ldg["brake_V"], ldg["brake_D"], 0),
    ]
    for name, v, d, s in rows:
        print(f"  {name:<24} {v:>8.0f} {d:>8.0f} {s:>8.0f} │ {v/G:>10.0f}")

    # ── Critical Summary ──
    print(f"\n{'═'*95}")
    print(f"  ■ CRITICAL DESIGN LOADS SUMMARY")
    print(f"{'═'*95}")

    # FAR 23 critical
    max_V = max_M = max_T = max_LT = 0
    cv = cm = ct = clt = ""
    for label, nz, V in far23_cases:
        LT = far23_tail_load(nz, V)
        Vr, Mr, Tr = far23_wing_root(nz, V, LT)
        if abs(Vr) > abs(max_V): max_V = Vr; cv = label
        if abs(Mr) > abs(max_M): max_M = Mr; cm = label
        if abs(Tr) > abs(max_T): max_T = Tr; ct = label
        if abs(LT) > abs(max_LT): max_LT = LT; clt = label

    print(f"\n  FAR 23 LOADS (Analytical):")
    print(f"    Wing max shear:    {max_V:>10.0f} N  = {max_V/G:>8.0f} kgf  ← {cv}")
    print(f"    Wing max bending:  {max_M/1000:>10.0f} N-m = {max_M/G/1000:>8.1f} kgf-m ← {cm}")
    print(f"    Wing max torsion:  {max_T/1000:>10.0f} N-m = {max_T/G/1000:>8.1f} kgf-m ← {ct}")
    print(f"    Max tail load:     {max_LT:>10.0f} N  = {max_LT/G:>8.0f} kgf  ← {clt}")
    print(f"    Landing main gear: {ldg['main_V']:>10.0f} N  = {ldg['main_V']/G:>8.0f} kgf")
    print(f"    Landing brake:     {ldg['brake_D']:>10.0f} N  = {ldg['brake_D']/G:>8.0f} kgf")

    # ASCENT-Load critical
    if na_data and rw_key:
        max_V_n = max_M_n = max_T_n = 0
        cv_n = cm_n = ct_n = ""
        for d in na_data:
            vmt = d["vmt"].get(rw_key, {})
            if vmt:
                if abs(vmt["shear"][0]) > abs(max_V_n):
                    max_V_n = vmt["shear"][0]; cv_n = d["label"]
                if abs(vmt["bending"][0]) > abs(max_M_n):
                    max_M_n = vmt["bending"][0]; cm_n = d["label"]
                if abs(vmt["torsion"][0]) > abs(max_T_n):
                    max_T_n = vmt["torsion"][0]; ct_n = d["label"]

        print(f"\n  ASCENT-Load SOL 144:")
        print(f"    Wing max shear:    {max_V_n:>10.0f} N  = {max_V_n/G:>8.0f} kgf  ← {cv_n}")
        print(f"    Wing max bending:  {max_M_n/1000:>10.0f} N-m = {max_M_n/G/1000:>8.1f} kgf-m ← {cm_n}")
        print(f"    Wing max torsion:  {max_T_n/1000:>10.0f} N-m = {max_T_n/G/1000:>8.1f} kgf-m ← {ct_n}")

        # Ratio
        if abs(max_V) > 1:
            print(f"\n  Ratio (ASCENT-Load / FAR 23):")
            print(f"    Wing shear:  {max_V_n/max_V:.3f}")
            print(f"    Wing bending: {max_M_n/max_M:.3f}")

    print(f"\n{'═'*95}")


if __name__ == "__main__":
    print("=" * 95)
    print("  GACOMP Structural Loads Analysis: FAR 23 LOADS vs ASCENT-Load SOL 144")
    print("=" * 95)

    na_data = []; comp_names = []
    try:
        na_data, comp_names = run_ascent_load()
    except Exception as e:
        print(f"\n  ASCENT-Load failed: {e}")
        import traceback; traceback.print_exc()

    print_all(na_data, comp_names)
