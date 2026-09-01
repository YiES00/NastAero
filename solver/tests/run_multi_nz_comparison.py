#!/usr/bin/env python3
"""Multi-nz trim comparison: Run GACOMP at 1g, 3.8g, -1.52g, 4.53g.

Creates a modified BDF with URDD3 as AESTAT and multiple TRIM cards,
then runs SOL 144 and compares VMT with FAR 23 LOADS.
"""
import os, sys, re, math, time, tempfile
import numpy as np

G = 9.80665; RHO0 = 1.225
W_KG = 1288.9; W_N = W_KG * G
S_M2 = 17.0; B_M = 11.233; MAC_M = 1.6
XCG = 3882.0; XW = 3500.0; XT = 8500.0; CMAC_WF = -0.05
TAIL_ARM = XT - XCG


def create_multi_nz_bdf(original_bdf, output_bdf):
    """Create modified BDF with multiple nz TRIM cases."""
    with open(original_bdf, 'r') as f:
        text = f.read()

    # Remove original SUBCASE and TRIM sections, replace with new ones
    # Replace case control: remove old SUBCASEs, add new ones
    new_case_control = """$
TITLE      =GACOMP MULTI-NZ TRIM COMPARISON
$
ECHO       =BOTH
MPC = 999
SPC        = 2
DISP       =ALL
$
SUBCASE 1
  SUBTITLE =M0.182 NZ=1.00 (1g LEVEL FLIGHT)
  TRIM     = 101
SUBCASE 2
  SUBTITLE =M0.182 NZ=3.80 (POINT A+ MANEUVER)
  TRIM     = 102
SUBCASE 3
  SUBTITLE =M0.182 NZ=-1.52 (NEGATIVE MANEUVER)
  TRIM     = 103
SUBCASE 4
  SUBTITLE =M0.235 NZ=3.80 (POINT C+ AT VC)
  TRIM     = 104
SUBCASE 5
  SUBTITLE =M0.235 NZ=4.53 (GUST VC+)
  TRIM     = 105
SUBCASE 6
  SUBTITLE =M0.235 NZ=-2.53 (GUST VC-)
  TRIM     = 106
SUBCASE 7
  SUBTITLE =M0.294 NZ=3.80 (POINT D+ AT VD)
  TRIM     = 107
"""
    # Replace everything between CEND and BEGIN BULK
    text = re.sub(
        r'CEND\n.*?BEGIN BULK',
        f'CEND\n{new_case_control}$\nBEGIN BULK',
        text, flags=re.DOTALL)

    # Remove old TRIM and add new ones before ENDDATA
    # Remove lines starting with TRIM or +TR
    lines = text.split('\n')
    new_lines = []
    skip_trim = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('TRIM') and not stripped.startswith('TITLE') and '=' not in stripped:
            skip_trim = True
            continue
        if skip_trim and stripped.startswith('+TR'):
            continue
        skip_trim = False
        if stripped == 'ENDDATA':
            # Add AESTAT for URDD3 and new TRIM cards before ENDDATA
            new_lines.append('$')
            new_lines.append('$ ADDED: URDD3 as trim variable for load factor control')
            new_lines.append('AESTAT  1720    URDD3')
            new_lines.append('$')
            new_lines.append('$ MULTI-NZ TRIM CONDITIONS')
            new_lines.append('$ VA=62 m/s → M=0.182, VC=80 m/s → M=0.235, VD=100 → M=0.294')
            new_lines.append('$')

            # q = 0.5 * rho * V^2 in N/mm^2 (MPa)
            # M=0.182: V=62 m/s, q = 0.5*1.225*62^2 = 2356 Pa = 2.356e-3 MPa
            # M=0.235: V=80 m/s, q = 0.5*1.225*80^2 = 3920 Pa = 3.920e-3 MPa
            # M=0.294: V=100 m/s, q = 0.5*1.225*100^2 = 6125 Pa = 6.125e-3 MPa

            def nastran_float(val):
                """Format float in Nastran 8-char field notation."""
                if val == 0.0:
                    return "0.0     "
                exp = int(math.floor(math.log10(abs(val))))
                mantissa = val / (10 ** exp)
                if exp >= 0:
                    return f"{val:<8.5f}"[:8]
                else:
                    return f"{mantissa:.4f}{exp:d}"[:8].ljust(8)

            trim_cases = [
                (101, 0.182, 2.356e-3, 1.00,  "1g at VA"),
                (102, 0.182, 2.356e-3, 3.80,  "3.8g at VA (A+)"),
                (103, 0.182, 2.356e-3, -1.52, "-1.52g at VA"),
                (104, 0.235, 3.920e-3, 3.80,  "3.8g at VC (C+)"),
                (105, 0.235, 3.920e-3, 4.53,  "4.53g gust at VC"),
                (106, 0.235, 3.920e-3, -2.53, "-2.53g gust at VC"),
                (107, 0.294, 6.125e-3, 3.80,  "3.8g at VD (D+)"),
            ]

            for tid, mach, q, nz, comment in trim_cases:
                # Use comma-separated free format for reliability
                new_lines.append(f'$ TRIM {tid}: {comment}')
                new_lines.append(
                    f'TRIM,{tid},{mach:.5f},{q:.6e},'
                    f'ROLL,0.0,YAW,0.0,+T{tid}')
                new_lines.append(
                    f'+T{tid},URDD2,0.0,URDD3,{nz:.4f},'
                    f'URDD4,0.0,URDD6,0.0,+T{tid}a')
                new_lines.append(
                    f'+T{tid}a,ARON,0.0,RUD,0.0')
                new_lines.append('$')

        new_lines.append(line)

    with open(output_bdf, 'w') as f:
        f.write('\n'.join(new_lines))

    print(f"  Created: {output_bdf}")
    return trim_cases


def far23_wing_root(nz, V_eas):
    """FAR 23 analytical wing root loads."""
    q = 0.5 * RHO0 * V_eas**2
    M_acwf = CMAC_WF * q * S_M2 * MAC_M * 1000
    LT = -(M_acwf + nz * W_N * (XCG - XW)) / TAIL_ARM
    L_wing = nz * W_N - LT
    L_semi = L_wing / 2.0
    W_relief = nz * 0.20 * W_N / 2.0
    V_root = L_semi - W_relief
    b_semi = B_M / 2.0
    y_aero = 4 * b_semi / (3 * math.pi)
    y_mass = b_semi * 0.38
    M_root = (L_semi * y_aero - W_relief * y_mass) * 1000
    T_root = -0.08 * q * S_M2/2 * MAC_M * 1000 + L_semi * 0.15 * MAC_M * 1000
    return V_root, M_root, T_root, LT


def run():
    from ascent_load.bdf.parser import BDFParser
    from ascent_load.solvers.sol144 import solve_trim
    from ascent_load.loads_analysis.vmt import compute_vmt
    from ascent_load.loads_analysis.component_id import identify_components

    orig_bdf = os.path.join(os.path.dirname(__file__),
        "validation", "GACOMP", "p400r3-free-trim.bdf")

    # Create modified BDF
    mod_bdf = os.path.join(os.path.dirname(__file__),
        "validation", "GACOMP", "p400r3-multi-nz.bdf")

    print("=" * 95)
    print("  GACOMP Multi-nz Trim: FAR 23 LOADS vs ASCENT-Load SOL 144")
    print("=" * 95)

    print("\n  Creating multi-nz BDF...")
    trim_defs = create_multi_nz_bdf(orig_bdf, mod_bdf)

    print("  Parsing modified BDF...")
    parser = BDFParser()
    model = parser.parse(mod_bdf)
    print(f"  {len(model.nodes)} nodes, {len(model.elements)} elements, "
          f"{len(model.trims)} TRIM cards")

    print("  Solving SOL 144 (7 subcases)...")
    t0 = time.time()
    result = solve_trim(model)
    dt = time.time() - t0
    print(f"  Solved in {dt:.1f}s, {len(result.subcases)} subcases")

    # Components
    comp_set = identify_components(model)
    rw = comp_set.get("Right Wing")
    rht = comp_set.get("Right HTP")
    lw = comp_set.get("Left Wing")

    if not rw:
        print("ERROR: Right Wing not found"); return

    # ── Results table ──
    case_labels = [
        ("1g (VA=62)",     1.00,  62.0),
        ("3.8g (A+, VA)",  3.80,  62.0),
        ("-1.52g (VA)",   -1.52,  62.0),
        ("3.8g (C+, VC)",  3.80,  80.0),
        ("4.53g gust VC+", 4.53,  80.0),
        ("-2.53g gust VC-",-2.53, 80.0),
        ("3.8g (D+, VD)",  3.80, 100.0),
    ]

    print(f"\n{'═'*95}")
    print(f"  RIGHT WING ROOT LOADS — ASCENT-Load SOL 144 vs FAR 23 LOADS")
    print(f"{'═'*95}")
    print(f"  {'Case':<22} {'nz':>5} │ {'NA Shear':>10} {'F23 Shear':>10} {'Ratio':>7}"
          f" │ {'NA Bend':>12} {'F23 Bend':>12} {'Ratio':>7}")
    print(f"  {'':22} {'':>5} │ {'(N)':>10} {'(N)':>10} {'':>7}"
          f" │ {'(N-m)':>12} {'(N-m)':>12} {'':>7}")
    print(f"  {'─'*92}")

    na_results = []

    for i, sc in enumerate(result.subcases):
        trim_vars = sc.trim_variables if hasattr(sc, 'trim_variables') else {}
        nz_actual = trim_vars.get('URDD3', 1.0)
        if nz_actual == 0: nz_actual = 1.0
        alpha = trim_vars.get('ANGLEA', 0)
        elev = trim_vars.get('ELEV', 0)

        label, nz_expected, V_eas = case_labels[i]

        # Get combined forces
        combined = {}
        if hasattr(sc, 'nodal_combined_forces') and sc.nodal_combined_forces:
            combined = sc.nodal_combined_forces
        elif hasattr(sc, 'nodal_aero_forces') and sc.nodal_aero_forces:
            aero = sc.nodal_aero_forces
            inertia = sc.nodal_inertial_forces or {}
            for nid in set(aero) | set(inertia):
                combined[nid] = aero.get(nid, np.zeros(6)) + inertia.get(nid, np.zeros(6))

        # VMT for Right Wing
        if combined:
            curve = compute_vmt(model, combined, rw, n_stations=30)
            V_na = curve.shear[0]
            M_na = curve.bending_moment[0] / 1000  # N-mm → N-m
            T_na = curve.torsion[0] / 1000

            # Also compute for HTP
            curve_ht = compute_vmt(model, combined, rht, n_stations=20) if rht else None
        else:
            V_na = M_na = T_na = 0
            curve_ht = None

        # FAR 23 analytical
        V_f, M_f, T_f, LT_f = far23_wing_root(nz_expected, V_eas)
        M_f /= 1000; T_f /= 1000  # to N-m

        rV = V_na / V_f if abs(V_f) > 1 else 0
        rM = M_na / M_f if abs(M_f) > 1 else 0

        print(f"  {label:<22} {nz_actual:>5.2f} │ {V_na:>10.0f} {V_f:>10.0f} {rV:>7.3f}"
              f" │ {M_na:>12.0f} {M_f:>12.0f} {rM:>7.3f}")

        na_results.append({
            "label": label, "nz": nz_actual, "V_eas": V_eas,
            "alpha_deg": math.degrees(alpha), "elev_deg": math.degrees(elev),
            "V_na": V_na, "M_na": M_na, "T_na": T_na,
            "V_f23": V_f, "M_f23": M_f, "T_f23": T_f,
            "LT_f23": LT_f,
            "ht_shear": curve_ht.shear[0] if curve_ht else 0,
        })

    # ── Torsion comparison ──
    print(f"\n{'─'*95}")
    print(f"  TORSION (EA=40% chord) & TAIL LOAD")
    print(f"{'─'*95}")
    print(f"  {'Case':<22} {'nz':>5} │ {'NA Tors':>10} {'F23 Tors':>10} {'Ratio':>7}"
          f" │ {'NA HTP V':>10} {'F23 LT':>10} {'Ratio':>7}")
    print(f"  {'':22} {'':>5} │ {'(N-m)':>10} {'(N-m)':>10} {'':>7}"
          f" │ {'(N)':>10} {'(N)':>10} {'':>7}")
    print(f"  {'─'*92}")
    for r in na_results:
        rT = r["T_na"] / r["T_f23"] if abs(r["T_f23"]) > 1 else 0
        rLT = r["ht_shear"] / r["LT_f23"] if abs(r["LT_f23"]) > 1 else 0
        print(f"  {r['label']:<22} {r['nz']:>5.2f} │ {r['T_na']:>10.0f} {r['T_f23']:>10.0f}"
              f" {rT:>7.3f} │ {r['ht_shear']:>10.0f} {r['LT_f23']:>10.0f} {rLT:>7.3f}")

    # ── Trim variables ──
    print(f"\n{'─'*95}")
    print(f"  TRIM VARIABLES (solved)")
    print(f"{'─'*95}")
    print(f"  {'Case':<22} {'nz':>5} {'Alpha(°)':>9} {'Elev(°)':>9}")
    print(f"  {'─'*48}")
    for r in na_results:
        print(f"  {r['label']:<22} {r['nz']:>5.2f} {r['alpha_deg']:>9.2f} {r['elev_deg']:>9.2f}")

    # ── Critical Design Loads ──
    print(f"\n{'═'*95}")
    print(f"  CRITICAL DESIGN LOADS")
    print(f"{'═'*95}")

    # Find max |shear|, |bending|, |torsion|
    max_V_na = max(na_results, key=lambda r: abs(r["V_na"]))
    max_M_na = max(na_results, key=lambda r: abs(r["M_na"]))
    max_T_na = max(na_results, key=lambda r: abs(r["T_na"]))
    max_V_f = max(na_results, key=lambda r: abs(r["V_f23"]))
    max_M_f = max(na_results, key=lambda r: abs(r["M_f23"]))

    print(f"\n  {'Quantity':<25} {'ASCENT-Load':>14} {'Case':>22} │ {'FAR 23':>14} {'Case':>22}")
    print(f"  {'─'*100}")
    print(f"  {'Wing Root Shear (N)':<25} {max_V_na['V_na']:>14.0f} {max_V_na['label']:>22}"
          f" │ {max_V_f['V_f23']:>14.0f} {max_V_f['label']:>22}")
    print(f"  {'Wing Root Bending (N-m)':<25} {max_M_na['M_na']:>14.0f} {max_M_na['label']:>22}"
          f" │ {max_M_f['M_f23']:>14.0f} {max_M_f['label']:>22}")
    max_T_f = max(na_results, key=lambda r: abs(r['T_f23']))
    print(f"  {'Wing Root Torsion (N-m)':<25} {max_T_na['T_na']:>14.0f} {max_T_na['label']:>22}"
          f" │ {max_T_f['T_f23']:>14.0f} {max_T_f['label']:>22}")

    ratio_V = max_V_na["V_na"] / max_V_f["V_f23"] if abs(max_V_f["V_f23"]) > 1 else 0
    ratio_M = max_M_na["M_na"] / max_M_f["M_f23"] if abs(max_M_f["M_f23"]) > 1 else 0
    print(f"\n  Critical load ratio (ASCENT-Load / FAR 23):")
    print(f"    Shear:   {ratio_V:.3f}")
    print(f"    Bending: {ratio_M:.3f}")

    print(f"\n{'═'*95}")

    # Cleanup
    try: os.unlink(mod_bdf)
    except: pass


if __name__ == "__main__":
    run()
