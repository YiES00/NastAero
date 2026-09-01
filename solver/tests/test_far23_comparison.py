"""FAR 23 LOADS (DARCorp) vs ASCENT-Load comparison for GACOMP.

Compares the analytical FAR 23 LOADS methodology (Hal C. McMaster)
against ASCENT-Load's V-n diagram and load case matrix for the GACOMP aircraft.
"""
from __future__ import annotations

import os
import sys
import numpy as np

# ── GACOMP Parameters ──
# Source: ASCENT-Load test_gacomp_cert_e2e.py + comparison-model public specifications
GACOMP_NAME = "comparison model (conventional GA aircraft)"
GACOMP = {
    "W_kg": 1288.9,          # MTOW (kg), from CONM2 sum
    "S_m2": 17.0,            # Wing area (m^2)
    "b_m": 11.233,           # Wing span (m), 2 * half-span 5616.6mm
    "MAC_m": 1.6,            # Mean aerodynamic chord (m)
    "CLalpha": 5.5,          # Lift curve slope (/rad)
    "VS1": 33.0,             # Stall speed clean (m/s)
    "VC": 80.0,              # Cruise speed (m/s)
    "VD": 100.0,             # Dive speed (m/s)
    "VF": 40.0,              # Flap speed (m/s)
    "category": "normal",
    "Ude_VC_fps": 50.0,      # Gust velocity at VC (fps)
    "Ude_VD_fps": 25.0,      # Gust velocity at VD (fps)
    # Geometry (approximate from BDF model)
    "XCG_m": 3.882,          # CG x-position
    "XW_m": 3.5,             # Wing AC (approx 25% MAC from LE)
    "XT_m": 8.5,             # Tail AC (approx)
    "Cmac_wf": -0.05,        # Pitching moment coeff
    "CD0": 0.03,
    # Landing gear
    "strut_efficiency": 0.7,
    "tire_efficiency": 0.47,
    "strut_stroke_m": 0.25,
    "tire_stroke_m": 0.05,
    "altitude_m": 0.0,
}


def run_far23_analysis():
    """Run FAR 23 LOADS analytical method for GACOMP."""
    from ascent_load.loads_analysis.far23_analytical import (
        FAR23Config, compute_far23_loads
    )

    cfg = FAR23Config(**GACOMP)
    results = compute_far23_loads(cfg)
    return cfg, results


def run_ascent_load_analysis():
    """Run ASCENT-Load V-n diagram and load case matrix for GACOMP."""
    from ascent_load.loads_analysis.certification.aircraft_config import (
        AircraftConfig, SpeedSchedule, WeightCGCondition,
        ControlSurfaceLimits, LandingGearConfig,
        part23_nz_max,
    )
    from ascent_load.loads_analysis.certification.vn_diagram import (
        compute_vn_diagram,
    )

    speeds = SpeedSchedule(
        VS1=33.0, VA=62.0, VC=80.0, VD=100.0, VF=40.0
    )

    weight_cg = WeightCGCondition(
        label="MTOW_mid",
        weight_N=1288.9 * 9.80665,
        cg_x=3882.0,
    )

    ctrl_limits = ControlSurfaceLimits(
        aileron_max_deg=20.0,
        rudder_max_deg=25.0,
        elevator_max_deg=25.0,
    )

    landing_gear = LandingGearConfig(
        main_gear_node_ids=[100, 101],
        nose_gear_node_ids=[102],
        main_gear_x=4200.0,
        nose_gear_x=1500.0,
        strut_efficiency=0.7,
        stroke=250.0,   # mm (model units)
        sink_rate_fps=10.0,
    )

    config = AircraftConfig(
        speeds=speeds,
        weight_cg_conditions=[weight_cg],
        altitudes_m=[0.0],
        wing_area_m2=17.0,
        CLalpha=5.5,
        mean_chord_m=1.6,
        ctrl_limits=ctrl_limits,
        landing_gear=landing_gear,
        gust_Ude_VC_fps=50.0,
        gust_Ude_VD_fps=25.0,
    )

    vn = compute_vn_diagram(config, weight_cg, altitude_m=0.0)

    return config, vn, weight_cg


def print_comparison():
    """Print detailed comparison between FAR 23 LOADS and ASCENT-Load."""
    cfg_f, far23 = run_far23_analysis()
    config_n, vn_n, wt = run_ascent_load_analysis()

    W_lb = cfg_f.W_lb
    WS_psf = cfg_f.WS_psf
    WS_pa = cfg_f.WS_pa

    print("=" * 80)
    print("FAR 23 LOADS (DARCorp) vs ASCENT-Load Comparison — GACOMP")
    print("=" * 80)

    # ── Aircraft Data ──
    print(f"\n■ Aircraft: {GACOMP_NAME}")
    print(f"  MTOW = {cfg_f.W_kg:.1f} kg = {W_lb:.1f} lb")
    print(f"  Wing Area = {cfg_f.S_m2:.1f} m² = {cfg_f.S_ft2:.1f} ft²")
    print(f"  W/S = {WS_pa:.1f} Pa = {WS_psf:.2f} psf")
    print(f"  Span = {cfg_f.b_m:.3f} m, MAC = {cfg_f.MAC_m:.2f} m")
    print(f"  AR = {cfg_f.AR:.2f}")
    print(f"  CLα = {cfg_f.CLalpha:.1f} /rad")
    print(f"  Category: {cfg_f.category}")

    # ── Design Speeds ──
    print(f"\n{'─'*60}")
    print(f"■ Design Speeds (§23.335)")
    print(f"{'─'*60}")
    print(f"  {'Speed':<8} {'FAR23':>10} {'ASCENT-Load':>10} {'Diff':>8}  Unit")
    print(f"  {'─'*50}")
    speed_pairs = [
        ("VS1", cfg_f.VS1, config_n.speeds.VS1),
        ("VA", far23.VA, config_n.speeds.VA),
        ("VC", cfg_f.VC, config_n.speeds.VC),
        ("VD", cfg_f.VD, config_n.speeds.VD),
        ("VF", cfg_f.VF, config_n.speeds.VF),
    ]
    for name, f23, na in speed_pairs:
        diff = f23 - na
        print(f"  {name:<8} {f23:>10.2f} {na:>10.2f} {diff:>+8.2f}  m/s")

    # ── Load Factors ──
    print(f"\n{'─'*60}")
    print(f"■ Maneuvering Load Factors (§23.337)")
    print(f"{'─'*60}")
    print(f"  {'Parameter':<20} {'FAR23':>10} {'ASCENT-Load':>10} {'Diff':>8}")
    print(f"  {'─'*50}")

    nz_max_na = vn_n.nz_max
    nz_min_na = vn_n.nz_min
    print(f"  {'nz_max':<20} {far23.nz_max:>10.4f} {nz_max_na:>10.4f} "
          f"{far23.nz_max - nz_max_na:>+8.4f}")
    print(f"  {'nz_min':<20} {far23.nz_min:>10.4f} {nz_min_na:>10.4f} "
          f"{far23.nz_min - nz_min_na:>+8.4f}")

    # Show intermediate calculation
    n_formula = 2.1 + 24000.0 / (W_lb + 10000.0)
    print(f"\n  Calculation: n = 2.1 + 24000/(W+10000)")
    print(f"             = 2.1 + 24000/({W_lb:.1f}+10000)")
    print(f"             = 2.1 + {24000/(W_lb+10000):.4f} = {n_formula:.4f}")
    print(f"             min(3.8, {n_formula:.4f}) → nz_max = {far23.nz_max:.4f}")

    # ── Gust Parameters ──
    print(f"\n{'─'*60}")
    print(f"■ Gust Load Factors (§23.341, Pratt Formula)")
    print(f"{'─'*60}")
    print(f"  Mass ratio μg = 2*(W/S) / (ρ*MAC*CLα*g)")
    print(f"               = 2*{WS_pa:.1f} / ({1.225:.3f}*{cfg_f.MAC_m:.2f}"
          f"*{cfg_f.CLalpha:.1f}*{9.807:.3f})")
    print(f"               = {far23.mu_g_VC:.4f}")
    print(f"  Kg = 0.88*μg / (5.3+μg) = 0.88*{far23.mu_g_VC:.4f} / "
          f"(5.3+{far23.mu_g_VC:.4f}) = {far23.Kg_VC:.4f}")
    print(f"\n  Δn = ρ₀*V*CLα*Kg*Ude / (2*W/S)")

    gust_data = [
        ("VC (50fps)", cfg_f.VC, far23.delta_n_gust_VC),
        ("VD (25fps)", cfg_f.VD, far23.delta_n_gust_VD),
        ("VB (50fps)", far23.VB, far23.delta_n_gust_VB),
    ]

    # Get ASCENT-Load gust points
    na_gust = {}
    for pt in vn_n.corner_points:
        if "Gust" in pt.label:
            na_gust[pt.label] = pt

    print(f"\n  {'Condition':<18} {'V(m/s)':>8} {'Δn_FAR23':>10} "
          f"{'n+':>8} {'n-':>8} {'n+_NA':>8} {'n-_NA':>8}")
    print(f"  {'─'*70}")

    for label, V, dn in gust_data:
        n_pos = 1.0 + dn
        n_neg = 1.0 - dn
        speed_label = label.split()[0]

        na_pos = na_gust.get(f"Gust_{speed_label}+")
        na_neg = na_gust.get(f"Gust_{speed_label}-")
        na_np = na_pos.nz if na_pos else 0
        na_nn = na_neg.nz if na_neg else 0

        print(f"  {label:<18} {V:>8.2f} {dn:>10.4f} "
              f"{n_pos:>8.4f} {n_neg:>8.4f} {na_np:>8.4f} {na_nn:>8.4f}")

    # ── V-n Diagram Points ──
    print(f"\n{'─'*60}")
    print(f"■ V-n Diagram Corner Points")
    print(f"{'─'*60}")
    print(f"  {'Label':<14} {'V_FAR23':>8} {'nz_FAR23':>10} "
          f"{'V_NA':>8} {'nz_NA':>10} {'Δnz':>8}")
    print(f"  {'─'*60}")

    label_map = {
        "A+": "A+", "C+": "C+", "D+": "D+",
        "C-": "C-", "D-": "D-", "G-": "A-",
        "Gust_VB+": "Gust_VB+", "Gust_VB-": "Gust_VB-",
        "Gust_VC+": "Gust_VC+", "Gust_VC-": "Gust_VC-",
        "Gust_VD+": "Gust_VD+", "Gust_VD-": "Gust_VD-",
        "Flap+": "Flap+", "Flap0": "Flap0",
    }

    for pt in far23.vn_points:
        na_label = label_map.get(pt.label, pt.label)
        na_match = None
        for na_pt in vn_n.corner_points:
            if na_pt.label == na_label:
                na_match = na_pt
                break

        if na_match:
            dn = pt.nz - na_match.nz
            print(f"  {pt.label:<14} {pt.V_eas_ms:>8.2f} {pt.nz:>10.4f} "
                  f"{na_match.V_eas:>8.2f} {na_match.nz:>10.4f} {dn:>+8.4f}")
        else:
            print(f"  {pt.label:<14} {pt.V_eas_ms:>8.2f} {pt.nz:>10.4f} "
                  f"{'—':>8} {'—':>10} {'—':>8}")

    # ── Tail Loads ──
    if far23.tail_loads:
        print(f"\n{'─'*60}")
        print(f"■ Balanced Tail Loads (§23.421)")
        print(f"{'─'*60}")
        print(f"  Tail arm = XT - XCG = {cfg_f.XT_m:.3f} - {cfg_f.XCG_m:.3f}"
              f" = {cfg_f.XT_m - cfg_f.XCG_m:.3f} m")
        print(f"  LT = [Macwf + L*(XCG-XW)] / (XT-XCG)")
        print(f"\n  {'Condition':<25} {'nz':>6} {'V(m/s)':>8} {'LT(N)':>10} {'LT(kgf)':>10}")
        print(f"  {'─'*60}")
        for label, LT in far23.tail_loads.items():
            parts = label.split(",")
            nz_str = parts[0].split("(")[1] if "(" in parts[0] else ""
            print(f"  {label:<25} {'':>6} {'':>8} {LT:>10.1f} {LT/9.807:>10.1f}")

    # ── Landing Loads ──
    print(f"\n{'─'*60}")
    print(f"■ Landing Loads (§23.473)")
    print(f"{'─'*60}")
    print(f"  V_sink = 4.4*(W/S)^0.25 = 4.4*{WS_psf:.2f}^0.25"
          f" = {far23.V_sink_fps:.2f} fps"
          f" (clamped to [{max(7,min(10,far23.V_sink_fps)):.1f}])")
    print(f"  Strut eff = {cfg_f.strut_efficiency}, stroke = {cfg_f.strut_stroke_m} m")
    print(f"  Tire eff = {cfg_f.tire_efficiency}, stroke = {cfg_f.tire_stroke_m} m")
    print(f"  nz_landing = {far23.nz_landing:.4f}")

    # ── Wing Root Loads ──
    if far23.wing_root_shear_N:
        print(f"\n{'─'*60}")
        print(f"■ Wing Root Loads (Semi-span, Simplified Elliptic)")
        print(f"{'─'*60}")
        print(f"  {'Condition':<12} {'Shear(N)':>12} {'Shear(kgf)':>12} "
              f"{'Moment(Nm)':>12} {'Moment(kgf.m)':>14}")
        print(f"  {'─'*60}")
        for label in far23.wing_root_shear_N:
            V = far23.wing_root_shear_N[label]
            M = far23.wing_root_moment_Nm[label]
            print(f"  {label:<12} {V:>12.1f} {V/9.807:>12.1f} "
                  f"{M:>12.1f} {M/9.807:>14.1f}")

    print(f"\n{'='*80}")
    print("비교 완료")
    print(f"{'='*80}")


if __name__ == "__main__":
    print_comparison()
