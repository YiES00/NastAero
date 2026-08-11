#!/usr/bin/env python3
"""Run all FAR 23 dynamic maneuver simulations at multiple altitudes.

For each altitude / maneuver / gust case, generates time-transient plots of:
 - Control surface inputs (δe, δa, δr)
 - Attitude angles (φ, θ, ψ)
 - Body-axis velocities (u, v, w) and total airspeed
 - Angle of attack (α), sideslip (β)
 - Angular rates (p, q, r)
 - Angular accelerations (ṗ, q̇, ṙ)
 - Load factors (nz, ny)
 - Earth position (xe, ye, ze)

Altitude effects per FAR 23.333(c):
 - ISA atmospheric density ρ(h) for aerodynamic forces
 - TAS = EAS × √(ρ₀/ρ) for correct flight speed
 - Gust Ude linearly reduced above 20,000 ft
 - Pratt mass ratio μg uses ρ(h), not ρ₀

Flags any anomalous results (divergence, excessive load factors, NaN).
"""
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── NastAero imports ──────────────────────────────────────────────────────
from nastaero.bdf.parser import parse_bdf
from nastaero.config import setup_logging
from nastaero.loads_analysis.certification.aircraft_config import (
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    ControlSurfaceLimits, LandingGearConfig, eas_to_tas, eas_to_mach,
    gust_Ude_at_altitude,
)
from nastaero.loads_analysis.certification.aero_derivatives import (
    build_derivative_set, compute_inertia_from_conm2,
)
from nastaero.loads_analysis.certification.flight_sim import (
    AircraftParams, integrate_6dof, trim_initial_state,
    compute_nz_from_history,
)
from nastaero.loads_analysis.certification.maneuver_profiles import (
    generate_all_maneuver_profiles,
)
from nastaero.loads_analysis.case_generator import isa_atmosphere
from nastaero.aero.dlm import compute_rigid_clalpha

# ── Altitude analysis grid ───────────────────────────────────────────────
# 6 altitudes from SL to 25,000 ft service ceiling
ANALYSIS_ALTITUDES = [
    (0,     "SL"),       # Sea level
    (1524,  "5kft"),     # 5,000 ft
    (3048,  "10kft"),    # 10,000 ft
    (4572,  "15kft"),    # 15,000 ft
    (6096,  "20kft"),    # 20,000 ft — Ude reduction boundary
    (7620,  "25kft"),    # 25,000 ft — service ceiling
]

M_PER_FT = 0.3048


# ── Anomaly detection ─────────────────────────────────────────────────────

def check_anomalies(label, history, nz_limit=8.0, alpha_limit=20.0,
                    phi_limit=60.0, beta_limit=25.0):
    """Check for non-physical behaviour and return list of warnings."""
    warnings = []
    t = history.t

    # NaN / Inf check
    if np.any(np.isnan(history.states)) or np.any(np.isinf(history.states)):
        warnings.append("CRITICAL: NaN/Inf in state vector — simulation diverged")
        return warnings

    # Excessive load factor
    nz_max = np.max(np.abs(history.nz))
    if nz_max > nz_limit:
        idx = np.argmax(np.abs(history.nz))
        warnings.append(f"nz = {history.nz[idx]:+.2f} g at t={t[idx]:.3f}s "
                        f"(limit ±{nz_limit:.1f})")

    # Excessive alpha
    alpha_max = np.max(np.abs(history.alpha_deg))
    if alpha_max > alpha_limit:
        idx = np.argmax(np.abs(history.alpha_deg))
        warnings.append(f"α = {history.alpha_deg[idx]:+.1f}° at t={t[idx]:.3f}s "
                        f"(limit ±{alpha_limit:.0f}°)")

    # Excessive roll angle
    phi_deg = np.degrees(history.states[:, 6])
    phi_max = np.max(np.abs(phi_deg))
    if phi_max > phi_limit:
        idx = np.argmax(np.abs(phi_deg))
        warnings.append(f"φ = {phi_deg[idx]:+.1f}° at t={t[idx]:.3f}s "
                        f"(limit ±{phi_limit:.0f}°)")

    # Excessive beta
    beta_max = np.max(np.abs(history.beta_deg))
    if beta_max > beta_limit:
        idx = np.argmax(np.abs(history.beta_deg))
        warnings.append(f"β = {history.beta_deg[idx]:+.1f}° at t={t[idx]:.3f}s "
                        f"(limit ±{beta_limit:.0f}°)")

    # Monotonic divergence check
    V = np.sqrt(history.states[:, 0]**2 + history.states[:, 1]**2
                + history.states[:, 2]**2)
    V_init = V[0]
    if V_init > 1.0 and V[-1] / V_init > 2.0:
        warnings.append(f"Speed divergence: V grew from {V_init:.1f} to "
                        f"{V[-1]:.1f} m/s ({V[-1]/V_init:.1f}×)")

    return warnings


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_transient(history, label, output_path, nz_pos=3.8, nz_neg=-1.6):
    """Generate a comprehensive 3×3 transient plot for one simulation."""
    t = history.t

    de_deg = np.degrees(history.controls[:, 0])
    da_deg = np.degrees(history.controls[:, 1])
    dr_deg = np.degrees(history.controls[:, 2])

    phi_deg = np.degrees(history.states[:, 6])
    theta_deg = np.degrees(history.states[:, 7])
    psi_deg = np.degrees(history.states[:, 8])

    p_dps = np.degrees(history.p_rate)
    q_dps = np.degrees(history.q_rate)
    r_dps = np.degrees(history.r_rate)

    pdot_dps2 = np.degrees(history.p_dot)
    qdot_dps2 = np.degrees(history.q_dot)
    rdot_dps2 = np.degrees(history.r_dot)

    u = history.states[:, 0]
    v = history.states[:, 1]
    w = history.states[:, 2]
    V_total = np.sqrt(u**2 + v**2 + w**2)

    xe = history.states[:, 9]
    ye = history.states[:, 10]
    ze = history.states[:, 11]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle(label, fontsize=12, fontweight='bold', y=0.98)

    # Row 0, Col 0: Control inputs
    ax = axes[0, 0]
    ax.plot(t, de_deg, 'b-', label='δe', linewidth=1.2)
    ax.plot(t, da_deg, 'r-', label='δa', linewidth=1.2)
    ax.plot(t, dr_deg, 'g-', label='δr', linewidth=1.2)
    ax.set_ylabel('Deflection (°)')
    ax.set_title('Control Inputs')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 0, Col 1: Aero angles
    ax = axes[0, 1]
    ax.plot(t, history.alpha_deg, 'c-', label='α', linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(t, history.beta_deg, 'orange', label='β', linewidth=1.2)
    ax.set_ylabel('α (°)', color='c')
    ax2.set_ylabel('β (°)', color='orange')
    ax.set_title('Aerodynamic Angles')
    lines1, lines2 = ax.get_lines(), ax2.get_lines()
    ax.legend(lines1 + lines2, [l.get_label() for l in lines1 + lines2],
              fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 0, Col 2: Airspeed
    ax = axes[0, 2]
    ax.plot(t, V_total, 'k-', linewidth=1.2)
    ax.set_ylabel('V_TAS (m/s)')
    ax.set_title('Total Airspeed (TAS)')
    ax.grid(True, alpha=0.3)

    # Row 1, Col 0: Attitude
    ax = axes[1, 0]
    ax.plot(t, phi_deg, 'r-', label='φ (roll)', linewidth=1.2)
    ax.plot(t, theta_deg, 'b-', label='θ (pitch)', linewidth=1.2)
    ax.plot(t, psi_deg, 'g-', label='ψ (yaw)', linewidth=1.2)
    ax.set_ylabel('Angle (°)')
    ax.set_title('Euler Angles')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 1, Col 1: Angular rates
    ax = axes[1, 1]
    ax.plot(t, p_dps, 'r-', label='p (roll)', linewidth=1.2)
    ax.plot(t, q_dps, 'b-', label='q (pitch)', linewidth=1.2)
    ax.plot(t, r_dps, 'g-', label='r (yaw)', linewidth=1.2)
    ax.set_ylabel('Rate (°/s)')
    ax.set_title('Angular Rates')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 1, Col 2: Load factors
    ax = axes[1, 2]
    ax.plot(t, history.nz, 'b-', label='nz', linewidth=1.2)
    ax.plot(t, history.ny, 'r--', label='ny', linewidth=1.0)
    ax.axhline(nz_pos, color='k', linestyle=':', alpha=0.5,
               label=f'nz_max={nz_pos:+.1f}')
    ax.axhline(nz_neg, color='k', linestyle=':', alpha=0.5,
               label=f'nz_min={nz_neg:+.1f}')
    ax.set_ylabel('Load Factor (g)')
    ax.set_title('Load Factors')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 2, Col 0: Angular accelerations
    ax = axes[2, 0]
    ax.plot(t, pdot_dps2, 'r-', label='ṗ (roll)', linewidth=1.0)
    ax.plot(t, qdot_dps2, 'b-', label='q̇ (pitch)', linewidth=1.0)
    ax.plot(t, rdot_dps2, 'g-', label='ṙ (yaw)', linewidth=1.0)
    ax.set_ylabel('Accel (°/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Angular Accelerations')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 2, Col 1: Body velocities
    ax = axes[2, 1]
    ax.plot(t, u, 'b-', label='u (fwd)', linewidth=1.2)
    ax.plot(t, v, 'r-', label='v (lat)', linewidth=1.2)
    ax.plot(t, w, 'g-', label='w (vert)', linewidth=1.2)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Body-Axis Velocities')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 2, Col 2: Earth position
    ax = axes[2, 2]
    ax.plot(t, xe, 'b-', label='xe', linewidth=1.2)
    ax.plot(t, ye, 'r-', label='ye', linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(t, -ze, 'g-', label='h (-ze)', linewidth=1.2)
    ax.set_ylabel('xe, ye (m)')
    ax2.set_ylabel('Altitude h (m)', color='g')
    ax.set_xlabel('Time (s)')
    ax.set_title('Earth Position (NED)')
    lines1, lines2 = ax.get_lines(), ax2.get_lines()
    ax.legend(lines1 + lines2, [l.get_label() for l in lines1 + lines2],
              fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_table(all_results, output_path, title_suffix=""):
    """Generate a summary table of peak responses across all cases."""
    fig, ax = plt.subplots(figsize=(24, max(6, 0.32 * len(all_results) + 2)))
    ax.axis('off')

    headers = ['Case', 'Alt (ft)', 'V_EAS', 'nz_max', 'nz_min', 'α_max',
               '|β|_max', '|φ|_max', '|p|_max', '|q|_max', '|r|_max', 'Status']

    rows = []
    colors = []
    # Altitude color bands
    alt_colors = {
        0:    '#e8f4fd',  # SL — light blue
        1524: '#e8fde8',  # 5k — light green
        3048: '#fdf8e8',  # 10k — light yellow
        4572: '#fde8e8',  # 15k — light red
        6096: '#f0e8fd',  # 20k — light purple
        7620: '#e8fdfd',  # 25k — light cyan
    }

    for label, history, warnings, alt_m, V_eas in all_results:
        alt_ft = int(round(alt_m / M_PER_FT))
        if history is None:
            rows.append([label, f'{alt_ft}', '-', '-', '-', '-', '-',
                        '-', '-', '-', '-', 'FAILED'])
            colors.append(['#ffcccc'] * len(headers))
            continue

        V = np.sqrt(history.states[0, 0]**2 + history.states[0, 1]**2
                    + history.states[0, 2]**2)
        nz_max = np.max(history.nz)
        nz_min = np.min(history.nz)
        alpha_max = np.max(history.alpha_deg)
        beta_max = np.max(np.abs(history.beta_deg))
        phi_max = np.max(np.abs(np.degrees(history.states[:, 6])))
        p_max = np.max(np.abs(np.degrees(history.p_rate)))
        q_max = np.max(np.abs(np.degrees(history.q_rate)))
        r_max = np.max(np.abs(np.degrees(history.r_rate)))

        status = 'OK' if not warnings else f'⚠ ({len(warnings)})'
        base_color = alt_colors.get(alt_m, '#ffffff')
        row_color = base_color if not warnings else '#fff3cd'
        rows.append([
            label[:40],
            f'{alt_ft}',
            f'{V_eas:.0f}',
            f'{nz_max:+.2f}',
            f'{nz_min:+.2f}',
            f'{alpha_max:.1f}',
            f'{beta_max:.1f}',
            f'{phi_max:.1f}',
            f'{p_max:.1f}',
            f'{q_max:.1f}',
            f'{r_max:.1f}',
            status,
        ])
        colors.append([row_color] * len(headers))

    if rows:
        table = ax.table(cellText=rows, colLabels=headers, loc='center',
                         cellLoc='center', colColours=['#4472C4'] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.2)

        for j in range(len(headers)):
            table[0, j].set_text_props(color='white', fontweight='bold')

        for i, row_colors in enumerate(colors):
            for j, c in enumerate(row_colors):
                table[i + 1, j].set_facecolor(c)

    title = f'Dynamic Maneuver Simulation — Peak Response Summary{title_suffix}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_logging("WARNING")

    print("=" * 70)
    print("GACOMP Dynamic Maneuver Analysis — Multi-Altitude")
    print("=" * 70)
    t0 = time.time()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dynamic_transients_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}/")
    print(f"  Altitudes: {len(ANALYSIS_ALTITUDES)} "
          f"({', '.join(lbl for _, lbl in ANALYSIS_ALTITUDES)})")

    # ── 1. Parse BDF ──
    bdf_path = "tests/validation/GACOMP/p400r3-free-trim.bdf"
    print(f"\n[1] Parsing {bdf_path} ...")
    model = parse_bdf(bdf_path)
    print(f"    {len(model.nodes)} nodes, {len(model.elements)} elements")

    # ── 2. Aircraft config (altitude-independent parts) ──
    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000
    weight_N = total_mass_kg * 9.80665
    wing_area_mm2 = 17.0e6
    mach_vc = 80.0 / 340.3
    clalpha_vlm = compute_rigid_clalpha(model, mach=mach_vc, ref_area=wing_area_mm2)

    config = AircraftConfig(
        speeds=SpeedSchedule(VS1=33.0, VA=62.0, VB=0.0, VC=80.0,
                             VD=100.0, VF=40.0),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW", weight_N=weight_N, cg_x=3882.0),
        ],
        altitudes_m=[a[0] for a in ANALYSIS_ALTITUDES],
        wing_area_m2=17.0,
        CLalpha=clalpha_vlm,
        mean_chord_m=1.6,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0, rudder_max_deg=25.0, elevator_max_deg=25.0,
        ),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101], nose_gear_node_ids=[102],
            main_gear_x=4200.0, nose_gear_x=1500.0,
            strut_efficiency=0.7, stroke=0.25, sink_rate_fps=10.0,
        ),
        gust_Ude_VC_fps=50.0,
        gust_Ude_VD_fps=25.0,
    )

    wc = config.weight_cg_conditions[0]
    nz_pos = config.nz_max(wc.weight_N)
    nz_neg = config.nz_min(wc.weight_N)
    print(f"    Mass: {total_mass_kg:.1f} kg, CLα={clalpha_vlm:.3f}/rad")
    print(f"    V-n: nz_max={nz_pos:+.2f}, nz_min={nz_neg:+.2f}")

    # ── 3. Compute VLM derivatives & inertia (altitude-independent) ──
    print(f"\n[2] Computing VLM derivatives and inertia ...")
    mach = eas_to_mach(config.speeds.VC, 0.0)
    derivs = build_derivative_set(model, config, wc, mach)
    cg_xyz = np.array([wc.cg_x, 0.0, 0.0])
    inertia = compute_inertia_from_conm2(model, cg_xyz)

    mm = 1e-3
    print(f"    Ixx={inertia['Ixx']:.0f}, Iyy={inertia['Iyy']:.0f}, "
          f"Izz={inertia['Izz']:.0f}, Ixz={inertia['Ixz']:.0f} kg·m²")

    # ── 4. Run simulations at each altitude ──
    all_results = []        # (label, history, warnings, alt_m, V_eas)
    all_warnings = []       # (label, alt_label, warnings)
    total_cases = 0
    total_anomaly = 0

    for alt_m, alt_label in ANALYSIS_ALTITUDES:
        alt_ft = int(round(alt_m / M_PER_FT))
        rho_h, T_h, a_h = isa_atmosphere(alt_m)

        # Altitude-dependent Ude
        Ude_VC_h = gust_Ude_at_altitude(alt_m, config.gust_Ude_VC_fps)
        Ude_VD_h = gust_Ude_at_altitude(alt_m, config.gust_Ude_VD_fps)

        print(f"\n{'─' * 70}")
        print(f"[ALT] {alt_ft:,d} ft ({alt_m:.0f} m)  "
              f"ρ={rho_h:.4f} kg/m³  T={T_h:.1f} K  a={a_h:.1f} m/s  "
              f"Ude_VC={Ude_VC_h:.1f} fps")
        print(f"{'─' * 70}")

        # Create altitude-specific AircraftParams (rho changes)
        params = AircraftParams(
            mass=inertia["mass_kg"],
            S=derivs.S_ref * mm**2,
            b=derivs.b_ref * mm,
            c_bar=derivs.c_bar * mm,
            Ixx=inertia["Ixx"],
            Iyy=inertia["Iyy"],
            Izz=inertia["Izz"],
            Ixz=inertia["Ixz"],
            derivs=derivs,
            rho=rho_h,
        )

        # Trim at VC TAS for this altitude
        V_tas_vc = eas_to_tas(config.speeds.VC, alt_m)
        _, de_trim = trim_initial_state(params, V_tas_vc)

        # Generate maneuver profiles (altitude-aware Ude)
        profiles = generate_all_maneuver_profiles(
            config, wc, alt_m, delta_e_trim=de_trim, derivs=derivs)

        # Create altitude output sub-directory
        alt_dir = os.path.join(output_dir, f"alt_{alt_ft:05d}ft")
        os.makedirs(alt_dir, exist_ok=True)

        n_alt_anomaly = 0
        for i, (profile, V_eas, label) in enumerate(profiles):
            V_tas = eas_to_tas(V_eas, alt_m)
            if V_tas < 1.0:
                continue

            # Trim for this speed at this altitude
            initial_state, de_t = trim_initial_state(params, V_tas)

            # Update control function's trim
            ctrl_func = profile.control_func
            if hasattr(ctrl_func, 'de_trim'):
                ctrl_func.de_trim = de_t

            # Integrate
            history = integrate_6dof(
                params=params,
                initial_state=initial_state,
                control_func=ctrl_func,
                t_span=(0.0, profile.t_end),
                dt=0.005,
                gust_func=profile.gust_func,
            )
            compute_nz_from_history(params, history)

            # Check anomalies
            warnings = check_anomalies(label, history, nz_limit=nz_pos * 1.1)
            status = "OK" if not warnings else f"⚠ {len(warnings)} issue(s)"

            nz_max_val = np.max(history.nz)
            nz_min_val = np.min(history.nz)
            print(f"  [{i+1:2d}/{len(profiles)}] {label:45s}  "
                  f"nz=[{nz_min_val:+.2f}, {nz_max_val:+.2f}]  {status}")
            if warnings:
                n_alt_anomaly += 1
                for w in warnings:
                    print(f"           ⚠ {w}")
                all_warnings.append((label, alt_label, warnings))

            # Plot with altitude info in title
            safe_name = (label.replace("/", "_").replace(" ", "_")
                         .replace("+", "p").replace("-", "m"))
            plot_path = os.path.join(alt_dir, f"{i+1:02d}_{safe_name}.png")
            plot_title = (f"{label}\n{profile.description} | "
                         f"V_EAS={V_eas:.0f} m/s | "
                         f"Alt={alt_ft:,d} ft | ρ={rho_h:.4f} kg/m³")
            plot_transient(history, plot_title, plot_path,
                          nz_pos=nz_pos, nz_neg=nz_neg)

            all_results.append((label, history, warnings, alt_m, V_eas))
            total_cases += 1

        total_anomaly += n_alt_anomaly
        print(f"  → {alt_label}: {len(profiles)} cases, "
              f"{n_alt_anomaly} anomalies")

    # ── 5. Summary table ──
    print(f"\n[5] Generating summary tables ...")
    summary_path = os.path.join(output_dir, "00_summary_all_altitudes.png")
    plot_summary_table(all_results, summary_path,
                      title_suffix=f"\n{len(ANALYSIS_ALTITUDES)} Altitudes "
                      f"(SL – 25,000 ft)")

    # Per-altitude summary tables
    for alt_m, alt_label in ANALYSIS_ALTITUDES:
        alt_ft = int(round(alt_m / M_PER_FT))
        alt_results = [(l, h, w, a, v) for l, h, w, a, v in all_results
                       if a == alt_m]
        if alt_results:
            alt_path = os.path.join(output_dir,
                                    f"00_summary_{alt_ft:05d}ft.png")
            plot_summary_table(alt_results, alt_path,
                              title_suffix=f" — {alt_ft:,d} ft")

    # ── 6. Print summary ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY — MULTI-ALTITUDE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Altitudes analyzed: {len(ANALYSIS_ALTITUDES)}")
    print(f"  Total simulations:  {total_cases}")
    print(f"  Anomalous cases:    {total_anomaly}")
    print(f"  Output directory:   {output_dir}/")
    print(f"  Elapsed time:       {elapsed:.1f}s")

    # Altitude summary
    print(f"\n  PER-ALTITUDE SUMMARY:")
    for alt_m, alt_label in ANALYSIS_ALTITUDES:
        alt_ft = int(round(alt_m / M_PER_FT))
        rho_h, _, _ = isa_atmosphere(alt_m)
        n_cases = sum(1 for _, _, _, a, _ in all_results if a == alt_m)
        n_warn = sum(1 for _, _, w, a, _ in all_results
                     if a == alt_m and w)
        Ude_VC_h = gust_Ude_at_altitude(alt_m, 50.0)
        status_str = "✅ OK" if n_warn == 0 else f"⚠ {n_warn} anomalies"
        print(f"    {alt_ft:6,d} ft  ρ={rho_h:.4f}  "
              f"Ude_VC={Ude_VC_h:.1f}fps  "
              f"{n_cases:3d} cases  {status_str}")

    if all_warnings:
        print(f"\n  ANOMALY DETAILS:")
        print(f"  {'─' * 66}")
        for label, alt_label, warnings in all_warnings:
            print(f"  [{alt_label}] {label}:")
            for w in warnings:
                print(f"    ⚠ {w}")
        print(f"  {'─' * 66}")
    else:
        print(f"\n  ✅ All {total_cases} simulations produced "
              f"physically reasonable results.")

    print(f"\nDone.")
    return total_anomaly


if __name__ == "__main__":
    sys.exit(main())
