#!/usr/bin/env python3
"""OEI/Jam Recovery Maneuver Analysis.

Simulates full failure→recognition→recovery sequence for OEI and rotor-jam
events, comparing "no recovery" (open-loop divergence) vs "FCC recovery"
(closed-loop attitude controller with thrust redistribution).

Generates:
  - Overlay comparison plots (attitude, rates, accelerations)
  - Recovery thrust schedule per rotor (non-uniform allocation)
  - Peak loads summary table for structural design
  - Height loss / max attitude excursion metrics

4 simulation cases:
  1. OEI — no recovery  (5 s, open-loop)
  2. OEI — with recovery (10 s, closed-loop after 0.3 s FCC detection)
  3. Jam — no recovery  (5 s, open-loop)
  4. Jam — with recovery (10 s, closed-loop after 0.3 s FCC detection)
"""
import os
import sys
import time
import math
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from nastaero.bdf.parser import parse_bdf
from nastaero.config import setup_logging
from nastaero.rotor.rotor_config import VTOLConfig, RotorType
from nastaero.rotor.bemt_solver import BEMTSolver
from nastaero.rotor.rotor_dynamics import (
    make_oei_force_func, make_rotor_jam_force_func,
    make_oei_recovery_force_func, make_jam_recovery_force_func,
    compute_recovery_thrust_schedule,
)
from nastaero.loads_analysis.case_generator import isa_atmosphere
from nastaero.loads_analysis.certification.aero_derivatives import (
    build_derivative_set, compute_inertia_from_conm2,
)
from nastaero.loads_analysis.certification.aircraft_config import (
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    ControlSurfaceLimits, LandingGearConfig,
)
from nastaero.loads_analysis.certification.flight_sim import (
    AircraftParams, AircraftState, ControlInput,
    SimTimeHistory, integrate_6dof, trim_initial_state,
    compute_nz_from_history, six_dof_derivatives,
)
from nastaero.aero.dlm import compute_rigid_clalpha


# ---------------------------------------------------------------------------
# Thrust schedule for no-recovery cases
# ---------------------------------------------------------------------------

def compute_thrusts_no_recovery(vtol_config, failed_rotor_id, failure_time,
                                 weight_N, rho, t_arr):
    """Compute per-rotor thrust time history (no recovery)."""
    hover_rotors = vtol_config.hover_rotors
    n = len(hover_rotors)
    T_per = weight_N / n

    nominal_T = {}
    for rotor in hover_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(T_per, rotor.rpm_hover, rho)
        nominal_T[rotor.rotor_id] = loads.thrust

    thrusts = {}
    for rotor in hover_rotors:
        T_arr = np.full(len(t_arr), nominal_T[rotor.rotor_id])
        if rotor.rotor_id == failed_rotor_id:
            for i, t in enumerate(t_arr):
                if t >= failure_time:
                    T_arr[i] = 0.0
        thrusts[rotor.label] = T_arr
    return thrusts


def compute_body_accels(params, history, ext_force_func):
    """Compute body-axis linear accelerations from EOM."""
    N = len(history.t)
    du, dv, dw = np.zeros(N), np.zeros(N), np.zeros(N)
    ctrl = ControlInput()
    for i in range(N):
        dydt = six_dof_derivatives(
            history.states[i], history.t[i], params, ctrl,
            external_force_func=ext_force_func)
        du[i], dv[i], dw[i] = dydt[0], dydt[1], dydt[2]
    return du, dv, dw


# ---------------------------------------------------------------------------
# Peak loads extraction
# ---------------------------------------------------------------------------

def extract_peak_loads(history, params, thrusts, ext_force_func, failure_time):
    """Extract peak load metrics from simulation history."""
    t = history.t
    # Only analyze post-failure
    mask = t >= failure_time
    t_post = t[mask]

    phi = history.states[mask, 6]
    theta = history.states[mask, 7]
    p = history.states[mask, 3]
    q = history.states[mask, 4]
    r = history.states[mask, 5]
    ze = history.states[mask, 11]
    p_dot = history.p_dot[mask]
    q_dot = history.q_dot[mask]
    r_dot = history.r_dot[mask]

    # Max per-rotor thrust
    max_rotor_T = 0.0
    max_rotor_label = ""
    for label, T_arr in thrusts.items():
        T_post = T_arr[mask]
        peak = np.max(T_post)
        if peak > max_rotor_T:
            max_rotor_T = peak
            max_rotor_label = label

    return {
        "max_phi_deg": float(np.max(np.abs(np.degrees(phi)))),
        "max_theta_deg": float(np.max(np.abs(np.degrees(theta)))),
        "max_p_deg_s": float(np.max(np.abs(np.degrees(p)))),
        "max_q_deg_s": float(np.max(np.abs(np.degrees(q)))),
        "max_r_deg_s": float(np.max(np.abs(np.degrees(r)))),
        "max_p_dot_deg_s2": float(np.max(np.abs(np.degrees(p_dot)))),
        "max_q_dot_deg_s2": float(np.max(np.abs(np.degrees(q_dot)))),
        "max_r_dot_deg_s2": float(np.max(np.abs(np.degrees(r_dot)))),
        "max_height_loss_m": float(np.max(ze) - ze[0]),  # ze positive = down
        "max_rotor_thrust_N": max_rotor_T,
        "max_rotor_label": max_rotor_label,
        "final_phi_deg": float(np.degrees(phi[-1])),
        "final_theta_deg": float(np.degrees(theta[-1])),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_recovery_comparison(hist_open, hist_recov,
                              thrusts_open, thrusts_recov,
                              params, ext_open, ext_recov,
                              failure_time, t_rec_start, t_rec_end,
                              event_label, rotor_label,
                              peaks_open, peaks_recov,
                              output_dir):
    """Generate comparison plots: open-loop vs recovery."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Common formatting
    fail_kw = dict(color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    rec_kw = dict(color='green', linewidth=1.0, linestyle='--', alpha=0.7)
    end_kw = dict(color='blue', linewidth=1.0, linestyle=':', alpha=0.6)
    grid_kw = dict(alpha=0.3, linewidth=0.5)

    def add_phase_lines(ax):
        ax.axvline(x=failure_time, **fail_kw, label='Failure')
        ax.axvline(x=t_rec_start, **rec_kw, label='FCC detect')
        ax.axvline(x=t_rec_end, **end_kw, label='Ramp end')

    def style_ax(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(**grid_kw)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlabel('Time (s)', fontsize=8)

    # ================================================================
    # Figure 1: Attitude + Rates comparison (3×2)
    # ================================================================
    fig1, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig1.suptitle(
        f'{event_label}: {rotor_label} — Open-Loop vs Recovery Comparison\n'
        f'KC-100 Tilt-Rotor-12, Hover, t_rec={t_rec_start-failure_time:.1f}s, '
        f't_ramp={t_rec_end-t_rec_start:.1f}s',
        fontsize=12, fontweight='bold')

    t_o = hist_open.t
    t_r = hist_recov.t

    # Roll angle
    ax = axes[0, 0]
    ax.plot(t_o, np.degrees(hist_open.states[:, 6]),
            'r-', linewidth=1.5, alpha=0.8, label=r'$\phi$ open-loop')
    ax.plot(t_r, np.degrees(hist_recov.states[:, 6]),
            'b-', linewidth=1.5, label=r'$\phi$ recovery')
    add_phase_lines(ax)
    style_ax(ax, 'Angle (deg)', r'(a) Roll Angle $\phi$')

    # Pitch angle
    ax = axes[0, 1]
    ax.plot(t_o, np.degrees(hist_open.states[:, 7]),
            'r-', linewidth=1.5, alpha=0.8, label=r'$\theta$ open-loop')
    ax.plot(t_r, np.degrees(hist_recov.states[:, 7]),
            'b-', linewidth=1.5, label=r'$\theta$ recovery')
    add_phase_lines(ax)
    style_ax(ax, 'Angle (deg)', r'(b) Pitch Angle $\theta$')

    # Roll rate
    ax = axes[1, 0]
    ax.plot(t_o, np.degrees(hist_open.states[:, 3]),
            'r-', linewidth=1.5, alpha=0.8, label='p open-loop')
    ax.plot(t_r, np.degrees(hist_recov.states[:, 3]),
            'b-', linewidth=1.5, label='p recovery')
    add_phase_lines(ax)
    style_ax(ax, 'Rate (deg/s)', '(c) Roll Rate p')

    # Pitch rate
    ax = axes[1, 1]
    ax.plot(t_o, np.degrees(hist_open.states[:, 4]),
            'r-', linewidth=1.5, alpha=0.8, label='q open-loop')
    ax.plot(t_r, np.degrees(hist_recov.states[:, 4]),
            'b-', linewidth=1.5, label='q recovery')
    add_phase_lines(ax)
    style_ax(ax, 'Rate (deg/s)', '(d) Pitch Rate q')

    # Roll acceleration
    ax = axes[2, 0]
    ax.plot(t_o, np.degrees(hist_open.p_dot),
            'r-', linewidth=1.2, alpha=0.8, label=r'$\dot{p}$ open-loop')
    ax.plot(t_r, np.degrees(hist_recov.p_dot),
            'b-', linewidth=1.2, label=r'$\dot{p}$ recovery')
    add_phase_lines(ax)
    style_ax(ax, r'Accel (deg/s$^2$)', r'(e) Roll Accel $\dot{p}$')

    # Height (ze positive = down in NED)
    ax = axes[2, 1]
    ax.plot(t_o, hist_open.states[:, 11],
            'r-', linewidth=1.5, alpha=0.8, label='ze open-loop')
    ax.plot(t_r, hist_recov.states[:, 11],
            'b-', linewidth=1.5, label='ze recovery')
    add_phase_lines(ax)
    style_ax(ax, 'Height loss (m, +down)', '(f) Altitude (ze, NED down)')

    fig1.tight_layout()
    path1 = os.path.join(output_dir, f'{event_label.lower()}_comparison.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ================================================================
    # Figure 2: Recovery thrust schedule (individual rotors)
    # ================================================================
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 10))
    fig2.suptitle(
        f'{event_label}: {rotor_label} — Per-Rotor Thrust Schedule\n'
        f'Open-loop (top) vs Recovery (bottom)',
        fontsize=12, fontweight='bold')

    # Color palette for 12 rotors
    cmap = plt.colormaps['tab20'].resampled(len(thrusts_recov))
    W = params.mass * 9.80665

    for idx, (label, T_arr) in enumerate(thrusts_open.items()):
        lw = 2.0 if T_arr.min() < T_arr.max() * 0.5 else 0.8
        ax_top.plot(t_o, T_arr, '-', linewidth=lw, color=cmap(idx),
                    label=label, alpha=0.9)
    ax_top.axvline(x=failure_time, **fail_kw, label='Failure')
    ax_top.axhline(y=W, color='gray', linestyle=':', alpha=0.5,
                   label=f'Weight ({W:.0f} N)')
    total_o = sum(T_arr for T_arr in thrusts_open.values())
    ax_top.plot(t_o, total_o, 'k--', linewidth=2.0, alpha=0.5,
                label='Total')
    ax_top.set_ylabel('Thrust (N)', fontsize=10)
    ax_top.set_title('Open-loop (no recovery)', fontsize=10)
    ax_top.legend(fontsize=6, ncol=4, loc='lower left')
    ax_top.grid(**grid_kw)
    ax_top.set_ylim(bottom=-100)
    ax_top.set_xlabel('Time (s)', fontsize=9)

    for idx, (label, T_arr) in enumerate(thrusts_recov.items()):
        lw = 2.0 if (T_arr.max() - T_arr.min()) > 100 else 0.8
        ax_bot.plot(t_r, T_arr, '-', linewidth=lw, color=cmap(idx),
                    label=label, alpha=0.9)
    ax_bot.axvline(x=failure_time, **fail_kw)
    ax_bot.axvline(x=t_rec_start, **rec_kw, label='FCC detect')
    ax_bot.axvline(x=t_rec_end, **end_kw, label='Ramp end')
    ax_bot.axhline(y=W, color='gray', linestyle=':', alpha=0.5,
                   label=f'Weight ({W:.0f} N)')
    total_r = sum(T_arr for T_arr in thrusts_recov.values())
    ax_bot.plot(t_r, total_r, 'k--', linewidth=2.0, alpha=0.5,
                label='Total')
    ax_bot.set_ylabel('Thrust (N)', fontsize=10)
    ax_bot.set_title('With FCC recovery', fontsize=10)
    ax_bot.legend(fontsize=6, ncol=4, loc='lower left')
    ax_bot.grid(**grid_kw)
    ax_bot.set_ylim(bottom=-100)
    ax_bot.set_xlabel('Time (s)', fontsize=9)

    fig2.tight_layout()
    path2 = os.path.join(output_dir, f'{event_label.lower()}_thrust_schedule.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ================================================================
    # Figure 3: Peak loads summary table
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.axis('off')
    fig3.suptitle(
        f'{event_label}: {rotor_label} — Peak Loads Summary',
        fontsize=13, fontweight='bold')

    col_labels = ['Metric', 'Open-loop', 'Recovery', 'Unit']
    rows = [
        ['Max |φ| (roll)',
         f'{peaks_open["max_phi_deg"]:.1f}',
         f'{peaks_recov["max_phi_deg"]:.1f}', 'deg'],
        ['Max |θ| (pitch)',
         f'{peaks_open["max_theta_deg"]:.1f}',
         f'{peaks_recov["max_theta_deg"]:.1f}', 'deg'],
        ['Max |p| (roll rate)',
         f'{peaks_open["max_p_deg_s"]:.1f}',
         f'{peaks_recov["max_p_deg_s"]:.1f}', 'deg/s'],
        ['Max |q| (pitch rate)',
         f'{peaks_open["max_q_deg_s"]:.1f}',
         f'{peaks_recov["max_q_deg_s"]:.1f}', 'deg/s'],
        ['Max |r| (yaw rate)',
         f'{peaks_open["max_r_deg_s"]:.1f}',
         f'{peaks_recov["max_r_deg_s"]:.1f}', 'deg/s'],
        [r'Max |p_dot| (roll accel)',
         f'{peaks_open["max_p_dot_deg_s2"]:.1f}',
         f'{peaks_recov["max_p_dot_deg_s2"]:.1f}', 'deg/s²'],
        [r'Max |q_dot| (pitch accel)',
         f'{peaks_open["max_q_dot_deg_s2"]:.1f}',
         f'{peaks_recov["max_q_dot_deg_s2"]:.1f}', 'deg/s²'],
        ['Height loss (max ze)',
         f'{peaks_open["max_height_loss_m"]:.2f}',
         f'{peaks_recov["max_height_loss_m"]:.2f}', 'm'],
        ['Max single-rotor thrust',
         f'{peaks_open["max_rotor_thrust_N"]:.0f}',
         f'{peaks_recov["max_rotor_thrust_N"]:.0f}', 'N'],
        ['  (rotor)',
         peaks_open["max_rotor_label"],
         peaks_recov["max_rotor_label"], ''],
        ['Final φ',
         f'{peaks_open["final_phi_deg"]:.1f}',
         f'{peaks_recov["final_phi_deg"]:.1f}', 'deg'],
        ['Final θ',
         f'{peaks_open["final_theta_deg"]:.1f}',
         f'{peaks_recov["final_theta_deg"]:.1f}', 'deg'],
    ]

    table = ax3.table(
        cellText=rows, colLabels=col_labels,
        cellLoc='center', loc='center',
        colWidths=[0.30, 0.22, 0.22, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    fig3.tight_layout()
    path3 = os.path.join(output_dir, f'{event_label.lower()}_peak_loads.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {path3}")

    return [path1, path2, path3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging("WARNING")
    print("=" * 80)
    print("  OEI / Jam Recovery Maneuver Analysis")
    print("  Failure → Recognition → Recovery → Steady-State")
    print("=" * 80)

    t0_wall = time.time()
    timestamp = datetime.now()
    output_dir = f"recovery_analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Setup ----
    model = parse_bdf("tests/validation/KC100/p400r3-free-trim.bdf")
    print(f"[1] BDF parsed: {len(model.nodes)} nodes")

    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000
    weight_N = total_mass_kg * 9.80665
    vtol_config = VTOLConfig.kc100_tilt_rotor_12()

    wing_area_mm2 = 17.0e6
    mach_vc = 80.0 / 340.3
    clalpha_vlm = compute_rigid_clalpha(model, mach=mach_vc,
                                         ref_area=wing_area_mm2)

    config = AircraftConfig(
        speeds=SpeedSchedule(
            VS1=33.0, VA=62.0, VB=0.0, VC=80.0, VD=100.0, VF=40.0),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW", weight_N=weight_N, cg_x=3882.0)],
        altitudes_m=[0.0],
        wing_area_m2=17.0, CLalpha=clalpha_vlm, mean_chord_m=1.6,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0, rudder_max_deg=25.0, elevator_max_deg=25.0),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101], nose_gear_node_ids=[102],
            main_gear_x=4200.0, nose_gear_x=1500.0,
            strut_efficiency=0.7, stroke=0.25, sink_rate_fps=10.0),
        gust_Ude_VC_fps=50.0, gust_Ude_VD_fps=25.0,
    )

    wc = config.weight_cg_conditions[0]
    derivs = build_derivative_set(model, config, wc, mach_vc)
    cg_xyz = np.array([wc.cg_x, 0.0, 0.0])
    inertia = compute_inertia_from_conm2(model, cg_xyz)

    mm_to_m = 1e-3
    params = AircraftParams(
        mass=inertia["mass_kg"],
        S=derivs.S_ref * mm_to_m**2,
        b=derivs.b_ref * mm_to_m,
        c_bar=derivs.c_bar * mm_to_m,
        Ixx=inertia["Ixx"], Iyy=inertia["Iyy"],
        Izz=inertia["Izz"], Ixz=inertia["Ixz"],
        derivs=derivs,
    )

    rho_sl, _, _ = isa_atmosphere(0.0)
    params.rho = rho_sl
    Ixx = inertia["Ixx"]
    Iyy = inertia["Iyy"]

    print(f"[2] Aircraft: mass={params.mass:.0f} kg, "
          f"Ixx={Ixx:.0f}, Iyy={Iyy:.0f}, Izz={inertia['Izz']:.0f}")

    # ---- Simulation parameters ----
    t_sim_open = 5.0     # open-loop (no recovery)
    t_sim_recov = 10.0   # with recovery (longer to see settling)
    t_fail = 1.0
    t_recognition = 0.3   # FCC auto-detection delay (s)
    t_ramp = 0.5           # thrust redistribution ramp (s)
    dt = 0.005
    omega_att = 2.0
    zeta_att = 0.7

    t_rec_start = t_fail + t_recognition
    t_rec_end = t_rec_start + t_ramp

    print(f"    t_fail={t_fail}s, t_recognition={t_recognition}s, "
          f"t_ramp={t_ramp}s")
    print(f"    FCC detect at {t_rec_start}s, ramp complete at {t_rec_end}s")
    print(f"    Controller: ω_n={omega_att} rad/s, ζ={zeta_att}")

    # ---- Find worst-case rotor (outboard, largest |Y|) ----
    hover_rotors = vtol_config.hover_rotors
    worst_rotor = None
    for rotor in hover_rotors:
        if 'FL3' in rotor.label or 'L3' in rotor.label:
            worst_rotor = rotor
            break
    if worst_rotor is None:
        worst_rotor = max(hover_rotors,
                          key=lambda r: abs(r.hub_position[1]))

    failed_id = worst_rotor.rotor_id
    print(f"\n[3] Worst-case rotor: {worst_rotor.label} "
          f"(Y={worst_rotor.hub_position[1]:.0f} mm)")

    initial_state = AircraftState(u=0.1, v=0.0, w=0.0)
    ctrl = ControlInput()
    control_func = lambda t: ctrl

    # ==================================================================
    # Case 1: OEI — no recovery
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Case 1: OEI — No Recovery ({t_sim_open}s)")
    print(f"{'='*60}")

    oei_open_ext = make_oei_force_func(
        vtol_config, failed_rotor_id=failed_id,
        failure_time=t_fail, weight_N=weight_N, rho=rho_sl,
        cg_position=cg_xyz)

    hist_oei_open = integrate_6dof(
        params, initial_state, control_func,
        t_span=(0.0, t_sim_open), dt=dt,
        external_force_func=oei_open_ext)
    compute_nz_from_history(params, hist_oei_open)

    thrusts_oei_open = compute_thrusts_no_recovery(
        vtol_config, failed_id, t_fail, weight_N, rho_sl,
        hist_oei_open.t)

    print(f"    Done. Max |phi|={np.max(np.abs(np.degrees(hist_oei_open.states[:, 6]))):.1f} deg")

    # ==================================================================
    # Case 2: OEI — with recovery
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Case 2: OEI — With Recovery ({t_sim_recov}s)")
    print(f"{'='*60}")

    oei_recov_ext = make_oei_recovery_force_func(
        vtol_config, failed_rotor_id=failed_id,
        failure_time=t_fail, weight_N=weight_N, rho=rho_sl,
        cg_position=cg_xyz,
        t_recognition=t_recognition, t_ramp=t_ramp,
        Ixx=Ixx, Iyy=Iyy, omega_att=omega_att, zeta_att=zeta_att)

    hist_oei_recov = integrate_6dof(
        params, initial_state, control_func,
        t_span=(0.0, t_sim_recov), dt=dt,
        external_force_func=oei_recov_ext)
    compute_nz_from_history(params, hist_oei_recov)

    thrusts_oei_recov = compute_recovery_thrust_schedule(
        vtol_config, failed_id, t_fail, weight_N, rho_sl,
        cg_xyz, hist_oei_recov.t, hist_oei_recov.states,
        t_recognition=t_recognition, t_ramp=t_ramp,
        Ixx=Ixx, Iyy=Iyy, omega_att=omega_att, zeta_att=zeta_att)

    phi_recov = np.degrees(hist_oei_recov.states[:, 6])
    print(f"    Done. Max |phi|={np.max(np.abs(phi_recov)):.1f} deg, "
          f"Final phi={phi_recov[-1]:.2f} deg")

    # ==================================================================
    # Case 3: Jam — no recovery
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Case 3: Rotor Jam — No Recovery ({t_sim_open}s)")
    print(f"{'='*60}")

    jam_open_ext = make_rotor_jam_force_func(
        vtol_config, jammed_rotor_id=failed_id,
        jam_time=t_fail, weight_N=weight_N, rho=rho_sl,
        cg_position=cg_xyz)

    hist_jam_open = integrate_6dof(
        params, initial_state, control_func,
        t_span=(0.0, t_sim_open), dt=dt,
        external_force_func=jam_open_ext)
    compute_nz_from_history(params, hist_jam_open)

    thrusts_jam_open = compute_thrusts_no_recovery(
        vtol_config, failed_id, t_fail, weight_N, rho_sl,
        hist_jam_open.t)

    print(f"    Done. Max |phi|={np.max(np.abs(np.degrees(hist_jam_open.states[:, 6]))):.1f} deg")

    # ==================================================================
    # Case 4: Jam — with recovery
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Case 4: Rotor Jam — With Recovery ({t_sim_recov}s)")
    print(f"{'='*60}")

    jam_recov_ext = make_jam_recovery_force_func(
        vtol_config, jammed_rotor_id=failed_id,
        jam_time=t_fail, weight_N=weight_N, rho=rho_sl,
        cg_position=cg_xyz,
        t_recognition=t_recognition, t_ramp=t_ramp,
        Ixx=Ixx, Iyy=Iyy, omega_att=omega_att, zeta_att=zeta_att)

    hist_jam_recov = integrate_6dof(
        params, initial_state, control_func,
        t_span=(0.0, t_sim_recov), dt=dt,
        external_force_func=jam_recov_ext)
    compute_nz_from_history(params, hist_jam_recov)

    thrusts_jam_recov = compute_recovery_thrust_schedule(
        vtol_config, failed_id, t_fail, weight_N, rho_sl,
        cg_xyz, hist_jam_recov.t, hist_jam_recov.states,
        t_recognition=t_recognition, t_ramp=t_ramp,
        Ixx=Ixx, Iyy=Iyy, omega_att=omega_att, zeta_att=zeta_att)

    phi_jam_recov = np.degrees(hist_jam_recov.states[:, 6])
    print(f"    Done. Max |phi|={np.max(np.abs(phi_jam_recov)):.1f} deg, "
          f"Final phi={phi_jam_recov[-1]:.2f} deg")

    # ==================================================================
    # Extract peak loads
    # ==================================================================
    print(f"\n[4] Extracting peak loads...")

    peaks_oei_open = extract_peak_loads(
        hist_oei_open, params, thrusts_oei_open, oei_open_ext, t_fail)
    peaks_oei_recov = extract_peak_loads(
        hist_oei_recov, params, thrusts_oei_recov, oei_recov_ext, t_fail)
    peaks_jam_open = extract_peak_loads(
        hist_jam_open, params, thrusts_jam_open, jam_open_ext, t_fail)
    peaks_jam_recov = extract_peak_loads(
        hist_jam_recov, params, thrusts_jam_recov, jam_recov_ext, t_fail)

    # ==================================================================
    # Generate comparison plots
    # ==================================================================
    print(f"\n[5] Generating comparison plots...")

    plot_recovery_comparison(
        hist_oei_open, hist_oei_recov,
        thrusts_oei_open, thrusts_oei_recov,
        params, oei_open_ext, oei_recov_ext,
        t_fail, t_rec_start, t_rec_end,
        "OEI", worst_rotor.label,
        peaks_oei_open, peaks_oei_recov,
        output_dir)

    plot_recovery_comparison(
        hist_jam_open, hist_jam_recov,
        thrusts_jam_open, thrusts_jam_recov,
        params, jam_open_ext, jam_recov_ext,
        t_fail, t_rec_start, t_rec_end,
        "Jam", worst_rotor.label,
        peaks_jam_open, peaks_jam_recov,
        output_dir)

    # ==================================================================
    # Print summary table
    # ==================================================================
    print(f"\n{'='*80}")
    print("  PEAK LOADS SUMMARY — OEI Recovery")
    print(f"{'='*80}")
    _print_comparison(peaks_oei_open, peaks_oei_recov)

    print(f"\n{'='*80}")
    print("  PEAK LOADS SUMMARY — Rotor Jam Recovery")
    print(f"{'='*80}")
    _print_comparison(peaks_jam_open, peaks_jam_recov)

    # Design loads for structural analysis
    print(f"\n{'='*80}")
    print("  DESIGN LOAD CONDITIONS FOR FEA")
    print(f"{'='*80}")
    T_nom = weight_N / len(vtol_config.hover_rotors)
    print(f"  Nominal thrust/rotor:  {T_nom:.0f} N")
    print(f"  OEI recovery max T:   {peaks_oei_recov['max_rotor_thrust_N']:.0f} N "
          f"({peaks_oei_recov['max_rotor_label']}) = "
          f"{peaks_oei_recov['max_rotor_thrust_N']/T_nom:.2f}× nominal")
    print(f"  Jam recovery max T:   {peaks_jam_recov['max_rotor_thrust_N']:.0f} N "
          f"({peaks_jam_recov['max_rotor_label']}) = "
          f"{peaks_jam_recov['max_rotor_thrust_N']/T_nom:.2f}× nominal")
    print(f"  OEI max p_dot:        {peaks_oei_recov['max_p_dot_deg_s2']:.1f} deg/s²")
    print(f"  Jam max p_dot:        {peaks_jam_recov['max_p_dot_deg_s2']:.1f} deg/s²")
    print(f"  OEI height loss:      {peaks_oei_recov['max_height_loss_m']:.2f} m")
    print(f"  Jam height loss:      {peaks_jam_recov['max_height_loss_m']:.2f} m")
    print(f"  Recovery settles:     phi→{peaks_oei_recov['final_phi_deg']:.2f}°, "
          f"theta→{peaks_oei_recov['final_theta_deg']:.2f}°")

    total_time = time.time() - t0_wall
    print(f"\n{'='*80}")
    print(f"  Complete ({total_time:.1f}s)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")


def _print_comparison(peaks_open, peaks_recov):
    """Print formatted comparison table."""
    fmt = "  {:<28s} {:>12s}  {:>12s}  {:>6s}"
    print(fmt.format("Metric", "Open-loop", "Recovery", "Unit"))
    print("  " + "-" * 64)
    rows = [
        ("Max |φ| (roll)", "max_phi_deg", "deg"),
        ("Max |θ| (pitch)", "max_theta_deg", "deg"),
        ("Max |p| (roll rate)", "max_p_deg_s", "deg/s"),
        ("Max |q| (pitch rate)", "max_q_deg_s", "deg/s"),
        ("Max |r| (yaw rate)", "max_r_deg_s", "deg/s"),
        ("Max |p_dot| (roll accel)", "max_p_dot_deg_s2", "deg/s²"),
        ("Max |q_dot| (pitch accel)", "max_q_dot_deg_s2", "deg/s²"),
        ("Height loss", "max_height_loss_m", "m"),
        ("Max single-rotor thrust", "max_rotor_thrust_N", "N"),
        ("Final φ", "final_phi_deg", "deg"),
        ("Final θ", "final_theta_deg", "deg"),
    ]
    for label, key, unit in rows:
        v1 = peaks_open[key]
        v2 = peaks_recov[key]
        if isinstance(v1, float):
            s1 = f"{v1:.2f}" if abs(v1) < 10 else f"{v1:.1f}"
            s2 = f"{v2:.2f}" if abs(v2) < 10 else f"{v2:.1f}"
        else:
            s1, s2 = str(v1), str(v2)
        print(fmt.format(label, s1, s2, unit))


if __name__ == "__main__":
    main()
