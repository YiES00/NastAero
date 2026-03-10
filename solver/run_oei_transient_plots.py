#!/usr/bin/env python3
"""OEI Time Transient Analysis — Comprehensive Dynamics Plots.

Runs the worst-case OEI simulation (outboard L3 rotor failure) and
generates detailed time-history plots for verification of aircraft
dynamic response reasonableness:

 - Position (xe, ye, ze)
 - Body-axis velocities (u, v, w)
 - Body-axis accelerations (du, dv, dw)
 - Euler angles (phi, theta, psi)
 - Angular rates (p, q, r)
 - Angular accelerations (p_dot, q_dot, r_dot)
 - Load factors (nz, ny)
 - Rotor thrust schedule per rotor

Also generates a rotor jam worst-case for comparison.
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
from nastaero.rotor.rotor_dynamics import make_oei_force_func, make_rotor_jam_force_func
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


def compute_body_accels(params, history, ext_force_func):
    """Compute body-axis linear accelerations (du, dv, dw) from EOM."""
    N = len(history.t)
    du = np.zeros(N)
    dv = np.zeros(N)
    dw = np.zeros(N)

    ctrl = ControlInput()
    for i in range(N):
        t = history.t[i]
        y = history.states[i]
        dydt = six_dof_derivatives(y, t, params, ctrl,
                                    external_force_func=ext_force_func)
        du[i] = dydt[0]
        dv[i] = dydt[1]
        dw[i] = dydt[2]

    return du, dv, dw


def compute_rotor_thrusts_oei(vtol_config, failed_rotor_id, failure_time,
                                weight_N, rho, cg_xyz, t_arr):
    """Compute per-rotor thrust time history for OEI scenario."""
    mm_to_m = 1e-3
    lift_rotors = vtol_config.lift_rotors
    n_rotors = len(lift_rotors)
    thrust_per = weight_N / n_rotors

    # BEMT for nominal thrust
    nominal_T = {}
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(thrust_per, rotor.rpm_hover, rho)
        nominal_T[rotor.rotor_id] = loads.thrust

    N = len(t_arr)
    thrusts = {}
    for rotor in lift_rotors:
        T_arr = np.full(N, nominal_T[rotor.rotor_id])
        if rotor.rotor_id == failed_rotor_id:
            for i, t in enumerate(t_arr):
                if t >= failure_time:
                    T_arr[i] = 0.0
        thrusts[rotor.label] = T_arr

    return thrusts


def compute_rotor_thrusts_jam(vtol_config, jammed_rotor_id, jam_time,
                                weight_N, rho, t_arr):
    """Compute per-rotor thrust time history for rotor jam."""
    lift_rotors = vtol_config.lift_rotors
    n_rotors = len(lift_rotors)
    thrust_per = weight_N / n_rotors

    nominal_T = {}
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(thrust_per, rotor.rpm_hover, rho)
        nominal_T[rotor.rotor_id] = loads.thrust

    N = len(t_arr)
    thrusts = {}
    for rotor in lift_rotors:
        T_arr = np.full(N, nominal_T[rotor.rotor_id])
        if rotor.rotor_id == jammed_rotor_id:
            for i, t in enumerate(t_arr):
                if t >= jam_time:
                    T_arr[i] = 0.0  # Jammed rotor loses thrust
        thrusts[rotor.label] = T_arr

    return thrusts


def plot_comprehensive_transients(history, params, thrusts, ext_force_func,
                                    failure_time, event_label, rotor_label,
                                    nz_limits, output_path):
    """Generate 4-page comprehensive time transient plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t = history.t
    states = history.states

    # Unpack states
    u = states[:, 0]
    v = states[:, 1]
    w = states[:, 2]
    p = states[:, 3]
    q = states[:, 4]
    r = states[:, 5]
    phi = states[:, 6]
    theta = states[:, 7]
    psi = states[:, 8]
    xe = states[:, 9]
    ye = states[:, 10]
    ze = states[:, 11]

    # Compute body-axis accelerations
    du, dv, dw = compute_body_accels(params, history, ext_force_func)

    # Common formatting
    fail_kw = dict(color='red', linewidth=1.5, linestyle='--', alpha=0.7, zorder=3)
    grid_kw = dict(alpha=0.3, linewidth=0.5)
    ts_label = f'{event_label} t={failure_time}s'

    def add_fail(ax):
        ax.axvline(x=failure_time, **fail_kw, label=ts_label)

    def style_ax(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(**grid_kw)
        ax.legend(fontsize=7, loc='best')

    nz_pos, nz_neg = nz_limits

    # ================================================================
    # Figure 1: Attitude + Angular rates + Angular accelerations + nz
    # ================================================================
    fig1, axes1 = plt.subplots(4, 3, figsize=(18, 16))
    fig1.suptitle(
        f'{event_label}: {rotor_label} — Attitude & Angular Dynamics\n'
        f'KC-100 Lift+Cruise, Hover, mass={params.mass:.0f} kg, '
        f'Ixx={params.Ixx:.0f} kg-m²',
        fontsize=13, fontweight='bold')

    # Row 0: Euler angles
    ax = axes1[0, 0]
    ax.plot(t, np.degrees(phi), 'b-', linewidth=1.2, label=r'$\phi$ (roll)')
    add_fail(ax)
    style_ax(ax, 'Angle (deg)', r'(a) Roll Angle $\phi$')

    ax = axes1[0, 1]
    ax.plot(t, np.degrees(theta), 'g-', linewidth=1.2, label=r'$\theta$ (pitch)')
    add_fail(ax)
    style_ax(ax, 'Angle (deg)', r'(b) Pitch Angle $\theta$')

    ax = axes1[0, 2]
    ax.plot(t, np.degrees(psi), 'm-', linewidth=1.2, label=r'$\psi$ (yaw)')
    add_fail(ax)
    style_ax(ax, 'Angle (deg)', r'(c) Yaw Angle $\psi$')

    # Row 1: Angular rates
    ax = axes1[1, 0]
    ax.plot(t, np.degrees(p), 'b-', linewidth=1.2, label='p (roll rate)')
    add_fail(ax)
    style_ax(ax, 'Rate (deg/s)', '(d) Roll Rate p')

    ax = axes1[1, 1]
    ax.plot(t, np.degrees(q), 'g-', linewidth=1.2, label='q (pitch rate)')
    add_fail(ax)
    style_ax(ax, 'Rate (deg/s)', '(e) Pitch Rate q')

    ax = axes1[1, 2]
    ax.plot(t, np.degrees(r), 'm-', linewidth=1.2, label='r (yaw rate)')
    add_fail(ax)
    style_ax(ax, 'Rate (deg/s)', '(f) Yaw Rate r')

    # Row 2: Angular accelerations
    ax = axes1[2, 0]
    ax.plot(t, np.degrees(history.p_dot), 'b-', linewidth=1.2,
            label=r'$\dot{p}$ (roll accel)')
    add_fail(ax)
    style_ax(ax, r'Accel (deg/s$^2$)', r'(g) Roll Accel $\dot{p}$')

    ax = axes1[2, 1]
    ax.plot(t, np.degrees(history.q_dot), 'g-', linewidth=1.2,
            label=r'$\dot{q}$ (pitch accel)')
    add_fail(ax)
    style_ax(ax, r'Accel (deg/s$^2$)', r'(h) Pitch Accel $\dot{q}$')

    ax = axes1[2, 2]
    ax.plot(t, np.degrees(history.r_dot), 'm-', linewidth=1.2,
            label=r'$\dot{r}$ (yaw accel)')
    add_fail(ax)
    style_ax(ax, r'Accel (deg/s$^2$)', r'(i) Yaw Accel $\dot{r}$')

    # Row 3: Load factors
    ax = axes1[3, 0]
    ax.plot(t, history.nz, 'k-', linewidth=1.5, label='nz')
    ax.axhline(y=nz_pos, color='red', linestyle=':', alpha=0.6,
               label=f'+{nz_pos:.1f}g limit')
    ax.axhline(y=nz_neg, color='red', linestyle=':', alpha=0.6,
               label=f'{nz_neg:.2f}g limit')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)
    add_fail(ax)
    style_ax(ax, 'Load Factor (g)', '(j) Normal Load Factor nz')

    ax = axes1[3, 1]
    ax.plot(t, history.ny, 'k-', linewidth=1.5, label='ny')
    add_fail(ax)
    style_ax(ax, 'Load Factor (g)', '(k) Lateral Load Factor ny')

    ax = axes1[3, 2]
    ax.plot(t, history.alpha_deg, 'c-', linewidth=1.2, label=r'$\alpha$')
    ax2 = ax.twinx()
    ax2.plot(t, history.beta_deg, 'orange', linewidth=1.2, label=r'$\beta$')
    add_fail(ax)
    ax.set_ylabel(r'$\alpha$ (deg)', fontsize=9, color='c')
    ax2.set_ylabel(r'$\beta$ (deg)', fontsize=9, color='orange')
    ax.set_title(r'(l) Aero Angles $\alpha$, $\beta$', fontsize=10,
                 fontweight='bold')
    ax.grid(**grid_kw)
    lines1 = ax.get_lines() + ax2.get_lines()
    ax.legend(lines1, [l.get_label() for l in lines1], fontsize=7)

    for ax_row in axes1:
        for ax in ax_row:
            ax.set_xlabel('Time (s)', fontsize=8)

    fig1.tight_layout()
    path1 = output_path.replace('.png', '_attitude_angular.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ================================================================
    # Figure 2: Position + Velocities + Accelerations
    # ================================================================
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))
    fig2.suptitle(
        f'{event_label}: {rotor_label} — Position, Velocity, Acceleration\n'
        f'KC-100 Lift+Cruise, Hover',
        fontsize=13, fontweight='bold')

    # Row 0: Earth-fixed position
    ax = axes2[0, 0]
    ax.plot(t, xe, 'r-', linewidth=1.2, label='xe (forward)')
    add_fail(ax)
    style_ax(ax, 'Position (m)', '(a) Earth X Position (Forward)')

    ax = axes2[0, 1]
    ax.plot(t, ye, 'g-', linewidth=1.2, label='ye (lateral)')
    add_fail(ax)
    style_ax(ax, 'Position (m)', '(b) Earth Y Position (Lateral)')

    ax = axes2[0, 2]
    ax.plot(t, ze, 'b-', linewidth=1.2, label='ze (vertical)')
    add_fail(ax)
    style_ax(ax, 'Position (m)', '(c) Earth Z Position (Down)')

    # Row 1: Body-axis velocities
    ax = axes2[1, 0]
    ax.plot(t, u, 'r-', linewidth=1.2, label='u (forward)')
    add_fail(ax)
    style_ax(ax, 'Velocity (m/s)', '(d) Body-axis u (Forward)')

    ax = axes2[1, 1]
    ax.plot(t, v, 'g-', linewidth=1.2, label='v (lateral)')
    add_fail(ax)
    style_ax(ax, 'Velocity (m/s)', '(e) Body-axis v (Lateral)')

    ax = axes2[1, 2]
    ax.plot(t, w, 'b-', linewidth=1.2, label='w (vertical)')
    add_fail(ax)
    style_ax(ax, 'Velocity (m/s)', '(f) Body-axis w (Down)')

    # Row 2: Body-axis linear accelerations
    ax = axes2[2, 0]
    ax.plot(t, du, 'r-', linewidth=1.2, label=r'$\dot{u}$')
    add_fail(ax)
    style_ax(ax, r'Accel (m/s$^2$)', r'(g) Forward Accel $\dot{u}$')

    ax = axes2[2, 1]
    ax.plot(t, dv, 'g-', linewidth=1.2, label=r'$\dot{v}$')
    add_fail(ax)
    style_ax(ax, r'Accel (m/s$^2$)', r'(h) Lateral Accel $\dot{v}$')

    ax = axes2[2, 2]
    ax.plot(t, dw, 'b-', linewidth=1.2, label=r'$\dot{w}$')
    add_fail(ax)
    style_ax(ax, r'Accel (m/s$^2$)', r'(i) Vertical Accel $\dot{w}$')

    for ax_row in axes2:
        for ax in ax_row:
            ax.set_xlabel('Time (s)', fontsize=8)

    fig2.tight_layout()
    path2 = output_path.replace('.png', '_position_velocity.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ================================================================
    # Figure 3: Rotor thrust schedule
    # ================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))
    fig3.suptitle(
        f'{event_label}: {rotor_label} — Per-Rotor Thrust Schedule\n'
        f'KC-100 Lift+Cruise, Hover',
        fontsize=13, fontweight='bold')

    colors_rotor = ['#1f77b4', '#ff7f0e', '#2ca02c',
                    '#d62728', '#9467bd', '#8c564b']
    for i, (label, T_arr) in enumerate(thrusts.items()):
        lw = 2.5 if T_arr.min() < T_arr.max() * 0.5 else 1.0
        ax3.plot(t, T_arr, '-', linewidth=lw,
                 color=colors_rotor[i % len(colors_rotor)],
                 label=label, alpha=0.9)

    ax3.axvline(x=failure_time, **fail_kw, label=ts_label)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Thrust (N)', fontsize=11)
    ax3.set_title('Individual Rotor Thrust vs Time', fontsize=11)
    ax3.legend(fontsize=9, ncol=3, loc='lower left')
    ax3.grid(**grid_kw)
    ax3.set_ylim(bottom=-50)

    # Add total thrust line
    total_T = np.zeros(len(t))
    for T_arr in thrusts.values():
        total_T += T_arr
    ax3.plot(t, total_T, 'k--', linewidth=2.0, alpha=0.6, label='Total Thrust')
    ax3.axhline(y=params.mass * 9.80665, color='gray', linestyle=':',
                alpha=0.5, label=f'Weight ({params.mass*9.80665:.0f} N)')
    ax3.legend(fontsize=9, ncol=3, loc='lower left')

    fig3.tight_layout()
    path3 = output_path.replace('.png', '_rotor_thrust.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {path3}")

    return [path1, path2, path3]


def main():
    setup_logging("WARNING")
    print("=" * 80)
    print("  OEI Time Transient Analysis — Comprehensive Plots")
    print("=" * 80)

    t0_wall = time.time()
    timestamp = datetime.now()
    output_dir = f"oei_transient_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Setup (same as main study) ----
    model = parse_bdf("tests/validation/KC100/p400r3-free-trim.bdf")
    print(f"[1] BDF parsed: {len(model.nodes)} nodes")

    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000
    weight_N = total_mass_kg * 9.80665
    vtol_config = VTOLConfig.kc100_lift_cruise()

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

    nz_pos = config.nz_max(weight_N)
    nz_neg = config.nz_min(weight_N)
    nz_limits = (nz_pos, nz_neg)

    print(f"[2] Aircraft: mass={params.mass:.0f} kg, "
          f"Ixx={params.Ixx:.0f}, Iyy={params.Iyy:.0f}, Izz={params.Izz:.0f}")

    # ---- Simulation parameters ----
    t_sim = 5.0
    t_fail = 1.0
    dt = 0.005

    # ---- Worst-case OEI: L3 (outboard left, Y=-4500mm) ----
    lift_rotors = vtol_config.lift_rotors
    # Find L3 — the outboard left rotor
    worst_rotor = None
    for rotor in lift_rotors:
        if 'L3' in rotor.label:
            worst_rotor = rotor
            break
    if worst_rotor is None:
        # Fallback: pick rotor with largest |Y|
        worst_rotor = max(lift_rotors,
                          key=lambda r: abs(r.hub_position[1]))

    print(f"\n[3] OEI worst-case: {worst_rotor.label} "
          f"(Y={worst_rotor.hub_position[1]:.0f} mm)")

    # OEI force callback
    oei_ext_force = make_oei_force_func(
        vtol_config, failed_rotor_id=worst_rotor.rotor_id,
        failure_time=t_fail, weight_N=weight_N, rho=rho_sl,
        cg_position=cg_xyz)

    # Initial state: hover
    initial_state = AircraftState(u=0.1, v=0.0, w=0.0)
    ctrl = ControlInput()
    control_func = lambda t: ctrl

    # Integrate
    print(f"    Integrating 6-DOF (t={t_sim}s, dt={dt*1000:.0f}ms)...")
    history_oei = integrate_6dof(
        params, initial_state, control_func,
        t_span=(0.0, t_sim), dt=dt,
        external_force_func=oei_ext_force)
    compute_nz_from_history(params, history_oei)

    # Rotor thrust schedule
    thrusts_oei = compute_rotor_thrusts_oei(
        vtol_config, worst_rotor.rotor_id, t_fail,
        weight_N, rho_sl, cg_xyz, history_oei.t)

    print(f"    Generating comprehensive plots...")
    oei_paths = plot_comprehensive_transients(
        history_oei, params, thrusts_oei, oei_ext_force,
        t_fail, "OEI Failure", worst_rotor.label,
        nz_limits, os.path.join(output_dir, "oei_L3.png"))

    # ---- Worst-case Jam: L3 (same rotor) ----
    print(f"\n[4] Rotor Jam worst-case: {worst_rotor.label}")

    jam_ext_force = make_rotor_jam_force_func(
        vtol_config, jammed_rotor_id=worst_rotor.rotor_id,
        jam_time=t_fail, weight_N=weight_N, rho=rho_sl,
        cg_position=cg_xyz)

    initial_state_jam = AircraftState(u=0.1, v=0.0, w=0.0)
    control_func_jam = lambda t: ctrl

    history_jam = integrate_6dof(
        params, initial_state_jam, control_func_jam,
        t_span=(0.0, t_sim), dt=dt,
        external_force_func=jam_ext_force)
    compute_nz_from_history(params, history_jam)

    thrusts_jam = compute_rotor_thrusts_jam(
        vtol_config, worst_rotor.rotor_id, t_fail,
        weight_N, rho_sl, history_jam.t)

    jam_paths = plot_comprehensive_transients(
        history_jam, params, thrusts_jam, jam_ext_force,
        t_fail, "Rotor Jam", worst_rotor.label,
        nz_limits, os.path.join(output_dir, "jam_L3.png"))

    # ---- Also regenerate fixed VTOL model plot ----
    print(f"\n[5] Regenerating VTOL model plot...")
    from nastaero.visualization.cert_plot import plot_vtol_model
    model_path = plot_vtol_model(
        model, vtol_config,
        output_path=os.path.join(output_dir, "00_vtol_model.png"))
    print(f"  Saved: {model_path}")

    total_time = time.time() - t0_wall
    print(f"\n{'='*80}")
    print(f"  Complete ({total_time:.1f}s)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
