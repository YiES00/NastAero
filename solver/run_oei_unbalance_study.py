#!/usr/bin/env python3
"""OEI Rotor Failure Unbalancing Study for KC-100 Lift+Cruise VTOL.

Hypothesis: Sudden single-rotor failure in hover creates asymmetric thrust
that produces transient angular accelerations and load factors potentially
exceeding conventional flight envelope limits. If true, OEI dynamic cases
must be included in the certification design load selection to avoid
missing critical structural loads.

Method:
  1. Compute hover trim for 6 lift rotors (balanced)
  2. For each rotor, simulate sudden failure at t=1.0s in hover
  3. Also simulate rotor jam (sudden seizure with brake torque)
  4. Extract peak nz, p_dot (roll accel), r_dot (yaw accel), roll angle
  5. Compute moment arm × thrust loss → expected rolling moment
  6. Compare OEI response peaks with conventional V-n envelope limits
  7. Identify which rotor failure produces the most severe unbalancing
"""
import os
import sys
import time
import csv
import math
from datetime import datetime

import numpy as np

# Add solver to path
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
    compute_nz_from_history,
)
from nastaero.aero.dlm import compute_rigid_clalpha


def main():
    setup_logging("WARNING")

    print("=" * 80)
    print("  OEI Rotor Failure Unbalancing Study")
    print("  KC-100 Lift+Cruise VTOL Configuration")
    print("=" * 80)

    t0 = time.time()
    timestamp = datetime.now()
    output_dir = f"oei_study_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Parse BDF + Setup ----
    model = parse_bdf("tests/validation/KC100/p400r3-free-trim.bdf")
    print(f"\n[1] BDF parsed: {len(model.nodes)} nodes")

    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000
    weight_N = total_mass_kg * 9.80665
    print(f"    Mass: {total_mass_kg:.1f} kg, Weight: {weight_N:.0f} N")

    # Aircraft config
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
        wing_area_m2=17.0,
        CLalpha=clalpha_vlm,
        mean_chord_m=1.6,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0, rudder_max_deg=25.0, elevator_max_deg=25.0),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101], nose_gear_node_ids=[102],
            main_gear_x=4200.0, nose_gear_x=1500.0,
            strut_efficiency=0.7, stroke=0.25, sink_rate_fps=10.0),
        gust_Ude_VC_fps=50.0, gust_Ude_VD_fps=25.0,
    )

    # VTOL config
    vtol_config = VTOLConfig.kc100_lift_cruise()

    # ---- 2. Aircraft Parameters for 6-DOF ----
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
        Ixx=inertia["Ixx"],
        Iyy=inertia["Iyy"],
        Izz=inertia["Izz"],
        Ixz=inertia["Ixz"],
        derivs=derivs,
    )

    print(f"\n[2] Aircraft inertia properties:")
    print(f"    Mass: {params.mass:.1f} kg")
    print(f"    Ixx: {params.Ixx:.0f} kg-m2 (roll)")
    print(f"    Iyy: {params.Iyy:.0f} kg-m2 (pitch)")
    print(f"    Izz: {params.Izz:.0f} kg-m2 (yaw)")
    print(f"    Ixz: {params.Ixz:.0f} kg-m2")

    # ---- 3. Rotor analysis ----
    rho_sl, _, _ = isa_atmosphere(0.0)
    lift_rotors = vtol_config.lift_rotors
    n_lift = len(lift_rotors)
    thrust_per_rotor = weight_N / n_lift

    print(f"\n[3] Rotor hover analysis (rho={rho_sl:.4f} kg/m3)")
    print(f"    Lift rotors: {n_lift}")
    print(f"    Thrust per rotor: {thrust_per_rotor:.1f} N")

    # Compute BEMT loads for each rotor
    rotor_loads = {}
    for rotor in lift_rotors:
        solver = BEMTSolver(rotor.blade, rotor.n_blades)
        loads = solver.solve_for_thrust(thrust_per_rotor, rotor.rpm_hover, rho_sl)
        rotor_loads[rotor.rotor_id] = loads
        print(f"    {rotor.label}: T={loads.thrust:.1f}N, "
              f"Q={loads.torque:.2f}N-m, P={loads.power:.0f}W")

    # ---- 4. Geometric analysis of rotor positions ----
    print(f"\n[4] Rotor position analysis (moment arms from CG):")
    print(f"    CG position: ({cg_xyz[0]:.0f}, {cg_xyz[1]:.0f}, {cg_xyz[2]:.0f}) mm")

    rotor_arms = {}
    for rotor in lift_rotors:
        dx = (rotor.hub_position[0] - cg_xyz[0]) * mm_to_m  # m
        dy = rotor.hub_position[1] * mm_to_m  # m
        dz = (rotor.hub_position[2] - cg_xyz[2]) * mm_to_m  # m
        arm_roll = abs(dy)       # Y-arm for rolling moment
        arm_pitch = abs(dx)      # X-arm for pitching moment
        arm_total = math.sqrt(dx**2 + dy**2)

        # Expected rolling moment from loss of one rotor
        roll_moment = thrust_per_rotor * arm_roll  # N-m
        # Expected roll acceleration
        roll_accel = roll_moment / params.Ixx  # rad/s2

        rotor_arms[rotor.rotor_id] = {
            'dx': dx, 'dy': dy, 'dz': dz,
            'arm_roll': arm_roll, 'arm_pitch': arm_pitch,
            'arm_total': arm_total,
            'roll_moment': roll_moment,
            'roll_accel': roll_accel,
        }

        print(f"    {rotor.label}: Y-arm={dy*1000:.0f}mm ({arm_roll:.2f}m), "
              f"X-arm={dx*1000:.0f}mm ({arm_pitch:.2f}m)")
        print(f"      ΔM_roll = {roll_moment:.1f} N-m  →  "
              f"p_dot = {math.degrees(roll_accel):.1f} deg/s2")

    # ---- 5. OEI Dynamic Simulations ----
    print(f"\n[5] OEI Dynamic Simulations (hover, t_fail=1.0s, t_sim=5.0s)")
    print(f"    {'='*75}")

    t_sim = 5.0       # Total simulation time
    t_fail = 1.0       # Failure time
    dt = 0.005         # 200 Hz
    altitudes = [0.0]  # Sea level

    oei_results = []
    jam_results = []

    for alt in altitudes:
        rho, _, _ = isa_atmosphere(alt)
        params.rho = rho

        # ------ OEI simulations (thrust loss) ------
        print(f"\n    --- OEI (Thrust Loss) at {alt:.0f}m ---")

        for rotor in lift_rotors:
            rid = rotor.rotor_id

            # Create OEI force callback (pass CG position for moment arms)
            ext_force = make_oei_force_func(
                vtol_config, failed_rotor_id=rid,
                failure_time=t_fail, weight_N=weight_N, rho=rho,
                cg_position=cg_xyz)

            # Initial state: near-hover
            initial_state = AircraftState(u=0.1, v=0.0, w=0.0)
            ctrl = ControlInput()
            control_func = lambda t: ctrl

            # Integrate
            history = integrate_6dof(
                params, initial_state, control_func,
                t_span=(0.0, t_sim), dt=dt,
                external_force_func=ext_force)

            compute_nz_from_history(params, history)

            # Extract peaks AFTER failure
            i_fail = int(t_fail / dt)
            post_fail_slice = slice(i_fail, None)

            nz_post = history.nz[post_fail_slice]
            p_rate_post = history.p_rate[post_fail_slice]
            q_rate_post = history.q_rate[post_fail_slice]
            r_rate_post = history.r_rate[post_fail_slice]
            p_dot_post = history.p_dot[post_fail_slice]
            q_dot_post = history.q_dot[post_fail_slice]
            r_dot_post = history.r_dot[post_fail_slice]
            phi_post = history.states[post_fail_slice, 6]  # Roll angle
            theta_post = history.states[post_fail_slice, 7]  # Pitch angle

            # Peak values
            nz_max = float(np.max(nz_post))
            nz_min = float(np.min(nz_post))
            p_dot_max = float(np.max(np.abs(p_dot_post)))
            q_dot_max = float(np.max(np.abs(q_dot_post)))
            r_dot_max = float(np.max(np.abs(r_dot_post)))
            p_max = float(np.max(np.abs(p_rate_post)))
            q_max = float(np.max(np.abs(q_rate_post)))
            r_max = float(np.max(np.abs(r_rate_post)))
            phi_max = float(np.max(np.abs(phi_post)))
            theta_max = float(np.max(np.abs(theta_post)))

            # Time to reach 5 deg roll
            t_5deg = None
            for i in range(len(phi_post)):
                if abs(phi_post[i]) > math.radians(5.0):
                    t_5deg = i * dt
                    break

            # Time to reach 30 deg roll
            t_30deg = None
            for i in range(len(phi_post)):
                if abs(phi_post[i]) > math.radians(30.0):
                    t_30deg = i * dt
                    break

            # Expected vs actual roll accel
            expected_pdot = rotor_arms[rid]['roll_accel']

            result = {
                'rotor_label': rotor.label,
                'rotor_id': rid,
                'event_type': 'OEI',
                'altitude_m': alt,
                'y_arm_m': rotor_arms[rid]['dy'],
                'arm_roll_m': rotor_arms[rid]['arm_roll'],
                'expected_pdot_rads2': expected_pdot,
                'nz_max': nz_max,
                'nz_min': nz_min,
                'p_dot_max_rads2': p_dot_max,
                'p_dot_max_degs2': math.degrees(p_dot_max),
                'q_dot_max_rads2': q_dot_max,
                'q_dot_max_degs2': math.degrees(q_dot_max),
                'r_dot_max_rads2': r_dot_max,
                'r_dot_max_degs2': math.degrees(r_dot_max),
                'p_max_rads': p_max,
                'p_max_degs': math.degrees(p_max),
                'q_max_rads': q_max,
                'r_max_rads': r_max,
                'phi_max_deg': math.degrees(phi_max),
                'theta_max_deg': math.degrees(theta_max),
                't_5deg_roll_s': t_5deg,
                't_30deg_roll_s': t_30deg,
                'roll_moment_Nm': rotor_arms[rid]['roll_moment'],
            }
            oei_results.append(result)

            # Store history for worst-case plotting
            result['_history'] = history

            print(f"    {rotor.label} (Y={rotor.hub_position[1]:+.0f}mm):")
            print(f"      p_dot_max = {math.degrees(p_dot_max):6.1f} deg/s2  "
                  f"(expected: {math.degrees(expected_pdot):6.1f} deg/s2)")
            print(f"      p_max     = {math.degrees(p_max):6.1f} deg/s    "
                  f"phi_max = {math.degrees(phi_max):6.1f} deg")
            print(f"      nz: [{nz_min:.3f}, {nz_max:.3f}]")
            if t_5deg is not None:
                print(f"      t(5 deg roll) = {t_5deg:.3f}s  ", end="")
            if t_30deg is not None:
                print(f"t(30 deg roll) = {t_30deg:.3f}s")
            else:
                print()

        # ------ Rotor Jam simulations (seizure with brake torque) ------
        print(f"\n    --- Rotor Jam (Seizure) at {alt:.0f}m ---")

        for rotor in lift_rotors:
            rid = rotor.rotor_id

            ext_force = make_rotor_jam_force_func(
                vtol_config, jammed_rotor_id=rid,
                jam_time=t_fail, weight_N=weight_N, rho=rho,
                cg_position=cg_xyz)

            initial_state = AircraftState(u=0.1, v=0.0, w=0.0)
            ctrl = ControlInput()
            control_func = lambda t: ctrl

            history = integrate_6dof(
                params, initial_state, control_func,
                t_span=(0.0, t_sim), dt=dt,
                external_force_func=ext_force)

            compute_nz_from_history(params, history)

            i_fail = int(t_fail / dt)
            post_fail_slice = slice(i_fail, None)

            r_dot_post = history.r_dot[post_fail_slice]
            p_dot_post = history.p_dot[post_fail_slice]
            r_rate_post = history.r_rate[post_fail_slice]
            phi_post = history.states[post_fail_slice, 6]

            r_dot_max = float(np.max(np.abs(r_dot_post)))
            p_dot_max = float(np.max(np.abs(p_dot_post)))
            r_max = float(np.max(np.abs(r_rate_post)))
            phi_max = float(np.max(np.abs(phi_post)))

            # Jam produces torque impulse → primarily yaw response
            result = {
                'rotor_label': rotor.label,
                'rotor_id': rid,
                'event_type': 'JAM',
                'altitude_m': alt,
                'r_dot_max_rads2': r_dot_max,
                'r_dot_max_degs2': math.degrees(r_dot_max),
                'p_dot_max_rads2': p_dot_max,
                'p_dot_max_degs2': math.degrees(p_dot_max),
                'r_max_rads': r_max,
                'r_max_degs': math.degrees(r_max),
                'phi_max_deg': math.degrees(phi_max),
                'nz_max': float(np.max(history.nz[post_fail_slice])),
                'nz_min': float(np.min(history.nz[post_fail_slice])),
                'jam_torque_Nm': rotor_loads[rid].torque * 3.0,
            }
            jam_results.append(result)
            result['_history'] = history

            print(f"    {rotor.label} (Y={rotor.hub_position[1]:+.0f}mm):")
            print(f"      r_dot_max = {math.degrees(r_dot_max):6.1f} deg/s2  "
                  f"(jam torque: {rotor_loads[rid].torque*3.0:.2f} N-m)")
            print(f"      p_dot_max = {math.degrees(p_dot_max):6.1f} deg/s2  "
                  f"(cross-coupling via Ixz)")
            print(f"      r_max = {math.degrees(r_max):6.1f} deg/s")

    # ---- 6. Find worst-case rotor failure ----
    print(f"\n{'='*80}")
    print(f"  WORST-CASE ANALYSIS")
    print(f"{'='*80}")

    # OEI worst case = max roll moment arm (outboard rotors)
    worst_oei = max(oei_results, key=lambda r: r['arm_roll_m'])
    print(f"\n  Worst OEI (max roll arm): {worst_oei['rotor_label']}")
    print(f"    Roll arm: {worst_oei['arm_roll_m']:.2f} m")
    print(f"    Roll moment: {worst_oei['roll_moment_Nm']:.1f} N-m")
    print(f"    Peak roll accel: {worst_oei['p_dot_max_degs2']:.1f} deg/s2")
    print(f"    Peak roll rate: {worst_oei['p_max_degs']:.1f} deg/s")
    print(f"    Max bank angle: {worst_oei['phi_max_deg']:.1f} deg")
    if worst_oei['t_5deg_roll_s'] is not None:
        print(f"    Time to 5 deg: {worst_oei['t_5deg_roll_s']:.3f} s")
    if worst_oei['t_30deg_roll_s'] is not None:
        print(f"    Time to 30 deg: {worst_oei['t_30deg_roll_s']:.3f} s")

    # Jam worst case = max yaw response
    worst_jam = max(jam_results, key=lambda r: r['r_dot_max_rads2'])
    print(f"\n  Worst Rotor Jam (max yaw accel): {worst_jam['rotor_label']}")
    print(f"    Jam torque: {worst_jam['jam_torque_Nm']:.2f} N-m")
    print(f"    Peak yaw accel: {worst_jam['r_dot_max_degs2']:.1f} deg/s2")
    print(f"    Peak yaw rate: {worst_jam['r_max_degs']:.1f} deg/s")

    # ---- 7. Comparison with conventional V-n envelope ----
    print(f"\n{'='*80}")
    print(f"  COMPARISON: OEI LOADS vs CONVENTIONAL V-n ENVELOPE")
    print(f"{'='*80}")

    # FAR 23 normal category V-n limits
    nz_pos_limit = config.nz_max(weight_N)  # Typically +3.8g
    nz_neg_limit = config.nz_min(weight_N)  # Typically -1.52g

    print(f"\n  Conventional V-n envelope:")
    print(f"    nz_pos_limit = +{nz_pos_limit:.2f}g")
    print(f"    nz_neg_limit = {nz_neg_limit:.2f}g")

    # Compute equivalent incremental nz from OEI roll acceleration
    # When aircraft rolls to angle phi, vertical component of thrust decreases:
    # nz_eff = cos(phi) * T_remaining / W
    # At phi = 30 deg: nz_eff = cos(30) * (5/6) = 0.722 → Δnz = -0.278
    # At phi = 60 deg: nz_eff = cos(60) * (5/6) = 0.417 → Δnz = -0.583

    print(f"\n  OEI load factor analysis (worst case: {worst_oei['rotor_label']}):")
    phi_max_rad = math.radians(worst_oei['phi_max_deg'])
    thrust_remaining_frac = (n_lift - 1) / n_lift
    nz_oei_vert = math.cos(phi_max_rad) * thrust_remaining_frac
    delta_nz_vertical = 1.0 - nz_oei_vert

    print(f"    Remaining thrust fraction: {thrust_remaining_frac:.3f}")
    print(f"    At phi_max={worst_oei['phi_max_deg']:.1f} deg:")
    print(f"      nz_vertical = cos(phi) * T_rem/W = {nz_oei_vert:.3f}")
    print(f"      Delta_nz (vertical) = {delta_nz_vertical:.3f}")

    # Lateral load factor from roll acceleration
    # ny = p_dot * y_arm / g for a structural station at y from CG
    g = 9.80665
    wing_tip_y = 5.617  # KC-100 wing tip Y position in meters
    wing_root_y = 0.465  # Wing root Y

    ny_wing_tip = worst_oei['p_dot_max_rads2'] * wing_tip_y / g
    ny_wing_root = worst_oei['p_dot_max_rads2'] * wing_root_y / g

    print(f"\n  Lateral load factors from roll acceleration:")
    print(f"    ny at wing tip (Y={wing_tip_y:.3f}m): {ny_wing_tip:.3f}g")
    print(f"    ny at wing root (Y={wing_root_y:.3f}m): {ny_wing_root:.3f}g")

    # Combined load factor at wing tip: nz component + ny component
    # This is the key comparison for structural loads
    combined_nz_tip = math.sqrt(nz_oei_vert**2 + ny_wing_tip**2)
    print(f"\n  Combined load factor at wing tip:")
    print(f"    |n_combined| = sqrt(nz^2 + ny^2) = {combined_nz_tip:.3f}g")

    # Incremental bending moment at wing root from OEI
    # M_roll = p_dot * Ixx at wing root
    # Additional bending from asymmetric thrust loss
    wing_mass_kg = total_mass_kg * 0.15  # Approx 15% of mass in each wing
    M_bending_oei = (worst_oei['p_dot_max_rads2'] * params.Ixx)
    M_bending_thrust = worst_oei['roll_moment_Nm']

    print(f"\n  Wing root bending moment contributions:")
    print(f"    From roll inertia (p_dot * Ixx): {M_bending_oei:.0f} N-m")
    print(f"    From thrust asymmetry: {M_bending_thrust:.0f} N-m")
    print(f"    Total OEI roll moment: {M_bending_oei + M_bending_thrust:.0f} N-m")

    # ---- 8. Hypothesis verification ----
    print(f"\n{'='*80}")
    print(f"  HYPOTHESIS VERIFICATION")
    print(f"  'OEI unbalancing produces design-critical loads'")
    print(f"{'='*80}")

    # Severity scoring
    findings = []

    # Check 1: Does OEI nz exceed conventional limits?
    oei_nz_range = worst_oei['nz_max'] - worst_oei['nz_min']
    if worst_oei['nz_min'] < nz_neg_limit:
        findings.append(f"CRITICAL: OEI nz_min ({worst_oei['nz_min']:.2f}) "
                       f"< V-n neg limit ({nz_neg_limit:.2f})")
    else:
        findings.append(f"  nz range [{worst_oei['nz_min']:.3f}, "
                       f"{worst_oei['nz_max']:.3f}] within V-n limits")

    # Check 2: Does roll acceleration create significant wing loads?
    p_dot_threshold = math.radians(10.0)  # 10 deg/s2 is significant
    if worst_oei['p_dot_max_rads2'] > p_dot_threshold:
        findings.append(f"SIGNIFICANT: Peak roll accel "
                       f"{worst_oei['p_dot_max_degs2']:.1f} deg/s2 > 10 deg/s2 threshold")
    else:
        findings.append(f"  Roll accel {worst_oei['p_dot_max_degs2']:.1f} deg/s2 "
                       f"< 10 deg/s2 threshold")

    # Check 3: Time to reach large roll angles
    if worst_oei['t_30deg_roll_s'] is not None and worst_oei['t_30deg_roll_s'] < 2.0:
        findings.append(f"CRITICAL: Reaches 30 deg roll in "
                       f"{worst_oei['t_30deg_roll_s']:.2f}s (< 2s)")
    elif worst_oei['t_5deg_roll_s'] is not None and worst_oei['t_5deg_roll_s'] < 0.5:
        findings.append(f"SIGNIFICANT: Reaches 5 deg roll in "
                       f"{worst_oei['t_5deg_roll_s']:.3f}s (< 0.5s)")

    # Check 4: Wing tip lateral load factor
    if ny_wing_tip > 0.1:
        findings.append(f"SIGNIFICANT: Wing tip ny = {ny_wing_tip:.3f}g from OEI roll")
    if ny_wing_tip > 0.3:
        findings.append(f"CRITICAL: Wing tip ny = {ny_wing_tip:.3f}g exceeds 0.3g")

    # Check 5: Jam yaw response
    if worst_jam['r_dot_max_degs2'] > 5.0:
        findings.append(f"SIGNIFICANT: Rotor jam yaw accel "
                       f"{worst_jam['r_dot_max_degs2']:.1f} deg/s2")

    print(f"\n  Findings:")
    for f in findings:
        print(f"    {f}")

    # Overall verdict
    critical_count = sum(1 for f in findings if f.startswith("CRITICAL"))
    significant_count = sum(1 for f in findings if f.startswith("SIGNIFICANT"))

    print(f"\n  Verdict: {critical_count} CRITICAL, "
          f"{significant_count} SIGNIFICANT findings")
    if critical_count > 0:
        print(f"\n  >>> HYPOTHESIS CONFIRMED: OEI rotor failure produces load")
        print(f"  >>> conditions that may exceed conventional V-n envelope limits.")
        print(f"  >>> These cases MUST be included in design load selection.")
    elif significant_count > 0:
        print(f"\n  >>> HYPOTHESIS PARTIALLY CONFIRMED: OEI produces significant")
        print(f"  >>> transient loads. While within V-n limits, the asymmetric")
        print(f"  >>> loads may drive local structural sizing (pylon, wing root).")
    else:
        print(f"\n  >>> HYPOTHESIS NOT CONFIRMED: OEI loads are within")
        print(f"  >>> conventional V-n envelope for this configuration.")

    # ---- 9. Export results CSV ----
    csv_path = os.path.join(output_dir, "oei_study_results.csv")
    fieldnames = [k for k in oei_results[0].keys() if not k.startswith('_')]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in oei_results:
            writer.writerow({k: v for k, v in r.items() if not k.startswith('_')})
    print(f"\n  OEI results CSV: {csv_path}")

    jam_csv = os.path.join(output_dir, "jam_study_results.csv")
    jam_fields = [k for k in jam_results[0].keys() if not k.startswith('_')]
    with open(jam_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=jam_fields)
        writer.writeheader()
        for r in jam_results:
            writer.writerow({k: v for k, v in r.items() if not k.startswith('_')})
    print(f"  Jam results CSV: {jam_csv}")

    # ---- 10. Plots ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # --- Plot A: OEI severity comparison (bar chart) ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('OEI Rotor Failure Severity Comparison\nKC-100 Lift+Cruise',
                     fontsize=14, fontweight='bold')

        labels = [r['rotor_label'].replace('Lift Rotor ', '') for r in oei_results]
        x = np.arange(len(labels))
        bar_w = 0.6

        # Colors based on position: left=blue, right=red
        colors = []
        for r in oei_results:
            if r['y_arm_m'] < 0:
                colors.append('#2196F3')  # blue = left
            else:
                colors.append('#F44336')  # red = right

        # (a) Roll acceleration
        ax = axes[0, 0]
        vals = [r['p_dot_max_degs2'] for r in oei_results]
        ax.bar(x, vals, bar_w, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Peak Roll Acceleration (deg/s2)')
        ax.set_title('(a) Roll Acceleration (p_dot)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Threshold (10 deg/s2)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # (b) Max roll angle
        ax = axes[0, 1]
        vals = [r['phi_max_deg'] for r in oei_results]
        ax.bar(x, vals, bar_w, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Max Roll Angle (deg)')
        ax.set_title(f'(b) Roll Angle at t={t_sim:.0f}s')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='30 deg limit')
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5 deg controllability')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # (c) Roll moment arm
        ax = axes[1, 0]
        vals = [r['arm_roll_m'] for r in oei_results]
        moments = [r['roll_moment_Nm'] for r in oei_results]
        bars = ax.bar(x, vals, bar_w, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Roll Moment Arm (m)')
        ax.set_title('(c) Thrust Asymmetry Arm')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # Add moment annotation
        for i, (bar, m) in enumerate(zip(bars, moments)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{m:.0f}\nN-m', ha='center', va='bottom', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        # (d) Load factor range
        ax = axes[1, 1]
        for i, r in enumerate(oei_results):
            ax.plot([i, i], [r['nz_min'], r['nz_max']], 'o-',
                    color=colors[i], linewidth=2, markersize=6)
        ax.axhline(y=nz_pos_limit, color='red', linestyle='--', alpha=0.7,
                   label=f'+{nz_pos_limit:.1f}g limit')
        ax.axhline(y=nz_neg_limit, color='red', linestyle='--', alpha=0.7,
                   label=f'{nz_neg_limit:.1f}g limit')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('Load Factor (g)')
        ax.set_title('(d) nz Range During OEI')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_a_path = os.path.join(output_dir, "oei_severity_comparison.png")
        fig.savefig(plot_a_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved: {plot_a_path}")

        # --- Plot B: Worst-case OEI time history ---
        worst_hist = worst_oei['_history']
        t = worst_hist.t

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Worst-Case OEI Time History: {worst_oei["rotor_label"]} Failure\n'
                     f'KC-100 Hover, t_fail={t_fail:.1f}s',
                     fontsize=14, fontweight='bold')

        # Failure time marker
        def add_fail_line(ax):
            ax.axvline(x=t_fail, color='red', linewidth=1.5, linestyle='--',
                       alpha=0.8, label=f'Failure t={t_fail}s')

        # (a) Roll angle
        ax = axes[0, 0]
        phi_deg = np.degrees(worst_hist.states[:, 6])
        ax.plot(t, phi_deg, 'b-', linewidth=1.5)
        add_fail_line(ax)
        ax.set_ylabel('Roll Angle phi (deg)')
        ax.set_title('(a) Roll Angle')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (b) Pitch angle
        ax = axes[0, 1]
        theta_deg = np.degrees(worst_hist.states[:, 7])
        ax.plot(t, theta_deg, 'g-', linewidth=1.5)
        add_fail_line(ax)
        ax.set_ylabel('Pitch Angle theta (deg)')
        ax.set_title('(b) Pitch Angle')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (c) Roll rate
        ax = axes[1, 0]
        p_degs = np.degrees(worst_hist.p_rate)
        ax.plot(t, p_degs, 'b-', linewidth=1.5)
        add_fail_line(ax)
        ax.set_ylabel('Roll Rate p (deg/s)')
        ax.set_title('(c) Roll Rate')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (d) Roll acceleration
        ax = axes[1, 1]
        pdot_degs = np.degrees(worst_hist.p_dot)
        ax.plot(t, pdot_degs, 'r-', linewidth=1.5)
        add_fail_line(ax)
        ax.set_ylabel('Roll Accel p_dot (deg/s2)')
        ax.set_title('(d) Roll Acceleration')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (e) Load factor nz
        ax = axes[2, 0]
        ax.plot(t, worst_hist.nz, 'k-', linewidth=1.5)
        add_fail_line(ax)
        ax.axhline(y=nz_pos_limit, color='red', linestyle=':', alpha=0.5,
                   label=f'+{nz_pos_limit:.1f}g')
        ax.axhline(y=nz_neg_limit, color='red', linestyle=':', alpha=0.5,
                   label=f'{nz_neg_limit:.1f}g')
        ax.set_ylabel('Load Factor nz (g)')
        ax.set_xlabel('Time (s)')
        ax.set_title('(e) Normal Load Factor')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (f) Yaw rate
        ax = axes[2, 1]
        r_degs = np.degrees(worst_hist.r_rate)
        ax.plot(t, r_degs, 'm-', linewidth=1.5)
        add_fail_line(ax)
        ax.set_ylabel('Yaw Rate r (deg/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title('(f) Yaw Rate')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_b_path = os.path.join(output_dir, "oei_worst_case_history.png")
        fig.savefig(plot_b_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved: {plot_b_path}")

        # --- Plot C: OEI vs Jam comparison (key finding summary) ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('OEI vs Rotor Jam: Response Severity\nKC-100 Lift+Cruise',
                     fontsize=14, fontweight='bold')

        # (a) Roll acceleration: OEI vs Jam
        ax = axes[0]
        oei_pdots = [r['p_dot_max_degs2'] for r in oei_results]
        jam_pdots = [r['p_dot_max_degs2'] for r in jam_results]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, oei_pdots, w, label='OEI (thrust loss)',
               color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x + w/2, jam_pdots, w, label='Rotor Jam (seizure)',
               color='#FF5722', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Peak Roll Accel (deg/s2)')
        ax.set_title('(a) Roll Acceleration')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # (b) Yaw acceleration: OEI vs Jam
        ax = axes[1]
        oei_rdots = [r.get('r_dot_max_degs2', 0) for r in oei_results]
        jam_rdots = [r['r_dot_max_degs2'] for r in jam_results]
        ax.bar(x - w/2, oei_rdots, w, label='OEI (thrust loss)',
               color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(x + w/2, jam_rdots, w, label='Rotor Jam (seizure)',
               color='#FF5722', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Peak Yaw Accel (deg/s2)')
        ax.set_title('(b) Yaw Acceleration')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_c_path = os.path.join(output_dir, "oei_vs_jam_comparison.png")
        fig.savefig(plot_c_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved: {plot_c_path}")

    except ImportError:
        print("  (matplotlib not available — skipping plots)")

    # ---- 11. Summary ----
    total_time = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  OEI Study Complete ({total_time:.1f}s)")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")

    return {
        'oei_results': [{k: v for k, v in r.items() if not k.startswith('_')}
                        for r in oei_results],
        'jam_results': [{k: v for k, v in r.items() if not k.startswith('_')}
                        for r in jam_results],
        'worst_oei': worst_oei['rotor_label'],
        'worst_jam': worst_jam['rotor_label'],
        'findings': findings,
        'output_dir': output_dir,
    }


if __name__ == "__main__":
    main()
