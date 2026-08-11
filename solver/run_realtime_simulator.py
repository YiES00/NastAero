#!/usr/bin/env python3
"""Real-time flight simulator with live structural loads visualization.

Launches a PyQt5 window showing:
- 3D structural mesh with real-time deformation (color-coded)
- Time history plots: nz, alpha, beta, angular rates
- V-n diagram with live operating point
- Control surface sliders + keyboard input

Architecture:
    Startup (~5-8s): BDF parse → FEM assembly → SOL 103 → spline → AIC → ROM
    Runtime: 6-DOF dynamics @ 200 Hz + ModalROM @ 30 Hz + GUI @ 20 Hz

Usage:
    cd solver
    python run_realtime_simulator.py [--altitude 0] [--modes 20] [--speed 80]
"""
import sys
import time
import argparse
import numpy as np

from nastaero.bdf.parser import parse_bdf
from nastaero.config import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="NastAero Real-Time Flight Simulator")
    parser.add_argument('--bdf', type=str,
                        default='tests/validation/GACOMP/p400r3-free-trim.bdf',
                        help='Path to BDF model file')
    parser.add_argument('--altitude', type=float, default=0.0,
                        help='Simulation altitude in meters (default: 0 = sea level)')
    parser.add_argument('--modes', type=int, default=20,
                        help='Number of structural modes for ROM (default: 20)')
    parser.add_argument('--speed', type=float, default=80.0,
                        help='Initial cruise speed V_TAS in m/s (default: 80)')
    parser.add_argument('--mach', type=float, default=0.0,
                        help='Mach number for AIC (default: auto from speed)')
    parser.add_argument('--log', type=str, default='INFO',
                        help='Log level (default: INFO)')
    args = parser.parse_args()

    setup_logging(args.log)

    print("=" * 70)
    print("  NastAero Real-Time Flight Simulator")
    print("  Live Structural Loads Visualization")
    print("=" * 70)
    print()

    t_total = time.time()

    # ------------------------------------------------------------------
    # 1. Parse BDF model
    # ------------------------------------------------------------------
    print(f"[1/6] Parsing BDF model: {args.bdf}")
    t0 = time.time()
    model = parse_bdf(args.bdf)
    model.cross_reference()
    print(f"       {len(model.nodes):,} nodes, {len(model.elements):,} elements, "
          f"{len(model.masses)} CONM2  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Build aircraft configuration and derivatives
    # ------------------------------------------------------------------
    print(f"\n[2/6] Computing aerodynamic derivatives...")
    t0 = time.time()

    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig, SpeedSchedule, WeightCGCondition,
        ControlSurfaceLimits, LandingGearConfig, eas_to_tas, eas_to_mach,
    )
    from nastaero.loads_analysis.certification.aero_derivatives import (
        build_derivative_set, compute_inertia_from_conm2,
    )
    from nastaero.loads_analysis.case_generator import isa_atmosphere
    from nastaero.aero.dlm import compute_rigid_clalpha

    # Mass and weight
    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000  # Mg → kg
    weight_N = total_mass_kg * 9.80665
    mm = 1e-3  # mm → m conversion

    # Wing area and CLalpha
    wing_area_mm2 = 17.0e6  # 17.0 m² → mm²
    mach_vc = args.speed / 340.3 if args.mach <= 0 else args.mach
    clalpha_vlm = compute_rigid_clalpha(model, mach=mach_vc, ref_area=wing_area_mm2)

    config = AircraftConfig(
        speeds=SpeedSchedule(
            VS1=33.0, VA=62.0, VB=0.0, VC=80.0, VD=100.0, VF=40.0,
        ),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW", weight_N=weight_N, cg_x=3882.0),
        ],
        altitudes_m=[args.altitude],
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
    mach = eas_to_mach(config.speeds.VC, 0.0)
    derivs = build_derivative_set(model, config, wc, mach)

    # Inertia from CONM2 masses
    cg_xyz = np.array([wc.cg_x, 0.0, 0.0])
    inertia = compute_inertia_from_conm2(model, cg_xyz)

    print(f"       Mass: {total_mass_kg:.1f} kg, CLα={clalpha_vlm:.3f}/rad")
    print(f"       Ixx={inertia['Ixx']:.0f}, Iyy={inertia['Iyy']:.0f}, "
          f"Izz={inertia['Izz']:.0f}, Ixz={inertia['Ixz']:.0f} kg·m²")
    print(f"       ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Build AircraftParams for 6-DOF
    # ------------------------------------------------------------------
    from nastaero.loads_analysis.certification.flight_sim import (
        AircraftParams, trim_initial_state,
    )

    rho_h, _, _ = isa_atmosphere(args.altitude)
    params = AircraftParams(
        mass=inertia["mass_kg"],
        S=derivs.S_ref * mm**2,    # mm² → m²
        b=derivs.b_ref * mm,       # mm → m
        c_bar=derivs.c_bar * mm,   # mm → m
        Ixx=inertia["Ixx"],
        Iyy=inertia["Iyy"],
        Izz=inertia["Izz"],
        Ixz=inertia["Ixz"],
        derivs=derivs,
        rho=rho_h,
    )

    # Verify trim
    V_tas = eas_to_tas(args.speed, args.altitude) if args.altitude > 0 else args.speed
    initial_state, de_trim = trim_initial_state(params, V_tas, nz=1.0)
    print(f"\n[3/6] Trim: V_TAS={V_tas:.1f} m/s, α={initial_state.alpha*57.3:.2f}°, "
          f"δe_trim={de_trim*57.3:.2f}°")

    # ------------------------------------------------------------------
    # 4. Build Modal ROM (this is the heavy computation)
    # ------------------------------------------------------------------
    print(f"\n[4/6] Building Modal Reduced-Order Model ({args.modes} modes)...")
    t0 = time.time()

    from nastaero.loads_analysis.certification.modal_rom import ModalROM
    rom = ModalROM.build(model, n_modes=args.modes, mach=mach_vc)

    rom_time = time.time() - t0
    print(f"       {rom.n_modes} elastic modes extracted")
    if rom.n_modes > 0:
        print(f"       Freq range: {rom.frequencies_hz[0]:.1f} - "
              f"{rom.frequencies_hz[-1]:.1f} Hz")
    print(f"       {rom.n_boxes} aero boxes, {rom.n_free} free DOFs")
    print(f"       ({rom_time:.1f}s)")

    # Quick verification: compute response at trim
    print(f"\n       ROM verification at trim...")
    t0 = time.time()
    disp, forces = rom.compute_response(
        alpha=initial_state.alpha,
        V=V_tas,
        de=de_trim,
        nz=1.0,
        rho=rho_h,
    )
    rom_ms = (time.time() - t0) * 1000
    max_d = max(np.linalg.norm(d[:3]) for d in disp.values())
    print(f"       Trim response: max_disp={max_d:.3f} mm, "
          f"compute_time={rom_ms:.2f} ms")

    # ------------------------------------------------------------------
    # 5. Compute V-n diagram
    # ------------------------------------------------------------------
    print(f"\n[5/6] Computing V-n diagram...")
    t0 = time.time()

    from nastaero.loads_analysis.certification.vn_diagram import compute_vn_diagram
    vn = compute_vn_diagram(config, wc, altitude_m=args.altitude)
    print(f"       nz_max={vn.nz_max:.2f}, nz_min={vn.nz_min:.2f}, "
          f"{len(vn.corner_points)} corner points ({time.time()-t0:.2f}s)")

    # ------------------------------------------------------------------
    # 6. Launch GUI
    # ------------------------------------------------------------------
    print(f"\n[6/6] Launching simulator GUI...")
    print(f"       Total startup: {time.time()-t_total:.1f}s")
    print()
    print("  Controls:")
    print("    ↑/↓  Elevator  ±1°")
    print("    ←/→  Aileron   ±1°")
    print("    ,/.  Rudder    ±1°")
    print("    Space  Pause/Resume")
    print("    R      Reset to trim")
    print()

    from PyQt5.QtWidgets import QApplication
    from nastaero.visualization.realtime_gui import RealtimeSimulatorWindow

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = RealtimeSimulatorWindow(
        bdf_model=model,
        rom=rom,
        params=params,
        vn_data=vn,
        altitude_m=args.altitude,
    )
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
