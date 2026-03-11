#!/usr/bin/env python3
"""Run VTOL certification loads analysis on KC-100 Lift+Cruise configuration.

Extends the conventional Part 23 certification loads analysis with
VTOL-specific load cases (hover, OEI, transition, VTOL landing, rotor jam).
Uses BEMT for rotor loads and the existing SOL 144 / 6-DOF pipeline.

Full pipeline:
  1. Parse BDF + setup
  2. Aircraft + VTOL config
  3. BEMT rotor validation
  4. Model visualization (FEM + rotor disks)
  5. Load case matrix (conventional + VTOL)
  6. Batch SOL 144 solver
  7. Rotor hub 6-component loads table
  8. VMT integration
  9. Envelope processing + potato plots
 10. Critical case table + force export
"""
import os
import sys
import time
from datetime import datetime
import numpy as np

from nastaero.bdf.parser import parse_bdf
from nastaero.config import setup_logging


def main():
    """Run VTOL certification loads analysis."""
    setup_logging("WARNING")

    print("=" * 70)
    print("KC-100 Lift+Cruise VTOL Certification Loads Analysis")
    print("=" * 70)

    t0 = time.time()

    # Create output directory
    analysis_time = datetime.now()
    timestamp_str = analysis_time.strftime("%Y%m%d_%H%M%S")
    timestamp_label = analysis_time.strftime("Analysis: %Y-%m-%d %H:%M:%S")
    output_dir = f"vtol_cert_results_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}/\n")

    # ---- 1. Parse KC-100 BDF ----
    model = parse_bdf("tests/validation/KC100/p400r3-free-trim.bdf")
    print(f"[1] BDF parsed: {len(model.nodes)} nodes, "
          f"{len(model.elements)} elements  ({time.time()-t0:.1f}s)")

    # ---- 2. Aircraft Config (conventional) ----
    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig, SpeedSchedule, WeightCGCondition,
        ControlSurfaceLimits, LandingGearConfig,
    )
    from nastaero.aero.dlm import compute_rigid_clalpha

    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000
    weight_N = total_mass_kg * 9.80665
    print(f"    Total mass: {total_mass_kg:.1f} kg ({total_mass_kg*2.20462:.0f} lb)")
    print(f"    Total weight: {weight_N:.0f} N")

    wing_area_mm2 = 17.0 * 1e6
    mach_vc = 80.0 / 340.3
    clalpha_vlm = compute_rigid_clalpha(model, mach=mach_vc,
                                         ref_area=wing_area_mm2)
    print(f"    CLa (VLM, M={mach_vc:.3f}): {clalpha_vlm:.3f} /rad")

    config = AircraftConfig(
        speeds=SpeedSchedule(
            VS1=33.0, VA=62.0, VB=0.0, VC=80.0, VD=100.0, VF=40.0,
        ),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW", weight_N=weight_N, cg_x=3882.0),
        ],
        altitudes_m=[0.0],
        wing_area_m2=17.0,
        CLalpha=clalpha_vlm,
        mean_chord_m=1.6,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0, rudder_max_deg=25.0, elevator_max_deg=25.0,
        ),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101],
            nose_gear_node_ids=[102],
            main_gear_x=4200.0,
            nose_gear_x=1500.0,
            strut_efficiency=0.7,
            stroke=0.25,
            sink_rate_fps=10.0,
        ),
        gust_Ude_VC_fps=50.0,
        gust_Ude_VD_fps=25.0,
    )

    # ---- 3. VTOL Configuration ----
    from nastaero.rotor.rotor_config import VTOLConfig

    vtol_config = VTOLConfig.kc100_tilt_rotor_12()
    config.vtol_config = vtol_config

    print(f"\n[2] VTOL Configuration: {vtol_config.config_type}")
    print(f"    Hover rotors: {vtol_config.n_hover_rotors}")
    print(f"    Tilt rotors: {len(vtol_config.tilt_rotors)}")
    print(f"    Lift rotors: {vtol_config.n_lift_rotors}")
    print(f"    Cruise rotors: {len(vtol_config.cruise_rotors)}")
    print(f"    Total rotor mass: {vtol_config.total_rotor_mass_kg:.1f} kg")
    for r in vtol_config.rotors:
        print(f"      {r.label}: R={r.blade.radius*1000:.0f}mm, "
              f"{r.n_blades} blades, {r.rotor_type.value}, "
              f"hub=({r.hub_position[0]:.0f},{r.hub_position[1]:.0f},{r.hub_position[2]:.0f})")

    # ---- 4. BEMT Rotor Analysis ----
    from nastaero.rotor.bemt_solver import BEMTSolver
    from nastaero.loads_analysis.case_generator import isa_atmosphere

    rho_sl, _, _ = isa_atmosphere(0.0)
    print(f"\n[3] BEMT Rotor Analysis (ρ={rho_sl:.3f} kg/m³)")

    # Test hover thrust for one hover rotor
    test_rotor = vtol_config.hover_rotors[0]
    solver = BEMTSolver(test_rotor.blade, test_rotor.n_blades)
    target_per_rotor = weight_N / vtol_config.n_hover_rotors
    hover_loads = solver.solve_for_thrust(
        target_per_rotor, test_rotor.rpm_hover, rho_sl)

    # Momentum theory validation
    A = np.pi * test_rotor.blade.radius ** 2
    vi_mt = np.sqrt(target_per_rotor / (2 * rho_sl * A))
    T_mt = 2 * rho_sl * A * vi_mt ** 2
    print(f"    Hover thrust per rotor (BEMT): {hover_loads.thrust:.1f} N "
          f"(target: {target_per_rotor:.1f} N)")
    print(f"    Hover thrust (momentum theory): {T_mt:.1f} N")
    print(f"    CT = {hover_loads.CT:.6f}, CQ = {hover_loads.CQ:.6f}")
    print(f"    Power per rotor: {hover_loads.power:.0f} W "
          f"({hover_loads.power/745.7:.1f} hp)")
    print(f"    Collective: {np.degrees(hover_loads.collective_rad):.1f}°")

    # ---- 5. Model Visualization (FEM + rotor disks) ----
    try:
        from nastaero.visualization.cert_plot import plot_vtol_model

        vtol_model_path = plot_vtol_model(
            model, vtol_config,
            output_path=os.path.join(output_dir, "00_vtol_model.png"),
            timestamp=timestamp_label)
        print(f"\n[4] VTOL model plot saved: {vtol_model_path}")
    except Exception as e:
        print(f"\n[4] Model visualization failed: {e}")

    # ---- 6. Conventional Load Case Matrix ----
    from nastaero.loads_analysis.certification.load_case_matrix import LoadCaseMatrix

    conv_matrix = LoadCaseMatrix(config)
    conv_matrix.generate_all()
    n_conventional = conv_matrix.total_cases
    print(f"\n[5] Conventional load cases: {n_conventional}")
    for cat, count in sorted(conv_matrix.summary().items()):
        print(f"      {cat:15s}: {count:3d}")

    # ---- 7. VTOL Load Case Matrix ----
    from nastaero.loads_analysis.certification.vtol_load_case_matrix import (
        VTOLLoadCaseMatrix,
    )

    vtol_matrix = VTOLLoadCaseMatrix(vtol_config, config)
    vtol_cases = vtol_matrix.generate_all()
    n_vtol = len(vtol_cases)
    print(f"\n[6] VTOL load cases: {n_vtol}")
    for cat, count in sorted(vtol_matrix.summary().items()):
        print(f"      {cat:15s}: {count:3d}")

    # Merge VTOL cases into conventional matrix
    conv_matrix.merge_vtol_cases(vtol_cases)
    print(f"\n[7] Combined load case matrix: {conv_matrix.total_cases} total")

    # Export case matrix CSV
    csv_path = os.path.join(output_dir, "vtol_case_matrix.csv")
    conv_matrix.to_csv(csv_path)
    print(f"    Case matrix: {csv_path}")

    # Save case matrix summary plot
    from nastaero.visualization.cert_plot import plot_case_matrix_summary

    class _PlotCase:
        """Lightweight wrapper for plot_case_matrix_summary."""
        __slots__ = ('category', 'nz', 'mach', 'altitude_m')
        def __init__(self, c):
            self.category = c.category
            tc = c.trim_condition
            self.nz = tc.nz if tc else 1.0
            self.mach = tc.mach if tc else 0.0
            self.altitude_m = c.altitude_m

    plot_cases = [_PlotCase(c) for c in conv_matrix.flight_cases]
    matrix_path = plot_case_matrix_summary(plot_cases,
        output_path=os.path.join(output_dir, "01_case_matrix_summary.png"),
        title=f"VTOL Load Case Matrix ({conv_matrix.total_cases} cases)",
        timestamp=timestamp_label)
    print(f"    →Plot saved: {matrix_path}")

    # ---- 8. Batch SOL 144 Solver ----
    from nastaero.loads_analysis.certification.batch_runner import BatchRunner

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    print(f"\n[8] Running SOL 144 trim solver for "
          f"{len(conv_matrix.flight_cases)} flight cases "
          f"({n_cpus} workers)...")
    t_solve = time.time()

    runner = BatchRunner(conv_matrix, bdf_model=model, n_workers=n_cpus)
    batch_result = runner.run()

    solve_time = time.time() - t_solve
    print(f"    Solver completed in {solve_time:.1f}s")
    print(f"    Converged: {batch_result.n_converged}/{batch_result.n_total}")

    # Convergence by category
    by_cat = {}
    for r in batch_result.case_results:
        cat = r.category
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "converged": 0}
        by_cat[cat]["total"] += 1
        if r.converged:
            by_cat[cat]["converged"] += 1

    for cat, info in sorted(by_cat.items()):
        print(f"      {cat:15s}: {info['converged']}/{info['total']} converged")

    # ---- 9. Rotor Hub 6-Component Loads Table ----
    print(f"\n[9] Rotor hub 6-component loads table")
    from nastaero.rotor.rotor_loads_applicator import rotor_loads_to_nodal_forces

    hub_loads_table = []

    # Collect rotor hub loads from all VTOL cases
    for case in vtol_cases:
        if case.rotor_forces:
            for rotor in vtol_config.rotors:
                nid = rotor.hub_node_id
                if nid in case.rotor_forces:
                    fv = case.rotor_forces[nid]
                    hub_loads_table.append({
                        'rotor_label': rotor.label,
                        'rotor_id': rotor.rotor_id,
                        'condition': case.label,
                        'category': case.category,
                        'case_id': case.case_id,
                        'nz': case.trim_condition.nz if case.trim_condition else 1.0,
                        'Fx': fv[0], 'Fy': fv[1], 'Fz': fv[2],
                        'Mx': fv[3], 'My': fv[4], 'Mz': fv[5],
                    })

    if hub_loads_table:
        # Print summary table header
        print(f"    {'Rotor':20s} {'Condition':30s} "
              f"{'Fz (N)':>10s} {'Mx (N-mm)':>12s} {'Mz (N-mm)':>12s}")
        print(f"    {'-'*20} {'-'*30} {'-'*10} {'-'*12} {'-'*12}")

        # Show max absolute Fz per rotor (key design driver)
        rotors_seen = set()
        for rl in sorted(set(r['rotor_label'] for r in hub_loads_table)):
            rotor_entries = [r for r in hub_loads_table
                            if r['rotor_label'] == rl]
            # Find max |Fz| entry
            max_fz_entry = max(rotor_entries, key=lambda r: abs(r['Fz']))
            print(f"    {rl:20s} {max_fz_entry['condition'][:30]:30s} "
                  f"{max_fz_entry['Fz']:10.1f} "
                  f"{max_fz_entry['Mx']:12.1f} "
                  f"{max_fz_entry['Mz']:12.1f}")

        # Export hub loads CSV
        import csv
        hub_csv = os.path.join(output_dir, "rotor_hub_loads.csv")
        with open(hub_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'rotor_label', 'rotor_id', 'condition', 'category',
                'case_id', 'nz', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
            writer.writeheader()
            for row in hub_loads_table:
                writer.writerow(row)
        print(f"    →Hub loads CSV: {hub_csv} ({len(hub_loads_table)} entries)")

        # Plot hub loads
        try:
            from nastaero.visualization.cert_plot import plot_rotor_hub_loads
            hub_plot_path = plot_rotor_hub_loads(
                hub_loads_table,
                output_path=os.path.join(output_dir, "02_rotor_hub_loads.png"),
                timestamp=timestamp_label)
            print(f"    →Hub loads plot: {hub_plot_path}")
        except Exception as e:
            print(f"    Hub loads plot failed: {e}")

    # ---- 10. VMT Integration ----
    from nastaero.loads_analysis.certification.vmt_bridge import compute_vmt_for_batch

    _wc0 = config.weight_cg_conditions[0] if config.weight_cg_conditions else None
    fuselage_cg_x = _wc0.cg_x if _wc0 else None

    print(f"\n[10] Computing VMT internal loads...")
    if fuselage_cg_x is not None:
        print(f"    Fuselage VMT: CG-based integration (CG_x={fuselage_cg_x:.1f} mm)")
    t_vmt = time.time()
    vmt_data = compute_vmt_for_batch(model, batch_result,
                                      fuselage_cg_x=fuselage_cg_x)
    vmt_time = time.time() - t_vmt
    print(f"    VMT computed for {len(vmt_data)} cases in {vmt_time:.1f}s")

    comp_names = []
    if vmt_data:
        first_case = next(iter(vmt_data.values()))
        comp_names = list(first_case.keys())
        print(f"    Components: {', '.join(comp_names)}")

        # VMT stats per component
        print(f"\n    VMT summary (max absolute values across ALL cases):")
        print(f"    {'Component':20s} {'Max Shear (N)':>14s} "
              f"{'Max Bending (N·mm)':>20s} {'Max Torsion (N·mm)':>20s}")
        print(f"    {'-'*20} {'-'*14} {'-'*20} {'-'*20}")
        for comp in comp_names:
            max_s = 0
            max_b = 0
            max_t = 0
            for cid, case_data in vmt_data.items():
                if comp in case_data:
                    d = case_data[comp]
                    max_s = max(max_s, np.max(np.abs(d["shear"])))
                    max_b = max(max_b, np.max(np.abs(d["bending"])))
                    max_t = max(max_t, np.max(np.abs(d["torsion"])))
            print(f"    {comp:20s} {max_s:14,.0f} {max_b:20,.0f} {max_t:20,.0f}")

    # ---- 11. Envelope Processing ----
    from nastaero.loads_analysis.certification.envelope import EnvelopeProcessor

    print(f"\n[11] Envelope processing...")
    proc = EnvelopeProcessor(batch_result, vmt_data)
    proc.compute_envelopes()
    proc.identify_critical_cases()

    env_summary = proc.summary()
    print(f"    Envelope components: {len(env_summary['components'])}")
    print(f"    Critical cases identified: {env_summary['n_critical']}")

    # Category distribution of critical cases
    cat_dist = proc.critical_category_distribution()
    if cat_dist:
        print(f"    Critical case distribution:")
        for cat, count in sorted(cat_dist.items()):
            print(f"      {cat:15s}: {count:3d}")

    # ---- 12. VMT Envelope + Potato Plots ----
    from nastaero.visualization.cert_plot import (
        plot_vmt_envelope as _plot_vmt_env,
        plot_potato as _plot_potato,
        plot_critical_frequency as _plot_crit_freq,
    )
    from nastaero.loads_analysis.certification.monitoring_stations import (
        identify_monitoring_stations,
    )
    from nastaero.loads_analysis.component_id import identify_components

    _components = identify_components(model)
    monitoring_stations = identify_monitoring_stations(
        model, config=config, components=_components,
        mass_threshold_kg=5.0, offset_mm=50.0,
        vtol_config=vtol_config)

    print(f"\n[12] Monitoring stations identified:")
    for _comp_name, _sta_list in monitoring_stations.items():
        print(f"      {_comp_name}: {len(_sta_list)} stations")
        for _s in _sta_list:
            print(f"        {_s.position:8.1f} mm  {_s.label} [{_s.reason}]")

    if vmt_data:
        for comp in comp_names:
            env = proc.get_envelope(comp)
            if env:
                safe = comp.replace(' ', '_').lower()
                _cg = fuselage_cg_x if 'fuselage' in comp.lower() else None
                p = _plot_vmt_env(env,
                    output_path=os.path.join(output_dir,
                                              f"03_vmt_envelope_{safe}.png"),
                    timestamp=timestamp_label, cg_x=_cg)
                print(f"    →Plot saved: {p}")

                # Multi-station potato plots
                comp_stations = monitoring_stations.get(comp, [])
                if comp_stations and env.stations:
                    potato_dir = os.path.join(output_dir, f"potato_{safe}")
                    os.makedirs(potato_dir, exist_ok=True)
                    n_potato = 0
                    for idx, ms in enumerate(comp_stations):
                        potato = proc.compute_potato(comp, station=ms.position)
                        if potato and potato.n_points >= 3:
                            sta_label = ms.label.replace(' ', '_').replace('/', '_')
                            fname = (f"04_potato_{safe}_{idx:02d}_"
                                     f"{sta_label}.png")
                            p = _plot_potato(potato,
                                output_path=os.path.join(potato_dir, fname),
                                timestamp=timestamp_label)
                            n_potato += 1
                    print(f"    →{n_potato} potato plots saved: {potato_dir}/")

    freq = proc.critical_case_frequency()
    if freq:
        p = _plot_crit_freq(freq, batch_result=batch_result,
            output_path=os.path.join(output_dir,
                                      "05_critical_frequency.png"),
            timestamp=timestamp_label)
        print(f"    →Plot saved: {p}")

    # ---- 13. Critical Case Table ----
    all_critical = proc.get_critical_cases()
    if all_critical:
        print(f"\n    ┌──────────────────────────────────────────────"
              f"───────────────────────────────────────────────┐")
        print(f"    │                          "
              f"CRITICAL DESIGN LOAD CONDITIONS (CONV + VTOL)"
              f"                               │")
        print(f"    ├──────────────┬──────────┬────────┬──────────────"
              f"┬──────────────┬────────┬───────────────┤")
        print(f"    │ Component    │ Quantity │ Ext.   │ Value        "
              f"│ Station      │ CaseID │ Category      │")
        print(f"    ├──────────────┼──────────┼────────┼──────────────"
              f"┼──────────────┼────────┼───────────────┤")
        for cc in all_critical:
            comp = cc.component[:12]
            qty = cc.quantity[:8]
            ext = cc.extreme[:6]
            val = f"{cc.value:12,.0f}"
            sta = f"{cc.station:12.1f}"
            cid = f"{cc.case_id:6d}"
            cat = cc.category[:13]
            print(f"    │ {comp:12s} │ {qty:8s} │ {ext:6s} │ "
                  f"{val:>12s} │ {sta:>12s} │ {cid:>6s} │ {cat:13s} │")
        print(f"    └──────────────┴──────────┴────────┴──────────────"
              f"┴──────────────┴────────┴───────────────┘")

    # ---- 14. Force Card Export ----
    from nastaero.loads_analysis.certification.force_export import export_critical_forces

    print(f"\n[14] Exporting critical design load FORCE cards...")
    t_force = time.time()
    force_dir = os.path.join(output_dir, "force_cards")
    export_result = export_critical_forces(
        batch_result, proc, model, force_dir)
    force_time = time.time() - t_force

    print(f"    Exported {export_result['n_cases']} critical cases "
          f"in {force_time:.1f}s")
    print(f"    FORCE cards: {export_result['n_force_cards']:,}")
    print(f"    MOMENT cards: {export_result['n_moment_cards']:,}")
    print(f"    →Master BDF: {export_result['master_bdf']}")
    print(f"    →Summary CSV: {export_result['summary_csv']}")

    # ---- 15. Final Summary ----
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"VTOL Certification Loads Analysis Complete")
    print(f"{'='*70}")
    print(f"  Configuration: KC-100 {vtol_config.config_type}")
    print(f"  Conventional cases: {n_conventional}")
    print(f"  VTOL cases: {n_vtol}")
    print(f"  Total cases: {conv_matrix.total_cases}")
    print(f"  Converged: {batch_result.n_converged}/{batch_result.n_total}")
    print(f"  Critical cases: {len(all_critical)}")
    print(f"  Solver time: {solve_time:.1f}s")
    print(f"  VMT time: {vmt_time:.1f}s")
    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Output: {output_dir}/")
    print(f"  Force cards: {force_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
