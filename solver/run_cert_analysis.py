#!/usr/bin/env python3
"""Run full certification loads analysis on KC-100 and print results."""
import os
import sys
import time
from datetime import datetime
import numpy as np

from nastaero.bdf.parser import parse_bdf
from nastaero.config import setup_logging


def main():
    """Run full certification loads analysis."""
    setup_logging("WARNING")

    # ---- 1. Parse KC-100 BDF ----
    print("=" * 70)
    print("KC-100 FAR Part 23 Certification Loads Analysis")
    print("=" * 70)

    t0 = time.time()

    # Create timestamped output directory for intermediate plots
    analysis_time = datetime.now()
    timestamp_str = analysis_time.strftime("%Y%m%d_%H%M%S")
    timestamp_label = analysis_time.strftime("Analysis: %Y-%m-%d %H:%M:%S")
    output_dir = f"cert_results_{timestamp_str}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}/\n")

    model = parse_bdf("tests/validation/KC100/p400r3-free-trim.bdf")
    print(f"\n[1] BDF parsed: {len(model.nodes)} nodes, "
          f"{len(model.elements)} elements, "
          f"{len(model.masses)} masses  ({time.time()-t0:.1f}s)")

    # ---- 1b. Generate Model Screenshots ----
    try:
        import pyvista as pv
        from nastaero.visualization.viewer import NastAeroViewer
        from nastaero.visualization.mesh_builder import (
            build_structural_mesh, build_beam_tubes, build_rbe_lines,
        )

        pv.OFF_SCREEN = True

        print(f"\n[1b] Generating model screenshots...")

        viewer = NastAeroViewer(model, off_screen=True)

        # ① Combined FE + aero (aircraft overview)
        viewer.plot_model(
            screenshot=os.path.join(output_dir, "06_aircraft_model.png"),
            title="KC-100 Aircraft Configuration (FE + Aero Model)",
            window_size=(1400, 900),
        )
        print(f"    → 06_aircraft_model.png")

        # ② FE mesh only (shells + beams, NO aero panels)
        pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
        shell_mesh = build_structural_mesh(model, include_beams=False)
        if shell_mesh.n_cells > 0:
            pl.add_mesh(shell_mesh, color='lightblue', show_edges=True,
                        edge_color='gray', line_width=1, opacity=1.0,
                        label='Shells')
        tubes = build_beam_tubes(model)
        if tubes is not None:
            pl.add_mesh(tubes, color='steelblue', opacity=1.0, label='Beams')
        rbe = build_rbe_lines(model)
        if rbe is not None:
            pl.add_mesh(rbe, color='red', line_width=3, label='RBE')
        n_elem = len(model.elements)
        n_node = len(model.nodes)
        pl.add_title(f"Structural FE Model ({n_node:,} nodes, {n_elem:,} elements)",
                     font_size=14)
        pl.add_axes()
        pl.show_bounds(grid=False, location='outer')
        pl.show(screenshot=os.path.join(output_dir, "07_fe_model.png"))
        print(f"    → 07_fe_model.png")

        # ③ Aero panels only
        viewer.plot_aero_model(
            show_structure=False,
            show_normals=True,
            screenshot=os.path.join(output_dir, "08_aero_panels.png"),
            window_size=(1400, 900),
        )
        print(f"    → 08_aero_panels.png")

    except ImportError:
        print(f"\n[1b] PyVista not available — skipping model screenshots")
    except Exception as e:
        print(f"\n[1b] Screenshot generation failed: {e}")

    # ---- 2. Aircraft Config ----
    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig, SpeedSchedule, WeightCGCondition,
        ControlSurfaceLimits, LandingGearConfig,
    )
    from nastaero.aero.dlm import compute_rigid_clalpha

    total_mass_kg = sum(m.mass for m in model.masses.values()) * 1000  # Mg->kg
    weight_N = total_mass_kg * 9.80665
    print(f"    Total mass: {total_mass_kg:.1f} kg ({total_mass_kg*2.20462:.0f} lb)")
    print(f"    Total weight: {weight_N:.0f} N")

    # Compute CLa from VLM at representative cruise Mach
    # ref_area in model units (mm2) must match wing_area_m2 for Pratt consistency
    wing_area_mm2 = 17.0 * 1e6  # 17.0 m2 -> mm2
    mach_vc = 80.0 / 340.3  # VC / speed of sound ~ 0.235
    clalpha_vlm = compute_rigid_clalpha(model, mach=mach_vc, ref_area=wing_area_mm2)
    print(f"    CLa (VLM, M={mach_vc:.3f}, S_ref={17.0} m2): {clalpha_vlm:.3f} /rad")

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

    # ---- 3. V-n Diagram ----
    from nastaero.loads_analysis.certification.vn_diagram import compute_vn_diagram

    wc = config.weight_cg_conditions[0]
    vn = compute_vn_diagram(config, wc, altitude_m=0.0)
    print(f"\n[2] V-n diagram: {len(vn.corner_points)} corner points")
    print(f"    nz_max = {vn.nz_max:.2f}, nz_min = {vn.nz_min:.2f}")
    print(f"    VA = {config.speeds.VA:.0f} m/s, VC = {config.speeds.VC:.0f} m/s, "
          f"VD = {config.speeds.VD:.0f} m/s")
    for cp in vn.corner_points:
        print(f"      {cp.label:12s}  V={cp.V_eas:6.1f} m/s  nz={cp.nz:+6.2f}")

    # Save V-n diagram plot
    from nastaero.visualization.cert_plot import plot_vn_diagram as _plot_vn
    vn_path = _plot_vn(vn,
        output_path=os.path.join(output_dir, "01_vn_diagram.png"),
        timestamp=timestamp_label)
    print(f"    →Plot saved: {vn_path}")

    # ---- 4. Load Case Matrix ----
    from nastaero.loads_analysis.certification.load_case_matrix import LoadCaseMatrix

    matrix = LoadCaseMatrix(config)
    matrix.generate_all()
    cat_counts = matrix.summary()
    far_sections = set()
    for c in matrix.flight_cases:
        far_sections.add(c.far_section)
    for c in matrix.landing_cases:
        far_sections.add(c.far_section)
    print(f"\n[3] Load case matrix: {matrix.total_cases} total cases")
    print(f"    Flight cases: {len(matrix.flight_cases)}")
    print(f"    Landing cases: {len(matrix.landing_cases)}")
    print(f"    FAR sections: {', '.join(sorted(far_sections))}")
    print(f"    Categories:")
    for cat, count in sorted(cat_counts.items()):
        print(f"      {cat:15s}: {count:3d} cases")

    # Save case matrix summary plot
    from nastaero.visualization.cert_plot import plot_case_matrix_summary

    class _PlotCase:
        """Lightweight wrapper to expose nz/mach for plot_case_matrix_summary."""
        __slots__ = ('category', 'nz', 'mach', 'altitude_m')
        def __init__(self, c):
            self.category = c.category
            tc = c.trim_condition
            self.nz = tc.nz if tc else 1.0
            self.mach = tc.mach if tc else 0.0
            self.altitude_m = c.altitude_m

    plot_cases = [_PlotCase(c) for c in matrix.flight_cases]
    matrix_path = plot_case_matrix_summary(plot_cases,
        output_path=os.path.join(output_dir, "02_case_matrix_summary.png"),
        timestamp=timestamp_label)
    print(f"    →Plot saved: {matrix_path}")

    # ---- 4b. 6-DOF Dynamic Simulations ----
    from nastaero.loads_analysis.certification.sim_runner import SimRunner
    from nastaero.loads_analysis.certification.sim_to_loads import (
        critical_points_to_load_cases, summarize_critical_points,
        deduplicate_critical_points,
    )

    print(f"\n[4b] Computing stability derivatives from VLM...")
    t_sim = time.time()

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    print(f"    Using {n_cpus} workers (half of {os.cpu_count()} CPUs)")
    sim_runner = SimRunner(config, bdf_model=model, n_workers=n_cpus)
    sim_results, sim_info = sim_runner.run_all()

    # Print derivative summary
    derivs = sim_info.get("derivs")
    if derivs:
        print(f"    CLα = {derivs.CLalpha:.3f}/rad (VLM)")
        print(f"    Cmα = {derivs.Cmalpha:.3f}/rad (VLM), "
              f"Cmq = {derivs.Cmq:.2f}/rad (empirical-C172)")
        if derivs.Cldelta_a != 0:
            print(f"    Clδa = {derivs.Cldelta_a:.4f}/rad (VLM), "
                  f"Cnr = {derivs.Cnr:.3f}/rad (empirical-C172)")
        if derivs.CL_aileron_halfwing > 1e-6:
            print(f"    CL_ail_halfwing = {derivs.CL_aileron_halfwing:.4f}/rad "
                  f"(VLM, one-sided wing Z-force)")
        print(f"    → {len(derivs.vlm_computed)} VLM-computed, "
              f"{len(derivs.empirical)} empirical derivatives")

    # Print inertia summary
    inertia = sim_info.get("inertia")
    if inertia:
        print(f"\n    Inertia: Ixx={inertia['Ixx']:.0f}, "
              f"Iyy={inertia['Iyy']:.0f}, "
              f"Izz={inertia['Izz']:.0f}, "
              f"Ixz={inertia['Ixz']:.0f} kg·m²")

    print(f"\n[4c] Running 6-DOF maneuver/gust simulations...")
    print(f"    {sim_info.get('n_sims', 0)} simulations completed")
    print(f"    →{sim_info.get('n_critical_points', 0)} critical time points "
          f"extracted ({sim_info.get('elapsed_s', 0):.2f}s)")

    # Summarize by maneuver type
    crit_counts = summarize_critical_points(sim_results)
    if crit_counts:
        print(f"    Categories: {', '.join(f'{k}({v})' for k, v in sorted(crit_counts.items()))}")

    print(f"\n[4d] Converting critical points to load cases...")
    n_raw = sim_info.get('n_critical_points', 0)
    n_static = len(matrix.flight_cases)
    max_dynamic = 50  # Target: total ≤ 100
    dynamic_cases = critical_points_to_load_cases(
        sim_results, config, max_dynamic_cases=max_dynamic)
    matrix.flight_cases.extend(dynamic_cases)
    sim_time = time.time() - t_sim

    if n_raw > len(dynamic_cases):
        print(f"    Deduplication: {n_raw} → {len(dynamic_cases)} "
              f"(budget {max_dynamic}, farthest-point sampling)")
        # Show per-type breakdown
        type_counts_before = summarize_critical_points(sim_results)
        type_counts_after: dict = {}
        for dc in dynamic_cases:
            cat = dc.category.replace("dynamic_", "")
            type_counts_after[cat] = type_counts_after.get(cat, 0) + 1
        for mt in sorted(type_counts_before):
            before = type_counts_before[mt]
            after = type_counts_after.get(mt, 0)
            print(f"      {mt}: {before} → {after}")

    print(f"    Dynamic load cases added: {len(dynamic_cases)} "
          f"(case IDs {10000}-{10000 + len(dynamic_cases) - 1 if dynamic_cases else 10000})")
    print(f"    Total flight cases: {n_static} static + {len(dynamic_cases)} dynamic "
          f"= {len(matrix.flight_cases)}")
    print(f"    Simulation phase: {sim_time:.2f}s")

    # ---- 5. Batch Solver Execution ----
    from nastaero.loads_analysis.certification.batch_runner import BatchRunner

    print(f"\n[4] Running SOL 144 trim solver for {len(matrix.flight_cases)} "
          f"flight cases (static + dynamic)...")
    t_solve = time.time()

    runner = BatchRunner(matrix, bdf_model=model, n_workers=n_cpus)
    batch_result = runner.run()

    solve_time = time.time() - t_solve
    print(f"    Solver completed in {solve_time:.1f}s")
    print(f"    Converged: {batch_result.n_converged}/{batch_result.n_total}")

    # Show convergence by category
    by_cat = {}
    for r in batch_result.case_results:
        cat = r.category
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "converged": 0}
        by_cat[cat]["total"] += 1
        if r.converged:
            by_cat[cat]["converged"] += 1

    print(f"    Convergence by category:")
    for cat, info in sorted(by_cat.items()):
        print(f"      {cat:15s}: {info['converged']}/{info['total']} converged")

    # ---- Quick sanity check: force magnitudes ----
    print(f"\n    Force magnitude check (max|F| per category):")
    cat_maxf = {}
    for r in batch_result.case_results:
        if r.nodal_forces:
            max_f = max(np.linalg.norm(f[:3]) for f in r.nodal_forces.values())
            cat_maxf.setdefault(r.category, 0.0)
            cat_maxf[r.category] = max(cat_maxf[r.category], max_f)
    for cat, mf in sorted(cat_maxf.items()):
        status = "OK" if mf < 100000 else "** HIGH **"
        print(f"      {cat:15s}: {mf:12,.0f} N  {status}")

    # ---- 6. VMT Integration ----
    from nastaero.loads_analysis.certification.vmt_bridge import compute_vmt_for_batch

    # Get CG position for fuselage VMT integration
    _wc0 = config.weight_cg_conditions[0] if config.weight_cg_conditions else None
    fuselage_cg_x = _wc0.cg_x if _wc0 else None

    print(f"\n[5] Computing VMT internal loads...")
    if fuselage_cg_x is not None:
        print(f"    Fuselage VMT: CG-based forward/aft integration (CG_x={fuselage_cg_x:.1f} mm)")
    t_vmt = time.time()
    vmt_data = compute_vmt_for_batch(model, batch_result,
                                      fuselage_cg_x=fuselage_cg_x)
    vmt_time = time.time() - t_vmt
    print(f"    VMT computed for {len(vmt_data)} cases in {vmt_time:.1f}s")

    if vmt_data:
        first_case = next(iter(vmt_data.values()))
        comp_names = list(first_case.keys())
        print(f"    Components: {', '.join(comp_names)}")

        # Print VMT stats per component
        print(f"\n    VMT summary (max absolute values across ALL cases):")
        print(f"    {'Component':20s} {'Max Shear (N)':>14s} {'Max Bending (N·mm)':>20s} {'Max Torsion (N·mm)':>20s}")
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

    # ---- 7. Envelope Processing ----
    from nastaero.loads_analysis.certification.envelope import EnvelopeProcessor

    print(f"\n[6] Envelope processing...")
    proc = EnvelopeProcessor(batch_result, vmt_data)
    proc.compute_envelopes()
    proc.identify_critical_cases()

    env_summary = proc.summary()
    print(f"    Envelope components: {len(env_summary['components'])}")
    print(f"    Critical cases identified: {env_summary['n_critical']}")

    # Save VMT envelope + potato + critical frequency plots
    from nastaero.visualization.cert_plot import (
        plot_vmt_envelope as _plot_vmt_env,
        plot_potato as _plot_potato,
        plot_critical_frequency as _plot_crit_freq,
    )

    # Identify critical monitoring stations for multi-station potato plots
    from nastaero.loads_analysis.certification.monitoring_stations import (
        identify_monitoring_stations,
    )
    from nastaero.loads_analysis.component_id import identify_components

    _components = identify_components(model)
    monitoring_stations = identify_monitoring_stations(
        model, config=config, components=_components,
        mass_threshold_kg=5.0, offset_mm=50.0)

    print(f"\n    Monitoring stations identified:")
    for _comp_name, _sta_list in monitoring_stations.items():
        print(f"      {_comp_name}: {len(_sta_list)} stations")
        for _s in _sta_list:
            print(f"        {_s.position:8.1f} mm  {_s.label} [{_s.reason}]")

    if vmt_data:
        for comp in comp_names:
            env = proc.get_envelope(comp)
            if env:
                safe = comp.replace(' ', '_').lower()
                # Pass CG position for fuselage plots
                _cg = fuselage_cg_x if 'fuselage' in comp.lower() else None
                p = _plot_vmt_env(env,
                    output_path=os.path.join(output_dir, f"03_vmt_envelope_{safe}.png"),
                    timestamp=timestamp_label, cg_x=_cg)
                print(f"    →Plot saved: {p}")

                # Multi-station potato plots at critical monitoring stations
                comp_stations = monitoring_stations.get(comp, [])
                if comp_stations and env.stations:
                    potato_dir = os.path.join(output_dir, f"potato_{safe}")
                    os.makedirs(potato_dir, exist_ok=True)
                    n_potato = 0
                    for idx, ms in enumerate(comp_stations):
                        potato = proc.compute_potato(comp, station=ms.position)
                        if potato and potato.n_points >= 3:
                            sta_label = ms.label.replace(' ', '_').replace('/', '_')
                            fname = f"04_potato_{safe}_{idx:02d}_{sta_label}.png"
                            p = _plot_potato(potato,
                                output_path=os.path.join(potato_dir, fname),
                                timestamp=timestamp_label)
                            n_potato += 1
                    print(f"    →{n_potato} potato plots saved: {potato_dir}/")

    freq = proc.critical_case_frequency()
    if freq:
        p = _plot_crit_freq(freq, batch_result=batch_result,
            output_path=os.path.join(output_dir, "05_critical_frequency.png"),
            timestamp=timestamp_label)
        print(f"    →Plot saved: {p}")

    # Print critical cases using get_critical_cases() method
    all_critical = proc.get_critical_cases()
    if all_critical:
        print(f"\n    ┌─────────────────────────────────────────────────────────────────────────────────────────┐")
        print(f"    │                          CRITICAL DESIGN LOAD CONDITIONS                               │")
        print(f"    ├──────────────┬──────────┬────────┬──────────────┬──────────────┬────────┬───────────────┤")
        print(f"    │ Component    │ Quantity │ Ext.   │ Value        │ Station      │ CaseID │ Category      │")
        print(f"    ├──────────────┼──────────┼────────┼──────────────┼──────────────┼────────┼───────────────┤")
        for cc in all_critical:
            comp = cc.component[:12]
            qty = cc.quantity[:8]
            ext = cc.extreme[:6]
            val = f"{cc.value:12,.0f}"
            sta = f"{cc.station:12.1f}"
            cid = f"{cc.case_id:6d}"
            cat = cc.category[:13]
            print(f"    │ {comp:12s} │ {qty:8s} │ {ext:6s} │ {val:>12s} │ {sta:>12s} │ {cid:>6s} │ {cat:13s} │")
        print(f"    └──────────────┴──────────┴────────┴──────────────┴──────────────┴────────┴───────────────┘")

    # ---- 8. Force Card Export ----
    from nastaero.loads_analysis.certification.force_export import export_critical_forces

    print(f"\n[7] Exporting critical design load FORCE cards...")
    t_force = time.time()
    force_dir = os.path.join(output_dir, "force_cards")
    export_result = export_critical_forces(
        batch_result, proc, model, force_dir)
    force_time = time.time() - t_force

    print(f"    Exported {export_result['n_cases']} critical cases in {force_time:.1f}s")
    print(f"    FORCE cards: {export_result['n_force_cards']:,}")
    print(f"    MOMENT cards: {export_result['n_moment_cards']:,}")
    print(f"    →Master BDF: {export_result['master_bdf']}")
    print(f"    →Summary CSV: {export_result['summary_csv']}")

    # List exported BDF files
    for cid in sorted(export_result['case_files'].keys()):
        cr = batch_result.get_result(cid)
        lbl = cr.label if cr else f"Case {cid}"
        reasons = len([cc for cc in all_critical if cc.case_id == cid])
        print(f"      Case {cid:4d}: {export_result['case_files'][cid]}"
              f"  (critical for {reasons} quantities)")

    # ---- 9. Report ----
    from nastaero.loads_analysis.certification.report import CertificationReport

    report = CertificationReport(matrix, batch_result, proc)
    rep_summary = report.summary()

    print(f"\n[8] Certification Report Summary")
    print(f"    Total cases: {rep_summary['total_cases']}")
    print(f"    Converged: {rep_summary['converged']}")
    print(f"    FAR sections covered: {rep_summary['far_sections_covered']}")
    print(f"    Compliance rate: {rep_summary['compliance_rate']*100:.1f}%")

    # Print which cases dominate which quantities
    print(f"\n    Dominant load cases by structural component:")
    if all_critical:
        # Group by component
        by_comp = {}
        for cc in all_critical:
            comp = cc.component
            if comp not in by_comp:
                by_comp[comp] = []
            by_comp[comp].append(cc)

        for comp, cases in sorted(by_comp.items()):
            print(f"\n    ──{comp} --")
            for cc in cases:
                # Find the CaseResult for this critical case
                cr = batch_result.get_result(cc.case_id)
                label = cr.label if cr and cr.label else f"Case {cc.case_id}"
                nz_str = f"nz={cr.nz:+.2f}" if cr else ""
                cat_str = cc.category
                print(f"       {cc.extreme:4s} {cc.quantity:8s} = {cc.value:>12,.0f}"
                      f"  at station {cc.station:.1f}"
                      f"  ←{label} ({cat_str}, {nz_str})")

    # ---- 10. Word Report ----
    from nastaero.loads_analysis.certification.report_docx import generate_cert_report

    print(f"\n[9] Generating Word report...")
    t_report = time.time()

    docx_path = generate_cert_report(
        config=config,
        vn_diagram=vn,
        matrix=matrix,
        batch_result=batch_result,
        envelope_proc=proc,
        report=report,
        model=model,
        plot_dir=output_dir,
        vmt_data=vmt_data,
        output_path=os.path.join(output_dir, "KC100_Cert_Report.docx"),
        analysis_time=analysis_time,
        monitoring_stations=monitoring_stations,
    )
    report_time = time.time() - t_report
    print(f"    Report generated in {report_time:.1f}s")
    print(f"    →{docx_path}")

    # ---- Total time ----
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total analysis time: {total_time:.1f}s")
    print(f"  6-DOF simulation: {sim_time:.1f}s")
    print(f"  Solver: {solve_time:.1f}s")
    print(f"  VMT: {vmt_time:.1f}s")
    print(f"  Force export: {force_time:.1f}s")
    print(f"  Report: {report_time:.1f}s")
    print(f"  Output: {output_dir}/")
    print(f"  Force cards: {output_dir}/force_cards/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
