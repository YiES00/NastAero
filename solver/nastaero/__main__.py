"""CLI entry point: python -m nastaero input.bdf"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from .config import setup_logging, logger
from .bdf.parser import parse_bdf
from .solvers.sol101 import solve_static
from .solvers.sol103 import solve_modes
from .solvers.sol144 import solve_trim
from .output.f06_writer import write_f06


def _run_cert_loads(args, bdf_model, bdf_path: Path) -> None:
    """Run FAR Part 23 certification loads analysis."""
    import os
    from .loads_analysis.certification.aircraft_config import AircraftConfig
    from .loads_analysis.certification.vn_diagram import VnDiagram
    from .loads_analysis.certification.load_case_matrix import LoadCaseMatrix

    # Output directory
    out_dir = args.cert_output or str(bdf_path.with_suffix('')) + '_cert'
    os.makedirs(out_dir, exist_ok=True)

    # Load configuration
    if args.cert_loads == 'auto':
        logger.info("Certification: auto-config from BDF model")
        config = AircraftConfig.from_model_defaults(bdf_model)
    else:
        config_path = Path(args.cert_loads)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)
        if config_path.suffix in ('.yaml', '.yml'):
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
        else:
            import json
            with open(config_path) as f:
                data = json.load(f)
        config = AircraftConfig.from_dict(data)

    logger.info("Certification: %d weight/CG conditions, %d altitudes",
                len(config.weight_cg_conditions), len(config.altitudes_m))

    # V-n diagram
    vn = VnDiagram(config)
    vn.compute()
    logger.info("V-n diagram: %d corner points", len(vn.corners))

    # Save V-n diagram plot
    try:
        from .visualization.cert_plot import plot_vn_diagram
        vn_path = os.path.join(out_dir, 'vn_diagram.png')
        plot_vn_diagram(vn, output_path=vn_path)
        logger.info("V-n diagram saved: %s", vn_path)
    except Exception as e:
        logger.warning("V-n plot failed: %s", e)

    if args.vn_only:
        logger.info("V-n only mode — done.")
        return

    # Generate load case matrix
    matrix = LoadCaseMatrix(config)
    matrix.generate_all()
    logger.info("Load case matrix: %d total cases "
                "(%d flight + %d landing)",
                matrix.total_cases, len(matrix.flight_cases),
                len(matrix.landing_cases))

    # Save matrix CSV
    csv_path = os.path.join(out_dir, 'load_case_matrix.csv')
    matrix.to_csv(csv_path)
    logger.info("Matrix CSV: %s", csv_path)

    # Print summary
    summary = matrix.summary()
    logger.info("FAR sections covered: %s",
                ', '.join(sorted(summary['far_sections'])))

    if args.dry_run:
        logger.info("Dry-run mode — matrix generated, no solver execution.")
        return

    # Run batch analysis
    from .loads_analysis.certification.batch_runner import BatchRunner

    cp_dir = os.path.join(out_dir, 'checkpoints')
    runner = BatchRunner(
        matrix, bdf_model=bdf_model,
        n_workers=args.parallel,
        checkpoint_dir=cp_dir,
    )
    batch_result = runner.run(resume=os.path.exists(
        os.path.join(cp_dir, 'batch_checkpoint.json')))

    logger.info("Batch complete: %d/%d converged (%.1fs)",
                batch_result.n_converged, batch_result.n_total,
                batch_result.wall_time_s)

    # VMT integration: convert nodal forces → shear/bending/torsion curves
    from .loads_analysis.certification.vmt_bridge import compute_vmt_for_batch

    vmt_data = compute_vmt_for_batch(bdf_model, batch_result)
    logger.info("VMT computed for %d cases across structural components",
                len(vmt_data))

    # Envelope processing
    from .loads_analysis.certification.envelope import EnvelopeProcessor

    proc = EnvelopeProcessor(batch_result, vmt_data)
    proc.compute_envelopes()
    proc.identify_critical_cases()

    env_summary = proc.summary()
    logger.info("Envelopes: %d components, %d critical cases",
                len(env_summary['components']), env_summary['n_critical'])

    # Report generation
    from .loads_analysis.certification.report import CertificationReport

    report = CertificationReport(matrix, batch_result, proc)

    # Save critical loads CSV
    crit_csv = os.path.join(out_dir, 'critical_loads.csv')
    report.to_csv(crit_csv)
    logger.info("Critical loads CSV: %s", crit_csv)

    # Save compliance CSV
    comp_csv = os.path.join(out_dir, 'compliance_matrix.csv')
    report.compliance_to_csv(comp_csv)
    logger.info("Compliance CSV: %s", comp_csv)

    # Print final summary
    rep_summary = report.summary()
    logger.info("=== Certification Loads Summary ===")
    logger.info("  Total cases: %d", rep_summary['total_cases'])
    logger.info("  Converged: %d", rep_summary['converged'])
    logger.info("  FAR sections covered: %d/%d",
                rep_summary['far_sections_covered'],
                len(rep_summary.get('compliance', [])))
    logger.info("  Compliance rate: %.1f%%",
                rep_summary['compliance_rate'] * 100)
    logger.info("  Output directory: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='nastaero',
        description='NastAero - Finite Element Analysis Solver')

    parser.add_argument('bdf_file', help='Input BDF file path')
    parser.add_argument('--run', action='store_true', default=True,
                        help='Run analysis (default: True)')
    parser.add_argument('-p', '--parallel', type=int, default=0,
                        help='Parallel workers: 0=sequential, -1=auto, N=explicit')
    parser.add_argument('--blas-threads', type=int, default=1,
                        help='BLAS threads per worker (default: 1)')
    parser.add_argument('--loads', action='store_true',
                        help='Write FORCE cards for trim loads')
    parser.add_argument('--loads-type', default='combined',
                        choices=['aero', 'inertial', 'combined', 'all'],
                        help='Type of loads to output (default: combined)')
    parser.add_argument('--force-cards', action='store_true',
                        help='Write FORCE card BDF files')
    parser.add_argument('--screenshot', type=str, default=None,
                        help='Save visualization screenshot')
    parser.add_argument('--save-results', type=str, default=None, metavar='FILE',
                        help='Save results to .naero file for later visualization')
    parser.add_argument('--save', action='store_true',
                        help='Auto-save results to <input>.naero')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')

    # Certification loads arguments
    parser.add_argument('--cert-loads', type=str, default=None, metavar='CONFIG',
                        help='Run FAR Part 23 certification loads analysis '
                             '(CONFIG = YAML/JSON config file or "auto")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate load case matrix without running (cert-loads)')
    parser.add_argument('--vn-only', action='store_true',
                        help='Compute V-n diagram only (cert-loads)')
    parser.add_argument('--cert-output', type=str, default=None, metavar='DIR',
                        help='Output directory for certification results (default: <bdf>_cert/)')

    args = parser.parse_args()

    setup_logging(args.log_level)

    bdf_path = Path(args.bdf_file)
    if not bdf_path.exists():
        print(f"Error: file not found: {bdf_path}")
        sys.exit(1)

    logger.info("NastAero v0.3.0 - Reading %s", bdf_path.name)
    bdf_model = parse_bdf(str(bdf_path))

    f06_path = str(bdf_path.with_suffix(".f06"))

    # --- Certification loads mode ---
    if args.cert_loads is not None:
        _run_cert_loads(args, bdf_model, bdf_path)
        return

    if bdf_model.sol == 101:
        results = solve_static(bdf_model)
    elif bdf_model.sol == 103:
        results = solve_modes(bdf_model)
    elif bdf_model.sol == 144:
        results = solve_trim(bdf_model,
                             n_workers=args.parallel,
                             blas_threads=args.blas_threads)
    elif bdf_model.sol == 146:
        from .solvers.sol146 import solve_aeroelastic_transient
        logger.info("SOL 146: Dynamic Aeroelastic Response")
        logger.info("  SOL 146 requires programmatic force_func input")
        logger.info("  Use: from nastaero.solvers.sol146 import "
                     "solve_aeroelastic_transient")
        sys.exit(0)
    else:
        logger.error("Unsupported SOL %d", bdf_model.sol)
        sys.exit(1)

    write_f06(results, bdf_model, f06_path)
    logger.info("Results written to %s", f06_path)

    # Save results to .naero file
    save_path = args.save_results or (str(bdf_path.with_suffix('.naero')) if args.save else None)
    if save_path:
        from .output.result_io import save_results
        save_results(results, bdf_model, save_path, bdf_file=str(bdf_path))

    # Write FORCE cards for SOL 144 trim loads
    if bdf_model.sol == 144 and results.subcases:
        sc = results.subcases[0]
        if sc.nodal_combined_forces is not None:
            from .loads_analysis.trim_loads import write_force_cards
            base = str(bdf_path.with_suffix(''))
            write_force_cards(sc.nodal_aero_forces,
                              base + '_aero_forces.bdf',
                              load_sid=101, label='AERODYNAMIC')
            write_force_cards(sc.nodal_inertial_forces,
                              base + '_inertial_forces.bdf',
                              load_sid=102, label='INERTIAL')
            write_force_cards(sc.nodal_combined_forces,
                              base + '_combined_forces.bdf',
                              load_sid=100, label='COMBINED')
            logger.info("FORCE cards written for trim loads")

    # Visualization screenshot
    if args.screenshot and bdf_model.sol == 144 and results.subcases:
        try:
            from .visualization.viewer import NastAeroViewer
            viewer = NastAeroViewer(bdf_model, results, off_screen=True)
            viewer.plot_nodal_forces(screenshot=args.screenshot, loads_type='all')
            logger.info("Screenshot saved: %s", args.screenshot)
        except Exception as e:
            logger.warning("Visualization failed: %s", e)


if __name__ == "__main__":
    main()
