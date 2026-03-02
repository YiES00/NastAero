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
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')

    args = parser.parse_args()

    setup_logging(args.log_level)

    bdf_path = Path(args.bdf_file)
    if not bdf_path.exists():
        print(f"Error: file not found: {bdf_path}")
        sys.exit(1)

    logger.info("NastAero v0.3.0 - Reading %s", bdf_path.name)
    bdf_model = parse_bdf(str(bdf_path))

    f06_path = str(bdf_path.with_suffix(".f06"))

    if bdf_model.sol == 101:
        results = solve_static(bdf_model)
    elif bdf_model.sol == 103:
        results = solve_modes(bdf_model)
    elif bdf_model.sol == 144:
        results = solve_trim(bdf_model,
                             n_workers=args.parallel,
                             blas_threads=args.blas_threads)
    else:
        logger.error("Unsupported SOL %d", bdf_model.sol)
        sys.exit(1)

    write_f06(results, bdf_model, f06_path)
    logger.info("Results written to %s", f06_path)

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
