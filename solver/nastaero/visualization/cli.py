"""Command-line interface for NastAero 3D visualization.

Usage:
    python -m nastaero.visualization model.bdf              # View undeformed model
    python -m nastaero.visualization model.bdf --run        # Run analysis + view results
    python -m nastaero.visualization model.bdf --mode 1     # View mode shape
    python -m nastaero.visualization model.bdf --aero       # View aero panels
    python -m nastaero.visualization model.bdf --export out.vtk  # Export to VTK
"""
from __future__ import annotations
import argparse
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    parser = argparse.ArgumentParser(
        prog='nastaero-viz',
        description='NastAero 3D Model and Results Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nastaero.visualization cantilever.bdf            # View model
  python -m nastaero.visualization cantilever.bdf --run      # Run + view displacement
  python -m nastaero.visualization plate.bdf --run --modes   # Run + view modes
  python -m nastaero.visualization goland.bdf --run --aero   # Run + view aero results
  python -m nastaero.visualization model.bdf --screenshot fig.png
  python -m nastaero.visualization model.bdf --export out.vtk
        """,
    )

    parser.add_argument('bdf_file', help='BDF input file')

    # Analysis
    parser.add_argument('--run', action='store_true',
                        help='Run analysis before visualization')

    # Display mode
    parser.add_argument('--disp', action='store_true',
                        help='Show displacement contour (requires results)')
    parser.add_argument('--modes', action='store_true',
                        help='Show mode shapes (SOL 103)')
    parser.add_argument('--mode', type=int, default=None,
                        help='Show specific mode number (1-based)')
    parser.add_argument('--aero', action='store_true',
                        help='Show aerodynamic panels')
    parser.add_argument('--pressure', action='store_true',
                        help='Show aerodynamic pressure (SOL 144)')
    parser.add_argument('--trim', action='store_true',
                        help='Show combined trim results (SOL 144)')

    # Display options
    parser.add_argument('--component', type=str, default='magnitude',
                        choices=['magnitude', 't1', 't2', 't3', 'r1', 'r2', 'r3'],
                        help='Displacement component to display')
    parser.add_argument('--scale', type=float, default=0.0,
                        help='Deformation scale factor (0 = auto)')
    parser.add_argument('--cmap', type=str, default=None,
                        help='Colormap (jet, coolwarm, viridis, etc.)')
    parser.add_argument('--no-edges', action='store_true',
                        help='Hide element edges')
    parser.add_argument('--nodes', action='store_true',
                        help='Show node points')
    parser.add_argument('--node-ids', action='store_true',
                        help='Show node ID labels')
    parser.add_argument('--no-arrows', action='store_true',
                        help='Hide aerodynamic force direction arrows')

    # Output
    parser.add_argument('--screenshot', type=str, default=None,
                        help='Save screenshot to file (PNG)')
    parser.add_argument('--export', type=str, default=None,
                        help='Export mesh to VTK file')
    parser.add_argument('--window-size', type=str, default='1200x800',
                        help='Window size WxH (default: 1200x800)')

    args = parser.parse_args()

    # Parse window size
    ws = args.window_size.split('x')
    window_size = (int(ws[0]), int(ws[1]))

    # Determine off-screen mode: screenshot-only or env var
    off_screen = bool(args.screenshot) or os.environ.get('PYVISTA_OFF_SCREEN', '').lower() == 'true'

    # Import NastAero
    from ..config import setup_logging
    from ..bdf.parser import parse_bdf

    setup_logging("WARNING")

    # Parse BDF
    print(f"Parsing: {args.bdf_file}")
    bdf_model = parse_bdf(args.bdf_file)
    print(f"  Nodes: {len(bdf_model.nodes):,}")
    print(f"  Elements: {len(bdf_model.elements):,}")
    print(f"  SOL: {bdf_model.sol}")

    results = None

    # Run analysis if requested
    if args.run:
        print(f"\nRunning SOL {bdf_model.sol} analysis...")
        if bdf_model.sol == 101:
            from ..solvers.sol101 import solve_static
            results = solve_static(bdf_model)
            print("  Static analysis complete.")
        elif bdf_model.sol == 103:
            from ..solvers.sol103 import solve_modes
            results = solve_modes(bdf_model)
            n_modes = len(results.subcases[0].mode_shapes) if results.subcases else 0
            print(f"  Modal analysis complete ({n_modes} modes).")
        elif bdf_model.sol == 144:
            from ..solvers.sol144 import solve_trim
            results = solve_trim(bdf_model)
            print("  Trim analysis complete.")
        else:
            print(f"  Warning: SOL {bdf_model.sol} not supported, showing model only.")

    # Create viewer
    from .viewer import NastAeroViewer
    viewer = NastAeroViewer(bdf_model, results, off_screen=off_screen)

    # Export VTK if requested
    if args.export:
        viewer.export_vtk(args.export)
        if not any([args.disp, args.modes, args.mode is not None, args.aero,
                     args.pressure, args.trim]) and not args.screenshot:
            return

    # Determine what to plot
    show_edges = not args.no_edges
    show_arrows = not args.no_arrows
    cmap = args.cmap

    if args.trim and results:
        viewer.plot_trim_results(
            disp_scale=args.scale,
            show_aero_arrows=show_arrows,
            screenshot=args.screenshot,
            window_size=window_size,
        )
    elif args.pressure and results:
        viewer.plot_aero_pressure(
            cmap=cmap or 'RdBu_r',
            show_aero_arrows=show_arrows,
            screenshot=args.screenshot,
            window_size=window_size,
        )
    elif args.modes and results:
        viewer.plot_all_modes(
            scale=args.scale,
            cmap=cmap or 'coolwarm',
            screenshot=args.screenshot,
            window_size=window_size,
        )
    elif args.mode is not None:
        if results:
            viewer.plot_mode_shape(
                mode=args.mode - 1,  # Convert to 0-based
                scale=args.scale,
                cmap=cmap or 'coolwarm',
                screenshot=args.screenshot,
                window_size=window_size,
            )
        else:
            print("Error: --mode requires --run or results data")
            sys.exit(1)
    elif (args.disp or args.run) and results and results.subcases:
        sc = results.subcases[0]
        if sc.displacements:
            viewer.plot_displacement(
                component=args.component,
                scale=args.scale,
                show_edges=show_edges,
                show_aero_arrows=show_arrows,
                cmap=cmap or 'jet',
                screenshot=args.screenshot,
                window_size=window_size,
            )
        elif sc.mode_shapes:
            viewer.plot_all_modes(
                scale=args.scale,
                cmap=cmap or 'coolwarm',
                screenshot=args.screenshot,
                window_size=window_size,
            )
        elif sc.trim_variables is not None:
            viewer.plot_trim_results(
                disp_scale=args.scale,
                show_aero_arrows=show_arrows,
                screenshot=args.screenshot,
                window_size=window_size,
            )
        else:
            viewer.plot_model(
                show_edges=show_edges,
                show_nodes=args.nodes,
                show_node_ids=args.node_ids,
                screenshot=args.screenshot,
                window_size=window_size,
            )
    elif args.aero:
        try:
            viewer.plot_aero_model(
                show_control_points=True,
                show_normals=show_arrows,
                screenshot=args.screenshot,
                window_size=window_size,
            )
        except ValueError as e:
            print(f"Warning: {e}")
            viewer.plot_model(
                show_edges=show_edges,
                screenshot=args.screenshot,
                window_size=window_size,
            )
    else:
        viewer.plot_model(
            show_edges=show_edges,
            show_nodes=args.nodes,
            show_node_ids=args.node_ids,
            screenshot=args.screenshot,
            window_size=window_size,
        )

    print("\nVisualization complete.")


if __name__ == '__main__':
    main()
