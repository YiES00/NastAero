"""Command-line interface for NastAero 3D visualization.

Usage:
    python -m nastaero.visualization model.bdf              # View undeformed model
    python -m nastaero.visualization model.bdf --run        # Run analysis + view results
    python -m nastaero.visualization model.bdf --mode 1     # View mode shape
    python -m nastaero.visualization model.bdf --aero       # View aero panels
    python -m nastaero.visualization model.bdf --export out.vtk  # Export to VTK
    python -m nastaero.visualization --load model.naero     # Load saved results
    python -m nastaero.visualization --load model.naero --subcase 3
"""
from __future__ import annotations
import argparse
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _handle_cert_visualization(args):
    """Handle certification-specific visualization commands."""
    from .cert_plot import (
        plot_vn_diagram, plot_potato, plot_vmt_envelope,
        plot_critical_frequency,
    )

    save_prefix = args.cert_save

    # --- V-n diagram ---
    if args.cert_vn:
        if not args.cert_config:
            print("Error: --cert-vn requires --cert-config CONFIG")
            sys.exit(1)

        from ..loads_analysis.certification.aircraft_config import AircraftConfig
        from ..loads_analysis.certification.vn_diagram import VnDiagram

        config_path = args.cert_config
        if config_path.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
        else:
            import json
            with open(config_path) as f:
                data = json.load(f)

        config = AircraftConfig.from_dict(data)
        vn = VnDiagram(config)
        vn.compute()

        out = f"{save_prefix}_vn.png" if save_prefix else None
        plot_vn_diagram(vn, output_path=out)
        print(f"V-n diagram: {len(vn.corners)} corners"
              + (f" → {out}" if out else ""))
        return

    # --- Certification envelope, potato, critical ---
    # These all need batch results; build from cert-dir or load from config
    if not args.cert_dir and not args.cert_config:
        print("Error: cert visualization requires --cert-dir DIR "
              "or --cert-config CONFIG")
        sys.exit(1)

    # Try to load results from cert directory
    if args.cert_dir:
        _cert_plots_from_dir(args, save_prefix)
    else:
        # Generate from config (quick: matrix + placeholder batch)
        _cert_plots_from_config(args, save_prefix)

    print("\nCertification visualization complete.")


def _cert_plots_from_config(args, save_prefix):
    """Generate cert plots from config (placeholder solver mode)."""
    import json
    from ..loads_analysis.certification.aircraft_config import AircraftConfig
    from ..loads_analysis.certification.load_case_matrix import LoadCaseMatrix
    from ..loads_analysis.certification.batch_runner import BatchRunner
    from ..loads_analysis.certification.envelope import EnvelopeProcessor
    from .cert_plot import (
        plot_potato, plot_vmt_envelope, plot_critical_frequency,
    )
    import numpy as np

    config_path = args.cert_config
    if config_path.endswith(('.yaml', '.yml')):
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            data = json.load(f)

    config = AircraftConfig.from_dict(data)
    matrix = LoadCaseMatrix(config)
    matrix.generate_all()

    runner = BatchRunner(matrix, bdf_model=None)
    batch_result = runner.run()

    # Build synthetic VMT for visualization demo
    stations = np.array([0, 2500, 5000, 7500, 10000], dtype=float)
    vmt_data = {}
    for r in batch_result.case_results:
        nz = r.nz if r.nz != 0 else 1.0
        vmt_data[r.case_id] = {
            "Wing": {
                "stations": stations,
                "shear": np.array([50000, 40000, 30000, 15000, 0]) * abs(nz),
                "bending": np.array([0, 5e7, 8e7, 9e7, 1e8]) * abs(nz),
                "torsion": np.array([1e6, 8e5, 5e5, 2e5, 0]) * abs(nz),
            },
        }

    proc = EnvelopeProcessor(batch_result, vmt_data)
    proc.compute_envelopes()
    proc.identify_critical_cases()

    _generate_cert_plots(args, proc, save_prefix)


def _cert_plots_from_dir(args, save_prefix):
    """Generate cert plots from existing certification results directory."""
    # This would load saved results from a previous --cert-loads run
    # For now, print guidance
    print(f"Loading certification results from: {args.cert_dir}")
    print("Note: Full result loading requires a prior --cert-loads run.")
    print("Use --cert-config for quick visualization from config.")


def _generate_cert_plots(args, proc, save_prefix):
    """Generate requested cert plots from an EnvelopeProcessor."""
    from .cert_plot import (
        plot_potato, plot_vmt_envelope, plot_critical_frequency,
    )

    component = args.cert_component or "Wing"

    if args.cert_envelope:
        env = proc.get_envelope(component)
        if env:
            out = f"{save_prefix}_{component}_envelope.png" if save_prefix else None
            plot_vmt_envelope(env, output_path=out)
            print(f"VMT envelope ({component}): {env.n_cases} cases"
                  + (f" → {out}" if out else ""))
        else:
            print(f"Warning: No envelope data for component '{component}'")

    if args.cert_potato is not None:
        station = args.cert_potato
        potato = proc.compute_potato(component, station=station)
        if potato:
            out = f"{save_prefix}_{component}_potato_{int(station)}.png" if save_prefix else None
            plot_potato(potato, output_path=out)
            print(f"Potato plot ({component} @ {station:.0f}mm): "
                  f"{potato.n_points} points"
                  + (f" → {out}" if out else ""))
        else:
            print(f"Warning: No potato data for {component} @ {station}")

    if args.cert_critical:
        freq = proc.critical_case_frequency()
        if freq:
            out = f"{save_prefix}_critical_frequency.png" if save_prefix else None
            plot_critical_frequency(freq, output_path=out)
            print(f"Critical frequency: {len(freq)} cases"
                  + (f" → {out}" if out else ""))


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
  python -m nastaero.visualization --load model.naero --trim # Load saved results
  python -m nastaero.visualization --load model.naero --loads --subcase 3
  python -m nastaero.visualization --load model.naero --vmt  # VMT diagrams (all components)
  python -m nastaero.visualization --load model.naero --vmt --vmt-component "Right Wing"
  python -m nastaero.visualization --load model.naero --vmt --vmt-loads all
  python -m nastaero.visualization --load model.naero --vmt --vmt-envelope
  python -m nastaero.visualization --load model.naero --vmt --vmt-save vmt_plots

Certification visualization:
  python -m nastaero.visualization --cert-vn --cert-config config.yaml
  python -m nastaero.visualization --cert-envelope --cert-config config.yaml
  python -m nastaero.visualization --cert-potato 5000 --cert-config config.yaml
  python -m nastaero.visualization --cert-critical --cert-config config.yaml --cert-save out
        """,
    )

    parser.add_argument('bdf_file', nargs='?', default=None,
                        help='BDF input file')
    parser.add_argument('--load', type=str, default=None, metavar='NAERO_FILE',
                        help='Load results from .naero file (skip BDF parsing and solving)')
    parser.add_argument('--subcase', type=int, default=0,
                        help='Subcase index (0-based) for visualization (default: 0)')

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
    parser.add_argument('--loads', action='store_true',
                        help='Show nodal force vectors (SOL 144 trim loads)')
    parser.add_argument('--loads-type', type=str, default='all',
                        choices=['all', 'aero', 'inertial', 'combined'],
                        help='Type of loads to display (default: all)')
    parser.add_argument('--force-cards', type=str, default=None,
                        help='Write FORCE cards to file (BDF format)')

    # VMT (Shear/Bending Moment/Torsion) diagrams
    parser.add_argument('--vmt', action='store_true',
                        help='Plot VMT (V-M-T) internal loads diagrams (matplotlib)')
    parser.add_argument('--vmt-component', type=str, default=None,
                        help='Show VMT for specific component (e.g., "Right Wing")')
    parser.add_argument('--vmt-loads', type=str, default='combined',
                        choices=['all', 'aero', 'inertial', 'combined'],
                        help='Load type for VMT (default: combined)')
    parser.add_argument('--vmt-stations', type=int, default=50,
                        help='Number of span stations for VMT (default: 50)')
    parser.add_argument('--vmt-save', type=str, default=None,
                        help='Save VMT plots to files (prefix, e.g., "vmt_output")')
    parser.add_argument('--vmt-envelope', action='store_true',
                        help='Plot VMT envelope across all subcases')

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

    # Certification visualization (matplotlib, no PyVista needed)
    parser.add_argument('--cert-vn', action='store_true',
                        help='Plot V-n diagram from certification config')
    parser.add_argument('--cert-envelope', action='store_true',
                        help='Plot certification VMT envelopes')
    parser.add_argument('--cert-potato', type=float, default=None, metavar='STATION',
                        help='Plot potato diagram at given span station (mm)')
    parser.add_argument('--cert-critical', action='store_true',
                        help='Plot critical case frequency chart')
    parser.add_argument('--cert-config', type=str, default=None, metavar='CONFIG',
                        help='Aircraft config for cert plots (YAML/JSON)')
    parser.add_argument('--cert-dir', type=str, default=None, metavar='DIR',
                        help='Directory with certification results')
    parser.add_argument('--cert-component', type=str, default=None,
                        help='Component name for cert plots (default: all)')
    parser.add_argument('--cert-save', type=str, default=None, metavar='PREFIX',
                        help='Save cert plots to files with this prefix')

    # VTOL
    parser.add_argument('--vtol', type=str, nargs='?', const='kc100-tilt12',
                        default=None, metavar='CONFIG',
                        help='Show rotor blades (kc100-tilt12 | kc100-lc | JSON path)')

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
    setup_logging("WARNING")

    results = None

    if args.load:
        # Load from .naero file (no BDF parsing or solving needed)
        from ..output.result_io import load_results
        print(f"Loading: {args.load}")
        results, bdf_model = load_results(args.load)
        print(f"  Nodes: {len(bdf_model.nodes):,}")
        print(f"  Elements: {len(bdf_model.elements):,}")
        print(f"  SOL: {bdf_model.sol}")
        print(f"  Subcases: {len(results.subcases)}")
        if args.subcase >= len(results.subcases):
            print(f"Error: --subcase {args.subcase} out of range "
                  f"(0..{len(results.subcases)-1})")
            sys.exit(1)
    else:
        if not args.bdf_file:
            parser.error("bdf_file is required (or use --load NAERO_FILE)")

        from ..bdf.parser import parse_bdf

        # Parse BDF
        print(f"Parsing: {args.bdf_file}")
        bdf_model = parse_bdf(args.bdf_file)
        print(f"  Nodes: {len(bdf_model.nodes):,}")
        print(f"  Elements: {len(bdf_model.elements):,}")
        print(f"  SOL: {bdf_model.sol}")

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

    # Select subcase index for visualization
    sc_idx = args.subcase

    # Write FORCE cards if requested
    if args.force_cards and results and results.subcases:
        sc = results.subcases[sc_idx]
        if sc.nodal_combined_forces:
            from ..loads_analysis.trim_loads import write_force_cards
            base = os.path.splitext(args.force_cards)[0]
            write_force_cards(sc.nodal_aero_forces,
                              base + '_aero.bdf', load_sid=101,
                              label='AERODYNAMIC')
            write_force_cards(sc.nodal_inertial_forces,
                              base + '_inertial.bdf', load_sid=102,
                              label='INERTIAL')
            write_force_cards(sc.nodal_combined_forces,
                              base + '_combined.bdf', load_sid=100,
                              label='COMBINED')
            print(f"  FORCE cards written: {base}_aero.bdf, "
                  f"{base}_inertial.bdf, {base}_combined.bdf")

    # --- Certification visualization (matplotlib, early return) ---
    cert_mode = any([args.cert_vn, args.cert_envelope,
                     args.cert_potato is not None, args.cert_critical])
    if cert_mode:
        _handle_cert_visualization(args)
        return

    # VMT diagrams (matplotlib, no PyVista needed — early return)
    if args.vmt and results and results.subcases:
        from ..loads_analysis.component_id import identify_components
        from ..loads_analysis.vmt import compute_vmt, compute_vmt_all, VMTResult
        from .vmt_plot import plot_vmt_component, plot_vmt_all, plot_vmt_envelope

        print("\n--- VMT Internal Loads ---")
        components = identify_components(bdf_model)
        if not components.components:
            print("Error: No structural components detected.")
            sys.exit(1)

        print(f"  Detected components: {', '.join(components.names())}")
        for comp in components.components:
            print(f"    {comp.name}: {len(comp.node_ids):,} nodes "
                  f"(span axis={comp.span_axis})")

        sc_idx = args.subcase

        if args.vmt_envelope:
            # Envelope across ALL subcases for each component
            vmt_all = VMTResult()
            for sc_i, sc in enumerate(results.subcases):
                if not sc.nodal_combined_forces:
                    continue
                for comp in components.components:
                    curve = compute_vmt(bdf_model, sc.nodal_combined_forces,
                                        comp, n_stations=args.vmt_stations,
                                        load_type='combined', subcase_id=sc_i)
                    vmt_all.curves.append(curve)

            print(f"  Computed VMT for {len(results.subcases)} subcases")

            if args.vmt_component:
                # Envelope for specific component
                save_path = (f"{args.vmt_save}_{args.vmt_component.replace(' ', '_')}"
                             f"_envelope.png" if args.vmt_save else None)
                plot_vmt_envelope(vmt_all, args.vmt_component,
                                  save_path=save_path, show=not args.vmt_save)
            else:
                # Envelope for each component
                for comp_name in vmt_all.component_names:
                    save_path = (f"{args.vmt_save}_{comp_name.replace(' ', '_')}"
                                 f"_envelope.png" if args.vmt_save else None)
                    plot_vmt_envelope(vmt_all, comp_name,
                                      save_path=save_path, show=not args.vmt_save)
        else:
            # Single subcase VMT
            sc = results.subcases[sc_idx]
            load_types_to_plot = (['aero', 'inertial', 'combined']
                                  if args.vmt_loads == 'all'
                                  else [args.vmt_loads])

            vmt_result = VMTResult()
            for lt in load_types_to_plot:
                force_dict = {
                    'aero': sc.nodal_aero_forces,
                    'inertial': sc.nodal_inertial_forces,
                    'combined': sc.nodal_combined_forces,
                }.get(lt)
                if force_dict:
                    r = compute_vmt_all(bdf_model, force_dict, components,
                                         n_stations=args.vmt_stations,
                                         load_type=lt, subcase_id=sc_idx)
                    vmt_result.curves.extend(r.curves)

            print(f"  Computed {len(vmt_result.curves)} VMT curves "
                  f"(SC{sc_idx}, loads={args.vmt_loads})")

            if args.vmt_component:
                # Single component with all requested load types overlaid
                curves = vmt_result.get_curves(
                    component_name=args.vmt_component)
                save_path = (f"{args.vmt_save}_{args.vmt_component.replace(' ', '_')}"
                             f".png" if args.vmt_save else None)
                plot_vmt_component(curves, save_path=save_path,
                                    show=not args.vmt_save)
            else:
                if args.vmt_loads == 'all':
                    # Per-component plots with all load types overlaid
                    for comp_name in vmt_result.component_names:
                        curves = vmt_result.get_curves(
                            component_name=comp_name)
                        save_path = (f"{args.vmt_save}_{comp_name.replace(' ', '_')}"
                                     f".png" if args.vmt_save else None)
                        plot_vmt_component(curves, save_path=save_path,
                                            show=not args.vmt_save)
                else:
                    # Grid overview: all components, single load type
                    save_path = (f"{args.vmt_save}_all.png"
                                 if args.vmt_save else None)
                    plot_vmt_all(vmt_result, load_type=args.vmt_loads,
                                 subcase_id=sc_idx, save_path=save_path,
                                 show=not args.vmt_save)

        print("\nVMT diagram complete.")
        return

    # Resolve VTOL configuration if requested
    vtol_config = None
    if args.vtol:
        from ..rotor.rotor_config import VTOLConfig
        if args.vtol == 'kc100-tilt12':
            vtol_config = VTOLConfig.kc100_tilt_rotor_12()
        elif args.vtol == 'kc100-lc':
            vtol_config = VTOLConfig.kc100_lift_cruise()
        else:
            import json
            with open(args.vtol) as f:
                vtol_config = VTOLConfig.from_dict(json.load(f))

    # Create PyVista viewer (only for 3D visualization modes)
    from .viewer import NastAeroViewer
    viewer = NastAeroViewer(bdf_model, results, off_screen=off_screen,
                            vtol_config=vtol_config)

    # Export VTK if requested
    if args.export:
        viewer.export_vtk(args.export)
        if not any([args.disp, args.modes, args.mode is not None, args.aero,
                     args.pressure, args.trim, args.loads]) and not args.screenshot:
            return

    # Determine what to plot
    show_edges = not args.no_edges
    show_arrows = not args.no_arrows
    cmap = args.cmap

    if args.loads and results:
        viewer.plot_nodal_forces(
            subcase=sc_idx,
            loads_type=args.loads_type,
            screenshot=args.screenshot,
            window_size=window_size,
        )
    elif args.trim and results:
        viewer.plot_trim_results(
            subcase=sc_idx,
            disp_scale=args.scale,
            show_aero_arrows=show_arrows,
            screenshot=args.screenshot,
            window_size=window_size,
        )
    elif args.pressure and results:
        viewer.plot_aero_pressure(
            subcase=sc_idx,
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
    elif (args.disp or args.run or args.load) and results and results.subcases:
        sc = results.subcases[sc_idx]
        if sc.displacements:
            viewer.plot_displacement(
                subcase=sc_idx,
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
                subcase=sc_idx,
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
