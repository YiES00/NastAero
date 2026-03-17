"""Interactive 3D viewer for NastAero FEA models and results.

Provides high-level plotting functions for structural mesh, deformed shapes,
mode shapes, and aerodynamic panels with result contours.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    import pyvista as pv
    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'
    pv.global_theme.font.size = 12
except ImportError:
    raise ImportError(
        "PyVista is required for visualization. Install with: pip install pyvista"
    )

from .mesh_builder import (
    build_structural_mesh,
    build_deformed_mesh,
    build_mode_shape_mesh,
    build_aero_mesh,
    build_aero_pressure_mesh,
    build_aero_force_arrows,
    build_aero_normal_arrows,
    build_nodal_force_arrows,
    build_rbe_lines,
    build_beam_tubes,
    build_deformed_beam_tubes,
    add_displacement_data,
    build_rotor_blades,
    build_rotor_disks,
)


def _finalize_plot(pl, screenshot: Optional[str], off_screen: bool) -> None:
    """Finalize a plotter: show interactively or save screenshot.

    Parameters
    ----------
    pl : pv.Plotter
    screenshot : str or None
    off_screen : bool
    """
    if screenshot:
        # Always use show(screenshot=...) for reliability
        pl.show(screenshot=screenshot)
        print(f"Screenshot saved: {screenshot}")
    else:
        pl.show()


class NastAeroViewer:
    """Interactive 3D viewer for FEA models and results.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model.
    results : ResultData, optional
        Analysis results (from SOL 101/103/144).
    off_screen : bool
        If True, render off-screen (for saving screenshots).
    vtol_config : VTOLConfig, optional
        VTOL rotor configuration. If provided, rotor blades and disks
        are rendered in 3D alongside the structural model.
    """

    def __init__(self, bdf_model, results=None, off_screen: bool = False,
                 vtol_config=None):
        self.bdf_model = bdf_model
        self.results = results
        self.off_screen = off_screen
        self.vtol_config = vtol_config

        # Pre-compute mesh
        self._struct_mesh = None
        self._aero_mesh = None

    @property
    def struct_mesh(self) -> pv.UnstructuredGrid:
        if self._struct_mesh is None:
            self._struct_mesh = build_structural_mesh(self.bdf_model)
        return self._struct_mesh

    def _has_beams(self) -> bool:
        """Check if model has CBAR/CROD elements."""
        return any(e.type in ("CBAR", "CROD")
                    for e in self.bdf_model.elements.values())

    def _has_aero_panels(self) -> bool:
        """Check if model has CAERO1 aerodynamic panels."""
        return bool(self.bdf_model.caero_panels)

    def _add_beam_tubes(self, pl, color='steelblue', opacity=1.0,
                         label='Beams') -> None:
        """Add 3D tube beams to plotter if model has CBAR/CROD."""
        tubes = build_beam_tubes(self.bdf_model)
        if tubes is not None:
            pl.add_mesh(
                tubes,
                color=color,
                opacity=opacity,
                label=label,
            )

    def _add_aero_force_arrows(self, pl, subcase: int = 0,
                                arrow_color: str = 'red',
                                arrow_scale: float = 0.0,
                                label: str = 'Aero Forces') -> None:
        """Add aerodynamic force direction arrows to plotter.

        Uses aero_forces from results if available, otherwise shows panel
        normals as a fallback to indicate the positive lift direction.

        Parameters
        ----------
        pl : pv.Plotter
        subcase : int
        arrow_color : str
        arrow_scale : float
            Arrow length scale factor. 0 = auto.
        label : str
        """
        if self.results and self.results.subcases:
            sc = self.results.subcases[subcase]
            if sc.aero_forces is not None and sc.aero_boxes is not None:
                arrows = build_aero_force_arrows(
                    sc.aero_boxes, sc.aero_forces, scale=arrow_scale)
                if arrows is not None:
                    pl.add_mesh(
                        arrows,
                        color=arrow_color,
                        opacity=0.9,
                        label=label,
                    )
                    return

        # Fallback: show panel normals if no force data
        if self._has_aero_panels():
            try:
                from ..aero.panel import generate_all_panels
                aero_boxes = generate_all_panels(self.bdf_model)
                if aero_boxes:
                    arrows = build_aero_normal_arrows(aero_boxes, scale=arrow_scale)
                    if arrows is not None:
                        pl.add_mesh(
                            arrows,
                            color='green',
                            opacity=0.7,
                            label='Panel Normals',
                        )
            except Exception:
                pass

    def _get_trim_variables(self, subcase: int = 0) -> Optional[dict]:
        """Get trim variables from results if available."""
        if self.results and self.results.subcases:
            sc = self.results.subcases[subcase]
            if sc.trim_variables:
                return sc.trim_variables
        return None

    def _add_aero_panels(self, pl, color='cyan', opacity=0.5,
                          label='Aero Panels', subcase: int = 0) -> None:
        """Add aero panels to plotter if model has CAERO1.

        When trim results are available, control surface panels are
        deflected to show the actual elevator/aileron deflection angle.
        """
        if not self._has_aero_panels():
            return
        try:
            from ..aero.panel import generate_all_panels
            aero_boxes = generate_all_panels(self.bdf_model)
            if aero_boxes:
                trim_vars = self._get_trim_variables(subcase)
                aero_mesh = build_aero_mesh(
                    aero_boxes,
                    bdf_model=self.bdf_model,
                    trim_variables=trim_vars,
                )
                pl.add_mesh(
                    aero_mesh,
                    color=color,
                    show_edges=True,
                    edge_color='darkblue',
                    line_width=1,
                    opacity=opacity,
                    label=label,
                )
        except Exception:
            pass

    def _add_rotors(self, pl, blade_opacity: float = 0.9,
                     disk_opacity: float = 0.15) -> None:
        """Add rotor blades and disks to plotter if VTOLConfig is provided.

        CW blades are shown in darkorange, CCW in steelblue.
        Disks are semi-transparent gray annuli.
        """
        if self.vtol_config is None:
            return
        try:
            from ..rotor.rotor_config import RotationDir
            cw = build_rotor_blades(self.vtol_config,
                                    rotation_filter=RotationDir.CW)
            ccw = build_rotor_blades(self.vtol_config,
                                     rotation_filter=RotationDir.CCW)
            disks = build_rotor_disks(self.vtol_config)
            if cw is not None:
                pl.add_mesh(cw, color='darkorange', opacity=blade_opacity,
                            label='CW Blades')
            if ccw is not None:
                pl.add_mesh(ccw, color='steelblue', opacity=blade_opacity,
                            label='CCW Blades')
            if disks is not None:
                pl.add_mesh(disks, color='gray', opacity=disk_opacity,
                            show_edges=False, label='Rotor Disks')
        except Exception:
            pass

    def plot_model(
        self,
        show_edges: bool = True,
        show_nodes: bool = False,
        show_node_ids: bool = False,
        show_elem_ids: bool = False,
        show_rbe: bool = True,
        color: str = 'lightblue',
        screenshot: Optional[str] = None,
        title: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot the undeformed structural model.

        Automatically renders CBAR/CROD as 3D tubes and shows
        aerodynamic panels if CAERO1 cards are present.
        """
        pl = pv.Plotter(off_screen=self.off_screen, window_size=window_size)

        has_beams = self._has_beams()

        # For beam-only models, use tubes instead of lines
        if has_beams:
            # Add shell/solid elements from the mesh (exclude beams)
            shell_mesh = build_structural_mesh(self.bdf_model, include_beams=False)
            if shell_mesh.n_cells > 0:
                pl.add_mesh(
                    shell_mesh,
                    color=color,
                    show_edges=show_edges,
                    edge_color='gray',
                    line_width=1,
                    opacity=1.0,
                    label='Shells',
                )
            # Add 3D beam tubes
            self._add_beam_tubes(pl)
        else:
            pl.add_mesh(
                self.struct_mesh,
                color=color,
                show_edges=show_edges,
                edge_color='gray',
                line_width=1,
                opacity=1.0,
                label='Structure',
            )

        if show_nodes:
            pl.add_mesh(
                pv.PolyData(self.struct_mesh.points),
                color='black',
                point_size=5,
                render_points_as_spheres=True,
                label='Nodes',
            )

        if show_node_ids and len(self.struct_mesh.points) < 500:
            nids = self.struct_mesh.point_data['NodeID']
            labels = [str(n) for n in nids]
            pl.add_point_labels(
                self.struct_mesh.points,
                labels,
                font_size=8,
                text_color='blue',
                point_size=1,
                shape=None,
            )

        if show_rbe:
            rbe_mesh = build_rbe_lines(self.bdf_model)
            if rbe_mesh is not None:
                pl.add_mesh(rbe_mesh, color='red', line_width=3, label='RBE')

        # Auto-detect and show aero panels
        self._add_aero_panels(pl)

        # Show rotor blades and disks if VTOLConfig provided
        self._add_rotors(pl)

        if title:
            pl.add_title(title, font_size=14)
        else:
            n_elem = len(self.bdf_model.elements)
            n_node = len(self.bdf_model.nodes)
            parts = [f"{n_node:,} nodes", f"{n_elem:,} elements"]
            if self._has_aero_panels():
                n_panels = sum(max(c.nspan,1) * max(c.nchord,1)
                               for c in self.bdf_model.caero_panels.values())
                parts.append(f"{n_panels} aero panels")
            if self.vtol_config:
                n_rotors = len(self.vtol_config.rotors)
                n_blades = sum(r.n_blades for r in self.vtol_config.rotors)
                parts.append(f"{n_rotors} rotors ({n_blades} blades)")
            pl.add_title(f"NastAero Model ({', '.join(parts)})", font_size=14)

        pl.add_axes()
        pl.show_bounds(grid=False, location='outer')

        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_displacement(
        self,
        subcase: int = 0,
        component: str = 'magnitude',
        scale: float = 0.0,
        show_undeformed: bool = True,
        show_edges: bool = True,
        show_aero_arrows: bool = True,
        cmap: str = 'jet',
        screenshot: Optional[str] = None,
        title: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot displacement results with contour coloring.

        When aero results are available and show_aero_arrows is True,
        aerodynamic force direction arrows are displayed on panels to
        help verify that displacement directions are physically consistent.
        """
        if not self.results or not self.results.subcases:
            raise ValueError("No results available for plotting")

        sc = self.results.subcases[subcase]
        if not sc.displacements:
            raise ValueError(f"No displacements in subcase {subcase}")

        displacements = sc.displacements

        # Auto-scale deformation
        if scale == 0.0:
            max_disp = max(np.linalg.norm(d[:3]) for d in displacements.values())
            if max_disp > 1e-30:
                pts = np.array([self.bdf_model.nodes[n].xyz_global
                                for n in self.bdf_model.nodes])
                bbox_size = np.max(pts.max(axis=0) - pts.min(axis=0))
                scale = 0.1 * bbox_size / max_disp
            else:
                scale = 1.0

        deformed = build_deformed_mesh(self.bdf_model, displacements, scale)
        add_displacement_data(deformed, self.bdf_model, displacements)

        scalar_map = {
            'magnitude': 'Displacement_Magnitude',
            't1': 'T1', 't2': 'T2', 't3': 'T3',
            'r1': 'R1', 'r2': 'R2', 'r3': 'R3',
        }
        scalar_name = scalar_map.get(component.lower(), 'Displacement_Magnitude')

        pl = pv.Plotter(off_screen=self.off_screen, window_size=window_size)

        has_beams = self._has_beams()

        if show_undeformed:
            if has_beams:
                # Ghost wireframe for undeformed beams
                tubes_undef = build_beam_tubes(self.bdf_model)
                if tubes_undef is not None:
                    pl.add_mesh(tubes_undef, color='lightgray', opacity=0.2)
            pl.add_mesh(
                self.struct_mesh,
                color='lightgray',
                style='wireframe',
                line_width=0.5,
                opacity=0.3,
                label='Undeformed',
            )

        # Add deformed beam tubes with displacement coloring
        if has_beams:
            def_tubes = build_deformed_beam_tubes(
                self.bdf_model, displacements, scale)
            if def_tubes is not None:
                pl.add_mesh(
                    def_tubes,
                    scalars='Displacement_Magnitude',
                    cmap=cmap,
                    show_scalar_bar=False,
                    label='Deformed Beams',
                )

        pl.add_mesh(
            deformed,
            scalars=scalar_name,
            cmap=cmap,
            show_edges=show_edges,
            edge_color='gray',
            line_width=0.5,
            scalar_bar_args={
                'title': component.upper(),
                'n_labels': 7,
                'fmt': '%.4e',
                'position_x': 0.8,
            },
            label='Deformed',
        )

        # Auto-show aero panels and force arrows
        self._add_aero_panels(pl, opacity=0.3)
        if show_aero_arrows:
            self._add_aero_force_arrows(pl, subcase=subcase)

        max_val = deformed.point_data[scalar_name].max()
        if title:
            pl.add_title(title, font_size=14)
        else:
            pl.add_title(
                f"Displacement ({component.upper()}) | Scale: {scale:.1f}x | Max: {max_val:.4e}",
                font_size=12,
            )

        pl.add_axes()
        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_mode_shape(
        self,
        mode: int = 0,
        subcase: int = 0,
        scale: float = 0.0,
        show_undeformed: bool = True,
        cmap: str = 'coolwarm',
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot a natural vibration mode shape."""
        if not self.results or not self.results.subcases:
            raise ValueError("No results available")

        sc = self.results.subcases[subcase]
        if not sc.mode_shapes:
            raise ValueError("No mode shapes in results")
        if mode >= len(sc.mode_shapes):
            raise ValueError(f"Mode {mode} not available (max {len(sc.mode_shapes)-1})")

        mode_shape = sc.mode_shapes[mode]
        freq = sc.frequencies[mode] if sc.frequencies is not None else 0.0

        if scale == 0.0:
            max_mode = max(np.linalg.norm(d[:3]) for d in mode_shape.values())
            if max_mode > 1e-30:
                pts = np.array([self.bdf_model.nodes[n].xyz_global
                                for n in self.bdf_model.nodes])
                bbox_size = np.max(pts.max(axis=0) - pts.min(axis=0))
                scale = 0.15 * bbox_size / max_mode
            else:
                scale = 1.0

        deformed = build_mode_shape_mesh(self.bdf_model, mode_shape, scale)

        pl = pv.Plotter(off_screen=self.off_screen, window_size=window_size)

        if show_undeformed:
            pl.add_mesh(
                self.struct_mesh,
                color='lightgray',
                style='wireframe',
                line_width=0.5,
                opacity=0.3,
            )

        pl.add_mesh(
            deformed,
            scalars='Mode_Magnitude',
            cmap=cmap,
            show_edges=True,
            edge_color='gray',
            line_width=0.5,
            scalar_bar_args={
                'title': 'Mode Amplitude',
                'n_labels': 5,
                'fmt': '%.4e',
            },
        )

        pl.add_title(
            f"Mode {mode + 1} | Freq = {freq:.4f} Hz | Scale: {scale:.1f}x",
            font_size=14,
        )
        pl.add_axes()
        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_all_modes(
        self,
        subcase: int = 0,
        max_modes: int = 6,
        scale: float = 0.0,
        cmap: str = 'coolwarm',
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1600, 1000),
    ) -> None:
        """Plot multiple mode shapes in a grid layout."""
        if not self.results or not self.results.subcases:
            raise ValueError("No results available")

        sc = self.results.subcases[subcase]
        if not sc.mode_shapes:
            raise ValueError("No mode shapes in results")

        n_modes = min(max_modes, len(sc.mode_shapes))

        if n_modes <= 3:
            n_rows, n_cols = 1, n_modes
        elif n_modes <= 6:
            n_rows, n_cols = 2, 3
        elif n_modes <= 9:
            n_rows, n_cols = 3, 3
        else:
            n_rows, n_cols = 3, 4

        pl = pv.Plotter(
            shape=(n_rows, n_cols),
            off_screen=self.off_screen,
            window_size=window_size,
        )

        for i in range(n_modes):
            row = i // n_cols
            col = i % n_cols
            pl.subplot(row, col)

            mode_shape = sc.mode_shapes[i]
            freq = sc.frequencies[i] if sc.frequencies is not None else 0.0

            if scale == 0.0:
                max_mode = max(np.linalg.norm(d[:3]) for d in mode_shape.values())
                if max_mode > 1e-30:
                    pts = np.array([self.bdf_model.nodes[n].xyz_global
                                    for n in self.bdf_model.nodes])
                    bbox_size = np.max(pts.max(axis=0) - pts.min(axis=0))
                    s = 0.15 * bbox_size / max_mode
                else:
                    s = 1.0
            else:
                s = scale

            deformed = build_mode_shape_mesh(self.bdf_model, mode_shape, s)

            pl.add_mesh(
                self.struct_mesh,
                color='lightgray',
                style='wireframe',
                line_width=0.5,
                opacity=0.2,
            )
            pl.add_mesh(
                deformed,
                scalars='Mode_Magnitude',
                cmap=cmap,
                show_edges=True,
                edge_color='gray',
                line_width=0.5,
                show_scalar_bar=False,
            )
            pl.add_title(f"Mode {i+1}: {freq:.2f} Hz", font_size=10)
            pl.add_axes()

        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_aero_model(
        self,
        show_structure: bool = True,
        show_control_points: bool = False,
        show_normals: bool = True,
        aero_color: str = 'cyan',
        struct_color: str = 'lightblue',
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot aerodynamic panels with optional structural mesh.

        When show_normals is True, panel normal arrows are displayed to
        indicate the positive pressure direction of each DLM box.
        """
        from ..aero.panel import generate_all_panels

        aero_boxes = generate_all_panels(self.bdf_model)
        if not aero_boxes:
            raise ValueError("No aerodynamic panels in model")

        trim_vars = self._get_trim_variables()
        aero_mesh = build_aero_mesh(
            aero_boxes, bdf_model=self.bdf_model, trim_variables=trim_vars)

        pl = pv.Plotter(off_screen=self.off_screen, window_size=window_size)

        pl.add_mesh(
            aero_mesh,
            color=aero_color,
            show_edges=True,
            edge_color='darkblue',
            line_width=1,
            opacity=0.6,
            label='Aero Panels',
        )

        if show_structure:
            pl.add_mesh(
                self.struct_mesh,
                color=struct_color,
                show_edges=True,
                edge_color='gray',
                line_width=1,
                opacity=0.8,
                label='Structure',
            )

        if show_control_points:
            cp_pts = np.array([b.control_point for b in aero_boxes])
            dp_pts = np.array([b.doublet_point for b in aero_boxes])
            pl.add_mesh(
                pv.PolyData(cp_pts),
                color='red',
                point_size=6,
                render_points_as_spheres=True,
                label='Control Points (3/4c)',
            )
            pl.add_mesh(
                pv.PolyData(dp_pts),
                color='green',
                point_size=6,
                render_points_as_spheres=True,
                label='Doublet Points (1/4c)',
            )

        if show_normals:
            normal_arrows = build_aero_normal_arrows(aero_boxes)
            if normal_arrows is not None:
                pl.add_mesh(
                    normal_arrows,
                    color='green',
                    opacity=0.8,
                    label='Panel Normals',
                )

        n_boxes = len(aero_boxes)
        pl.add_title(f"Aero Model ({n_boxes} DLM panels)", font_size=14)
        pl.add_axes()
        pl.add_legend()

        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_aero_pressure(
        self,
        subcase: int = 0,
        cmap: str = 'RdBu_r',
        show_structure: bool = True,
        show_aero_arrows: bool = True,
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot aerodynamic pressure distribution from trim results.

        When show_aero_arrows is True, force direction arrows are overlaid
        on each panel showing the direction and relative magnitude of the
        aerodynamic force.
        """
        if not self.results or not self.results.subcases:
            raise ValueError("No results available")

        sc = self.results.subcases[subcase]
        if sc.aero_pressures is None or sc.aero_boxes is None:
            raise ValueError("No aerodynamic pressure data in results")

        aero_mesh = build_aero_pressure_mesh(
            sc.aero_boxes, sc.aero_pressures,
            bdf_model=self.bdf_model,
            trim_variables=sc.trim_variables,
        )

        pl = pv.Plotter(off_screen=self.off_screen, window_size=window_size)

        pl.add_mesh(
            aero_mesh,
            scalars='Pressure',
            cmap=cmap,
            show_edges=True,
            edge_color='gray',
            line_width=0.5,
            scalar_bar_args={
                'title': 'Cp (Pressure Coefficient)',
                'n_labels': 7,
                'fmt': '%.3f',
            },
        )

        if show_structure:
            pl.add_mesh(
                self.struct_mesh,
                color='lightblue',
                show_edges=True,
                edge_color='gray',
                line_width=1,
                opacity=0.5,
            )

        # Add force direction arrows
        if show_aero_arrows:
            self._add_aero_force_arrows(pl, subcase=subcase)

        if sc.trim_variables:
            trim_text = "Trim Variables:\n"
            # Get control surface names for degree display
            angle_vars = {'ANGLEA', 'SIDES'}
            if hasattr(self.bdf_model, 'aesurfs'):
                for surf in self.bdf_model.aesurfs.values():
                    angle_vars.add(surf.label.upper())
            for var, val in sc.trim_variables.items():
                if var.upper() in angle_vars:
                    trim_text += f"  {var} = {np.degrees(val):.4f} deg\n"
                else:
                    trim_text += f"  {var} = {val:.6e}\n"
            pl.add_text(trim_text, position='lower_left', font_size=9)

        pl.add_title("Aerodynamic Pressure Distribution", font_size=14)
        pl.add_axes()

        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_trim_results(
        self,
        subcase: int = 0,
        disp_scale: float = 0.0,
        show_aero_arrows: bool = True,
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1600, 800),
    ) -> None:
        """Plot combined trim results: deformed structure + aero pressure.

        Creates a side-by-side view with structural deformation on the left
        and aerodynamic pressure distribution on the right.  When
        show_aero_arrows is True, force direction arrows are overlaid on the
        structural deformation view (left panel) so that the relationship
        between aero loads and structural response is clearly visible.
        """
        if not self.results or not self.results.subcases:
            raise ValueError("No results available")

        sc = self.results.subcases[subcase]

        pl = pv.Plotter(
            shape=(1, 2),
            off_screen=self.off_screen,
            window_size=window_size,
        )

        # Left: Structural deformation
        pl.subplot(0, 0)
        if sc.displacements:
            displacements = sc.displacements

            if disp_scale == 0.0:
                max_disp = max(np.linalg.norm(d[:3]) for d in displacements.values())
                if max_disp > 1e-30:
                    pts = np.array([self.bdf_model.nodes[n].xyz_global
                                    for n in self.bdf_model.nodes])
                    bbox_size = np.max(pts.max(axis=0) - pts.min(axis=0))
                    disp_scale = 0.1 * bbox_size / max_disp
                else:
                    disp_scale = 1.0

            deformed = build_deformed_mesh(self.bdf_model, displacements, disp_scale)
            add_displacement_data(deformed, self.bdf_model, displacements)

            # Undeformed ghost
            if self._has_beams():
                tubes_undef = build_beam_tubes(self.bdf_model)
                if tubes_undef is not None:
                    pl.add_mesh(tubes_undef, color='lightgray', opacity=0.2)
            pl.add_mesh(
                self.struct_mesh,
                color='lightgray',
                style='wireframe',
                line_width=0.5,
                opacity=0.3,
            )

            # Deformed beam tubes
            if self._has_beams():
                def_tubes = build_deformed_beam_tubes(
                    self.bdf_model, displacements, disp_scale)
                if def_tubes is not None:
                    pl.add_mesh(
                        def_tubes,
                        scalars='Displacement_Magnitude',
                        cmap='jet',
                        show_scalar_bar=False,
                    )

            pl.add_mesh(
                deformed,
                scalars='Displacement_Magnitude',
                cmap='jet',
                show_edges=True,
                edge_color='gray',
                scalar_bar_args={'title': 'Disp. Mag.', 'fmt': '%.3e'},
            )

            # Add aero panels + force arrows on structural deformation view
            self._add_aero_panels(pl, opacity=0.3)
            if show_aero_arrows:
                self._add_aero_force_arrows(pl, subcase=subcase)

            pl.add_title("Structural Deformation + Aero Forces", font_size=12)
        else:
            if self._has_beams():
                self._add_beam_tubes(pl)
            pl.add_mesh(self.struct_mesh, color='lightblue', show_edges=True)
            pl.add_title("Structure (no deformation)", font_size=12)
        pl.add_axes()

        # Right: Aero pressure + force arrows
        pl.subplot(0, 1)
        if sc.aero_pressures is not None and sc.aero_boxes is not None:
            aero_mesh = build_aero_pressure_mesh(
                sc.aero_boxes, sc.aero_pressures,
                bdf_model=self.bdf_model,
                trim_variables=sc.trim_variables,
            )
            pl.add_mesh(
                aero_mesh,
                scalars='Pressure',
                cmap='RdBu_r',
                show_edges=True,
                edge_color='gray',
                scalar_bar_args={'title': 'Cp', 'fmt': '%.3f'},
            )

            # Force arrows on pressure view
            if show_aero_arrows:
                self._add_aero_force_arrows(pl, subcase=subcase)

            pl.add_title("Aero Pressure (Cp) + Force Vectors", font_size=12)
        else:
            try:
                from ..aero.panel import generate_all_panels
                aero_boxes = generate_all_panels(self.bdf_model)
                if aero_boxes:
                    aero_mesh = build_aero_mesh(aero_boxes)
                    pl.add_mesh(aero_mesh, color='cyan', show_edges=True, opacity=0.6)
            except Exception:
                pass
            pl.add_title("Aero Panels (no pressure data)", font_size=12)
        pl.add_axes()

        if sc.trim_variables:
            trim_text = "Trim: "
            parts = []
            # Get control surface names for degree display
            angle_vars = {'ANGLEA', 'SIDES'}
            if hasattr(self.bdf_model, 'aesurfs'):
                for surf in self.bdf_model.aesurfs.values():
                    angle_vars.add(surf.label.upper())
            for var, val in sc.trim_variables.items():
                if var.upper() in angle_vars:
                    parts.append(f"{var}={np.degrees(val):.3f}deg")
                else:
                    parts.append(f"{var}={val:.4e}")
            trim_text += ", ".join(parts)
            pl.add_text(trim_text, position='upper_edge', font_size=9)

        _finalize_plot(pl, screenshot, self.off_screen)

    def plot_nodal_forces(
        self,
        subcase: int = 0,
        loads_type: str = 'all',
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (2000, 700),
    ) -> None:
        """Plot nodal force vectors: aerodynamic, inertial, and combined.

        Creates a 1x3 subplot layout suitable for loads analysis reports:
        - Left: Aerodynamic forces (blue arrows)
        - Center: Inertial forces (red arrows)
        - Right: Combined forces (green arrows)

        Parameters
        ----------
        subcase : int
            Subcase index (0-based).
        loads_type : str
            'all' for 1x3 layout, 'aero', 'inertial', or 'combined' for single.
        screenshot : str, optional
            Save to file.
        window_size : tuple
        """
        if not self.results or not self.results.subcases:
            raise ValueError("No results available")

        sc = self.results.subcases[subcase]
        if sc.nodal_aero_forces is None:
            raise ValueError("No nodal force data. Run SOL 144 with trim loads.")

        load_sets = []
        if loads_type == 'all':
            load_sets = [
                ('Aerodynamic Forces', sc.nodal_aero_forces, 'dodgerblue'),
                ('Inertial Forces', sc.nodal_inertial_forces, 'red'),
                ('Combined Forces (Aero + Inertial)', sc.nodal_combined_forces, 'green'),
            ]
        elif loads_type == 'aero':
            load_sets = [('Aerodynamic Forces', sc.nodal_aero_forces, 'dodgerblue')]
        elif loads_type == 'inertial':
            load_sets = [('Inertial Forces', sc.nodal_inertial_forces, 'red')]
        elif loads_type == 'combined':
            load_sets = [
                ('Combined Forces (Aero + Inertial)', sc.nodal_combined_forces, 'green')]

        n_plots = len(load_sets)
        if n_plots == 1:
            window_size = (1200, 800)

        pl = pv.Plotter(
            shape=(1, n_plots),
            off_screen=self.off_screen,
            window_size=window_size,
        )

        for col, (title, forces, color) in enumerate(load_sets):
            if n_plots > 1:
                pl.subplot(0, col)

            # Structural mesh background
            if self._has_beams():
                shell_mesh = build_structural_mesh(self.bdf_model,
                                                    include_beams=False)
                if shell_mesh.n_cells > 0:
                    pl.add_mesh(shell_mesh, color='lightgray', opacity=0.3,
                                show_edges=True, edge_color='whitesmoke')
                self._add_beam_tubes(pl, color='lightgray', opacity=0.3)
            else:
                pl.add_mesh(self.struct_mesh, color='lightgray', opacity=0.3,
                            show_edges=True, edge_color='whitesmoke')

            # Aero panels as reference
            self._add_aero_panels(pl, color='cyan', opacity=0.15)

            # Force arrows
            if forces is not None:
                arrows = build_nodal_force_arrows(self.bdf_model, forces)
                if arrows is not None:
                    pl.add_mesh(
                        arrows,
                        color=color,
                        opacity=0.9,
                        label=title,
                    )

            pl.add_title(title, font_size=11)
            pl.add_axes()

        # Add trim balance text if available
        if sc.trim_balance and n_plots > 1:
            b = sc.trim_balance
            bal_text = (f"Trim Balance: "
                       f"Fx={b['Fx']:.1f}  Fy={b['Fy']:.1f}  Fz={b['Fz']:.1f}  "
                       f"Mx={b['Mx']:.0f}  My={b['My']:.0f}  Mz={b['Mz']:.0f}")
            pl.add_text(bal_text, position='upper_edge', font_size=9)

        _finalize_plot(pl, screenshot, self.off_screen)

    def export_vtk(
        self,
        filename: str,
        subcase: int = 0,
        include_results: bool = True,
    ) -> None:
        """Export mesh with results to VTK file."""
        grid = build_structural_mesh(self.bdf_model)

        if include_results and self.results and self.results.subcases:
            sc = self.results.subcases[subcase]
            if sc.displacements:
                add_displacement_data(grid, self.bdf_model, sc.displacements)

        grid.save(filename)
        print(f"VTK file saved: {filename}")
