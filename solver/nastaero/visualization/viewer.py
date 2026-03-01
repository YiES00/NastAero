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
    build_rbe_lines,
    add_displacement_data,
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
    """

    def __init__(self, bdf_model, results=None, off_screen: bool = False):
        self.bdf_model = bdf_model
        self.results = results
        self.off_screen = off_screen

        # Pre-compute mesh
        self._struct_mesh = None
        self._aero_mesh = None

    @property
    def struct_mesh(self) -> pv.UnstructuredGrid:
        if self._struct_mesh is None:
            self._struct_mesh = build_structural_mesh(self.bdf_model)
        return self._struct_mesh

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
        """Plot the undeformed structural model."""
        pl = pv.Plotter(off_screen=self.off_screen, window_size=window_size)
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

        if title:
            pl.add_title(title, font_size=14)
        else:
            n_elem = len(self.bdf_model.elements)
            n_node = len(self.bdf_model.nodes)
            pl.add_title(f"NastAero Model ({n_node:,} nodes, {n_elem:,} elements)",
                         font_size=14)

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
        cmap: str = 'jet',
        screenshot: Optional[str] = None,
        title: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot displacement results with contour coloring."""
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

        if show_undeformed:
            pl.add_mesh(
                self.struct_mesh,
                color='lightgray',
                style='wireframe',
                line_width=0.5,
                opacity=0.3,
                label='Undeformed',
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
        aero_color: str = 'cyan',
        struct_color: str = 'lightblue',
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot aerodynamic panels with optional structural mesh."""
        from ..aero.panel import generate_all_panels

        aero_boxes = generate_all_panels(self.bdf_model)
        if not aero_boxes:
            raise ValueError("No aerodynamic panels in model")

        aero_mesh = build_aero_mesh(aero_boxes)

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
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1200, 800),
    ) -> None:
        """Plot aerodynamic pressure distribution from trim results."""
        if not self.results or not self.results.subcases:
            raise ValueError("No results available")

        sc = self.results.subcases[subcase]
        if sc.aero_pressures is None or sc.aero_boxes is None:
            raise ValueError("No aerodynamic pressure data in results")

        aero_mesh = build_aero_pressure_mesh(sc.aero_boxes, sc.aero_pressures)

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

        if sc.trim_variables:
            trim_text = "Trim Variables:\n"
            for var, val in sc.trim_variables.items():
                if var.upper() in ('ANGLEA', 'SIDES'):
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
        screenshot: Optional[str] = None,
        window_size: Tuple[int, int] = (1600, 800),
    ) -> None:
        """Plot combined trim results: deformed structure + aero pressure.

        Creates a side-by-side view with structural deformation on the left
        and aerodynamic pressure distribution on the right.
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

            pl.add_mesh(
                self.struct_mesh,
                color='lightgray',
                style='wireframe',
                line_width=0.5,
                opacity=0.3,
            )
            pl.add_mesh(
                deformed,
                scalars='Displacement_Magnitude',
                cmap='jet',
                show_edges=True,
                edge_color='gray',
                scalar_bar_args={'title': 'Disp. Mag.', 'fmt': '%.3e'},
            )
            pl.add_title("Structural Deformation", font_size=12)
        else:
            pl.add_mesh(self.struct_mesh, color='lightblue', show_edges=True)
            pl.add_title("Structure (no deformation)", font_size=12)
        pl.add_axes()

        # Right: Aero pressure
        pl.subplot(0, 1)
        if sc.aero_pressures is not None and sc.aero_boxes is not None:
            aero_mesh = build_aero_pressure_mesh(sc.aero_boxes, sc.aero_pressures)
            pl.add_mesh(
                aero_mesh,
                scalars='Pressure',
                cmap='RdBu_r',
                show_edges=True,
                edge_color='gray',
                scalar_bar_args={'title': 'Cp', 'fmt': '%.3f'},
            )
            pl.add_title("Aero Pressure (Cp)", font_size=12)
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
            for var, val in sc.trim_variables.items():
                if var.upper() in ('ANGLEA', 'SIDES'):
                    parts.append(f"{var}={np.degrees(val):.3f}deg")
                else:
                    parts.append(f"{var}={val:.4e}")
            trim_text += ", ".join(parts)
            pl.add_text(trim_text, position='upper_edge', font_size=9)

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
