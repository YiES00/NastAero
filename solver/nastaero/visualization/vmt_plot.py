"""VMT (Shear/Bending Moment/Torsion) diagram plotting with matplotlib.

Usage:
    from nastaero.visualization.vmt_plot import plot_vmt_component, plot_vmt_all

    plot_vmt_component(curves, title="Right Wing")
    plot_vmt_all(vmt_result, save_path="vmt_all.png")
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend by default
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError(
        "matplotlib is required for VMT plots. "
        "Install with: pip install matplotlib")

from ..loads_analysis.vmt import VMTCurve, VMTResult

# Load type styling
_LOAD_STYLES = {
    'aero':      {'color': 'dodgerblue', 'linestyle': '-',  'linewidth': 1.5},
    'inertial':  {'color': 'red',        'linestyle': '-',  'linewidth': 1.5},
    'combined':  {'color': 'black',      'linestyle': '-',  'linewidth': 2.0},
}


def plot_vmt_component(
    curves: List[VMTCurve],
    title: str = None,
    figsize: Tuple[float, float] = (12, 10),
    save_path: str = None,
    show: bool = True,
) -> None:
    """Plot V, M, T diagrams for a single component.

    Creates a 3-row subplot (Shear, Bending Moment, Torsion) vs span station.
    Multiple curves are overlaid (e.g., different load types or subcases).

    Parameters
    ----------
    curves : List[VMTCurve]
        Curves to plot (same component, different load types/subcases).
    title : str, optional
        Plot title. Defaults to component name.
    figsize : tuple
    save_path : str, optional
        Save figure to file.
    show : bool
        Whether to display the figure interactively.
    """
    if not curves:
        print("No VMT curves to plot.")
        return

    if show:
        matplotlib.use('TkAgg')
        import importlib
        importlib.reload(plt)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for curve in curves:
        style = _LOAD_STYLES.get(curve.load_type, {})
        label = f"{curve.load_type}"
        if curve.subcase_id > 0:
            label += f" (SC{curve.subcase_id})"

        axes[0].plot(curve.stations, curve.shear,
                     label=label, **style)
        axes[1].plot(curve.stations, curve.bending_moment,
                     label=label, **style)
        axes[2].plot(curve.stations, curve.torsion,
                     label=label, **style)

    comp_title = title or curves[0].component_name
    axes[0].set_title(f'{comp_title} - VMT Diagram', fontsize=14, fontweight='bold')

    labels = ['Shear Force V (N)', 'Bending Moment M (N-mm)', 'Torsion T (N-mm)']
    for i, ax in enumerate(axes):
        ax.set_ylabel(labels[i])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

    axes[2].set_xlabel(curves[0].station_label)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  VMT plot saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vmt_all(
    vmt_result: VMTResult,
    load_type: str = 'combined',
    subcase_id: int = None,
    figsize: Tuple[float, float] = None,
    save_path: str = None,
    show: bool = True,
) -> None:
    """Plot VMT for all components in a grid layout.

    Grid: rows = V, M, T; columns = components.

    Parameters
    ----------
    vmt_result : VMTResult
    load_type : str
        Which load type to show.
    subcase_id : int, optional
        Filter by subcase ID.
    figsize : tuple, optional
        Auto-sized if None.
    save_path : str, optional
    show : bool
    """
    # Collect unique component names
    comp_names = vmt_result.component_names
    if not comp_names:
        print("No VMT results to plot.")
        return

    if show:
        matplotlib.use('TkAgg')
        import importlib
        importlib.reload(plt)

    n_comp = len(comp_names)
    if figsize is None:
        figsize = (5 * n_comp, 10)

    fig, axes = plt.subplots(3, n_comp, figsize=figsize, squeeze=False)

    for col, comp_name in enumerate(comp_names):
        curves = vmt_result.get_curves(component_name=comp_name,
                                        load_type=load_type,
                                        subcase_id=subcase_id)
        if not curves:
            for row in range(3):
                axes[row, col].text(0.5, 0.5, 'No data',
                                    ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            axes[0, col].set_title(comp_name, fontsize=11, fontweight='bold')
            continue

        curve = curves[0]
        style = _LOAD_STYLES.get(load_type, {'color': 'black', 'linewidth': 1.5})

        axes[0, col].plot(curve.stations, curve.shear, **style)
        axes[1, col].plot(curve.stations, curve.bending_moment, **style)
        axes[2, col].plot(curve.stations, curve.torsion, **style)

        axes[0, col].set_title(comp_name, fontsize=11, fontweight='bold')

        for row in range(3):
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axhline(y=0, color='k', linewidth=0.5)
            axes[row, col].tick_params(labelsize=8)

        axes[2, col].set_xlabel(curve.station_label, fontsize=9)

    # Y-labels on left column only
    row_labels = ['V (N)', 'M (N-mm)', 'T (N-mm)']
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10)

    sc_str = f" SC{subcase_id}" if subcase_id is not None else ""
    fig.suptitle(f'VMT Diagram - {load_type}{sc_str}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  VMT plot saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vmt_envelope(
    vmt_result: VMTResult,
    component_name: str,
    load_type: str = 'combined',
    figsize: Tuple[float, float] = (12, 10),
    save_path: str = None,
    show: bool = True,
) -> None:
    """Plot VMT envelope across all subcases for a component.

    Shows max/min envelope with individual subcase curves in light gray.

    Parameters
    ----------
    vmt_result : VMTResult
    component_name : str
    load_type : str
    figsize : tuple
    save_path : str, optional
    show : bool
    """
    curves = vmt_result.get_curves(component_name=component_name,
                                    load_type=load_type)
    if not curves:
        print(f"No VMT curves for {component_name} ({load_type}).")
        return

    if show:
        matplotlib.use('TkAgg')
        import importlib
        importlib.reload(plt)

    # Use the stations from the first curve (they should all be the same)
    stations = curves[0].stations
    n = len(stations)

    # Stack all curves
    all_V = np.array([c.shear for c in curves])
    all_M = np.array([c.bending_moment for c in curves])
    all_T = np.array([c.torsion for c in curves])

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    data_sets = [
        (all_V, 'Shear Force V (N)'),
        (all_M, 'Bending Moment M (N-mm)'),
        (all_T, 'Torsion T (N-mm)'),
    ]

    for ax, (data, ylabel) in zip(axes, data_sets):
        # Individual subcases in light gray
        for j in range(len(curves)):
            ax.plot(stations, data[j], color='lightgray', linewidth=0.5,
                    alpha=0.7)

        # Envelope
        env_max = data.max(axis=0)
        env_min = data.min(axis=0)
        ax.fill_between(stations, env_min, env_max,
                         alpha=0.2, color='steelblue', label='Envelope')
        ax.plot(stations, env_max, color='steelblue', linewidth=1.5,
                label='Max')
        ax.plot(stations, env_min, color='steelblue', linewidth=1.5,
                linestyle='--', label='Min')

        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

    axes[0].set_title(
        f'{component_name} - VMT Envelope ({len(curves)} subcases, {load_type})',
        fontsize=14, fontweight='bold')
    axes[2].set_xlabel(curves[0].station_label)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  VMT envelope saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
