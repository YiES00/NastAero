"""Certification loads visualization — V-n diagrams, case matrices, potato plots.

Provides matplotlib-based plotting for FAA Part 23 certification loads analysis.
"""
from __future__ import annotations
import os
import numpy as np
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _add_timestamp(fig, timestamp: str) -> None:
    """Add timestamp annotation to bottom-right of figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    timestamp : str
        Timestamp text to display (e.g. "Analysis: 2026-03-04 14:30:22").
    """
    fig.text(0.98, 0.01, timestamp,
             fontsize=7, color='gray', alpha=0.6,
             ha='right', va='bottom', fontfamily='monospace')


def plot_vn_diagram(vn_diagram, output_path: str = None,
                     title: str = None, show: bool = False,
                     figsize: tuple = (12, 7), dpi: int = 150,
                     timestamp: str = None) -> str:
    """Plot V-n diagram with maneuver and gust envelopes.

    Parameters
    ----------
    vn_diagram : VnDiagram
        Computed V-n diagram from vn_diagram.compute_vn_diagram().
    output_path : str, optional
        PNG save path. Auto-generated if None.
    title : str, optional
        Plot title.
    show : bool
        Whether to display interactively.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution.

    Returns
    -------
    str
        Path to saved PNG file.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Maneuver envelope (solid blue)
    if vn_diagram.maneuver_curve:
        Vm = [p[0] for p in vn_diagram.maneuver_curve]
        Nm = [p[1] for p in vn_diagram.maneuver_curve]
        ax.plot(Vm, Nm, 'b-', linewidth=1.5, label='Maneuver', zorder=2)
        ax.fill(Vm, Nm, alpha=0.08, color='blue')

    # Gust envelope (dashed green)
    if vn_diagram.gust_curve_pos:
        Vg_pos = [p[0] for p in vn_diagram.gust_curve_pos]
        Ng_pos = [p[1] for p in vn_diagram.gust_curve_pos]
        ax.plot(Vg_pos, Ng_pos, 'g--', linewidth=1.2, label='Gust (+)', zorder=2)

    if vn_diagram.gust_curve_neg:
        Vg_neg = [p[0] for p in vn_diagram.gust_curve_neg]
        Ng_neg = [p[1] for p in vn_diagram.gust_curve_neg]
        ax.plot(Vg_neg, Ng_neg, 'g--', linewidth=1.2, label='Gust (-)', zorder=2)

    # Corner points
    category_colors = {
        'maneuver': 'blue',
        'gust': 'green',
        'flap': 'orange',
        'stall': 'red',
    }
    category_markers = {
        'maneuver': 'o',
        'gust': 's',
        'flap': 'D',
        'stall': '^',
    }

    for pt in vn_diagram.corner_points:
        color = category_colors.get(pt.category, 'black')
        marker = category_markers.get(pt.category, 'o')
        ax.plot(pt.V_eas, pt.nz, marker, color=color, markersize=8,
                zorder=5, markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(pt.label,
                     xy=(pt.V_eas, pt.nz),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=7, fontweight='bold', color=color,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor=color, alpha=0.8))

    # Reference lines
    ax.axhline(y=0.0, color='gray', linewidth=0.5, linestyle='-')
    ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=vn_diagram.nz_max, color='red', linewidth=0.5,
               linestyle=':', alpha=0.5, label=f'nz_max={vn_diagram.nz_max:.2f}')
    ax.axhline(y=vn_diagram.nz_min, color='red', linewidth=0.5,
               linestyle=':', alpha=0.5, label=f'nz_min={vn_diagram.nz_min:.2f}')

    # Speed lines
    speeds = vn_diagram.speeds
    if speeds:
        for label, V in [('VS1', speeds.VS1), ('VA', speeds.VA),
                           ('VB', speeds.VB), ('VC', speeds.VC),
                           ('VD', speeds.VD)]:
            if V > 0:
                ax.axvline(x=V, color='gray', linewidth=0.5, linestyle=':',
                            alpha=0.4)
                ax.text(V, ax.get_ylim()[1] * 0.95, f' {label}',
                         fontsize=7, color='gray', ha='left', va='top')

    # Labels and formatting
    wc = vn_diagram.weight_condition
    if title is None:
        title_parts = ["V-n Diagram (Part 23)"]
        if wc:
            title_parts.append(f"W={wc.weight_N:.0f}N")
        title_parts.append(f"H={vn_diagram.altitude_m:.0f}m")
        title = " | ".join(title_parts)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('V_EAS (m/s)', fontsize=10)
    ax.set_ylabel('Load Factor nz (g)', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if timestamp:
        _add_timestamp(fig, timestamp)

    if output_path is None:
        output_path = 'vn_diagram.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def plot_case_matrix_summary(cases, output_path: str = None,
                              title: str = "Load Case Matrix Summary",
                              figsize: tuple = (16, 10), dpi: int = 150,
                              timestamp: str = None) -> str:
    """Plot load case matrix summary with category breakdown.

    Parameters
    ----------
    cases : list
        List of CertLoadCase or TrimCondition objects with .category and .nz.
    output_path : str, optional
        PNG save path.
    title : str
        Plot title.

    Returns
    -------
    str
        Path to saved PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    # Category counts
    categories = {}
    machs = []
    nzs = []
    altitudes = []

    for case in cases:
        cat = getattr(case, 'category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        machs.append(getattr(case, 'mach', 0.0))
        nzs.append(getattr(case, 'nz', 1.0))
        altitudes.append(getattr(case, 'altitude_m', 0.0))

    # 1. Pie chart — category breakdown
    ax = axes[0, 0]
    if categories:
        cat_colors = {
            'symmetric': '#4ECDC4', 'gust': '#45B7D1',
            'rolling': '#FF6B6B', 'yaw': '#FFA07A',
            'checked': '#98D8C8', 'flap': '#F7DC6F',
            'landing': '#BB8FCE', 'ground': '#D5DBDB',
        }
        labels = list(categories.keys())
        sizes = list(categories.values())
        colors = [cat_colors.get(l, '#AAAAAA') for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.0f%%',
            colors=colors, startangle=90, pctdistance=0.8)
        for t in texts:
            t.set_fontsize(8)
        for t in autotexts:
            t.set_fontsize(7)
    ax.set_title(f'Category Distribution (N={len(cases)})', fontsize=10)

    # 2. Mach-nz scatter
    ax = axes[0, 1]
    ax.scatter(machs, nzs, c='steelblue', alpha=0.5, s=15, edgecolors='none')
    ax.set_xlabel('Mach', fontsize=9)
    ax.set_ylabel('nz (g)', fontsize=9)
    ax.set_title('Mach vs Load Factor', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # 3. Altitude histogram
    ax = axes[1, 0]
    if altitudes:
        unique_alts = sorted(set(altitudes))
        alt_counts = [altitudes.count(a) for a in unique_alts]
        ax.barh([f'{a:.0f}m' for a in unique_alts], alt_counts,
                 color='#45B7D1', edgecolor='white')
    ax.set_xlabel('Number of Cases', fontsize=9)
    ax.set_ylabel('Altitude', fontsize=9)
    ax.set_title('Cases per Altitude', fontsize=10)

    # 4. Category bar chart
    ax = axes[1, 1]
    if categories:
        cats = sorted(categories.keys())
        counts = [categories[c] for c in cats]
        bars = ax.bar(range(len(cats)), counts, color='#4ECDC4',
                       edgecolor='white')
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(count), ha='center', fontsize=8)
    ax.set_ylabel('Number of Cases', fontsize=9)
    ax.set_title('Cases per Category', fontsize=10)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if timestamp:
        _add_timestamp(fig, timestamp)

    if output_path is None:
        output_path = 'case_matrix_summary.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_potato(potato_data, output_path: str = None,
                 title: str = None, figsize: tuple = (10, 8),
                 dpi: int = 150, timestamp: str = None) -> str:
    """Plot potato diagram (V-M or M-T scatter with convex hull).

    Parameters
    ----------
    potato_data : PotatoData
        Data from EnvelopeProcessor.compute_potato().
    output_path : str, optional
        PNG save path.
    title : str, optional
        Plot title.

    Returns
    -------
    str
        Path to saved PNG.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    cat_colors = {
        'symmetric': '#2196F3', 'gust': '#4CAF50',
        'rolling': '#F44336', 'yaw': '#FF9800',
        'checked': '#009688', 'flap': '#FFC107',
        'landing': '#9C27B0', 'ground': '#607D8B',
    }

    # Plot points colored by category
    for cat in set(potato_data.categories):
        mask = [c == cat for c in potato_data.categories]
        x = [potato_data.x_values[i] for i, m in enumerate(mask) if m]
        y = [potato_data.y_values[i] for i, m in enumerate(mask) if m]
        color = cat_colors.get(cat, '#888888')
        ax.scatter(x, y, c=color, label=cat, alpha=0.6, s=30,
                    edgecolors='black', linewidths=0.3, zorder=3)

    # Convex hull
    if potato_data.hull_x is not None:
        ax.plot(potato_data.hull_x, potato_data.hull_y,
                 'k-', linewidth=1.5, alpha=0.7, zorder=2,
                 label='Convex hull')
        ax.fill(potato_data.hull_x, potato_data.hull_y,
                 alpha=0.05, color='gray')

    # Annotate critical (extreme) points on the envelope
    if (potato_data.x_values and potato_data.y_values
            and len(potato_data.case_ids) == len(potato_data.x_values)):
        xv = potato_data.x_values
        yv = potato_data.y_values
        cids = potato_data.case_ids
        cats = potato_data.categories

        # Find extreme points: max/min of X (shear) and Y (bending)
        extremes = []
        for arr, dim, lbl in [
            (xv, 'x', potato_data.x_label.split('(')[0].strip()),
            (yv, 'y', potato_data.y_label.split('(')[0].strip()),
        ]:
            i_max = int(np.argmax(arr))
            i_min = int(np.argmin(arr))
            extremes.append((i_max, f"{lbl} max"))
            extremes.append((i_min, f"{lbl} min"))

        # De-duplicate by index
        seen = set()
        for idx, elbl in extremes:
            if idx in seen:
                continue
            seen.add(idx)
            ax.scatter([xv[idx]], [yv[idx]], s=120, facecolors='none',
                        edgecolors='red', linewidths=2.0, zorder=5)
            cat_str = cats[idx] if idx < len(cats) else ""
            ax.annotate(
                f"C{cids[idx]} ({cat_str})\n{elbl}",
                xy=(xv[idx], yv[idx]),
                xytext=(12, 12), textcoords='offset points',
                fontsize=7, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red',
                                 lw=0.8),
                zorder=6,
            )

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    if title is None:
        title = (f"Potato Plot — {potato_data.component} "
                 f"Station={potato_data.station:.0f}")

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(potato_data.x_label, fontsize=10)
    ax.set_ylabel(potato_data.y_label, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if timestamp:
        _add_timestamp(fig, timestamp)

    if output_path is None:
        output_path = 'potato_plot.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_vmt_envelope(component_envelope, output_path: str = None,
                        title: str = None, figsize: tuple = (14, 10),
                        dpi: int = 150, timestamp: str = None,
                        cg_x: float = None) -> str:
    """Plot VMT envelope for a structural component.

    Parameters
    ----------
    component_envelope : ComponentEnvelope
        Envelope data from EnvelopeProcessor.
    output_path : str, optional
        PNG save path.
    title : str, optional
        Plot title.

    Returns
    -------
    str
        Path to saved PNG.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)

    stations = np.array(component_envelope.stations)

    # Shear
    ax = axes[0]
    ax.fill_between(stations, component_envelope.V_max_array,
                      component_envelope.V_min_array,
                      alpha=0.2, color='steelblue')
    ax.plot(stations, component_envelope.V_max_array,
             'b-', linewidth=1.2, label='V max')
    ax.plot(stations, component_envelope.V_min_array,
             'b--', linewidth=1.2, label='V min')
    ax.set_ylabel('Shear V (N)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # Bending
    ax = axes[1]
    ax.fill_between(stations, component_envelope.M_max_array,
                      component_envelope.M_min_array,
                      alpha=0.2, color='green')
    ax.plot(stations, component_envelope.M_max_array,
             'g-', linewidth=1.2, label='M max')
    ax.plot(stations, component_envelope.M_min_array,
             'g--', linewidth=1.2, label='M min')
    ax.set_ylabel('Bending M (N-mm)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # Torsion
    ax = axes[2]
    ax.fill_between(stations, component_envelope.T_max_array,
                      component_envelope.T_min_array,
                      alpha=0.2, color='red')
    ax.plot(stations, component_envelope.T_max_array,
             'r-', linewidth=1.2, label='T max')
    ax.plot(stations, component_envelope.T_min_array,
             'r--', linewidth=1.2, label='T min')
    ax.set_ylabel('Torsion T (N-mm)', fontsize=10)
    ax.set_xlabel('Station (mm)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # CG position marker for fuselage
    if cg_x is not None:
        for ax in axes:
            ax.axvline(x=cg_x, color='red', linewidth=1.5, linestyle='--',
                        alpha=0.7, zorder=10)
            ax.text(cg_x, ax.get_ylim()[1] * 0.95, ' CG',
                     fontsize=9, color='red', fontweight='bold',
                     ha='left', va='top')
        # Add forward/aft labels
        axes[0].text(stations.min() + (cg_x - stations.min()) * 0.5,
                      axes[0].get_ylim()[1] * 0.85,
                      '← Forward (Nose)', fontsize=8, color='gray',
                      ha='center', style='italic')
        axes[0].text(cg_x + (stations.max() - cg_x) * 0.5,
                      axes[0].get_ylim()[1] * 0.85,
                      'Aft (Tail) →', fontsize=8, color='gray',
                      ha='center', style='italic')

    if title is None:
        title = (f"VMT Envelope — {component_envelope.component} "
                 f"({component_envelope.n_cases} cases)")
        if cg_x is not None:
            title += f" [CG at X={cg_x:.0f}mm]"
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if timestamp:
        _add_timestamp(fig, timestamp)

    if output_path is None:
        output_path = 'vmt_envelope.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


def plot_critical_frequency(frequency_data: dict,
                              batch_result=None,
                              output_path: str = None,
                              top_n: int = 15,
                              figsize: tuple = (12, 6),
                              dpi: int = 150,
                              timestamp: str = None) -> str:
    """Plot critical case frequency bar chart.

    Parameters
    ----------
    frequency_data : dict of {case_id: count}
        From EnvelopeProcessor.critical_case_frequency().
    batch_result : BatchResult, optional
        For looking up case labels.
    output_path : str, optional
        PNG save path.
    top_n : int
        Number of top cases to show.

    Returns
    -------
    str
        Path to saved PNG.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Sort by frequency
    sorted_items = sorted(frequency_data.items(),
                           key=lambda x: x[1], reverse=True)[:top_n]

    if not sorted_items:
        ax.text(0.5, 0.5, 'No critical cases', ha='center', va='center')
    else:
        case_ids = [f"Case {item[0]}" for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Color by category if batch_result available
        colors = []
        cat_colors = {
            'symmetric': '#2196F3', 'gust': '#4CAF50',
            'rolling': '#F44336', 'yaw': '#FF9800',
            'checked': '#009688', 'flap': '#FFC107',
            'landing': '#9C27B0',
        }

        for cid, count in sorted_items:
            cat = ""
            if batch_result:
                r = batch_result.get_result(cid)
                if r:
                    cat = r.category
            colors.append(cat_colors.get(cat, '#888888'))

        bars = ax.barh(range(len(case_ids)), counts, color=colors,
                        edgecolor='white')
        ax.set_yticks(range(len(case_ids)))
        ax.set_yticklabels(case_ids, fontsize=8)
        ax.invert_yaxis()

        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(count), va='center', fontsize=8)

    ax.set_xlabel('Times Appearing as Critical', fontsize=10)
    ax.set_title('Critical Case Frequency (Top Cases)', fontsize=12,
                  fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if timestamp:
        _add_timestamp(fig, timestamp)

    if output_path is None:
        output_path = 'critical_frequency.png'
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path
