#!/usr/bin/env python3
"""Diagnostic: Visualize LDRV mode shapes from the ModalROM.

For each mode, shows:
- Mode number, frequency (Hz)
- Max displacement location (component: fuselage/wing/HTP/VTP)
- Classification: global bending vs local panel mode
- PyVista 3D visualization of the mode shape on the structural mesh

Usage:
    python diagnose_modes.py
"""
from __future__ import annotations

import sys
import os
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

from ascent_load.bdf.parser import parse_bdf
from ascent_load.loads_analysis.certification.modal_rom import ModalROM
from ascent_load.visualization.mesh_builder import build_structural_mesh


def classify_node(nid: int) -> str:
    """GACOMP node ID range → component name."""
    if 100000 <= nid < 300000:
        return "Fuselage"
    elif 300000 <= nid < 400000:
        return "L Wing"
    elif 400000 <= nid < 500000:
        return "R Wing"
    elif 500000 <= nid < 600000:
        return "L HTP"
    elif 600000 <= nid < 700000:
        return "R HTP"
    elif 700000 <= nid < 800000:
        return "VTP"
    else:
        return "Other"


def main():
    bdf_path = "tests/validation/GACOMP/p400r3-free-trim.bdf"

    print("=" * 70)
    print("  Mode Shape Diagnostic — ASCENT-Load LDRV Modes")
    print("=" * 70)

    # Parse model
    print("\n[1] Parsing BDF model...")
    bdf_model = parse_bdf(bdf_path)

    # Build ROM (this includes LDRV mode extraction)
    print("\n[2] Building ROM with LDRV modes...")
    rom = ModalROM.build(bdf_model, n_modes=20)

    # Check that Phi_viz is stored
    if rom.Phi_viz is None or rom.Phi_viz.size == 0:
        print("ERROR: Phi_viz not stored in ROM. Aborting.")
        return

    Phi = rom.Phi_viz  # (n_free, n_modes)
    n_free, n_modes = Phi.shape
    print(f"\n[3] Phi_viz shape: ({n_free}, {n_modes})")
    print(f"    Frequencies: {rom.frequencies_hz}")
    print(f"    n_modes = {n_modes}, n_free_dofs = {n_free}")

    # Build node mapping: sorted_nids → fdof_idx
    sorted_nids = rom.sorted_nids
    nid_fdof = rom._nid_fdof_idx  # (n_nodes, 6)
    n_nodes = len(sorted_nids)

    # For each mode, extract physical displacement at each node
    print(f"\n[4] Analyzing {n_modes} mode shapes...")
    print(f"    {'Mode':>4} {'Freq(Hz)':>10} {'MaxDisp':>10} "
          f"{'MaxNode':>10} {'Component':>12} {'Type':>20}")
    print("    " + "-" * 72)

    # Prepare summary arrays
    mode_node_disp = np.zeros((n_modes, n_nodes, 3))  # xyz displacement per node per mode

    for m in range(n_modes):
        phi_m = Phi[:, m]  # (n_free,) — mode shape vector in free DOF space

        # Map to physical node displacements
        for i in range(n_nodes):
            for comp in range(3):  # xyz only
                fidx = nid_fdof[i, comp]
                if fidx >= 0:
                    mode_node_disp[m, i, comp] = phi_m[fidx]

    # Analyze each mode
    for m in range(n_modes):
        disp_mag = np.linalg.norm(mode_node_disp[m], axis=1)
        max_idx = np.argmax(disp_mag)
        max_nid = sorted_nids[max_idx]
        max_val = disp_mag[max_idx]
        comp = classify_node(max_nid)

        # Classify mode: compute "participation" by component
        wing_mask = np.array([(300000 <= nid < 500000)
                               for nid in sorted_nids])
        htp_mask = np.array([(500000 <= nid < 700000)
                              for nid in sorted_nids])
        vtp_mask = np.array([(700000 <= nid < 800000)
                              for nid in sorted_nids])
        fuse_mask = np.array([(100000 <= nid < 300000)
                               for nid in sorted_nids])

        # RMS displacement by component
        wing_rms = np.sqrt(np.mean(disp_mag[wing_mask]**2)) if np.any(wing_mask) else 0
        htp_rms = np.sqrt(np.mean(disp_mag[htp_mask]**2)) if np.any(htp_mask) else 0
        vtp_rms = np.sqrt(np.mean(disp_mag[vtp_mask]**2)) if np.any(vtp_mask) else 0
        fuse_rms = np.sqrt(np.mean(disp_mag[fuse_mask]**2)) if np.any(fuse_mask) else 0
        total_rms = np.sqrt(np.mean(disp_mag**2))

        # Determine mode type
        # "Global" = wing or HTP dominates and displacement is spread across the span
        # "Local" = concentrated in a few nodes
        wing_frac = wing_rms / max(total_rms, 1e-30)
        htp_frac = htp_rms / max(total_rms, 1e-30)
        vtp_frac = vtp_rms / max(total_rms, 1e-30)
        fuse_frac = fuse_rms / max(total_rms, 1e-30)

        # Check how concentrated the mode is
        sorted_mag = np.sort(disp_mag)[::-1]
        top10_energy = np.sum(sorted_mag[:10]**2)
        total_energy = np.sum(sorted_mag**2)
        concentration = top10_energy / max(total_energy, 1e-30)

        if wing_frac > 1.0 and concentration < 0.3:
            mode_type = "Wing global"
        elif htp_frac > 1.0 and concentration < 0.3:
            mode_type = "HTP global"
        elif vtp_frac > 1.0 and concentration < 0.3:
            mode_type = "VTP global"
        elif concentration > 0.5:
            mode_type = f"Local ({comp})"
        elif fuse_frac > 1.0:
            mode_type = "Fuselage"
        else:
            dominant = max(
                [("Wing", wing_frac), ("HTP", htp_frac),
                 ("VTP", vtp_frac), ("Fuse", fuse_frac)],
                key=lambda x: x[1])
            mode_type = f"{dominant[0]} ({concentration:.0%} conc)"

        freq = rom.frequencies_hz[m] if m < len(rom.frequencies_hz) else 0.0
        print(f"    {m:4d} {freq:10.2f} {max_val:10.4e} "
              f"{max_nid:10d} {comp:>12} {mode_type:>20}")

    # ------------------------------------------------------------------
    # Participation analysis: which modes contribute to wing z-bending?
    # ------------------------------------------------------------------
    print(f"\n[5] Modal participation in wing z-bending (U_alpha)...")
    print(f"    {'Mode':>4} {'Freq(Hz)':>10} {'q_alpha':>12} {'q_nz':>12} "
          f"{'Wing Z frac':>12}")

    # The modal coordinates for U_alpha: q = Phi^T @ M @ U_alpha
    # But we need M_lump... let's just compute the wing z-displacement fraction
    for m in range(n_modes):
        # Wing z-displacement for this mode (average absolute)
        wing_z = mode_node_disp[m, :, 2]  # z-component
        wing_z_abs = np.abs(wing_z[wing_mask])
        all_z_abs = np.abs(wing_z)

        wing_z_mean = np.mean(wing_z_abs) if len(wing_z_abs) > 0 else 0
        all_z_mean = np.mean(all_z_abs) if len(all_z_abs) > 0 else 0
        frac = wing_z_mean / max(all_z_mean, 1e-30)

        # Z-displacement gradient along span (global bending indicator)
        wing_nid_y = np.array([
            abs(bdf_model.nodes[nid].xyz_global[1])
            for nid in sorted_nids if 300000 <= nid < 500000])
        wing_z_vals = wing_z[wing_mask]
        if len(wing_nid_y) > 10:
            # Correlation between |Y| and |z_displacement|
            corr = np.corrcoef(wing_nid_y, np.abs(wing_z_vals))[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        freq = rom.frequencies_hz[m] if m < len(rom.frequencies_hz) else 0.0
        print(f"    {m:4d} {freq:10.2f} "
              f"{'—':>12} {'—':>12} "
              f"  {frac:8.3f}  corr={corr:+.3f}")

    # ------------------------------------------------------------------
    # PyVista visualization of first N modes
    # ------------------------------------------------------------------
    print(f"\n[6] Opening PyVista mode shape viewer...")
    import pyvista as pv

    mesh = build_structural_mesh(bdf_model, include_beams=False)
    sorted_nids_mesh = sorted(bdf_model.nodes.keys())
    nid_to_mesh_idx = {nid: i for i, nid in enumerate(sorted_nids_mesh)}

    # Map ROM sorted_nids to mesh point indices
    rom_to_mesh = {}
    for i, nid in enumerate(sorted_nids):
        if nid in nid_to_mesh_idx:
            rom_to_mesh[i] = nid_to_mesh_idx[nid]

    n_show = min(n_modes, 12)  # Show first 12 modes
    cols = 4
    rows = (n_show + cols - 1) // cols

    plotter = pv.Plotter(shape=(rows, cols), window_size=(1800, 400 * rows))

    for m in range(n_show):
        row = m // cols
        col = m % cols
        plotter.subplot(row, col)

        # Create deformed mesh
        mesh_copy = mesh.copy()
        disp_mag = np.zeros(mesh_copy.n_points)
        deformed_pts = mesh_copy.points.copy()

        for rom_i, mesh_i in rom_to_mesh.items():
            dxyz = mode_node_disp[m, rom_i, :]
            mag = np.linalg.norm(dxyz)
            disp_mag[mesh_i] = mag

        # Normalize mode shape for visualization
        max_mag = np.max(disp_mag)
        if max_mag > 1e-20:
            vis_scale = 500.0 / max_mag  # Scale so max = 500 mm
        else:
            vis_scale = 1.0

        for rom_i, mesh_i in rom_to_mesh.items():
            dxyz = mode_node_disp[m, rom_i, :]
            deformed_pts[mesh_i] += dxyz * vis_scale

        mesh_copy.points = deformed_pts
        mesh_copy.point_data['Mode_Disp'] = disp_mag

        freq = rom.frequencies_hz[m] if m < len(rom.frequencies_hz) else 0.0
        plotter.add_mesh(mesh_copy, scalars='Mode_Disp', cmap='jet',
                         show_edges=True, edge_color='gray', opacity=0.9,
                         show_scalar_bar=False)
        plotter.add_text(f"Mode {m}: {freq:.1f} Hz",
                         font_size=10, position='upper_left')
        plotter.camera.position = (15000, -15000, 8000)
        plotter.camera.focal_point = (4000, 0, 1000)
        plotter.camera.up = (0, 0, 1)

    plotter.show()


if __name__ == '__main__':
    main()
