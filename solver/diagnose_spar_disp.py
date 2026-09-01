#!/usr/bin/env python3
"""Quick diagnostic: check spar node z-displacement values used by bending interpolation."""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from ascent_load.bdf.parser import parse_bdf
from ascent_load.loads_analysis.certification.modal_rom import ModalROM

bdf_path = "tests/validation/GACOMP/p400r3-free-trim.bdf"
print("Parsing BDF...")
bdf_model = parse_bdf(bdf_path)
print("Building ROM...")
rom = ModalROM.build(bdf_model, n_modes=20)

# Get raw displacement at 2.5g
print("\nComputing raw displacement at 2.5g...")
disp_arr, _ = rom.compute_response_arrays(
    alpha=0.05, V=80.0, de=-0.04, nz=2.5, rho=1.225, viz=False)

# Also get modal-filtered for comparison
disp_viz, _ = rom.compute_response_arrays(
    alpha=0.05, V=80.0, de=-0.04, nz=2.5, rho=1.225, viz=True)

sorted_nids = rom.sorted_nids
nid_fdof = rom._nid_fdof_idx

# Find CBAR/CBEAM nodes
bar_nids = set()
for eid, elem in bdf_model.elements.items():
    etype = type(elem).__name__
    if etype in ('CBAR', 'CBEAM'):
        for nid in elem.node_ids:
            bar_nids.add(nid)

print(f"\nTotal CBAR/CBEAM nodes: {len(bar_nids)}")

# Check wing spar nodes (300k-499k)
print("\n" + "=" * 80)
print("RIGHT WING (400k-499k) SPAR NODES — z-displacement at 2.5g")
print("=" * 80)
print(f"{'NID':>10} {'Y (mm)':>10} {'Z (mm)':>10} {'dz_raw':>12} {'dz_viz':>12} {'|d|_raw':>12}")
print("-" * 70)

r_wing_spar = []
for r, nid in enumerate(sorted_nids):
    if 400000 <= nid < 500000 and nid in bar_nids:
        # Get node position
        node_xyz = bdf_model.nodes[nid].xyz_global
        y = node_xyz[1]
        z = node_xyz[2]

        # Get displacement from disp_arr (ROM output row r)
        dz_raw = disp_arr[r, 2]  # z-component
        dz_viz = disp_viz[r, 2]
        d_mag = np.linalg.norm(disp_arr[r, :3])

        r_wing_spar.append((y, nid, dz_raw, dz_viz, d_mag, z))

r_wing_spar.sort(key=lambda x: x[0])
for y, nid, dz_raw, dz_viz, d_mag, z in r_wing_spar:
    print(f"{nid:10d} {y:10.1f} {z:10.1f} {dz_raw:12.4f} {dz_viz:12.4f} {d_mag:12.4f}")

print(f"\nTotal R wing spar nodes: {len(r_wing_spar)}")

# Check a few wing SKIN nodes for comparison
print("\n" + "=" * 80)
print("RIGHT WING SKIN NODES (sample) — z-displacement at 2.5g")
print("=" * 80)
r_wing_skin = []
for r, nid in enumerate(sorted_nids):
    if 400000 <= nid < 500000 and nid not in bar_nids:
        node_xyz = bdf_model.nodes[nid].xyz_global
        y = node_xyz[1]
        z = node_xyz[2]
        dz_raw = disp_arr[r, 2]
        d_mag = np.linalg.norm(disp_arr[r, :3])
        r_wing_skin.append((y, nid, dz_raw, d_mag, z))

r_wing_skin.sort(key=lambda x: x[0])
print(f"{'NID':>10} {'Y (mm)':>10} {'Z (mm)':>10} {'dz_raw':>12} {'|d|_raw':>12}")
print("-" * 60)
# Show every 50th node
for i in range(0, len(r_wing_skin), max(1, len(r_wing_skin) // 30)):
    y, nid, dz_raw, d_mag, z = r_wing_skin[i]
    print(f"{nid:10d} {y:10.1f} {z:10.1f} {dz_raw:12.4f} {d_mag:12.4f}")
print(f"\nTotal R wing skin nodes: {len(r_wing_skin)}")

# Check what the max z-displacement is at the wing tip spar
if r_wing_spar:
    tip_dz = r_wing_spar[-1][2]  # last spar node (outboard)
    root_dz = r_wing_spar[0][2]  # first spar node (inboard)
    print(f"\n>>> Wing tip spar dz = {tip_dz:.4f} mm")
    print(f">>> Wing root spar dz = {root_dz:.4f} mm")
    print(f">>> Bending range = {tip_dz - root_dz:.4f} mm")
    print(f">>> With scale 2839×: tip visual = {tip_dz * 2839:.1f} mm = {tip_dz * 2839 / 1000:.2f} m")

# Also check: what are the largest displacement nodes overall?
print("\n" + "=" * 80)
print("TOP 20 NODES BY |displacement| (raw, 2.5g)")
print("=" * 80)
all_mag = np.linalg.norm(disp_arr[:, :3], axis=1)
top_idx = np.argsort(all_mag)[::-1][:20]
print(f"{'NID':>10} {'Component':>12} {'Y':>8} {'Z':>8} {'|d|':>10} {'dz':>10} {'CBAR?':>6}")
print("-" * 70)
for idx in top_idx:
    nid = sorted_nids[idx]
    xyz = bdf_model.nodes[nid].xyz_global
    comp = ("Fuse" if 100000 <= nid < 300000 else
            "L Wing" if 300000 <= nid < 400000 else
            "R Wing" if 400000 <= nid < 500000 else
            "L HTP" if 500000 <= nid < 600000 else
            "R HTP" if 600000 <= nid < 700000 else
            "VTP" if 700000 <= nid < 800000 else "Other")
    is_bar = "Y" if nid in bar_nids else ""
    print(f"{nid:10d} {comp:>12} {xyz[1]:8.0f} {xyz[2]:8.0f} "
          f"{all_mag[idx]:10.4f} {disp_arr[idx, 2]:10.4f} {is_bar:>6}")

# Check max displacement on wing spar vs skin
wing_spar_mag = [np.linalg.norm(disp_arr[r, :3]) for r, nid in enumerate(sorted_nids)
                 if 300000 <= nid < 500000 and nid in bar_nids]
wing_skin_mag = [np.linalg.norm(disp_arr[r, :3]) for r, nid in enumerate(sorted_nids)
                 if 300000 <= nid < 500000 and nid not in bar_nids]
print(f"\n>>> Wing SPAR max |d| = {max(wing_spar_mag):.4f} mm (mean={np.mean(wing_spar_mag):.4f})")
print(f">>> Wing SKIN max |d| = {max(wing_skin_mag):.4f} mm (mean={np.mean(wing_skin_mag):.4f})")

# Check CROD nodes too — spar caps might be CROD elements
crod_nids = set()
for eid, elem in bdf_model.elements.items():
    etype = type(elem).__name__
    if etype == 'CROD':
        for nid in elem.node_ids:
            crod_nids.add(nid)

wing_crod_spar = [(bdf_model.nodes[nid].xyz_global[1], nid,
                   disp_arr[sorted_nids.index(nid), 2] if nid in sorted_nids else 0)
                  for nid in crod_nids if 400000 <= nid < 500000]
wing_crod_spar.sort(key=lambda x: x[0])
if wing_crod_spar:
    print(f"\n>>> R Wing CROD nodes: {len(wing_crod_spar)}")
    print(f">>> CROD tip dz = {wing_crod_spar[-1][2]:.4f} mm")
    print(f">>> CROD root dz = {wing_crod_spar[0][2]:.4f} mm")
    # Print a few
    print(f"\n{'NID':>10} {'Y (mm)':>10} {'dz_raw':>12}")
    for y, nid, dz in wing_crod_spar[::max(1, len(wing_crod_spar)//15)]:
        print(f"{nid:10d} {y:10.1f} {dz:12.4f}")
