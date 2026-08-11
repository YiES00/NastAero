#!/usr/bin/env python3
"""Generate large-scale BDF models for scalability benchmarking.

Creates parametric flat plate models with CQUAD4 elements:
- Adjustable mesh density (nx × ny elements)
- Simply-supported or clamped boundary conditions
- Uniform pressure or gravity loading
- Material: Aluminum 2024-T3

Usage:
    python generate_large_model.py --nx 100 --ny 100   # 60K DOF
    python generate_large_model.py --nx 300 --ny 300   # 540K DOF
    python generate_large_model.py --nx 400 --ny 400   # ~1M DOF
"""
import argparse
import os
import time
import math


def generate_plate_model(nx: int, ny: int, Lx: float = 10.0, Ly: float = 10.0,
                          thickness: float = 0.01, bc: str = "clamped",
                          load_type: str = "pressure", output_path: str = None):
    """Generate a flat plate BDF model with CQUAD4 elements.

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions.
    Lx, Ly : float
        Plate dimensions (meters).
    thickness : float
        Plate thickness (meters).
    bc : str
        Boundary condition: "clamped" or "ssss" (simply-supported all edges).
    load_type : str
        "pressure" for uniform pressure, "gravity" for body force.
    output_path : str
        Output BDF file path.
    """
    n_nodes = (nx + 1) * (ny + 1)
    n_elems = nx * ny
    n_dof = n_nodes * 6
    dx = Lx / nx
    dy = Ly / ny

    if output_path is None:
        output_path = f"plate_{nx}x{ny}_{n_dof // 1000}k_dof.bdf"

    print(f"Generating plate model: {nx}x{ny} elements, {n_nodes} nodes, {n_dof} DOFs ({n_dof/1e6:.2f}M)")
    t_start = time.perf_counter()

    # Material: Aluminum 2024-T3
    E = 7.31e10  # Pa
    nu = 0.33
    rho = 2780.0  # kg/m³

    # Pressure magnitude (small enough for linear assumption)
    pressure = 1000.0  # Pa

    with open(output_path, 'w') as f:
        # Executive Control
        f.write("$ NastAero Large-Scale Benchmark Model\n")
        f.write(f"$ Plate: {Lx}m x {Ly}m x {thickness}m, {nx}x{ny} CQUAD4\n")
        f.write(f"$ Nodes: {n_nodes}, Elements: {n_elems}, DOFs: {n_dof}\n")
        f.write("$\n")
        f.write("SOL 101\n")
        f.write("CEND\n")

        # Case Control
        f.write("$ Case Control Section\n")
        f.write("TITLE = Large Scale Plate Benchmark\n")
        f.write(f"SUBTITLE = {nx}x{ny} CQUAD4, {n_dof} DOF\n")
        f.write("SPC = 1\n")
        f.write("LOAD = 1\n")
        f.write("DISPLACEMENT(PRINT) = ALL\n")
        f.write("BEGIN BULK\n")

        # Grid points
        f.write("$\n$ GRID POINTS\n$\n")
        nid = 1
        for j in range(ny + 1):
            y = j * dy
            for i in range(nx + 1):
                x = i * dx
                z = 0.0
                # Use fixed-8 field format for speed
                f.write(f"GRID    {nid:8d}        {x:8.4f}{y:8.4f}{z:8.4f}\n")
                nid += 1

        # CQUAD4 elements
        f.write("$\n$ CQUAD4 ELEMENTS\n$\n")
        eid = 1
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i + 1
                n2 = n1 + 1
                n3 = n2 + (nx + 1)
                n4 = n1 + (nx + 1)
                f.write(f"CQUAD4  {eid:8d}{1:8d}{n1:8d}{n2:8d}{n3:8d}{n4:8d}\n")
                eid += 1

        # Property and material
        f.write("$\n$ PROPERTY AND MATERIAL\n$\n")
        f.write(f"PSHELL  {1:8d}{1:8d}{thickness:8.6f}{1:8d}\n")
        f.write(f"MAT1    {1:8d}{E:8.2E}{'':8s}{nu:8.4f}{rho:8.1f}\n")

        # Boundary conditions
        f.write("$\n$ BOUNDARY CONDITIONS\n$\n")
        if bc == "clamped":
            # Clamp all edges (DOFs 123456)
            components = "123456"
        else:
            # Simply supported (DOFs 123 on edges)
            components = "123"

        # Collect boundary node IDs
        edge_nodes = set()
        # Bottom edge (j=0)
        for i in range(nx + 1):
            edge_nodes.add(i + 1)
        # Top edge (j=ny)
        for i in range(nx + 1):
            edge_nodes.add(ny * (nx + 1) + i + 1)
        # Left edge (i=0)
        for j in range(ny + 1):
            edge_nodes.add(j * (nx + 1) + 1)
        # Right edge (i=nx)
        for j in range(ny + 1):
            edge_nodes.add(j * (nx + 1) + (nx + 1))

        # Write SPC1 cards (batch nodes in groups of 6 per line)
        edge_list = sorted(edge_nodes)
        # Use THRU when possible
        if len(edge_list) > 10:
            # Write individual SPC1 entries in batches
            batch_size = 6
            for b in range(0, len(edge_list), batch_size):
                batch = edge_list[b:b + batch_size]
                line = f"SPC1    {1:8d}{components:>8s}"
                for nid in batch:
                    line += f"{nid:8d}"
                f.write(line + "\n")
        else:
            line = f"SPC1    {1:8d}{components:>8s}"
            for nid in edge_list:
                line += f"{nid:8d}"
            f.write(line + "\n")

        # Loading
        f.write("$\n$ LOADING\n$\n")
        if load_type == "pressure":
            # PLOAD2 on all elements (uniform pressure)
            # Write LOAD combination card first
            f.write(f"LOAD    {1:8d}{1.0:8.4f}{1.0:8.4f}{2:8d}\n")
            # PLOAD2 cards in batches
            for eid in range(1, n_elems + 1):
                f.write(f"PLOAD2  {2:8d}{pressure:8.1f}{eid:8d}\n")
        else:
            # GRAV card (gravity in -Z)
            f.write(f"GRAV    {1:8d}        {9.81:8.4f}{0.0:8.4f}{0.0:8.4f}{-1.0:8.4f}\n")

        f.write("ENDDATA\n")

    t_elapsed = time.perf_counter() - t_start
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Generated: {output_path} ({file_size:.1f} MB) in {t_elapsed:.2f} s")
    return output_path


def generate_gravity_model(nx: int, ny: int, output_path: str = None):
    """Generate a simpler model with gravity load (no PLOAD2 cards = smaller file)."""
    return generate_plate_model(nx, ny, load_type="gravity", output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate large-scale plate BDF models")
    parser.add_argument("--nx", type=int, default=100, help="Elements in X (default: 100)")
    parser.add_argument("--ny", type=int, default=100, help="Elements in Y (default: 100)")
    parser.add_argument("--lx", type=float, default=10.0, help="Plate length X (m)")
    parser.add_argument("--ly", type=float, default=10.0, help="Plate length Y (m)")
    parser.add_argument("--thickness", type=float, default=0.01, help="Plate thickness (m)")
    parser.add_argument("--bc", choices=["clamped", "ssss"], default="clamped")
    parser.add_argument("--load", choices=["pressure", "gravity"], default="gravity")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    generate_plate_model(args.nx, args.ny, args.lx, args.ly, args.thickness,
                         args.bc, args.load, args.output)
