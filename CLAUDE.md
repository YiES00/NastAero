# NastAero - Claude Code Project Context

## Project Overview

NastAero is an open-source aeroelastic FEA solver (Python, ~60 source files) that reads MSC Nastran BDF files and solves SOL 101/103/144 with 3D visualization.

- **Language**: Python 3.9+, numpy/scipy core, pyvista (3D viz), matplotlib (VMT plots)
- **Units**: N-mm-sec (SI, millimeters) — KC-100 validation model uses this
- **Install**: `cd solver && pip install -e ".[plot]"`
- **Tests**: `cd solver && python -m pytest tests/ -x -q` (244 tests, ~1s)

## Directory Structure

```
solver/
├── nastaero/
│   ├── __main__.py          # CLI entry: python -m nastaero model.bdf [--save] [-p N]
│   ├── config.py            # Logging setup
│   ├── bdf/                 # BDF parser
│   │   ├── parser.py        # parse_bdf() → BDFModel
│   │   ├── model.py         # BDFModel dataclass (nodes, elements, properties, aero...)
│   │   ├── cards/           # Card parsers (GRID, CBAR, CQUAD4, CAERO1, SPLINE1, etc.)
│   │   ├── field_parser.py  # Fixed-width 8/16 char field parsing
│   │   ├── bulk_data.py     # Bulk data dispatcher
│   │   ├── case_control.py  # Case control parser
│   │   └── executive_control.py
│   ├── elements/            # Element stiffness (CBAR, CQUAD4, CTRIA3)
│   ├── fem/                 # Assembly, DOF management, BCs, coordinate transforms
│   │   ├── assembly.py      # Global K, M assembly
│   │   ├── dof_manager.py   # DOF numbering, free/constrained partitioning
│   │   ├── boundary.py      # SPC enforcement
│   │   └── coordinate_systems.py
│   ├── aero/                # Aerodynamics
│   │   ├── dlm.py           # Doublet Lattice Method (AIC matrix)
│   │   ├── panel.py         # CAERO1 panel meshing → AeroBox
│   │   └── spline.py        # IPS spline (G_sp, G_disp two-matrix architecture)
│   ├── solvers/
│   │   ├── sol101.py        # Static analysis
│   │   ├── sol103.py        # Modal analysis (Lanczos)
│   │   └── sol144.py        # Trim aeroelastic (Schur complement, parallel subcases)
│   ├── loads_analysis/
│   │   ├── trim_loads.py    # Nodal aero/inertial/combined forces, FORCE card output
│   │   ├── component_id.py  # Geometric component identification (wing/HTP/VTP/fuselage)
│   │   ├── vmt.py           # VMT (V-M-T) integration engine
│   │   └── case_generator.py
│   ├── output/
│   │   ├── result_data.py   # SubcaseResult, ResultData dataclasses
│   │   ├── result_io.py     # .naero file save/load (ZIP + numpy arrays)
│   │   └── f06_writer.py    # Nastran-style F06 output
│   └── visualization/
│       ├── cli.py           # Viz CLI: python -m nastaero.visualization [--load] [--vmt]
│       ├── viewer.py        # NastAeroViewer (PyVista-based 3D)
│       ├── mesh_builder.py  # Mesh construction, optimized beam tubes
│       └── vmt_plot.py      # VMT matplotlib plots (component, grid, envelope)
├── tests/                   # 244 pytest tests
│   └── validation/KC100/    # KC-100 full aircraft model (22,640 nodes, 27,016 elements)
├── scripts/                 # Benchmarks, report generation
└── pyproject.toml
```

## Key Architecture Decisions

### Two-G-Matrix Spline (cedc682)
Aeroelastic coupling uses TWO separate matrices:
- `G_sp` (normalwash): structural DOFs → slope (dz/dx). Only theta_y (DOF 5).
- `G_disp` (displacement): structural DOFs → z-displacement. Both z (DOF 3) + theta_y.
- Aero stiffness: `Q_aa = G_disp^T @ A_jj @ G_sp` (asymmetric!)
- Force transfer: `F_struct = G_disp^T @ f_aero` (use G_disp, NOT G_sp)

### Result Serialization (.naero format, c72b237)
- ZIP archive containing numpy .npy arrays + JSON metadata
- VizModel: lightweight BDFModel proxy (no cross-references, no SPLINE/SET1)
- 259x faster than re-solving, zero numerical diff
- Round-trip: `save_results()` → `.naero` → `load_results()` → (ResultData, VizModel)

### Parallel Subcase Solving (sol144.py)
- `-p 0` sequential, `-p -1` auto (cpu_count-1), `-p N` explicit workers
- ProcessPoolExecutor with shared TrimSharedData (K, M, G matrices precomputed)

### VMT Diagrams (b85b7a8)
- Geometric component detection from node coordinates (no SPLINE/SET1 needed)
- Section integration: tip-to-root summation of outboard forces at each span station
- Elastic axis: 40% chord from LE (transport aircraft standard)
- `--vmt-loads all` overlays aero (blue), inertial (red), combined (black)
- `--vmt-envelope` shows max/min across all subcases

## Data Structures

### BDFModel (bdf/model.py)
Central data container: `nodes`, `elements`, `properties`, `materials`, `coords`, `sets`, `splines`, `rigids`, `caero_panels`, `aesurfs`, `trims`, etc. Cross-referenced via `cross_reference()`.

### SubcaseResult (output/result_data.py)
Per-subcase results: `displacements`, `spc_forces`, `eigenvalues`, `frequencies`, `mode_shapes`, `trim_variables`, `aero_pressures`, `aero_forces`, `nodal_aero_forces`, `nodal_inertial_forces`, `nodal_combined_forces`, `trim_balance`.

Nodal forces format: `Dict[int, np.ndarray]` where key=node_id, value=ndarray(6)=[Fx,Fy,Fz,Mx,My,Mz].

### VizModel (output/result_io.py)
Lightweight BDFModel proxy for visualization from .naero files. Has `nodes`, `elements`, `properties`, `rigids`, `sol` but NO cross-references, NO SPLINE/SET1 data.

## CLI Usage

```bash
# Run solver
python -m nastaero model.bdf                    # SOL auto-detect
python -m nastaero model.bdf --save             # Save .naero results
python -m nastaero model.bdf -p -1              # Parallel subcases

# 3D Visualization
python -m nastaero.visualization model.bdf --run --trim
python -m nastaero.visualization --load model.naero --loads --subcase 3
python -m nastaero.visualization --load model.naero --pressure

# VMT Diagrams
python -m nastaero.visualization --load model.naero --vmt
python -m nastaero.visualization --load model.naero --vmt --vmt-component "Right Wing" --vmt-loads all
python -m nastaero.visualization --load model.naero --vmt --vmt-envelope
python -m nastaero.visualization --load model.naero --vmt --vmt-save output_prefix
```

## Coding Conventions

- Python 3.9+, `from __future__ import annotations`
- Type hints on public functions
- Numpy vectorization preferred over Python loops
- Lazy imports for optional deps (pyvista, matplotlib)
- All solver math in N-mm-sec units
- Tests: pytest, no external timeout plugins
- Git: descriptive commit messages, `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

## KC-100 Validation Model

- Path: `tests/validation/KC100/p400r3-free-trim.bdf`
- 22,640 nodes, 27,016 elements, 7 subcases (free-free trim)
- Node ID ranges: 100k-299k=fuselage, 300k-399k=L wing, 400k-499k=R wing, 500k-599k=L HTP, 600k-699k=R HTP, 700k-799k=VTP
- Wing span: Y=±465 to ±5617 mm, HTP: Y=0 to ±2000 mm, VTP: Z=1822 to 3200 mm
- 44 SPLINE1 cards, CORD2R coordinate systems

## Recent Change History

| Commit | Description |
|--------|-------------|
| b85b7a8 | VMT diagrams + fix G_disp for aero nodal forces |
| bb52a4d | Beam tube rendering 326x speedup (62s→0.19s) |
| 0b4a242 | Fix duplicate import os in viz cli |
| c72b237 | Result serialization (.naero format, 259x speedup) |
| cedc682 | Two-G-matrix architecture + Schur complement solver |
| 30d9c80 | Trim loads: nodal forces + FORCE card output |
| a2d5e6a | SOL 144 performance 6.8x speedup |
| b6b2bb6 | KC-100 support: RBE2, CD transform, panel normals |

## Known Issues / Future Work

- Flutter analysis (SOL 145) not yet implemented
