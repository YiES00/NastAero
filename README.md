# NastAero

Open-source aeroelastic FEA framework with MSC Nastran BDF I/O
compatibility, static aeroelastic trim, rotor load models, and a
certification loads pipeline for eVTOL aircraft.

## Status

Implemented and covered by the test suite: SOL 101, SOL 103,
SOL 144, the rotor load module, and the full certification loads
pipeline (case-matrix generation, load recovery, V–M–T envelopes,
critical design-load selection, FORCE/MOMENT export).

Not implemented: SOL 145 (flutter, p-k) and SOL 146 (dynamic
aeroelastic response). The doublet-lattice infrastructure exists
and is retained for future flutter work.

## Features

### Input / output
- **BDF parser**: fixed-8, fixed-16, and free-field formats
  - Structure: GRID, CORD2R, CBAR, CROD, CQUAD4, CQUAD8, CTRIA3,
    CTRIA6, PBAR, PROD, PSHELL, PCOMP, PSOLID, MAT1, MAT8
  - Loads: FORCE, MOMENT, GRAV, LOAD (combination), PLOAD4
  - Constraints: SPC, SPC1, RBE2
  - Mass: CONM2 (full 6×6 inertia with offset / parallel-axis)
  - Aero: CAERO1, PAERO1, SPLINE1/2, SET1, AEROS, TRIM, AESTAT,
    AESURF, DMI (W2GJ initial downwash)
  - Eigenvalue: EIGRL
- **Output**: F06 (displacements, SPC forces, eigenvalues, mode
  shapes), FORCE/MOMENT design-load decks, PLOAD4 skin pressures,
  `.naero` result archives

### Solution sequences
- **SOL 101**: linear static analysis (sparse LU)
- **SOL 103**: normal modes (shift-invert Lanczos / ARPACK)
- **SOL 144**: static aeroelastic trim — Schur-complement reduction
  onto the trim degrees of freedom, inertia relief for free-free
  models, one factorization shared by all subcases at a flight
  condition, parallel subcase solution

### Elements
- CROD: axial rod
- CBAR: 12-DOF Euler–Bernoulli beam (stiffness + consistent mass)
- CQUAD4: 24-DOF Mindlin plate (2×2 Gauss, selective reduced
  integration)
- CQUAD8: 48-DOF serendipity shell (3×3 membrane/bending, 2×2
  reduced shear)
- CTRIA3: 18-DOF CST membrane + DKT bending plate
- CTRIA6: 36-DOF quadratic triangle
- Properties: PSHELL, PCOMP (composite laminate, CLT A-matrix)

### Aeroelasticity
- VLM and DLM aerodynamic influence coefficients
- Two-matrix IPS spline: separate normalwash (`G_sp`) and
  displacement (`G_disp`) mappings; the surface-derivative slope
  construction is the default

### Rotor loads
- Blade-element momentum theory with Prandtl tip loss (hover,
  axial, and forward flight), thrust-targeted inverse mode
- Reduced-order first-order inflow lag (scalar uniform-inflow
  reduction of the Pitt–Peters model) with a multi-rotor aggregate
  coupled to body plunge and lateral degrees of freedom

### Certification loads pipeline
- V-n diagram and fixed-wing load-case matrix (FAR 23 family)
- VTOL extensions: hover, transition, transition gust, one
  propulsion unit inoperative (OPI), rotor jam, vertical landing,
  tilt conversion corridor, tilt-actuator jam
- Failure × re-trim event screening with P × C adjudication
- Geometric component identification, V–M–T integration,
  envelopes, planar and 3-D convex-hull critical-case selection
- Desktop GUI pre/post processor (`nastaero-gui`)

## Installation

```bash
cd solver
pip install -e ".[dev]"
```

## Usage

```bash
python -m nastaero model.bdf --save        # solve, archive results
python -m nastaero.visualization --load model.naero --vmt
python -m nastaero.gui                     # desktop workbench
```

## Tests

```bash
cd solver && python -m pytest tests/ -q
```

The full suite is 841 tests (about 5–6 minutes). The tests that
read the proprietary comparison-model data skip automatically in
this archive (see the data note below); this archive reports
`815 passed, 26 skipped` (815 + 26 = 841).

## Reproducing the published results

```bash
cd solver && python run_ilc8_cert_analysis.py
```

Runs the full ILC-8 certification analysis (about 40 s). Figure and
table scripts for the papers live under `solver/scripts/`.

Reference environment: Python 3.12.8, NumPy 2.4.4, SciPy 1.17.1,
macOS 26.3.1 (arm64, 11 cores, 18 GB RAM).

## Verification and comparison status

- Element benchmarks: CROD/CBAR match analytical solutions to
  machine precision; MacNeal–Harder shell benchmarks agree within
  2% except one CQUAD8 cantilever case (7.8%)
- Hover BEMT: compared against the Knight–Hefner static-thrust
  experiment (NACA TN-626); mean 3.8% in the operating range
- Full aircraft: archived solver-to-solver comparison against MSC
  Nastran SOL 144 on a 22,640-node model. This is preliminary
  cross-code evidence, not validation — it predates the 2026-07
  trim-equilibrium corrections and used an unmatched W2GJ setup.

## Data note

The full-aircraft comparison model used for the archived
solver-to-solver comparison is proprietary aircraft geometry and is
not distributed here, nor are the archived commercial-solver
outputs derived from it. The tests that read those inputs skip
automatically when the data directory is absent. The ILC-8 and
ILC-8T application models are original to this project and are
included in full, so every ILC result in the papers reproduces from
this archive.

## License

To be determined.
