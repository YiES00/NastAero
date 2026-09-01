# ASCENT-Load

[![tests](https://github.com/YiES00/ASCENT-Load/actions/workflows/tests.yml/badge.svg)](https://github.com/YiES00/ASCENT-Load/actions/workflows/tests.yml)

Open-source aeroelastic FEA framework with MSC Nastran BDF I/O
compatibility, static aeroelastic trim, rotor load models, and a
certification loads pipeline for eVTOL aircraft.

Developed at the ASCENT Laboratory (structures), Department of
Aerospace Engineering, Inha University, as the laboratory's
loads-analysis framework.

> **Renamed.** This project was previously released as **NastAero**
> (tags `v1.0-paper1` through `v1.4-paper1`). The name now identifies
> the developing laboratory. Code, history, and results are unchanged;
> the Python package is `ascent_load`, the commands are `ascent-load`
> and `ascent-load-gui`, and result archives use `.aload` (older
> `.naero` archives still load without conversion). GitHub redirects
> the former repository address.

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
  `.aload` result archives

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
- Desktop GUI pre/post processor (`ascent-load-gui`)

## Installation

```bash
cd solver
pip install -e ".[dev]"          # solver + test suite
```

Optional extras, by what you need:

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov | running the test suite |
| `plot` | matplotlib | V–M–T plots, figure regeneration |
| `gui` | qtpy, pyside6, pyvista, pyvistaqt, matplotlib | `ascent-load-gui`, 3-D visualization |

`dev` alone does **not** pull in matplotlib, so reproducing the plots
needs `".[dev,plot]"` and the 3-D viewer needs `".[gui]"`:

```bash
pip install -e ".[dev,plot]"     # + figure regeneration
pip install -e ".[gui]"          # + desktop workbench and 3-D viewer
```

## Usage

```bash
python -m ascent_load model.bdf --save        # solve, archive results
python -m ascent_load.visualization --load model.aload --vmt
python -m ascent_load.gui                     # desktop workbench
```

## Tests

```bash
cd solver && python -m pytest tests/ -q
```

The suite has 997 tests and takes about 11 minutes on the reference
workstation. What your clone reports depends on two things: whether
the proprietary comparison-model data is present, and whether the
optional `gui` extra is installed — the 51 GUI tests are not collected
without it.

| Install | Collected | Public clone reports |
|---|---|---|
| `pip install -e ".[dev,plot]"` | 947 | **921 passed, 26 skipped** |
| `pip install -e ".[dev,plot,gui]"` | 997 | **971 passed, 26 skipped** |

Both rows are asserted by the CI workflow above on every push, on
Python 3.10 and 3.12. The two `26`s are not the same 26: in the first
row it is 25 data-dependent tests plus the uncollected GUI module,
which pytest counts once; in the second it is the 26 data-dependent
tests themselves, one of which lives in the GUI module.

That data is **not redistributable and therefore not part of this
repository**. If you hold it under your own agreement, place it at
`solver/tests/validation/GACOMP/` (the path is git-ignored) and those
tests run automatically — no configuration needed. With the data and
the `gui` extra both present the suite reports 997 passed.

## Reproducing the published results

```bash
cd solver && python run_ilc8_cert_analysis.py   # about 95 s
```

Runs the full ILC-8 certification analysis (about 40 s). Figure and
table scripts for the papers live under `solver/scripts/` and
`docs/dissertation/papers/*/generate_*.py`.

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

## License

ASCENT-Load is released under the **GNU Affero General Public License
v3.0 or later** (AGPL-3.0-or-later); see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

The AGPL requires derivative works — including software offered to
users over a network — to be released under the same terms. If those
terms do not suit your use, a **separate commercial license** can be
negotiated with the copyright holder; contact the corresponding
author of the accompanying publications.

The proprietary comparison aircraft model used by 26 of the
validation tests is third-party data. It is not covered by this
license and is not distributed here (see [Tests](#tests)).
