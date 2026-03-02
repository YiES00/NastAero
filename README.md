# NastAero

Open-source structural analysis FEA solver with MSC Nastran BDF I/O compatibility and aeroelasticity capabilities.

## Features

### Phase 1+2 (Current)
- **BDF Parser**: Fixed-8, fixed-16, and free-field format support
  - Cards: GRID, CORD2R, CBAR, CROD, CQUAD4, CQUAD8, CTRIA3, CTRIA6, PBAR, PROD, PSHELL, PCOMP, PSOLID, MAT1, MAT8
  - Loads: FORCE, MOMENT, GRAV, LOAD (combination)
  - Constraints: SPC, SPC1, RBE2
  - Mass: CONM2 (full 6x6 inertia with offset/parallel axis theorem)
  - Eigenvalue: EIGRL
- **SOL 101**: Linear static analysis (scipy.sparse.linalg.spsolve)
- **SOL 103**: Normal modes / real eigenvalue analysis (scipy.sparse.linalg.eigsh)
- **Elements**:
  - CBAR: 12-DOF Euler-Bernoulli beam (stiffness + consistent mass)
  - CQUAD4: 24-DOF Mindlin plate (2x2 Gauss, selective reduced integration)
  - CQUAD8: 48-DOF serendipity shell (3x3 Gauss membrane/bending, 2x2 reduced shear)
  - CTRIA3: 18-DOF CST membrane + DKT bending plate
  - CTRIA6: 36-DOF quadratic triangle (3-point Gauss, area coordinates)
- **Properties**: PSHELL, PCOMP (composite laminate, CLT A-matrix)
- **Output**: F06 format (displacements, SPC forces, eigenvalues, mode shapes)

### Future Phases
- Phase 3: C++ core (pybind11, Eigen, Spectra)
- Phase 4: SOL 144 (DLM, static aeroelastic trim)
- Phase 5: SOL 145 (Flutter, p-k method)
- Phase 6: SOL 146 (Dynamic aeroelastic response, RFA)

## Installation

```bash
cd solver
pip install -e ".[dev]"
```

## Usage

```bash
# Run analysis
python -m nastaero input.bdf

# Run tests
cd solver
pytest tests/ -v
```

## Validation

- **Cantilever beam (SOL 101)**: Tip deflection matches analytical PL³/3EI within 0.2%
- **Cantilever beam (SOL 103)**: First bending frequency matches analytical within 5%
- **Simply-supported plate (SOL 103)**: Natural frequencies in correct order of magnitude

## License

MIT
