# NastAero

Open-source structural analysis FEA solver with MSC Nastran BDF I/O compatibility and aeroelasticity capabilities.

## Features

### Phase 1+2 (Current)
- **BDF Parser**: Fixed-8, fixed-16, and free-field format support
  - Cards: GRID, CORD2R, CBAR, CROD, CQUAD4, CTRIA3, PBAR, PROD, PSHELL, PSOLID, MAT1
  - Loads: FORCE, MOMENT, GRAV, LOAD (combination)
  - Constraints: SPC, SPC1, RBE2
  - Mass: CONM2
  - Eigenvalue: EIGRL
- **SOL 101**: Linear static analysis (scipy.sparse.linalg.spsolve)
- **SOL 103**: Normal modes / real eigenvalue analysis (scipy.sparse.linalg.eigsh)
- **Elements**:
  - CBAR: 12-DOF Euler-Bernoulli beam (stiffness + consistent mass)
  - CQUAD4: 24-DOF Mindlin plate (2x2 Gauss, selective reduced integration)
  - CTRIA3: 18-DOF CST membrane + DKT bending plate
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
