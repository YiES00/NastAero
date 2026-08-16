# NastAero Solver - Project Memory

## Project Location
- NastAero repo (separate git repo): `D:\git\NastAero\`
- Solver code: `D:\git\NastAero\solver\`
- Python package: `solver/nastaero/`
- Tests: `solver/tests/`
- Validation BDFs: `solver/tests/validation/`

## Architecture
- **Language**: Python + C/C++ (Phase 1-2 is pure Python with SciPy)
- **BDF Parser**: `nastaero/bdf/` - field_parser.py (NOT cards.py, renamed to avoid conflict with cards/ directory)
- **FEM Core**: `nastaero/fem/` - DOF manager, assembly, boundary, load vector
- **Elements**: `nastaero/elements/` - bar.py (CBAR), quad4.py (CQUAD4), tria3.py (CTRIA3)
- **Solvers**: `nastaero/solvers/` - sol101.py (linear static), sol103.py (normal modes)
- **Output**: `nastaero/output/` - f06_writer.py, result_data.py

## Key Patterns
- BDF fixed-8 field: `nastaero/bdf/cards.py` was renamed to `field_parser.py` because `cards/` is also a package directory
- Nastran float parser handles implicit exponent (1.5+3 = 1.5e3) and FORTRAN D notation
- BDF free-field format used in validation BDFs to avoid 8-char field boundary issues
- CBAR orientation vector from x-field or G0 node direction

## Current Status: Phase 1+2 Complete
- 63 tests passing (19 parser + 12 element + 13 field + 5 SOL101 + 9 SOL103 + 5 misc)
- Cantilever beam tip deflection matches analytical PL^3/3EI within 0.2%
- Cantilever beam 1st bending freq matches analytical within 5%
- Simply-supported plate modes in correct order of magnitude
- Next: Phase 3 (C++ core porting with pybind11/Eigen/Spectra)

## Git Info
- NastAero is a separate git repo at `D:\git\NastAero\` (not inside FL-eCRM-001)
- The main git root at `D:\git\` is the FL-eCRM-001 repo (origin: YiES00/FL-eCRM-001)
- `gh` CLI is NOT installed on this system
- No remote configured yet for NastAero - user needs to create GitHub repo and push

## Environment Notes
- Python 3.13.6 on Windows 11
- `pip install -e ".[dev]"` from `solver/` directory
- `pytest tests/ -v` from `solver/` directory
