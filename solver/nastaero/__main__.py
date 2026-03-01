"""CLI entry point: python -m nastaero input.bdf"""
from __future__ import annotations
import sys
from pathlib import Path
from .config import setup_logging, logger
from .bdf.parser import parse_bdf
from .solvers.sol101 import solve_static
from .solvers.sol103 import solve_modes
from .solvers.sol144 import solve_trim
from .output.f06_writer import write_f06


def main() -> None:
    setup_logging("INFO")

    if len(sys.argv) < 2:
        print("Usage: python -m nastaero <input.bdf>")
        sys.exit(1)

    bdf_path = Path(sys.argv[1])
    if not bdf_path.exists():
        print(f"Error: file not found: {bdf_path}")
        sys.exit(1)

    logger.info("NastAero v0.2.0 - Reading %s", bdf_path.name)
    bdf_model = parse_bdf(str(bdf_path))

    f06_path = str(bdf_path.with_suffix(".f06"))

    if bdf_model.sol == 101:
        results = solve_static(bdf_model)
    elif bdf_model.sol == 103:
        results = solve_modes(bdf_model)
    elif bdf_model.sol == 144:
        results = solve_trim(bdf_model)
    else:
        logger.error("Unsupported SOL %d", bdf_model.sol)
        sys.exit(1)

    write_f06(results, bdf_model, f06_path)
    logger.info("Results written to %s", f06_path)


if __name__ == "__main__":
    main()
