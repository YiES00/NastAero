"""Executive control section parser."""
from __future__ import annotations
import re
from typing import Any, Dict
from ..config import logger

def parse_executive_line(line: str, exec_data: Dict[str, Any]) -> None:
    stripped = line.strip().upper()
    sol_match = re.match(r"SOL\s+(\d+)", stripped)
    if sol_match:
        exec_data["sol"] = int(sol_match.group(1))
        logger.debug("Executive: SOL %d", exec_data["sol"])
        return
    time_match = re.match(r"TIME\s+(\d+)", stripped)
    if time_match:
        exec_data["time"] = int(time_match.group(1))
        return
