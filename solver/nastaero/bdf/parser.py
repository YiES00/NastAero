"""Main BDF file parser."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Any
from .model import BDFModel
from .field_parser import parse_card_fields, detect_format
from .executive_control import parse_executive_line
from .case_control import parse_case_control
from .bulk_data import parse_bulk_card
from ..config import logger

class BDFParser:
    def __init__(self) -> None:
        self._include_depth = 0
        self._max_include_depth = 10

    def parse(self, filename: str) -> BDFModel:
        model = BDFModel()
        filepath = Path(filename).resolve()
        logger.info("Parsing BDF file: %s", filepath)
        raw_lines = self._read_file(filepath)
        exec_lines, cc_lines, bulk_lines = self._split_sections(raw_lines)
        exec_data: Dict[str, Any] = {}
        for line in exec_lines:
            parse_executive_line(line, exec_data)
        model.sol = exec_data.get("sol", 0)
        parse_case_control(cc_lines, model)
        card_groups = self._group_continuation_lines(bulk_lines)
        for card_lines in card_groups:
            fields = parse_card_fields(card_lines)
            parse_bulk_card(fields, model)
        self._transform_grid_coordinates(model)
        model.cross_reference()
        logger.info("Parsed: %d nodes, %d elements, %d properties, %d materials",
            len(model.nodes), len(model.elements), len(model.properties), len(model.materials))
        return model

    def _read_file(self, filepath: Path) -> List[str]:
        self._include_depth += 1
        if self._include_depth > self._max_include_depth:
            raise RuntimeError(f"INCLUDE depth exceeds {self._max_include_depth}")
        lines: List[str] = []
        base_dir = filepath.parent
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n\r")
                stripped = line.strip().upper()
                if stripped.startswith("INCLUDE"):
                    inc_part = line.strip()[7:].strip().strip("'\"")
                    inc_path = base_dir / inc_part
                    if inc_path.exists():
                        lines.extend(self._read_file(inc_path))
                    else:
                        logger.warning("INCLUDE not found: %s", inc_path)
                    continue
                lines.append(line)
        self._include_depth -= 1
        return lines

    def _split_sections(self, lines: List[str]):
        exec_lines, cc_lines, bulk_lines = [], [], []
        section = "executive"
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("$"): continue
            upper = stripped.upper()
            if section == "executive":
                if upper.startswith("CEND"): section = "case_control"; continue
                exec_lines.append(stripped)
            elif section == "case_control":
                if upper.startswith("BEGIN") and "BULK" in upper: section = "bulk"; continue
                cc_lines.append(stripped)
            elif section == "bulk":
                if upper.startswith("ENDDATA"): break
                bulk_lines.append(line)
        return exec_lines, cc_lines, bulk_lines

    def _group_continuation_lines(self, lines: List[str]) -> List[List[str]]:
        groups: List[List[str]] = []; current: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("$"): continue
            if not current: current = [line]; continue
            is_cont = False
            first_char = stripped[0] if stripped else ""
            if first_char in ("+", "*"): is_cont = True
            if not is_cont and line and line[0] == " ":
                if len(line) >= 8 and line[:8].strip() == "": is_cont = True
            prev_stripped = current[-1].strip()
            if prev_stripped.endswith(","): is_cont = True
            if not is_cont and current:
                prev_line = current[-1]
                if len(prev_line) >= 73:
                    cont_marker = prev_line[72:80].strip()
                    if cont_marker and cont_marker.startswith("+"):
                        current_start = stripped[:8].strip() if len(stripped) >= 8 else stripped.strip()
                        if current_start == cont_marker: is_cont = True
            if is_cont: current.append(line)
            else:
                if current: groups.append(current)
                current = [line]
        if current: groups.append(current)
        return groups

    def _transform_grid_coordinates(self, model: BDFModel) -> None:
        for nid, grid in model.nodes.items():
            if grid.cp != 0 and grid.cp in model.coords:
                grid.xyz_global = model.coords[grid.cp].to_global(grid.xyz)
            else:
                grid.xyz_global = grid.xyz.copy()
