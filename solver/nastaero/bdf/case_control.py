"""Case control section parser."""
from __future__ import annotations
import re
from typing import List
from .model import BDFModel, Subcase
from ..config import logger

def parse_case_control(lines: List[str], model: BDFModel) -> None:
    current_subcase: Subcase = model.global_case
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if not stripped or stripped.startswith("$"):
            continue
        sc_match = re.match(r"SUBCASE\s+(\d+)", upper)
        if sc_match:
            sc_id = int(sc_match.group(1))
            current_subcase = Subcase(id=sc_id)
            model.subcases.append(current_subcase)
            continue
        label_match = re.match(r"LABEL\s*=\s*(.*)", stripped, re.IGNORECASE)
        if label_match:
            current_subcase.label = label_match.group(1).strip()
            continue
        subtitle_match = re.match(r"SUBTITLE\s*=\s*(.*)", stripped, re.IGNORECASE)
        if subtitle_match:
            if not current_subcase.label:
                current_subcase.label = subtitle_match.group(1).strip()
            continue
        spc_match = re.match(r"SPC\s*=\s*(\d+)", upper)
        if spc_match:
            current_subcase.spc_id = int(spc_match.group(1))
            continue
        mpc_match = re.match(r"MPC\s*=\s*(\d+)", upper)
        if mpc_match:
            current_subcase.mpc_id = int(mpc_match.group(1))
            continue
        load_match = re.match(r"LOAD\s*=\s*(\d+)", upper)
        if load_match:
            current_subcase.load_id = int(load_match.group(1))
            continue
        method_match = re.match(r"METHOD\s*=\s*(\d+)", upper)
        if method_match:
            current_subcase.method_id = int(method_match.group(1))
            continue
        for kw in ["DISPLACEMENT", "STRESS", "SPCFORCES", "OLOAD", "FORCE",
                    "STRAIN", "AEROF", "APRES", "DISP", "ECHO"]:
            m = re.match(rf"{kw}.*=\s*(.*)", upper)
            if m:
                current_subcase.output_requests[kw] = m.group(1).strip()
                break
        flutter_match = re.match(r"FLUTTER\s*=\s*(\d+)", upper)
        if flutter_match:
            current_subcase.flutter_id = int(flutter_match.group(1))
            continue
        trim_match = re.match(r"TRIM\s*=\s*(\d+)", upper)
        if trim_match:
            current_subcase.trim_id = int(trim_match.group(1))
            continue
    if not model.subcases:
        default_sc = Subcase(id=1)
        default_sc.spc_id = model.global_case.spc_id
        default_sc.mpc_id = model.global_case.mpc_id
        default_sc.load_id = model.global_case.load_id
        default_sc.method_id = model.global_case.method_id
        default_sc.output_requests = dict(model.global_case.output_requests)
        model.subcases.append(default_sc)
