"""BDF card field parsing: fixed-8, fixed-16, and free-field formats.

Nastran BDF cards use 80-column lines with fields:
- Fixed 8-char: 10 fields of 8 characters each
- Fixed 16-char (large field): card name ends with *, fields are 16 chars
- Free-field: comma-separated values

Nastran float quirks:
- '1.5+3' means 1.5e3
- '1.5-3' means 1.5e-3
- '.5+3' means 0.5e3
- '1.5D3' means 1.5e3 (FORTRAN double)
- Blank or empty -> default value
"""
from __future__ import annotations
import re
from typing import List

_NASTRAN_FLOAT_RE = re.compile(
    r"^([+-]?\d*\.?\d*)"
    r"([+-])"
    r"(\d+)$"
)

def nastran_float(field: str, default: float = 0.0) -> float:
    s = field.strip()
    if not s:
        return default
    s = s.replace("d", "e").replace("D", "e")
    try:
        return float(s)
    except ValueError:
        pass
    m = _NASTRAN_FLOAT_RE.match(s)
    if m:
        mantissa = m.group(1) if m.group(1) else "0"
        exp_sign = m.group(2)
        exp_digits = m.group(3)
        try:
            return float(f"{mantissa}e{exp_sign}{exp_digits}")
        except ValueError:
            pass
    raise ValueError(f"Cannot parse Nastran float: '{field}'")

def nastran_int(field: str, default: int = 0) -> int:
    s = field.strip()
    if not s:
        return default
    return int(s)

def nastran_string(field: str, default: str = "") -> str:
    s = field.strip()
    return s if s else default

def parse_fixed8(line: str) -> List[str]:
    padded = line.ljust(80)
    return [padded[i:i + 8] for i in range(0, 80, 8)]

def parse_fixed16(lines: List[str]) -> List[str]:
    """대필드 카드 파싱 — MSC 규칙상 형식은 행 단위로 결정된다.

    연속행 마커가 '*'로 시작하면 그 행은 16자 필드(4개), 아니면(+ 또는
    공란 마커) 8자 소필드(8개)다. 혼합 카드(GACOMP CBAR*의 소필드 오프셋
    연속행 등)를 16자로 강제 절단하면 '0.      0.' 같은 병합 토큰이 생겨
    카드가 통째로 누락된다.
    """
    fields = []
    for line in lines:
        padded = line.ljust(80)
        if not fields:
            fields.append(padded[0:8])
            for s in (8, 24, 40, 56):
                fields.append(padded[s:s + 16])
        elif padded[0] == "*":
            for s in (8, 24, 40, 56):
                fields.append(padded[s:s + 16])
        else:
            for s in range(8, 72, 8):
                fields.append(padded[s:s + 8])
    return fields

def parse_free(line: str) -> List[str]:
    return [f.strip() for f in line.split(",")]

def detect_format(line: str) -> str:
    if "," in line:
        return "free"
    card_name = line[:8].strip()
    if card_name.endswith("*"):
        return "fixed16"
    return "fixed8"

def _is_continuation_marker(s: str) -> bool:
    """Check if a string is a free-field continuation marker (e.g. +CA1)."""
    s = s.strip()
    if not s or not s.startswith("+"):
        return False
    # Must contain at least one alpha character to be a marker
    # (pure numbers like +3 or +1.5 are values, not markers)
    rest = s[1:]
    if not rest:
        return False
    try:
        float(s)
        return False  # It's a number
    except ValueError:
        return True  # Not a number -> continuation marker


def parse_card_fields(lines: List[str]) -> List[str]:
    if not lines:
        return []
    fmt = detect_format(lines[0])
    if fmt == "free":
        all_parts = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if i == 0:
                # Remove trailing continuation marker
                if parts and _is_continuation_marker(parts[-1]):
                    parts = parts[:-1]
                all_parts.extend(parts)
            else:
                # Remove leading continuation marker
                if parts and _is_continuation_marker(parts[0]):
                    parts = parts[1:]
                # Remove trailing continuation marker
                if parts and _is_continuation_marker(parts[-1]):
                    parts = parts[:-1]
                all_parts.extend(parts)
        return all_parts
    if fmt == "fixed16":
        return parse_fixed16(lines)
    all_fields: List[str] = []
    for i, line in enumerate(lines):
        raw = parse_fixed8(line)
        if i == 0:
            all_fields.extend(raw[:9])
        else:
            all_fields.extend(raw[1:9])
    return all_fields


def expand_thru(tokens) -> list:
    """'THRU'가 섞인 ID 토큰 목록을 정수 목록으로 전개한다.

    Nastran의 리스트형 필드(SET1, SPC1, RBE2 등)는 어디서든
    ``a THRU b`` 를 쓸 수 있고 그 앞뒤에 낱개 ID가 더 올 수 있다.
    선두 한 번만 처리하면 중간 범위의 내부 ID가 통째로 빠진다.
    """
    ids: list = []
    raw = [str(t).strip() for t in tokens if str(t).strip()]
    i = 0
    while i < len(raw):
        token = raw[i].upper()
        if token == "THRU" and ids and i + 1 < len(raw):
            try:
                end = int(raw[i + 1])
            except ValueError:
                i += 1
                continue
            ids.extend(range(ids[-1] + 1, end + 1))
            i += 2
            continue
        try:
            ids.append(int(raw[i]))
        except ValueError:
            pass
        i += 1
    return ids
