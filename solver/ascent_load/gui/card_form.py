# 카드 폼 편집 — BDF 카드의 필드 추출/8열 재포맷(카드 단위 writer)과 편집 다이얼로그
from __future__ import annotations

from typing import List, Optional, Tuple

# 주요 카드의 필드 이름 (Nastran Quick Reference 순서). 스키마에 없으면 F1, F2…로 표시.
CARD_SCHEMAS = {
    "GRID":   ["ID", "CP", "X1", "X2", "X3", "CD", "PS", "SEID"],
    "CBAR":   ["EID", "PID", "GA", "GB", "X1", "X2", "X3"],
    "CBEAM":  ["EID", "PID", "GA", "GB", "X1", "X2", "X3"],
    "CROD":   ["EID", "PID", "G1", "G2"],
    "CQUAD4": ["EID", "PID", "G1", "G2", "G3", "G4", "THETA", "ZOFFS"],
    "CTRIA3": ["EID", "PID", "G1", "G2", "G3", "THETA", "ZOFFS"],
    "PSHELL": ["PID", "MID1", "T", "MID2", "12I/T**3", "MID3", "TS/T", "NSM"],
    "PBAR":   ["PID", "MID", "A", "I1", "I2", "J", "NSM"],
    "PROD":   ["PID", "MID", "A", "J", "C", "NSM"],
    "MAT1":   ["MID", "E", "G", "NU", "RHO", "A", "TREF", "GE"],
    "MAT8":   ["MID", "E1", "E2", "NU12", "G12", "G1Z", "G2Z", "RHO"],
    "CONM2":  ["EID", "G", "CID", "M", "X1", "X2", "X3", "",
               "I11", "I21", "I22", "I31", "I32", "I33"],
    "FORCE":  ["SID", "G", "CID", "F", "N1", "N2", "N3"],
    "MOMENT": ["SID", "G", "CID", "M", "N1", "N2", "N3"],
    "GRAV":   ["SID", "CID", "A", "N1", "N2", "N3"],
    "SPC":    ["SID", "G1", "C1", "D1", "G2", "C2", "D2"],
    "SPC1":   ["SID", "C", "G1", "G2", "G3", "G4", "G5", "G6"],
    "EIGRL":  ["SID", "V1", "V2", "ND"],
    "RBE2":   ["EID", "GN", "CM", "GM1", "GM2", "GM3", "GM4", "GM5"],
    "TRIM":   ["SID", "MACH", "Q", "LABEL1", "UX1", "LABEL2", "UX2"],
}


def is_continuation(prev_line: str, line: str) -> bool:
    """parser._group_continuation_lines와 동일한 규칙으로 연속행 판정."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped[0] in ("+", "*"):
        return True
    if line[0] == " " and len(line) >= 8 and line[:8].strip() == "":
        return True
    if prev_line.strip().endswith(","):
        return True
    if len(prev_line) >= 73:
        marker = prev_line[72:80].strip()
        if marker and marker.startswith("+"):
            start = stripped[:8].strip() if len(stripped) >= 8 else stripped
            if start == marker:
                return True
    return False


def card_extent(lines: List[str], idx: int) -> Optional[Tuple[int, int]]:
    """idx 행이 속한 논리 카드의 [start, end) 범위. 카드가 아니면 None."""
    if not (0 <= idx < len(lines)):
        return None

    def is_card_line(i: int) -> bool:
        s = lines[i].strip()
        return bool(s) and not s.startswith("$")

    if not is_card_line(idx):
        return None
    # 연속행이면 카드 시작까지 거슬러 올라감
    start = idx
    while start > 0 and is_card_line(start - 1) and \
            is_continuation(lines[start - 1], lines[start]):
        start -= 1
    keyword = lines[start].strip().split(",")[0][:8].strip().upper()
    if keyword.startswith(("ENDDATA", "BEGIN", "CEND", "INCLUDE")):
        return None
    # 아래로 연속행 확장
    end = start + 1
    while end < len(lines) and is_card_line(end) and \
            is_continuation(lines[end - 1], lines[end]):
        end += 1
    return (start, end)


def fit8(value: str) -> str:
    """필드 값을 8자 이내로 맞춘다 (숫자는 정밀도 축소, 문자열은 절단)."""
    s = str(value).strip()
    if len(s) <= 8:
        return s
    try:
        f = float(s)
    except ValueError:
        return s[:8]
    for prec in range(7, 0, -1):
        out = f"{f:.{prec}g}"
        if len(out) <= 8:
            return out
    return f"{f:.1g}"[:8]


def format_card(fields: List[str]) -> List[str]:
    """필드 목록을 small-field(8열) 카드 행들로 재구성한다.

    첫 행은 키워드 + 데이터 8개, 넘치면 '+' 연속행에 8개씩.
    뒤쪽의 빈 필드는 잘라낸다.
    """
    fields = [fit8(f) for f in fields]
    while len(fields) > 1 and fields[-1] == "":
        fields.pop()

    lines = []
    head, rest = fields[0], fields[1:]
    chunks = [rest[i:i + 8] for i in range(0, len(rest), 8)] or [[]]
    for i, chunk in enumerate(chunks):
        lead = head if i == 0 else "+"
        line = "".join(f"{f:<8s}" for f in [lead] + chunk).rstrip()
        lines.append(line)
    return lines


def parse_card(lines: List[str]) -> List[str]:
    """논리 카드 행들 → 필드 목록 (파서의 parse_card_fields 재사용).

    후행 빈 필드는 잘라 format_card와 왕복 일관성을 유지한다.
    """
    from ..bdf.field_parser import parse_card_fields

    fields = [f.strip() for f in parse_card_fields(lines)]
    while len(fields) > 1 and fields[-1] == "":
        fields.pop()
    return fields


class CardFormDialog:
    """카드 필드를 폼으로 편집하는 QDialog 래퍼. exec() 후 fields()로 결과 취득."""

    def __init__(self, fields: List[str], parent=None) -> None:
        from qtpy.QtWidgets import (
            QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
            QScrollArea, QVBoxLayout, QWidget,
        )

        keyword = (fields[0] if fields else "").upper().rstrip("*")
        names = CARD_SCHEMAS.get(keyword, [])

        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle(f"{keyword} 카드 편집")
        self._edits: List[QLineEdit] = []

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.addRow("Card", QLabel(keyword))
        # 데이터 필드 + 여분 2칸(필드 추가용)
        values = fields[1:] + ["", ""]
        for i, value in enumerate(values):
            label = names[i] if i < len(names) else f"F{i + 1}"
            edit = QLineEdit(value)
            self._edits.append(edit)
            form.addRow(label, edit)

        scroll = QScrollArea()
        scroll.setWidget(form_host)
        scroll.setWidgetResizable(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.dialog.accept)
        buttons.rejected.connect(self.dialog.reject)

        layout = QVBoxLayout(self.dialog)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.dialog.resize(380, min(560, 90 + 34 * len(values)))
        self._keyword = keyword

    def exec(self) -> bool:
        from qtpy.QtWidgets import QDialog

        return self.dialog.exec_() == QDialog.Accepted if \
            hasattr(self.dialog, "exec_") else \
            self.dialog.exec() == QDialog.Accepted

    def fields(self) -> List[str]:
        return [self._keyword] + [e.text().strip() for e in self._edits]
