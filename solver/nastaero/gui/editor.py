# BDF 텍스트 에디터 — 줄번호 거터, Nastran 구문 강조, 카드 폼 편집, INCLUDE 파일 나열
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from qtpy.QtCore import QRect, QSize, Qt, Signal
from qtpy.QtGui import (
    QColor, QFont, QFontDatabase, QKeySequence, QPainter, QSyntaxHighlighter,
    QTextCharFormat, QTextCursor,
)
from qtpy.QtWidgets import QMessageBox, QPlainTextEdit, QWidget

try:  # Qt5는 QtWidgets, Qt6은 QtGui
    from qtpy.QtWidgets import QShortcut
except ImportError:  # pragma: no cover
    from qtpy.QtGui import QShortcut


def list_bdf_files(master_path: str, max_depth: int = 10) -> List[str]:
    """마스터 BDF와 재귀 INCLUDE 파일들의 절대경로 목록 (마스터 먼저).

    parser._read_file과 같은 규칙: INCLUDE로 시작하는 행, 경로는 따옴표 제거,
    현재 파일의 디렉터리 기준 상대경로.
    """
    seen: List[str] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth or not path.exists():
            return
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.append(resolved)
        base_dir = path.parent
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("INCLUDE"):
                inc = stripped[7:].strip().strip("'\"")
                walk(base_dir / inc, depth + 1)

    walk(Path(master_path), 0)
    return seen

# 순수 데이터로 정의한 강조 규칙 (단위시험 대상)
HIGHLIGHT_RULES = [
    # (이름, 정규식, 전경색, bold 여부) — 위에서부터 적용, 뒤 규칙이 덮어씀
    ("keyword", r"^[A-Z][A-Z0-9]{0,7}\*?", "#0057b7", True),
    ("continuation", r"^[+*][^,]*", "#8b008b", False),
    ("comment", r"^\$.*", "#2e8b57", False),
]


class NastranHighlighter(QSyntaxHighlighter):
    """Nastran BDF 구문 강조 — 카드 키워드, 연속행, $ 주석."""

    def __init__(self, document) -> None:
        super().__init__(document)
        self._rules = []
        for _name, pattern, color, bold in HIGHLIGHT_RULES:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if bold:
                fmt.setFontWeight(QFont.Bold)
            self._rules.append((re.compile(pattern), fmt))

    def highlightBlock(self, text: str) -> None:
        for regex, fmt in self._rules:
            m = regex.match(text)
            if m:
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class _LineNumberArea(QWidget):
    def __init__(self, editor: "BdfEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self._editor.line_number_width(), 0)

    def paintEvent(self, event) -> None:
        self._editor.paint_line_numbers(event)


class BdfEditor(QPlainTextEdit):
    """BDF 원문 편집기. 저장은 MainWindow가 담당하고 여기서는 dirty 신호만 낸다."""

    dirty_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(12)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._highlighter = NastranHighlighter(self.document())
        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self.document().modificationChanged.connect(self.dirty_changed)
        self._update_line_area_width()
        QShortcut(QKeySequence("Ctrl+E"), self, self.edit_card_at_cursor)

    # ------------------------------------------------------------------
    # 카드 폼 편집
    # ------------------------------------------------------------------
    def contextMenuEvent(self, event) -> None:
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        menu.addAction("카드 편집 (Ctrl+E)", self.edit_card_at_cursor)
        menu.exec_(event.globalPos()) if hasattr(menu, "exec_") \
            else menu.exec(event.globalPos())

    def edit_card_at_cursor(self) -> None:
        """커서 위치의 논리 카드를 폼으로 편집해 제자리 텍스트 교체한다."""
        from .card_form import CardFormDialog, card_extent, format_card, parse_card

        lines = self.toPlainText().split("\n")
        idx = self.textCursor().blockNumber()
        extent = card_extent(lines, idx)
        if extent is None:
            QMessageBox.information(self, "NastAero",
                                    "커서 위치에 편집 가능한 카드가 없습니다")
            return
        start, end = extent
        try:
            fields = parse_card(lines[start:end])
        except Exception as exc:
            QMessageBox.warning(self, "NastAero", f"카드 파싱 실패:\n{exc}")
            return
        dialog = CardFormDialog(fields, parent=self)
        if not dialog.exec():
            return
        self._replace_lines(start, end, format_card(dialog.fields()))

    def insert_card(self, keyword: str) -> None:
        """빈 폼으로 새 카드를 만들어 ENDDATA 앞(없으면 끝)에 삽입한다."""
        from .card_form import CARD_SCHEMAS, CardFormDialog, format_card

        n_fields = max(len(CARD_SCHEMAS.get(keyword, [])), 4)
        dialog = CardFormDialog([keyword] + [""] * n_fields, parent=self)
        if not dialog.exec():
            return
        new_lines = format_card(dialog.fields())
        lines = self.toPlainText().split("\n")
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("ENDDATA"):
                insert_at = i
                break
        cursor = QTextCursor(self.document().findBlockByNumber(
            min(insert_at, self.blockCount() - 1)))
        if insert_at >= len(lines):
            cursor.movePosition(QTextCursor.End)
            cursor.insertText("\n" + "\n".join(new_lines))
        else:
            cursor.movePosition(QTextCursor.StartOfBlock)
            cursor.insertText("\n".join(new_lines) + "\n")

    def insert_text(self, text: str) -> None:
        """생성된 벌크 텍스트 블록을 ENDDATA 앞(없으면 끝)에 삽입한다."""
        lines = self.toPlainText().split("\n")
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("ENDDATA"):
                insert_at = i
                break
        cursor = QTextCursor(self.document().findBlockByNumber(
            min(insert_at, self.blockCount() - 1)))
        if insert_at >= len(lines):
            cursor.movePosition(QTextCursor.End)
            cursor.insertText("\n" + text.rstrip("\n"))
        else:
            cursor.movePosition(QTextCursor.StartOfBlock)
            cursor.insertText(text.rstrip("\n") + "\n")

    def _replace_lines(self, start: int, end: int,
                       new_lines: List[str]) -> None:
        doc = self.document()
        cursor = QTextCursor(doc.findBlockByNumber(start))
        cursor.movePosition(QTextCursor.StartOfBlock)
        end_block = doc.findBlockByNumber(end - 1)
        cursor.setPosition(end_block.position() + len(end_block.text()),
                           QTextCursor.KeepAnchor)
        cursor.insertText("\n".join(new_lines))

    # ------------------------------------------------------------------
    # 내용 로드/조회
    # ------------------------------------------------------------------
    def load_text(self, text: str) -> None:
        self.setPlainText(text)
        self.document().setModified(False)

    def is_dirty(self) -> bool:
        return self.document().isModified()

    def mark_saved(self) -> None:
        self.document().setModified(False)

    # ------------------------------------------------------------------
    # 줄번호 거터
    # ------------------------------------------------------------------
    def line_number_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 10 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_area_width(self, *_args) -> None:
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)

    def _update_line_area(self, rect, dy) -> None:
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(), self._line_area.width(),
                                   rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_area_width()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_width(), cr.height()))

    def paint_line_numbers(self, event) -> None:
        painter = QPainter(self._line_area)
        painter.fillRect(event.rect(), QColor("#f0f0f0"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block)
                    .translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor("#808080"))
                painter.drawText(0, top, self._line_area.width() - 4,
                                 self.fontMetrics().height(),
                                 Qt.AlignRight, str(block_number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1
        painter.end()
