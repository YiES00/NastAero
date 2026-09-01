# 결과 후처리 패널 — 서브케이스/표시모드/성분/모드번호/하중종류/배율 선택
from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QSpinBox, QWidget,
)

DISPLAY_MODES = ["Model", "Displacement", "Mode Shape", "Aero Pressure",
                 "Nodal Forces"]


class ResultsPanel(QWidget):
    """선택이 바뀔 때마다 view_requested(dict)를 발신한다."""

    view_requested = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)

        self._mode = QComboBox()
        self._mode.addItems(DISPLAY_MODES)
        form.addRow("Display", self._mode)

        self._subcase = QComboBox()
        form.addRow("Subcase", self._subcase)

        self._component = QComboBox()
        self._component.addItems(["Magnitude", "T1", "T2", "T3",
                                  "R1", "R2", "R3"])
        form.addRow("Component", self._component)

        self._mode_num = QSpinBox()
        self._mode_num.setRange(1, 1)
        form.addRow("Mode #", self._mode_num)

        self._load_type = QComboBox()
        self._load_type.addItems(["combined", "aero", "inertial"])
        form.addRow("Load type", self._load_type)

        self._auto_scale = QCheckBox("자동 배율")
        self._auto_scale.setChecked(True)
        form.addRow(self._auto_scale)

        self._scale = QDoubleSpinBox()
        self._scale.setRange(0.001, 1e6)
        self._scale.setValue(1.0)
        self._scale.setDecimals(3)
        self._scale.setEnabled(False)
        form.addRow("Scale", self._scale)

        for w in (self._mode, self._subcase, self._component,
                  self._load_type):
            w.currentIndexChanged.connect(self._emit)
        self._mode_num.valueChanged.connect(self._emit)
        self._scale.valueChanged.connect(self._emit)
        self._auto_scale.toggled.connect(self._on_auto_toggled)

        self.setEnabled(False)
        self._results = None

    # ------------------------------------------------------------------
    def set_results(self, results) -> None:
        """결과 로드 시 서브케이스 목록과 모드 수를 채운다. None이면 비활성."""
        self._results = results
        self._subcase.blockSignals(True)
        self._subcase.clear()
        if results is not None and results.subcases:
            for sc in results.subcases:
                self._subcase.addItem(f"Subcase {sc.subcase_id}")
            n_modes = max((len(sc.mode_shapes or [])
                           for sc in results.subcases), default=1)
            self._mode_num.setMaximum(max(1, n_modes))
            self.setEnabled(True)
        else:
            self.setEnabled(False)
        self._subcase.blockSignals(False)

    def current_view(self) -> dict:
        return {
            "mode": self._mode.currentText(),
            "subcase_idx": max(0, self._subcase.currentIndex()),
            "component": self._component.currentText(),
            "mode_idx": self._mode_num.value() - 1,
            "load_type": self._load_type.currentText(),
            "scale": None if self._auto_scale.isChecked()
                     else self._scale.value(),
        }

    # ------------------------------------------------------------------
    def _on_auto_toggled(self, checked: bool) -> None:
        self._scale.setEnabled(not checked)
        self._emit()

    def _emit(self, *_args) -> None:
        if self.isEnabled():
            self.view_requested.emit(self.current_view())
