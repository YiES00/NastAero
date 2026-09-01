# V-n 다이어그램 탭 — 기체 정보 탭의 AircraftConfig로 기동·돌풍 포락선을 그림
from __future__ import annotations

import logging

from qtpy.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

logger = logging.getLogger("ascent_load.gui")


class VnPanel(QWidget):
    """중량/고도 조합을 골라 V-n 다이어그램(§23.333-341)을 그린다."""

    def __init__(self, info_panel, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self._info = info_panel

        controls = QHBoxLayout()
        controls.addWidget(QLabel("중량/CG"))
        self._weight = QComboBox()
        controls.addWidget(self._weight, 2)
        controls.addWidget(QLabel("고도 (m)"))
        self._alt = QComboBox()
        controls.addWidget(self._alt, 1)
        draw_btn = QPushButton("V-n 그리기")
        draw_btn.clicked.connect(self.redraw)
        controls.addWidget(draw_btn)
        controls.addStretch()

        self._figure = Figure(figsize=(12, 7))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._status = QLabel("기체 정보 탭에서 속도·중량을 입력한 뒤 [V-n 그리기]")

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._status)

        info_panel.config_changed.connect(self._sync_combos)

    # ------------------------------------------------------------------
    def _sync_combos(self) -> None:
        d = self._info.to_dict()
        for combo, items in (
            (self._weight, [wc["label"] or f"W={wc['weight_N']:.0f}N"
                            for wc in d.get("weight_cg", [])]),
            (self._alt, [str(a) for a in d.get("altitudes_m", [0.0])]),
        ):
            cur = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentIndex(min(max(cur, 0), combo.count() - 1))
            combo.blockSignals(False)

    def redraw(self) -> None:
        from ..loads_analysis.certification.vn_diagram import (
            compute_vn_diagram,
        )
        from ..visualization.cert_plot import draw_vn_diagram

        try:
            config = self._info.get_config()
        except Exception as exc:
            self._status.setText(f"설정 오류: {exc}")
            return
        if not config.weight_cg_conditions:
            self._status.setText("중량/CG 조건을 먼저 추가하세요 (기체 정보 탭)")
            return
        if config.speeds.VC <= 0 or config.speeds.VS1 <= 0:
            self._status.setText("설계 속도(최소 VS1, VC)를 입력하세요")
            return
        wi = max(0, self._weight.currentIndex())
        wc = config.weight_cg_conditions[
            min(wi, len(config.weight_cg_conditions) - 1)]
        try:
            alt = float(self._alt.currentText() or 0.0)
        except ValueError:
            alt = 0.0
        try:
            vn = compute_vn_diagram(config, wc, altitude_m=alt)
        except Exception as exc:
            logger.exception("V-n computation failed")
            self._status.setText(f"V-n 계산 실패: {exc}")
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        draw_vn_diagram(ax, vn)
        self._figure.tight_layout()
        self._canvas.draw_idle()
        pts = ", ".join(f"{p.label}(V={p.V_eas:.1f}, nz={p.nz:.2f})"
                        for p in vn.corner_points[:6])
        self._status.setText(
            f"{wc.label}, H={alt:.0f}m — nz {vn.nz_min:.2f}~{vn.nz_max:.2f}, "
            f"코너점 {len(vn.corner_points)}개: {pts}…")
