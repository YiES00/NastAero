# VMT 다이어그램 패널 — matplotlib FigureCanvas를 임베드해 부재별 V/M/T·포락선 표시
from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

logger = logging.getLogger("ascent_load.gui")

PLOT_MODES = ["Subcase (aero/inertial/combined)", "Envelope (all subcases)"]


class VMTPanel(QWidget):
    """VMT 계산·그리기 컨트롤 + matplotlib 캔버스. set_data()로 모델/결과 주입."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self._model = None
        self._results = None
        self._components = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Component"))
        self._component = QComboBox()
        controls.addWidget(self._component, 2)
        controls.addWidget(QLabel("Mode"))
        self._mode = QComboBox()
        self._mode.addItems(PLOT_MODES)
        controls.addWidget(self._mode, 2)
        controls.addWidget(QLabel("Subcase"))
        self._subcase = QComboBox()
        controls.addWidget(self._subcase, 1)
        controls.addWidget(QLabel("Stations"))
        self._stations = QSpinBox()
        self._stations.setRange(10, 200)
        self._stations.setValue(50)
        controls.addWidget(self._stations)
        controls.addStretch()

        self._figure = Figure(figsize=(12, 10))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._status = QLabel("")

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._status)

        for w in (self._component, self._mode, self._subcase):
            w.currentIndexChanged.connect(self._redraw)
        self._stations.valueChanged.connect(self._redraw)
        self.setEnabled(False)

    # ------------------------------------------------------------------
    def set_data(self, model, results) -> None:
        """모델·결과를 주입한다. 절점 하중이 없으면 비활성."""
        self._model = model
        self._results = results
        self._components = None
        has_forces = (
            results is not None and results.subcases
            and any(sc.nodal_combined_forces for sc in results.subcases)
        )
        if not (model is not None and has_forces):
            self.setEnabled(False)
            self._figure.clear()
            self._canvas.draw_idle()
            return

        from ..loads_analysis.component_id import identify_components

        try:
            self._components = identify_components(model)
        except Exception:
            logger.exception("Component identification failed")
            self.setEnabled(False)
            return
        if not self._components.components:
            self._status.setText("구조 부재를 식별하지 못했습니다")
            self.setEnabled(False)
            return

        for combo, items in (
            (self._component, self._components.names()),
            (self._subcase, [f"Subcase {sc.subcase_id}"
                             for sc in results.subcases]),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.blockSignals(False)
        self.setEnabled(True)
        self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self, *_args) -> None:
        if self._components is None or not self.isEnabled():
            return
        comp_name = self._component.currentText()
        comp = next((c for c in self._components.components
                     if c.name == comp_name), None)
        if comp is None:
            return

        from ..loads_analysis.vmt import compute_vmt
        from ..visualization.vmt_plot import (
            draw_vmt_component, draw_vmt_envelope,
        )

        n_stations = self._stations.value()
        envelope = self._mode.currentIndex() == 1
        try:
            if envelope:
                curves = [
                    compute_vmt(self._model, sc.nodal_combined_forces, comp,
                                n_stations=n_stations, load_type="combined",
                                subcase_id=sc.subcase_id)
                    for sc in self._results.subcases
                    if sc.nodal_combined_forces
                ]
                if not curves:
                    return
                draw_vmt_envelope(self._figure, curves, comp_name)
            else:
                sc = self._results.subcases[max(0, self._subcase.currentIndex())]
                forces_by_type = {
                    "aero": sc.nodal_aero_forces,
                    "inertial": sc.nodal_inertial_forces,
                    "combined": sc.nodal_combined_forces,
                }
                curves = [
                    compute_vmt(self._model, forces, comp,
                                n_stations=n_stations, load_type=lt,
                                subcase_id=sc.subcase_id)
                    for lt, forces in forces_by_type.items() if forces
                ]
                if not curves:
                    return
                draw_vmt_component(self._figure, curves)
        except Exception:
            logger.exception("VMT computation failed")
            self._status.setText("VMT 계산 실패 — 로그를 확인하세요")
            return
        self._subcase.setEnabled(not envelope)
        self._status.setText(
            f"{comp_name}: {len(comp.node_ids):,} nodes, "
            f"{n_stations} stations")
        self._canvas.draw_idle()
