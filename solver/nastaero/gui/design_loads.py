# 설계하중 선정 탭 — VMT 포락선·potato(상호작용) 헐로 critical 설계하중을 선정·표시·내보내기
from __future__ import annotations

import logging
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

logger = logging.getLogger("nastaero.gui")

PLANES = ["V-M", "V-T", "M-T"]


def pload4_available(model, results) -> bool:
    """PLOAD4 내보내기 가능 여부 — CAERO 패널과 박스 공력이 필요."""
    if model is None or results is None:
        return False
    if not (getattr(model, "caero_panels", None) or {}):
        return False
    return any(getattr(sc, "aero_forces", None) is not None
               for sc in getattr(results, "subcases", []) or [])


def batch_from_results(results) -> "BatchResult":
    """ResultData(서브케이스 결과)를 선정 파이프라인 입력인 BatchResult로 감싼다.

    case_id = subcase_id, 하중은 combined 절점력. 트림변수는 라벨/메타로 전달.
    """
    from ..loads_analysis.certification.batch_runner import (
        BatchResult, CaseResult,
    )

    batch = BatchResult()
    for sc in results.subcases:
        if not sc.nodal_combined_forces:
            continue
        tv = sc.trim_variables or {}
        # 착륙/지상 케이스는 SubcaseResult에 실린 메타(label/category 등)를
        # 그대로 승계 — 설계하중 표·요약 CSV에 선정 근거가 남는다
        batch.case_results.append(CaseResult(
            case_id=sc.subcase_id,
            category=getattr(sc, "category", "") or "trim",
            far_section=getattr(sc, "far_section", "") or "",
            nz=getattr(sc, "nz_cg", None) or 0.0,
            converged=True,
            nodal_forces=sc.nodal_combined_forces,
            trim_vars=tv,
            label=getattr(sc, "label", "") or f"Subcase {sc.subcase_id}",
        ))
        batch.completed_ids.add(sc.subcase_id)
    return batch


class DesignLoadsPanel(QWidget):
    """설계하중 선정 절차 실행 → 요약·설계하중 표·potato plot 표시 → FORCE 카드 내보내기."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._model = None
        self._results = None
        self._batch = None
        self._selection = None   # select_critical_design_loads 반환 dict

        # --- 상단: 선정 설정 + 실행 ---
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Stations"))
        self._stations = QSpinBox()
        self._stations.setRange(10, 200)
        self._stations.setValue(50)
        controls.addWidget(self._stations)
        self._axis_chk = QCheckBox("축 극값 포함")
        self._axis_chk.setChecked(True)
        controls.addWidget(self._axis_chk)
        self._plane_chks = []
        for plane in PLANES:
            chk = QCheckBox(plane)
            chk.setChecked(True)
            self._plane_chks.append(chk)
            controls.addWidget(chk)
        self._run_btn = QPushButton("선정 실행")
        self._run_btn.clicked.connect(self.run_selection)
        controls.addWidget(self._run_btn)
        self._export_btn = QPushButton("FORCE 카드 내보내기…")
        self._export_btn.clicked.connect(self.export_forces)
        self._export_btn.setEnabled(False)
        controls.addWidget(self._export_btn)
        self._pload_btn = QPushButton("PLOAD4 압력 내보내기…")
        self._pload_btn.setToolTip(
            "트림 박스 공력을 외피 요소 압력(PLOAD4)으로 내보냅니다 — "
            "외피 패널 국부 설계용 보조 산출물. 박스별 합력 보존, "
            "스플라인 절점과 무관하며 서브케이스별 SID로 기록됩니다. "
            "단면하중의 공식 산출물은 FORCE 덱입니다")
        self._pload_btn.clicked.connect(self.export_pload4)
        self._pload_btn.setEnabled(False)
        controls.addWidget(self._pload_btn)
        controls.addStretch()

        self._summary = QLabel("결과를 로드한 뒤 [선정 실행]을 누르세요")

        # --- 좌: 설계하중 표 ---
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Case", "지배 극값", "근거", "부재", "선정 사유"])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._redraw_potato)

        # --- 우: potato plot ---
        potato_controls = QHBoxLayout()
        potato_controls.addWidget(QLabel("Component"))
        self._component = QComboBox()
        potato_controls.addWidget(self._component, 2)
        potato_controls.addWidget(QLabel("Station"))
        self._station = QComboBox()
        potato_controls.addWidget(self._station, 1)
        potato_controls.addWidget(QLabel("Plane"))
        self._plane = QComboBox()
        self._plane.addItems(PLANES)
        self._plane.setCurrentText("M-T")
        potato_controls.addWidget(self._plane)
        for w in (self._component, self._station, self._plane):
            w.currentIndexChanged.connect(self._on_potato_combo)

        self._figure = Figure(figsize=(8, 7))
        self._canvas = FigureCanvasQTAgg(self._figure)
        potato_box = QWidget()
        playout = QVBoxLayout(potato_box)
        playout.setContentsMargins(0, 0, 0, 0)
        playout.addLayout(potato_controls)
        playout.addWidget(self._canvas, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._table)
        splitter.addWidget(potato_box)
        splitter.setSizes([550, 650])

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._summary)
        layout.addWidget(splitter, 1)
        self.setEnabled(False)

    # ------------------------------------------------------------------
    def set_data(self, model, results) -> None:
        self._model = model
        self._results = results
        self._batch = None
        self._selection = None
        self._table.setRowCount(0)
        self._figure.clear()
        self._canvas.draw_idle()
        self._export_btn.setEnabled(False)
        ok = (model is not None and results is not None
              and results.subcases
              and any(sc.nodal_combined_forces for sc in results.subcases))
        self.setEnabled(bool(ok))
        self._pload_btn.setEnabled(pload4_available(model, results))
        if ok:
            self._summary.setText(
                f"{len(results.subcases)}개 서브케이스 로드됨 — [선정 실행]을 누르세요")

    # ------------------------------------------------------------------
    def run_selection(self) -> None:
        """포락선+potato 헐 기반 critical 설계하중 선정 전체 절차 실행."""
        if self._model is None or self._results is None:
            return
        planes = tuple(
            tuple(chk.text().split("-")) for chk in self._plane_chks
            if chk.isChecked()
        )
        if not planes and not self._axis_chk.isChecked():
            QMessageBox.information(
                self, "NastAero", "축 극값 또는 상호작용 평면을 하나 이상 선택하세요")
            return
        from ..loads_analysis.certification.envelope import (
            select_critical_design_loads,
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._batch = batch_from_results(self._results)
            self._selection = select_critical_design_loads(
                self._model, self._batch,
                n_stations=self._stations.value(),
                planes=planes,
                include_axis=self._axis_chk.isChecked(),
            )
        except Exception as exc:
            logger.exception("Design load selection failed")
            QMessageBox.critical(self, "NastAero", f"선정 실패:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        sel = self._selection
        self._summary.setText(
            f"하중 케이스 {sel['n_cases_in']}개 → critical 기록 "
            f"{sel['n_critical']}건 → 설계하중 {sel['n_design_cases']}개 "
            f"(압축비 {sel['compression']:.1f}:1, "
            f"부재 {len(sel['components'])}개)")
        self._populate_table(sel["design_cases"])
        self._populate_potato_combos(sel)
        self._export_btn.setEnabled(True)
        self._redraw_potato()

    def _populate_table(self, design_cases: List) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(len(design_cases))
        for row, dc in enumerate(design_cases):
            items = [
                f"{dc.label or dc.case_id}",
                str(dc.n_govern),
                dc.basis,
                ", ".join(dc.components),
                dc.why(),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, dc.case_id)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        self._table.blockSignals(False)

    def _populate_potato_combos(self, sel: dict) -> None:
        proc = sel["processor"]
        comps = sel["components"]
        self._component.blockSignals(True)
        self._component.clear()
        self._component.addItems(comps)
        self._component.blockSignals(False)
        self._populate_stations()

    def _populate_stations(self) -> None:
        """현재 부재의 스테이션 목록을 채운다 (첫 케이스의 VMT 스테이션)."""
        proc = self._selection["processor"] if self._selection else None
        comp = self._component.currentText()
        self._station.blockSignals(True)
        self._station.clear()
        if proc is not None and comp:
            for case_data in proc.vmt_data.values():
                if comp in case_data:
                    for s in case_data[comp]["stations"]:
                        self._station.addItem(f"{s:.0f}", float(s))
                    break
        self._station.setCurrentIndex(0)
        self._station.blockSignals(False)

    # ------------------------------------------------------------------
    def _on_potato_combo(self, *_args) -> None:
        if self.sender() is self._component:
            self._populate_stations()
        self._redraw_potato()

    def _redraw_potato(self, *_args) -> None:
        if self._selection is None or self._station.count() == 0:
            return
        from ..visualization.cert_plot import draw_potato

        proc = self._selection["processor"]
        comp = self._component.currentText()
        station = self._station.currentData()
        x_qty, y_qty = self._plane.currentText().split("-")
        try:
            potato = proc.compute_potato(comp, station,
                                         x_quantity=x_qty, y_quantity=y_qty)
        except Exception:
            logger.exception("Potato computation failed")
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        draw_potato(ax, potato)

        # 표에서 선택한 설계하중 케이스를 별표로 강조
        selected = self._table.selectedItems()
        if selected and potato.case_ids:
            case_id = selected[0].data(Qt.UserRole)
            if case_id in potato.case_ids:
                i = potato.case_ids.index(case_id)
                ax.scatter([potato.x_values[i]], [potato.y_values[i]],
                           marker="*", s=350, color="gold",
                           edgecolors="black", zorder=7,
                           label=f"C{case_id} (selected)")
                ax.legend(fontsize=8, framealpha=0.9)
        self._figure.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def export_pload4(self) -> None:
        """트림 박스 공력을 PLOAD4 외피 압력 덱으로 내보낸다."""
        if not pload4_available(self._model, self._results):
            QMessageBox.information(
                self, "NastAero",
                "PLOAD4 내보내기에는 CAERO 패널이 있는 모델과 박스 "
                "공력(aero_forces)이 담긴 트림 결과가 필요합니다")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "PLOAD4 덱 저장", "skin_pressures_pload4.bdf",
            "BDF (*.bdf)")
        if not path:
            return
        from ..aero.panel import generate_all_panels
        from ..loads_analysis.pload_export import export_pload4

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            boxes = generate_all_panels(self._model, use_nastran_eid=True)
            reports = export_pload4(self._model, boxes, self._results, path)
        except Exception as exc:
            logger.exception("PLOAD4 export failed")
            QMessageBox.critical(self, "NastAero",
                                 f"PLOAD4 내보내기 실패:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not reports:
            QMessageBox.information(self, "NastAero",
                                    "박스 공력이 있는 서브케이스가 없습니다")
            return
        worst = max(r["residual_pct"] for r in reports.values())
        cov = min(r["n_covered"] for r in reports.values())
        n_box = next(iter(reports.values()))["n_boxes"]
        QMessageBox.information(
            self, "NastAero",
            f"PLOAD4 내보내기 완료: {path}\n"
            f"서브케이스 {len(reports)}개, 박스 커버 {cov}/{n_box}, "
            f"합력 잔차 최대 {worst:.2f}%\n"
            f"(미커버 = 아래에 외피가 없는 박스 — 그 몫은 FORCE 덱에 "
            f"보존됩니다)")

    def export_forces(self) -> None:
        """선정된 설계하중의 FORCE/MOMENT BDF + 요약 CSV를 내보낸다."""
        if self._selection is None or self._batch is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "내보낼 폴더 선택")
        if not out_dir:
            return
        from ..loads_analysis.certification.envelope import (
            write_design_load_summary_csv,
        )
        from ..loads_analysis.certification.force_export import (
            export_critical_forces,
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            export = export_critical_forces(
                self._batch, self._selection["processor"], self._model,
                out_dir)
            import os

            csv_path = write_design_load_summary_csv(
                self._selection["design_cases"],
                os.path.join(out_dir, "design_loads_summary.csv"))
        except Exception as exc:
            logger.exception("Force export failed")
            QMessageBox.critical(self, "NastAero", f"내보내기 실패:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        n = export.get("n_cases", len(self._selection["design_cases"])) \
            if isinstance(export, dict) else len(self._selection["design_cases"])
        QMessageBox.information(
            self, "NastAero",
            f"내보내기 완료: {out_dir}\n설계하중 {n}개 FORCE 카드 + "
            f"design_loads_summary.csv")
