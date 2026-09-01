# 메인 윈도우 — 트리/3D 뷰포트/에디터/실행·결과 패널/로그 콘솔을 묶는 QMainWindow
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QScrollArea, QTabWidget,
    QVBoxLayout, QWidget,
)

from .editor import BdfEditor, list_bdf_files
from .model_tree import ModelTreeWidget
from .results_panel import ResultsPanel
from .run_panel import RunPanel
from .scene import SceneController

logger = logging.getLogger("ascent_load.gui")

# *.naero 는 개명 전 결과 아카이브 — 변환 없이 그대로 열린다
_FILE_FILTER = ("ASCENT-Load files (*.bdf *.dat *.aload *.naero *.yaml *.yml);;"
                "All files (*)")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ASCENT-Load")
        # 화면보다 큰 초기 크기를 요구하지 않는다 — 1500x950을 고정하면
        # 노트북 화면(예: 1512x982 논리 해상도)에서 창이 화면을 꽉 채워
        # 사실상 크기 조절이 막힌다.
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, int(avail.width() * 0.9)),
                    min(950, int(avail.height() * 0.9)))

        # 상태
        self.current_path: Optional[str] = None   # 열린 파일 (.bdf 또는 .aload)
        self.bdf_path: Optional[str] = None       # 해석 가능한 마스터 .bdf 경로
        self.editor_file_path: Optional[str] = None  # 에디터에 열린 파일
        self.model = None                          # BDFModel 또는 VizModel
        self.results = None                        # ResultData 또는 None

        self._build_central()
        self._build_docks()
        self._build_menus()
        self.statusBar().showMessage("파일 > 열기로 .bdf 또는 .aload를 여세요")

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        """넓은 패널을 스크롤 영역에 담아 창의 최소 크기에서 분리한다.

        여러 열을 가로로 배치한 패널(Panel 1364px, Design Loads 1089px,
        Cert Cases 1069px, VMT 860px)은 그 최소 너비가 탭 위젯을 거쳐
        QMainWindow의 최소 너비로 전달된다. 좌우 도크까지 더하면 최소
        너비가 화면 폭을 넘어 창을 줄일 수 없게 되므로, 스크롤 영역으로
        감싸 최소 크기를 끊고 좁을 때는 스크롤로 보게 한다.
        """
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _build_central(self) -> None:
        self.tabs = QTabWidget()

        # 워크플로우 앞단: 기체/운용조건 정보 → V-n → 인증 케이스
        from .cert_cases import CertCasesPanel
        from .cert_setup import AircraftInfoPanel
        from .vn_widget import VnPanel

        self.aircraft_panel = AircraftInfoPanel()
        self.aircraft_panel.open_bdf_requested = self.open_path
        self.tabs.addTab(self.aircraft_panel, "Aircraft")

        self.vn_panel = VnPanel(self.aircraft_panel)
        self.tabs.addTab(self.vn_panel, "V-n")

        self.cert_cases_panel = CertCasesPanel(self.aircraft_panel)
        self.cert_cases_panel.open_requested = self.open_path
        self.cert_cases_panel.landing_results_ready = self._on_landing_results
        # 재트림 결과도 동일 병합 경로(동일 subcase_id 교체 + 덧붙임)를 쓴다
        self.cert_cases_panel.retrim_results_ready = self._on_landing_results
        self.tabs.addTab(self._scrollable(self.cert_cases_panel), "Cert Cases")

        self.viewport_tab = QFrame()
        vlayout = QVBoxLayout(self.viewport_tab)
        vlayout.setContentsMargins(0, 0, 0, 0)
        self.scene = SceneController(self.viewport_tab)
        vlayout.addWidget(self.scene.plotter.interactor)
        self.tabs.addTab(self.viewport_tab, "3D View")

        from .panel_editor import PanelEditorPanel

        self.panel_editor = PanelEditorPanel()
        self.panel_editor.insert_bdf_text = self._insert_bulk_text
        self.panel_editor.scene = self.scene
        self.panel_editor.show_3d_view = (
            lambda: self.tabs.setCurrentWidget(self.viewport_tab))
        self.tabs.addTab(self._scrollable(self.panel_editor), "Panel")

        self.editor_tab = editor_tab = QWidget()
        elayout = QVBoxLayout(editor_tab)
        elayout.setContentsMargins(4, 4, 4, 4)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File"))
        self.editor_file_combo = QComboBox()
        self.editor_file_combo.currentIndexChanged.connect(
            self._on_editor_file_changed)
        file_row.addWidget(self.editor_file_combo, 1)
        elayout.addLayout(file_row)
        self.editor = BdfEditor()
        self.editor.dirty_changed.connect(self._update_title)
        elayout.addWidget(self.editor)
        self.tabs.addTab(editor_tab, "BDF Editor")

        from .load_cases import LoadCasesPanel

        self.load_cases_panel = LoadCasesPanel()
        self.tabs.addTab(self.load_cases_panel.widget, "Load Cases")

        from .vmt_widget import VMTPanel

        self.vmt_panel = VMTPanel()
        self.tabs.addTab(self._scrollable(self.vmt_panel), "VMT")

        from .design_loads import DesignLoadsPanel

        self.design_loads_panel = DesignLoadsPanel()
        self.tabs.addTab(self._scrollable(self.design_loads_panel), "Design Loads")

        self.setCentralWidget(self.tabs)

    def _build_docks(self) -> None:
        self.tree = ModelTreeWidget(self)
        self.tree.node_clicked = self.scene.highlight_node
        self.tree.elements_clicked = self.scene.highlight_elements
        dock = QDockWidget("Model Tree", self)
        dock.setWidget(self.tree.container)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.run_panel = RunPanel(self)
        self.run_panel.log_line.connect(self._append_log)
        self.run_panel.run_finished.connect(self._on_run_finished)
        dock = QDockWidget("Analysis", self)
        dock.setWidget(self.run_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.results_panel = ResultsPanel(self)
        self.results_panel.view_requested.connect(self._on_view_requested)
        dock = QDockWidget("Results", self)
        dock.setWidget(self.results_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(5000)
        dock = QDockWidget("Log", self)
        dock.setWidget(self.log_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("파일(&F)")
        file_menu.addAction("열기…", self._open_dialog, "Ctrl+O")
        file_menu.addAction("BDF 저장", self.save_bdf, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("종료", self.close, "Ctrl+Q")

        edit_menu = self.menuBar().addMenu("편집(&E)")
        edit_menu.addAction("카드 편집", self.editor.edit_card_at_cursor,
                            "Ctrl+Shift+E")
        insert_menu = edit_menu.addMenu("카드 추가")
        from .card_form import CARD_SCHEMAS

        for keyword in sorted(CARD_SCHEMAS):
            insert_menu.addAction(
                keyword, lambda kw=keyword: self._insert_card(kw))

        view_menu = self.menuBar().addMenu("보기(&V)")
        for label, attr in (("구조 FEM 전체", "show_structure"),
                            ("보 튜브", "show_beams"),
                            ("공력 패널", "show_aero"),
                            ("RBE 라인", "show_rbe"),
                            ("요소 모서리", "show_edges")):
            action = view_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(getattr(self.scene, attr))
            action.toggled.connect(
                lambda checked, a=attr: self._toggle_scene(a, checked))
        # 모델의 요소 종류에 따라 동적으로 채워지는 서브메뉴
        self.fem_type_menu = view_menu.addMenu("FEM 요소 종류")
        self.fem_type_menu.setEnabled(False)
        view_menu.addSeparator()
        zoom_action = view_menu.addAction("절점/요소 클릭 시 줌인")
        zoom_action.setCheckable(True)
        zoom_action.setChecked(self.scene.zoom_on_highlight)
        # 표시 모드 재렌더 없이 플래그만 바꾼다 (_toggle_scene과 다름)
        zoom_action.toggled.connect(
            lambda checked: setattr(self.scene, "zoom_on_highlight", checked))
        iso_action = view_menu.addAction("클릭 항목만 표시(격리)")
        iso_action.setCheckable(True)
        iso_action.setChecked(self.scene.isolate_mode)
        iso_action.toggled.connect(self._toggle_isolate)

        run_menu = self.menuBar().addMenu("해석(&A)")
        run_menu.addAction("실행", self.run_panel.start_run, "F5")
        run_menu.addAction("중단", self.run_panel.stop_run)

    # ------------------------------------------------------------------
    # 파일 열기/저장
    # ------------------------------------------------------------------
    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "모델/결과 열기", "",
                                              _FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        """파일 확장자에 따라 .bdf(모델) 또는 .aload(결과)를 연다."""
        if self.editor.is_dirty() and not self._confirm_discard():
            return
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(self, "ASCENT-Load", f"파일이 없습니다: {path}")
            return
        if p.suffix.lower() in (".yaml", ".yml"):
            # 기체 설정 — bdf_model 참조가 있으면 BDF까지 자동 오픈
            self.aircraft_panel.open_yaml_file(str(p))
            self.tabs.setCurrentWidget(self.aircraft_panel)
            self.statusBar().showMessage(f"기체 설정 로드: {p.name}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if p.suffix.lower() == ".aload":
                self._load_naero(str(p))
            else:
                self._load_bdf(str(p))
        except Exception as exc:  # 파싱/로드 실패를 사용자에게 표시
            logger.exception("Failed to open %s", path)
            QMessageBox.critical(self, "ASCENT-Load", f"열기 실패:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.current_path = str(p)
        self._update_title()

    def _load_bdf(self, path: str) -> None:
        from ..bdf.parser import parse_bdf

        self.model = parse_bdf(path)
        self.results = None
        self.bdf_path = path
        self._populate_editor_files(path)
        self._refresh_all()
        # 옆에 같은 이름의 .aload가 있으면 결과도 함께 로드
        naero = Path(path).with_suffix(".aload")
        if naero.exists():
            self._attach_results(str(naero))
        self.statusBar().showMessage(
            f"{Path(path).name} — SOL {self.model.sol}, "
            f"{len(self.model.nodes)} nodes, {len(self.model.elements)} elements")

    def _load_naero(self, path: str) -> None:
        from ..output.result_io import load_results

        self.results, self.model = load_results(path)
        self.bdf_path = None
        self.editor_file_path = None
        self.editor_file_combo.blockSignals(True)
        self.editor_file_combo.clear()
        self.editor_file_combo.blockSignals(False)
        self.editor.load_text("")
        self._refresh_all()
        self.statusBar().showMessage(
            f"{Path(path).name} — {len(self.results.subcases)} subcases (보기 전용)")

    def _attach_results(self, naero_path: str) -> None:
        """현재 BDF 모델을 유지한 채 .aload 결과만 붙인다."""
        from ..output.result_io import load_results

        try:
            self.results, _viz = load_results(naero_path)
        except Exception:
            logger.exception("Failed to load results %s", naero_path)
            return
        self.scene.results = self.results
        self.results_panel.set_results(self.results)
        self.tree.populate(self.model, self.results)
        self.load_cases_panel.set_data(self.model, self.results)
        self.vmt_panel.set_data(self.model, self.results)
        self.design_loads_panel.set_data(self.model, self.results)
        self._append_log(f"결과 로드: {naero_path}")

    def save_bdf(self) -> None:
        """에디터의 현재 파일(마스터 또는 INCLUDE)을 저장하고 마스터를 재파싱한다."""
        target = self.editor_file_path or self.bdf_path
        if target is None or self.bdf_path is None:
            QMessageBox.information(self, "ASCENT-Load",
                                    "저장할 BDF가 없습니다 (.aload는 보기 전용)")
            return
        Path(target).write_text(self.editor.toPlainText())
        self.editor.mark_saved()
        self._append_log(f"저장: {target} — 마스터 재파싱 중…")
        try:
            from ..bdf.parser import parse_bdf

            self.model = parse_bdf(self.bdf_path)
        except Exception as exc:
            QMessageBox.warning(self, "ASCENT-Load",
                                f"저장됨. 그러나 재파싱 실패:\n{exc}")
            return
        self.results = None  # 형상이 바뀌었으므로 기존 결과는 무효
        # INCLUDE 목록이 바뀌었을 수 있으니 갱신 (현재 파일 유지)
        self._populate_editor_files(self.bdf_path, keep_current=target)
        self._refresh_all()
        self.statusBar().showMessage("저장·재파싱 완료 — 3D 갱신됨")

    # ------------------------------------------------------------------
    # 에디터 파일(마스터/INCLUDE) 전환
    # ------------------------------------------------------------------
    def _populate_editor_files(self, master_path: str,
                               keep_current: Optional[str] = None) -> None:
        files = list_bdf_files(master_path)
        self.editor_file_combo.blockSignals(True)
        self.editor_file_combo.clear()
        master_dir = Path(master_path).parent
        for f in files:
            try:
                label = str(Path(f).relative_to(master_dir))
            except ValueError:
                label = Path(f).name
            self.editor_file_combo.addItem(label, f)
        target = keep_current if keep_current in files else files[0]
        self.editor_file_combo.setCurrentIndex(files.index(target))
        self.editor_file_combo.blockSignals(False)
        self._load_editor_file(target)

    def _on_editor_file_changed(self, index: int) -> None:
        if index < 0:
            return
        path = self.editor_file_combo.itemData(index)
        if path is None or path == self.editor_file_path:
            return
        if self.editor.is_dirty() and not self._confirm_discard():
            # 원래 파일로 콤보 되돌림
            prev = self.editor_file_path
            self.editor_file_combo.blockSignals(True)
            for i in range(self.editor_file_combo.count()):
                if self.editor_file_combo.itemData(i) == prev:
                    self.editor_file_combo.setCurrentIndex(i)
                    break
            self.editor_file_combo.blockSignals(False)
            return
        self._load_editor_file(path)

    def _load_editor_file(self, path: str) -> None:
        self.editor_file_path = path
        self.editor.load_text(Path(path).read_text(errors="replace"))
        self._update_title()

    def _insert_bulk_text(self, text: str) -> None:
        """Panel 탭 등이 생성한 벌크 텍스트를 에디터에 삽입하고 이동한다."""
        if self.bdf_path is None:
            QMessageBox.information(self, "ASCENT-Load",
                                    "카드를 추가하려면 .bdf를 여세요")
            return
        self.editor.insert_text(text)
        self.tabs.setCurrentWidget(self.editor_tab)
        self.statusBar().showMessage(
            "카드가 에디터에 삽입되었습니다 — 저장(⌘S)하면 재파싱되어 "
            "모델에 반영됩니다")

    def _insert_card(self, keyword: str) -> None:
        if self.bdf_path is None:
            QMessageBox.information(self, "ASCENT-Load",
                                    "카드를 추가하려면 .bdf를 여세요")
            return
        self.tabs.setCurrentWidget(self.editor_tab)
        self.editor.insert_card(keyword)

    # ------------------------------------------------------------------
    # 갱신/이벤트
    # ------------------------------------------------------------------
    def _on_landing_results(self, subcases) -> None:
        """착륙/지상 케이스 절점하중을 현재 결과에 합류시킨다.

        같은 subcase_id의 이전 착륙 결과는 교체하고, 트림 결과가
        열려 있으면 그 뒤에 덧붙여 VMT/Design Loads가 함께 소비한다.
        """
        from ..output.result_data import ResultData

        if self.results is None:
            self.results = ResultData(title="Landing cases")
        new_ids = {sc.subcase_id for sc in subcases}
        self.results.subcases = (
            [sc for sc in self.results.subcases
             if sc.subcase_id not in new_ids] + list(subcases))
        self.scene.results = self.results
        self.results_panel.set_results(self.results)
        self.tree.populate(self.model, self.results)
        self.load_cases_panel.set_data(self.model, self.results)
        self.vmt_panel.set_data(self.model, self.results)
        self.design_loads_panel.set_data(self.model, self.results)
        self.statusBar().showMessage(
            f"착륙/지상 {len(subcases)}개 케이스 결과 합류 완료")

    def _refresh_all(self) -> None:
        self.scene.set_model(self.model, self.results)
        self.tree.populate(self.model, self.results)
        self.results_panel.set_results(self.results)
        self.run_panel.set_bdf(self.bdf_path, getattr(self.model, "sol", None))
        self.load_cases_panel.set_data(self.model, self.results)
        self.vmt_panel.set_data(self.model, self.results)
        self.design_loads_panel.set_data(self.model, self.results)
        self.aircraft_panel.set_model(
            self.model if self.bdf_path else None, self.bdf_path)
        self.cert_cases_panel.set_context(
            self.model if self.bdf_path else None, self.bdf_path)
        self.panel_editor.set_model(
            self.model if self.bdf_path else None, self.bdf_path)
        self._populate_fem_type_menu()

    def _on_view_requested(self, view: dict) -> None:
        mode = view["mode"]
        idx = view["subcase_idx"]
        if mode == "Model":
            self.scene.display_model()
        elif mode == "Displacement":
            self.scene.display_displacement(idx, view["component"],
                                            view["scale"])
        elif mode == "Mode Shape":
            self.scene.display_mode(idx, view["mode_idx"], view["scale"])
        elif mode == "Aero Pressure":
            self.scene.display_pressure(idx)
        elif mode == "Nodal Forces":
            self.scene.display_forces(idx, view["load_type"])
        self.tabs.setCurrentWidget(self.viewport_tab)

    def _on_run_finished(self, naero_path: str) -> None:
        if naero_path and Path(naero_path).exists():
            self._attach_results(naero_path)
            self.statusBar().showMessage("해석 완료 — 결과 로드됨")
        elif naero_path:
            self._append_log(f"결과 파일을 찾지 못함: {naero_path}")

    def _toggle_scene(self, attr: str, checked: bool) -> None:
        setattr(self.scene, attr, checked)
        self.scene.display_model()

    def _populate_fem_type_menu(self) -> None:
        """모델의 요소 종류로 [보기 > FEM 요소 종류] 체크 항목을 재구성."""
        self.fem_type_menu.clear()
        types = self.scene.present_element_types()
        self.fem_type_menu.setEnabled(bool(types))
        counts = {}
        for e in (getattr(self.model, "elements", {}) or {}).values():
            counts[e.type] = counts.get(e.type, 0) + 1
        for t in types:
            action = self.fem_type_menu.addAction(f"{t} ({counts.get(t, 0)})")
            action.setCheckable(True)
            action.setChecked(t not in self.scene.hidden_types)
            action.toggled.connect(
                lambda checked, tt=t: self._toggle_fem_type(tt, checked))

    def _toggle_fem_type(self, etype: str, visible: bool) -> None:
        if visible:
            self.scene.hidden_types.discard(etype)
        else:
            self.scene.hidden_types.add(etype)
        self.scene.display_model()

    def _toggle_isolate(self, checked: bool) -> None:
        self.scene.isolate_mode = checked
        if not checked:
            # 격리 해제 → 전체 모델 복원
            self.scene.display_model()

    def _append_log(self, line: str) -> None:
        self.log_console.appendPlainText(line)

    def _confirm_discard(self) -> bool:
        answer = QMessageBox.question(
            self, "ASCENT-Load", "편집 중인 내용이 저장되지 않았습니다. 버릴까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def _update_title(self, *_args) -> None:
        name = Path(self.current_path).name if self.current_path else "제목 없음"
        dirty = " *" if self.editor.is_dirty() else ""
        self.setWindowTitle(f"ASCENT-Load — {name}{dirty}")

    def closeEvent(self, event) -> None:
        if self.editor.is_dirty() and not self._confirm_discard():
            event.ignore()
            return
        if self.run_panel.is_running():
            self.run_panel.stop_run()
        self.scene.close()
        super().closeEvent(event)
