# 공력 패널 모델링 탭 — CAERO1 편집·미러·3D 미리보기, 에어포일 지정→W2GJ 생성, 스플라인 도우미
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

logger = logging.getLogger("ascent_load.gui")


def _dspin(lo, hi, val=0.0, dec=2, step=10.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    return s


class PanelEditorPanel(QWidget):
    """공력 패널(CAERO1) 저작 탭.

    카드 생성물은 BDF 에디터에 삽입되고, 저장→재파싱을 거쳐 모델에
    반영된다(기존 편집 파이프라인 재사용). 에어포일 캠버는 표준 DMI
    W2GJ로 출력되므로 같은 덱을 MSC Nastran에서도 그대로 쓸 수 있다.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = None
        self._bdf_path: Optional[str] = None
        # main_window가 연결: 벌크 텍스트 삽입 콜백, 3D 씬 컨트롤러
        self.insert_bdf_text: Optional[Callable[[str], None]] = None
        self.scene = None
        # CAERO별 에어포일 지정 (eid -> (root_str, tip_str|""))
        self._airfoils: Dict[int, Tuple[str, str]] = {}

        root = QHBoxLayout(self)

        # ---------- 좌: CAERO1 편집기 ----------
        geo_box = QGroupBox("CAERO1 패널 정의")
        geo = QVBoxLayout(geo_box)
        self._list = QListWidget()
        self._list.currentTextChanged.connect(self._load_selected)
        geo.addWidget(self._list, 1)

        form = QFormLayout()
        self.f_eid = QSpinBox(); self.f_eid.setRange(1, 99_999_999)
        self.f_pid = QSpinBox(); self.f_pid.setRange(1, 99_999_999)
        self.f_nspan = QSpinBox(); self.f_nspan.setRange(1, 200); self.f_nspan.setValue(4)
        self.f_nchord = QSpinBox(); self.f_nchord.setRange(1, 50); self.f_nchord.setValue(4)
        form.addRow("EID", self.f_eid)
        form.addRow("PAERO PID", self.f_pid)
        form.addRow("NSPAN", self.f_nspan)
        form.addRow("NCHORD", self.f_nchord)
        self.f_p1 = [_dspin(-1e6, 1e6) for _ in range(3)]
        self.f_p4 = [_dspin(-1e6, 1e6) for _ in range(3)]
        self.f_c1 = _dspin(0.0, 1e6, 1000.0)
        self.f_c4 = _dspin(0.0, 1e6, 1000.0)
        for label, w3 in (("P1 내측 LE (x,y,z)", self.f_p1),
                          ("P4 외측 LE (x,y,z)", self.f_p4)):
            row = QHBoxLayout()
            for w in w3:
                row.addWidget(w)
            holder = QWidget(); holder.setLayout(row)
            form.addRow(label, holder)
        form.addRow("CHORD1 내측 시위", self.f_c1)
        form.addRow("CHORD4 외측 시위", self.f_c4)
        geo.addLayout(form)

        btns = QHBoxLayout()
        b_prev = QPushButton("3D 미리보기")
        b_prev.clicked.connect(self.preview_panel)
        b_mirror = QPushButton("미러 정의")
        b_mirror.clicked.connect(self.make_mirror)
        b_insert = QPushButton("CAERO1 카드 삽입")
        b_insert.clicked.connect(self.insert_caero)
        for b in (b_prev, b_mirror, b_insert):
            btns.addWidget(b)
        geo.addLayout(btns)
        b_pload = QPushButton("PLOAD4 대상 요소 3D 표시")
        b_pload.setToolTip(
            "PLOAD4 압력 매핑이 칠하게 될 외피 요소를 3D 뷰에 표시합니다 "
            "(매핑은 기하로만 결정 — 해석 결과 불필요). 초록 = 매핑 요소, "
            "빨강 윤곽 = 아래에 외피가 없어 미커버되는 박스(그 몫은 FORCE "
            "덱에 보존)")
        b_pload.clicked.connect(self.preview_pload4)
        geo.addWidget(b_pload)
        self.pload_info = QLabel("")
        self.pload_info.setWordWrap(True)
        geo.addWidget(self.pload_info)
        root.addWidget(geo_box, 4)
        # main_window가 연결: 3D View 탭으로 전환
        self.show_3d_view: Optional[Callable[[], None]] = None

        # ---------- 중: 에어포일 → W2GJ ----------
        af_box = QGroupBox("에어포일 캠버 → W2GJ DMI")
        af = QVBoxLayout(af_box)
        af_form = QFormLayout()
        self.af_caero = QComboBox()
        self.af_root = QLineEdit("NACA2412")
        self.af_tip = QLineEdit("")
        self.af_tip.setPlaceholderText("(비우면 루트와 동일)")
        af_form.addRow("대상 CAERO", self.af_caero)
        af_form.addRow("루트 에어포일", self.af_root)
        af_form.addRow("팁 에어포일", self.af_tip)
        af.addLayout(af_form)
        af_btns = QHBoxLayout()
        b_af_add = QPushButton("지정 추가/갱신")
        b_af_add.clicked.connect(self.add_airfoil)
        b_af_plot = QPushButton("캠버선 미리보기")
        b_af_plot.clicked.connect(self.plot_camber)
        af_btns.addWidget(b_af_add)
        af_btns.addWidget(b_af_plot)
        af.addLayout(af_btns)

        self.af_table = QTableWidget(0, 3)
        self.af_table.setHorizontalHeaderLabels(["CAERO", "루트", "팁"])
        self.af_table.setEditTriggers(QTableWidget.NoEditTriggers)
        af.addWidget(self.af_table, 1)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(4, 2.2))
        self._canvas = FigureCanvasQTAgg(self._fig)
        af.addWidget(self._canvas, 1)

        b_w2gj = QPushButton("W2GJ DMI 생성 → 에디터 삽입")
        b_w2gj.clicked.connect(self.generate_w2gj)
        af.addWidget(b_w2gj)
        root.addWidget(af_box, 4)

        # ---------- 우: 스플라인 도우미 ----------
        sp_box = QGroupBox("스플라인 도우미 (SET1 + SPLINE1)")
        sp = QVBoxLayout(sp_box)
        sp_form = QFormLayout()
        self.sp_caero = QComboBox()
        self.sp_tol = _dspin(0.0, 1e5, 0.0, dec=1)
        self.sp_tol.setToolTip("면외 거리 허용치 (mm). 0 = 평균 시위의 25%")
        self.sp_margin = _dspin(0.0, 100.0, 10.0, dec=0, step=5.0)
        self.sp_max = QSpinBox(); self.sp_max.setRange(4, 500); self.sp_max.setValue(60)
        sp_form.addRow("대상 CAERO", self.sp_caero)
        sp_form.addRow("면외 허용치 (mm, 0=자동)", self.sp_tol)
        sp_form.addRow("외곽 여유 (%)", self.sp_margin)
        sp_form.addRow("최대 절점 수", self.sp_max)
        from qtpy.QtWidgets import QCheckBox

        self.sp_hard = QCheckBox("스파·리브 절점 우선 (하드포인트)")
        self.sp_hard.setChecked(True)
        self.sp_hard.setToolTip(
            "보 요소가 물린 절점, 쉘 법선이 꺾이는 접합선(스파·리브 웹), "
            "고연결도 절점만 후보로 사용 — 국소 외피 변형 잡음을 차단하고 "
            "하중을 주 하중 경로에 인가합니다. 연속 곡면(붐 튜브 등)의 "
            "꺾임은 접합선으로 치지 않습니다. 스팬 스테이션별 최전방/최후방 "
            "절점을 지켜 비틀림 정의(시위 2열)를 보장합니다")
        sp_form.addRow("", self.sp_hard)
        sp.addLayout(sp_form)
        # PID(속성) 필터 — 체크된 속성의 요소에 물린 절점만 후보
        sp.addWidget(QLabel("대상 속성(PID) 필터 — 체크 없으면 전체"))
        self.sp_pids = QListWidget()
        self.sp_pids.setMaximumHeight(120)
        self.sp_pids.setToolTip(
            "스파 웹·리브 등 원하는 속성만 체크하면 그 요소들에 물린 "
            "절점만 후보가 됩니다 (휴리스틱보다 우선하는 확실한 방법)")
        sp.addWidget(self.sp_pids)
        b_suggest = QPushButton("절점 제안 + 3D 하이라이트")
        b_suggest.clicked.connect(self.suggest_nodes)
        sp.addWidget(b_suggest)
        self.sp_result = QLabel("—")
        self.sp_result.setWordWrap(True)
        sp.addWidget(self.sp_result, 1)
        b_sp_insert = QPushButton("SET1 + SPLINE1 카드 삽입")
        b_sp_insert.clicked.connect(self.insert_spline)
        sp.addWidget(b_sp_insert)
        root.addWidget(sp_box, 3)

        self._suggested: list = []
        self.setEnabled(False)

    # ------------------------------------------------------------------
    def set_model(self, model, bdf_path: Optional[str]) -> None:
        self._model = model
        self._bdf_path = bdf_path
        editable = model is not None and bdf_path is not None
        self.setEnabled(bool(editable))
        self._list.blockSignals(True)
        self._list.clear()
        self.af_caero.clear()
        self.sp_caero.clear()
        self._populate_pid_list(model)
        if model is not None:
            for eid in sorted(getattr(model, "caero_panels", {}) or {}):
                c = model.caero_panels[eid]
                self._list.addItem(
                    f"{eid}  ({c.nspan}x{c.nchord})  "
                    f"y {c.p1[1]:.0f}→{c.p4[1]:.0f}")
                self.af_caero.addItem(str(eid), eid)
                self.sp_caero.addItem(str(eid), eid)
            # 새 EID 기본값 제안
            eids = sorted(getattr(model, "caero_panels", {}) or {})
            self.f_eid.setValue((eids[-1] + 1000) if eids else 1001)
            splines = getattr(model, "splines", {}) or {}
            sets = getattr(model, "sets", {}) or {}
            self._next_spline = max(splines, default=500) + 1
            self._next_set = max(sets, default=70) + 1
        self._list.blockSignals(False)
        self._refresh_airfoil_table()

    def _populate_pid_list(self, model) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QListWidgetItem

        self.sp_pids.clear()
        if model is None:
            return
        counts: Dict[int, int] = {}
        for e in (getattr(model, "elements", {}) or {}).values():
            pid = getattr(e, "pid", None)
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        props = getattr(model, "properties", {}) or {}
        for pid in sorted(counts):
            ptype = getattr(props.get(pid), "type", "?")
            item = QListWidgetItem(
                f"PID {pid}  {ptype}  ({counts[pid]} elems)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, pid)
            self.sp_pids.addItem(item)

    def _checked_pids(self) -> set:
        from qtpy.QtCore import Qt

        pids = set()
        for i in range(self.sp_pids.count()):
            item = self.sp_pids.item(i)
            if item.checkState() == Qt.Checked:
                pids.add(item.data(Qt.UserRole))
        return pids

    def _load_selected(self, text: str) -> None:
        if not text or self._model is None:
            return
        eid = int(text.split()[0])
        c = self._model.caero_panels.get(eid)
        if c is None:
            return
        self.f_eid.setValue(c.eid)
        self.f_pid.setValue(max(c.pid, 1))
        self.f_nspan.setValue(max(c.nspan, 1))
        self.f_nchord.setValue(max(c.nchord, 1))
        for w, v in zip(self.f_p1, c.p1):
            w.setValue(float(v))
        for w, v in zip(self.f_p4, c.p4):
            w.setValue(float(v))
        self.f_c1.setValue(float(c.chord1))
        self.f_c4.setValue(float(c.chord4))

    # ------------------------------------------------------------------
    # CAERO 편집
    # ------------------------------------------------------------------
    def _form_geometry(self):
        p1 = np.array([w.value() for w in self.f_p1])
        p4 = np.array([w.value() for w in self.f_p4])
        return p1, self.f_c1.value(), p4, self.f_c4.value()

    def preview_panel(self) -> None:
        """현재 폼의 패널 외곽을 3D 뷰에 반투명 오버레이로 표시."""
        if self.scene is None:
            return
        p1, c1, p4, c4 = self._form_geometry()
        stream = np.array([1.0, 0.0, 0.0])
        pts = np.array([p1, p1 + c1 * stream, p4 + c4 * stream, p4])
        try:
            import pyvista as pv

            quad = pv.PolyData(pts, faces=[4, 0, 1, 2, 3])
            self.scene.plotter.add_mesh(
                quad, name="panel_preview", color="#00acc1", opacity=0.45,
                show_edges=True, line_width=2)
            self.scene.plotter.render()
        except Exception:
            logger.exception("Panel preview failed")

    def preview_pload4(self) -> None:
        """PLOAD4 매핑 대상 외피 요소·미커버 박스를 3D 뷰에 표시."""
        if self._model is None or self.scene is None:
            return
        from ..aero.panel import generate_all_panels
        from ..loads_analysis.pload_export import map_box_forces_to_skin

        try:
            boxes = generate_all_panels(self._model, use_nastran_eid=True)
            if not boxes:
                QMessageBox.information(self, "ASCENT-Load",
                                        "CAERO 패널이 없습니다")
                return
            # 매핑 대상은 기하로만 결정되므로 단위 법선력으로 충분
            unit_F = np.array([b.normal for b in boxes])
            pressures, rep = map_box_forces_to_skin(
                self._model, boxes, unit_F)
        except Exception as exc:
            logger.exception("PLOAD4 preview failed")
            QMessageBox.critical(self, "ASCENT-Load",
                                 f"PLOAD4 미리보기 실패:\n{exc}")
            return

        try:
            import pyvista as pv

            from .scene import build_element_overlay

            # 매핑 요소 오버레이 (초록)
            poly = build_element_overlay(self._model,
                                         sorted(pressures.keys()))
            if poly is not None:
                self.scene.plotter.add_mesh(
                    poly, color="#2e7d32", opacity=0.8, show_edges=True,
                    edge_color="darkgreen", name="pload_preview")
            # 미커버 박스 윤곽 (빨강)
            unc = set(rep.get("uncovered", []))
            if unc:
                pts, faces = [], []
                for b in boxes:
                    if b.box_id not in unc:
                        continue
                    k = len(pts)
                    pts.extend(b.corners)
                    faces.extend([4, k, k + 1, k + 2, k + 3])
                quads = pv.PolyData(np.array(pts), faces=faces)
                self.scene.plotter.add_mesh(
                    quads, color="red", style="wireframe", line_width=3,
                    name="pload_uncovered")
            else:
                self.scene.plotter.remove_actor("pload_uncovered")
            self.scene.plotter.render()
        except Exception:
            logger.exception("PLOAD4 preview render failed")

        self.pload_info.setText(
            f"PLOAD4 매핑: 외피 요소 {len(pressures)}개(초록), 박스 커버 "
            f"{rep['n_covered']}/{rep['n_boxes']}, 미커버 {len(rep['uncovered'])}개"
            f"(빨강 윤곽) — 3D View 탭에서 확인하세요")
        if callable(self.show_3d_view):
            self.show_3d_view()

    def make_mirror(self) -> None:
        """현재 폼 정의를 XZ면 대칭으로 뒤집는다 (EID +1000)."""
        from ..aero.panel_authoring import mirror_caero1

        p1, c1, p4, c4 = self._form_geometry()
        m1, mc1, m4, mc4 = mirror_caero1(self.f_eid.value(), p1, c1, p4, c4)
        for w, v in zip(self.f_p1, m1):
            w.setValue(float(v))
        for w, v in zip(self.f_p4, m4):
            w.setValue(float(v))
        self.f_eid.setValue(self.f_eid.value() + 1000)
        self.preview_panel()

    def insert_caero(self) -> None:
        from ..aero.panel_authoring import caero1_card_text

        if not callable(self.insert_bdf_text):
            return
        eid = self.f_eid.value()
        if self._model is not None \
                and eid in (getattr(self._model, "caero_panels", {}) or {}):
            answer = QMessageBox.question(
                self, "ASCENT-Load",
                f"CAERO {eid}이(가) 모델에 이미 있습니다. 같은 EID로 "
                "삽입하면 중복 정의가 됩니다.\n그래도 삽입할까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        p1, c1, p4, c4 = self._form_geometry()
        text = caero1_card_text(
            self.f_eid.value(), self.f_pid.value(),
            self.f_nspan.value(), self.f_nchord.value(),
            p1, c1, p4, c4)
        self.insert_bdf_text("$ CAERO1 (Panel tab)\n" + text + "\n")

    # ------------------------------------------------------------------
    # 에어포일 → W2GJ
    # ------------------------------------------------------------------
    def _refresh_airfoil_table(self) -> None:
        self.af_table.setRowCount(len(self._airfoils))
        for r, (eid, (root, tip)) in enumerate(sorted(self._airfoils.items())):
            for c, text in enumerate((str(eid), root, tip or "(루트)")):
                self.af_table.setItem(r, c, QTableWidgetItem(text))
        self.af_table.resizeColumnsToContents()

    def add_airfoil(self) -> None:
        from ..aero.airfoil_camber import AirfoilCamber

        eid = self.af_caero.currentData()
        if eid is None:
            return
        root = self.af_root.text().strip().upper()
        tip = self.af_tip.text().strip().upper()
        try:
            AirfoilCamber.from_naca_string(root)
            if tip:
                AirfoilCamber.from_naca_string(tip)
        except Exception as exc:
            QMessageBox.warning(self, "ASCENT-Load",
                                f"에어포일 명칭 해석 실패:\n{exc}")
            return
        self._airfoils[int(eid)] = (root, tip)
        self._refresh_airfoil_table()
        self.plot_camber()

    def plot_camber(self) -> None:
        from ..aero.airfoil_camber import AirfoilCamber

        name = self.af_root.text().strip().upper()
        try:
            af = AirfoilCamber.from_naca_string(name)
        except Exception:
            return
        x = np.linspace(0.0, 1.0, 101)
        slope = np.array([af.camber_slope(xi) for xi in x])
        z = np.concatenate([[0.0], np.cumsum(
            0.5 * (slope[1:] + slope[:-1]) * np.diff(x))])
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.plot(x, z, color="#1565c0")
        ax.set_title(f"{name} camber line (z/c)", fontsize=9)
        ax.set_xlabel("x/c", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _airfoil_config(self):
        from ..aero.airfoil_camber import AirfoilCamber, PanelAirfoilConfig

        pa = {}
        for eid, (root, tip) in self._airfoils.items():
            root_af = AirfoilCamber.from_naca_string(root)
            tip_af = AirfoilCamber.from_naca_string(tip) if tip else None
            pa[eid] = (root_af, tip_af)
        return PanelAirfoilConfig(panel_airfoils=pa)

    def generate_w2gj(self) -> None:
        """지정된 에어포일 캠버를 DMI W2GJ 덱으로 에디터에 삽입."""
        from ..aero.panel import generate_all_panels
        from ..aero.panel_authoring import w2gj_dmi_text

        if self._model is None or not callable(self.insert_bdf_text):
            return
        if not self._airfoils:
            QMessageBox.information(self, "ASCENT-Load",
                                    "먼저 에어포일 지정을 추가하세요")
            return
        if "W2GJ" in (getattr(self._model, "dmis", {}) or {}):
            QMessageBox.warning(
                self, "ASCENT-Load",
                "모델에 이미 W2GJ DMI가 있습니다. 기존 카드를 지운 뒤 "
                "삽입하세요 (중복 시 뒤에 오는 헤더가 행렬을 재생성합니다)")
        try:
            boxes = generate_all_panels(self._model, use_nastran_eid=True)
            text = w2gj_dmi_text(boxes, self._model.caero_panels,
                                 self._airfoil_config())
        except Exception as exc:
            logger.exception("W2GJ generation failed")
            QMessageBox.critical(self, "ASCENT-Load", f"W2GJ 생성 실패:\n{exc}")
            return
        self.insert_bdf_text(text)

    # ------------------------------------------------------------------
    # 스플라인 도우미
    # ------------------------------------------------------------------
    def suggest_nodes(self) -> None:
        from ..aero.panel_authoring import suggest_spline_nodes

        eid = self.sp_caero.currentData()
        if self._model is None or eid is None:
            return
        caero = self._model.caero_panels[int(eid)]
        self._suggested = suggest_spline_nodes(
            self._model, caero,
            offset_tol=self.sp_tol.value(),
            margin_frac=self.sp_margin.value() / 100.0,
            max_nodes=self.sp_max.value(),
            prefer_hard_points=self.sp_hard.isChecked(),
            pids=self._checked_pids() or None)
        self.sp_result.setText(
            f"CAERO {eid}: 절점 {len(self._suggested)}개 제안 — "
            + ", ".join(str(n) for n in self._suggested[:12])
            + (" …" if len(self._suggested) > 12 else ""))
        # 3D 하이라이트 (점 구름)
        if self.scene is not None and self._suggested:
            try:
                import pyvista as pv

                pts = np.array([
                    getattr(self._model.nodes[n], "xyz_global",
                            self._model.nodes[n].xyz)
                    for n in self._suggested])
                cloud = pv.PolyData(pts)
                self.scene.plotter.add_mesh(
                    cloud, name="spline_preview", color="#e91e63",
                    point_size=12, render_points_as_spheres=True)
                self.scene.plotter.render()
            except Exception:
                logger.exception("Spline node highlight failed")

    def insert_spline(self) -> None:
        from ..aero.panel_authoring import set1_card_text, spline1_card_text

        eid = self.sp_caero.currentData()
        if not self._suggested or eid is None \
                or not callable(self.insert_bdf_text):
            QMessageBox.information(self, "ASCENT-Load",
                                    "먼저 [절점 제안]을 실행하세요")
            return
        caero = self._model.caero_panels[int(eid)]
        n_boxes = max(caero.nspan, 1) * max(caero.nchord, 1)
        sid = self._next_set
        spline_eid = self._next_spline
        text = ("$ SPLINE helper (Panel tab)\n"
                + set1_card_text(sid, self._suggested) + "\n"
                + spline1_card_text(spline_eid, int(eid), int(eid),
                                    int(eid) + n_boxes - 1, sid) + "\n")
        self._next_set += 1
        self._next_spline += 1
        self.insert_bdf_text(text)
