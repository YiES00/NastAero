# 기체·운용조건·하중해석 정보 탭 — AircraftConfig 폼 편집, YAML 열기/저장, BDF 자동 추정
from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("nastaero.gui")


def resolve_bdf_ref(yaml_path: str, ref: str) -> Optional[str]:
    """YAML의 bdf_model 참조를 절대경로로 해석한다 (YAML 위치 기준 상대).

    존재하지 않으면 None.
    """
    from pathlib import Path

    if not ref:
        return None
    p = Path(ref)
    if not p.is_absolute():
        p = Path(yaml_path).parent / p
    return str(p.resolve()) if p.exists() else None


def _spin(minimum=0.0, maximum=1e9, decimals=3, value=0.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimum, maximum)
    s.setDecimals(decimals)
    s.setValue(value)
    # 숫자 필드 폭·정렬 통일 — 패널 폭에 따라 제각각 늘어나지 않게
    s.setFixedWidth(130)
    s.setAlignment(Qt.AlignRight)
    return s


def _form(box: QGroupBox) -> QFormLayout:
    """그룹박스 공통 폼 — 라벨 우측정렬, 간격 통일."""
    f = QFormLayout(box)
    f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    f.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    f.setHorizontalSpacing(12)
    f.setVerticalSpacing(6)
    return f


class AircraftInfoPanel(QWidget):
    """하중해석 대상 기체·운용조건·하중해석 축 정보. get_config()가 단일 진실 소스.

    폼에 없는 YAML 키(landing_gear, vtol 등)는 _extra로 보존해 저장 시 병합한다.
    """

    config_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = None
        self._bdf_path: Optional[str] = None
        self._extra: dict = {}   # 폼 미표현 키 보존 (landing_gear, vtol …)
        self.open_bdf_requested = None   # main window가 주입 (경로 → 열기)

        # --- 상단 버튼 ---
        buttons = QHBoxLayout()
        auto_btn = QPushButton("BDF에서 자동 추정")
        auto_btn.clicked.connect(self._auto_from_model)
        open_btn = QPushButton("기체 설정 열기…")
        open_btn.setToolTip("기체 제원·운용조건 YAML을 열고, 참조된 BDF 모델도 함께 엽니다")
        open_btn.clicked.connect(self._open_yaml)
        save_btn = QPushButton("기체 설정 저장…")
        save_btn.setToolTip("현재 폼을 기체 설정 YAML로 저장합니다")
        save_btn.clicked.connect(self._save_yaml)
        for b in (auto_btn, open_btn, save_btn):
            buttons.addWidget(b)
        buttons.addStretch()

        # --- 기체 일반 ---
        general = QGroupBox("기체")
        gform = _form(general)
        self.name = QLineEdit()
        self.name.setMinimumWidth(200)
        gform.addRow("이름", self.name)
        # YAML의 bdf_model 참조 / 실제 열린 해석 모델 표시 (읽기 전용)
        self.bdf_label = QLabel("—")
        self.bdf_label.setStyleSheet("color: gray;")
        self.bdf_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gform.addRow("해석 모델 (BDF)", self.bdf_label)
        self.wing_area = _spin(0, 1e4, 3)
        gform.addRow("날개 면적 S (m²)", self.wing_area)
        self.mean_chord = _spin(0, 100, 3)
        gform.addRow("평균 시위 MAC (m)", self.mean_chord)
        self.clalpha = _spin(0, 20, 3, 6.283)
        gform.addRow("양력선 기울기 CLα (/rad)", self.clalpha)

        # --- 설계 속도 (EAS m/s) ---
        speeds = QGroupBox("설계 속도 스케줄 (EAS, m/s) — §23.333-335")
        sform = _form(speeds)
        self.v_fields = {}
        for key, label in (("VS1", "VS1 실속"), ("VA", "VA 기동"),
                           ("VB", "VB 돌풍(선택)"), ("VC", "VC 순항"),
                           ("VD", "VD 급강하"), ("VF", "VF 플랩(선택)")):
            self.v_fields[key] = _spin(0, 500, 2)
            sform.addRow(label, self.v_fields[key])

        # --- 운용 조건 ---
        ops = QGroupBox("운용 조건")
        oform = _form(ops)
        self.altitudes = QLineEdit("0")
        self.altitudes.setToolTip("해석 고도 목록 (m), 쉼표 구분")
        self.altitudes.setFixedWidth(130)
        self.altitudes.setAlignment(Qt.AlignRight)
        oform.addRow("고도 (m, 쉼표)", self.altitudes)
        self.nz_max = _spin(0, 10, 2, 0.0)
        self.nz_max.setToolTip("0 = §23.337 공식 자동")
        oform.addRow("nz_max 지정 (0=자동)", self.nz_max)
        self.gust_vc = _spin(0, 100, 1, 50.0)
        oform.addRow("돌풍 Ude @VC (ft/s)", self.gust_vc)
        self.gust_vd = _spin(0, 100, 1, 25.0)
        oform.addRow("돌풍 Ude @VD (ft/s)", self.gust_vd)

        # --- 하중해석 설정 ---
        loads = QGroupBox("하중해석 설정")
        lform = _form(loads)
        self.elastic_axis = _spin(0, 1, 2, 0.40)
        self.elastic_axis.setToolTip("V-M-T 비틀림 기준축 (앞전 기준 시위 분율)")
        lform.addRow("탄성축 위치 (x/c)", self.elastic_axis)
        self.ail_max = _spin(0, 60, 1, 20.0)
        lform.addRow("에일러론 한계 (deg)", self.ail_max)
        self.elev_max = _spin(0, 60, 1, 25.0)
        lform.addRow("승강타 한계 (deg)", self.elev_max)
        self.rud_max = _spin(0, 60, 1, 25.0)
        lform.addRow("방향타 한계 (deg)", self.rud_max)

        # --- 중량/CG 조건 ---
        weights = QGroupBox("중량/CG 조건")
        wlayout = QVBoxLayout(weights)
        wlayout.setSpacing(6)
        self.wc_table = QTableWidget(0, 3)
        self.wc_table.setHorizontalHeaderLabels(
            ["라벨", "중량 (N)", "CG x (모델 좌표)"])
        self.wc_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.wc_table.verticalHeader().setVisible(False)
        self.wc_table.setAlternatingRowColors(True)
        self.wc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.wc_table.setMinimumHeight(110)
        wlayout.addWidget(self.wc_table)
        wbtns = QHBoxLayout()
        add_btn = QPushButton("행 추가")
        add_btn.clicked.connect(lambda: self._add_weight_row())
        del_btn = QPushButton("선택 삭제")
        del_btn.clicked.connect(self._remove_weight_row)
        wbtns.addWidget(add_btn)
        wbtns.addWidget(del_btn)
        wbtns.addStretch()
        wlayout.addLayout(wbtns)

        # --- 착륙장치 (§23.473 하중배수 입력) ---
        gear_box = QGroupBox("착륙장치 — §23.473 착륙 하중배수 (FAR 23 LOADS Manual Ch.20)")
        ghl = QHBoxLayout(gear_box)
        gleft = QFormLayout()
        gleft.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gleft.setHorizontalSpacing(12)
        gright = QFormLayout()
        gright.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gright.setHorizontalSpacing(12)

        self.lg_main_nodes = QLineEdit()
        self.lg_main_nodes.setFixedWidth(130)
        self.lg_main_nodes.setToolTip("메인 기어 부착 절점 ID (쉼표, 좌/우)")
        gleft.addRow("메인 기어 절점", self.lg_main_nodes)
        self.lg_nose_nodes = QLineEdit()
        self.lg_nose_nodes.setFixedWidth(130)
        gleft.addRow("노즈 기어 절점", self.lg_nose_nodes)
        self.lg_main_x = _spin(0, 1e6, 1)
        gleft.addRow("메인 기어 x (mm)", self.lg_main_x)
        self.lg_nose_x = _spin(0, 1e6, 1)
        gleft.addRow("노즈 기어 x (mm)", self.lg_nose_x)

        self.lg_strut_eff = _spin(0, 1, 2, 0.75)
        self.lg_strut_eff.setToolTip("스트럿 흡수 효율 η_s — 오레오 0.75 / 스프링 0.5")
        gright.addRow("스트럿 효율 η_s", self.lg_strut_eff)
        self.lg_stroke = _spin(0, 2, 3, 0.25)
        gright.addRow("스트로크 (m)", self.lg_stroke)
        self.lg_sink = _spin(0, 10, 1, 10.0)
        self.lg_sink.setToolTip("0 = §23.473(d) 규정식 V=4.4(W/S)^¼ 자동 (7~10 ft/s 클램프)")
        gright.addRow("침하율 (ft/s, 0=자동)", self.lg_sink)
        self.lg_tire_defl = _spin(0, 0.5, 3, 0.0)
        self.lg_tire_defl.setToolTip("타이어 압착 처짐 (m) — LGFACTOR: (외경−허브경)/6")
        gright.addRow("타이어 처짐 (m)", self.lg_tire_defl)
        self.lg_tire_eff = _spin(0, 1, 2, 0.3)
        gright.addRow("타이어 효율 η_t", self.lg_tire_eff)

        ghl.addLayout(gleft)
        ghl.addSpacing(24)
        ghl.addLayout(gright)
        ghl.addStretch()
        # 하중배수 미리보기 (LGFACTOR 에너지법 즉시 계산)
        self.lg_preview = QLabel("—")
        self.lg_preview.setStyleSheet("color: #205080; font-weight: bold;")
        self.lg_preview.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ghl.addWidget(self.lg_preview)

        # --- 기준 정보 (읽기 전용 안내) ---
        info = QLabel(
            "모멘트 기준점: 질량 CG (compute_node_masses 자동)   ·   "
            "단위계: N-mm-sec (모델), 속도/면적은 SI (m)")
        info.setStyleSheet("color: gray;")
        info.setAlignment(Qt.AlignLeft)

        # --- 배치: 2열 그리드, 각 셀 상단 정렬(높이 차이 나도 줄 맞음) ---
        content = QWidget()
        grid = QVBoxLayout(content)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(10)
        grid.addLayout(buttons)
        panels = QGridLayout()
        panels.setSpacing(10)
        panels.addWidget(general, 0, 0, Qt.AlignTop)
        panels.addWidget(speeds, 0, 1, 2, 1, Qt.AlignTop)
        panels.addWidget(ops, 1, 0, Qt.AlignTop)
        panels.addWidget(loads, 2, 0, Qt.AlignTop)
        panels.addWidget(weights, 2, 1)
        panels.addWidget(gear_box, 3, 0, 1, 2)
        panels.setColumnStretch(0, 1)
        panels.setColumnStretch(1, 1)
        grid.addLayout(panels)
        grid.addWidget(info)
        grid.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        for w in [self.wing_area, self.mean_chord, self.clalpha,
                  self.nz_max, self.gust_vc, self.gust_vd,
                  self.lg_main_x, self.lg_nose_x, self.lg_strut_eff,
                  self.lg_stroke, self.lg_sink, self.lg_tire_defl,
                  self.lg_tire_eff,
                  *self.v_fields.values()]:
            w.valueChanged.connect(lambda *_: self.config_changed.emit())
        self.altitudes.editingFinished.connect(self.config_changed.emit)
        for w in (self.lg_main_nodes, self.lg_nose_nodes):
            w.editingFinished.connect(self.config_changed.emit)
        self.wc_table.itemChanged.connect(
            lambda *_: self.config_changed.emit())
        self.config_changed.connect(self._update_landing_preview)

    # ------------------------------------------------------------------
    def set_model(self, model, bdf_path: Optional[str] = None) -> None:
        self._model = model
        self._bdf_path = bdf_path
        if bdf_path:
            from pathlib import Path

            self.bdf_label.setText(Path(bdf_path).name)
            self.bdf_label.setToolTip(str(bdf_path))

    def _add_weight_row(self, label="", weight="", cg_x="") -> None:
        r = self.wc_table.rowCount()
        self.wc_table.insertRow(r)
        for c, v in enumerate((label, str(weight), str(cg_x))):
            self.wc_table.setItem(r, c, QTableWidgetItem(v))

    @staticmethod
    def _parse_ids(text: str) -> list:
        out = []
        for tok in text.replace(",", " ").split():
            try:
                out.append(int(tok))
            except ValueError:
                continue
        return out

    def _gear_config(self):
        from ..loads_analysis.certification.aircraft_config import (
            LandingGearConfig,
        )

        return LandingGearConfig(
            main_gear_node_ids=self._parse_ids(self.lg_main_nodes.text()),
            nose_gear_node_ids=self._parse_ids(self.lg_nose_nodes.text()),
            main_gear_x=self.lg_main_x.value(),
            nose_gear_x=self.lg_nose_x.value(),
            strut_efficiency=self.lg_strut_eff.value(),
            stroke=self.lg_stroke.value(),
            sink_rate_fps=self.lg_sink.value(),
            tire_deflection=self.lg_tire_defl.value(),
            tire_efficiency=self.lg_tire_eff.value(),
        )

    def _update_landing_preview(self) -> None:
        """§23.473 하중배수 즉시 계산 표시 (LGFACTOR 에너지법)."""
        try:
            gear = self._gear_config()
            w = 0.0
            if self.wc_table.rowCount() > 0 and self.wc_table.item(0, 1):
                w = float(self.wc_table.item(0, 1).text() or 0)
            n, n_gear, v = gear.landing_load_factors(
                w, self.wing_area.value())
            self.lg_preview.setText(
                f"착륙 하중배수\nV = {v:.1f} ft/s\n"
                f"N = {n:.2f}  ·  N_gear = {n_gear:.2f}")
        except Exception:
            self.lg_preview.setText("—")

    def _remove_weight_row(self) -> None:
        rows = sorted({i.row() for i in self.wc_table.selectedItems()},
                      reverse=True)
        for r in rows:
            self.wc_table.removeRow(r)
        self.config_changed.emit()

    # ------------------------------------------------------------------
    # dict ↔ 폼
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(self._extra)
        d["name"] = self.name.text()
        d["speeds"] = {k: w.value() for k, w in self.v_fields.items()}
        d["wing_area_m2"] = self.wing_area.value()
        d["mean_chord_m"] = self.mean_chord.value()
        d["CLalpha"] = self.clalpha.value()
        d["gust_Ude_VC_fps"] = self.gust_vc.value()
        d["gust_Ude_VD_fps"] = self.gust_vd.value()
        if self.nz_max.value() > 0:
            d["nz_max"] = self.nz_max.value()
        else:
            d.pop("nz_max", None)
        try:
            d["altitudes_m"] = [float(x) for x in
                                self.altitudes.text().split(",") if x.strip()]
        except ValueError:
            d["altitudes_m"] = [0.0]
        wcs = []
        for r in range(self.wc_table.rowCount()):
            try:
                wcs.append({
                    "label": self.wc_table.item(r, 0).text(),
                    "weight_N": float(self.wc_table.item(r, 1).text()),
                    "cg_x": float(self.wc_table.item(r, 2).text() or 0),
                })
            except (AttributeError, ValueError):
                continue
        d["weight_cg"] = wcs
        d["ctrl_limits"] = {
            "aileron_max_deg": self.ail_max.value(),
            "elevator_max_deg": self.elev_max.value(),
            "rudder_max_deg": self.rud_max.value(),
        }
        d["elastic_axis_frac"] = self.elastic_axis.value()
        d["landing_gear"] = {
            "main_gear_node_ids": self._parse_ids(self.lg_main_nodes.text()),
            "nose_gear_node_ids": self._parse_ids(self.lg_nose_nodes.text()),
            "main_gear_x": self.lg_main_x.value(),
            "nose_gear_x": self.lg_nose_x.value(),
            "strut_efficiency": self.lg_strut_eff.value(),
            "stroke": self.lg_stroke.value(),
            "sink_rate_fps": self.lg_sink.value(),
            "tire_deflection": self.lg_tire_defl.value(),
            "tire_efficiency": self.lg_tire_eff.value(),
        }
        return d

    def load_dict(self, d: dict) -> None:
        known = {"name", "speeds", "wing_area_m2", "mean_chord_m", "CLalpha",
                 "gust_Ude_VC_fps", "gust_Ude_VD_fps", "nz_max",
                 "altitudes_m", "weight_cg", "ctrl_limits",
                 "elastic_axis_frac", "landing_gear"}
        self._extra = {k: v for k, v in d.items() if k not in known}
        # YAML의 bdf_model 참조 표시 — 실제 모델이 열리면 set_model이
        # 실측 경로(파일명 + 툴팁 절대경로)로 덮어쓴다
        ref = str(d.get("bdf_model", "") or "")
        if ref:
            from pathlib import Path

            self.bdf_label.setText(Path(ref).name)
            self.bdf_label.setToolTip(f"YAML 참조: {ref}")
        self.name.setText(str(d.get("name", "")))
        for k, w in self.v_fields.items():
            w.setValue(float(d.get("speeds", {}).get(k, 0.0)))
        self.wing_area.setValue(float(d.get("wing_area_m2", 0.0)))
        self.mean_chord.setValue(float(d.get("mean_chord_m", 0.0)))
        self.clalpha.setValue(float(d.get("CLalpha", 6.283)))
        self.gust_vc.setValue(float(d.get("gust_Ude_VC_fps", 50.0)))
        self.gust_vd.setValue(float(d.get("gust_Ude_VD_fps", 25.0)))
        self.nz_max.setValue(float(d.get("nz_max", 0.0)))
        self.altitudes.setText(", ".join(
            str(a) for a in d.get("altitudes_m", [0.0])))
        self.wc_table.setRowCount(0)
        for wc in d.get("weight_cg", []):
            self._add_weight_row(wc.get("label", ""),
                                 wc.get("weight_N", ""),
                                 wc.get("cg_x", ""))
        cl = d.get("ctrl_limits", {})
        self.ail_max.setValue(float(cl.get("aileron_max_deg", 20.0)))
        self.elev_max.setValue(float(cl.get("elevator_max_deg", 25.0)))
        self.rud_max.setValue(float(cl.get("rudder_max_deg", 25.0)))
        self.elastic_axis.setValue(float(d.get("elastic_axis_frac", 0.40)))
        lg = d.get("landing_gear", {})
        self.lg_main_nodes.setText(
            ", ".join(str(n) for n in lg.get("main_gear_node_ids", [])))
        self.lg_nose_nodes.setText(
            ", ".join(str(n) for n in lg.get("nose_gear_node_ids", [])))
        self.lg_main_x.setValue(float(lg.get("main_gear_x", 0.0)))
        self.lg_nose_x.setValue(float(lg.get("nose_gear_x", 0.0)))
        self.lg_strut_eff.setValue(float(lg.get("strut_efficiency", 0.75)))
        self.lg_stroke.setValue(float(lg.get("stroke", 0.25)))
        self.lg_sink.setValue(float(lg.get("sink_rate_fps", 10.0)))
        self.lg_tire_defl.setValue(float(lg.get("tire_deflection", 0.0)))
        self.lg_tire_eff.setValue(float(lg.get("tire_efficiency", 0.3)))
        self.config_changed.emit()

    def get_config(self):
        """폼 → AircraftConfig (from_dict 경로 재사용)."""
        from ..loads_analysis.certification.aircraft_config import (
            AircraftConfig,
        )

        return AircraftConfig.from_dict(self.to_dict())

    # ------------------------------------------------------------------
    def _auto_from_model(self) -> None:
        if self._model is None:
            QMessageBox.information(self, "NastAero",
                                    "먼저 .bdf 모델을 여세요")
            return
        from ..loads_analysis.certification.aircraft_config import (
            AircraftConfig,
        )

        try:
            cfg = AircraftConfig.from_model_defaults(self._model)
        except Exception as exc:
            logger.exception("Auto config failed")
            QMessageBox.warning(self, "NastAero", f"자동 추정 실패:\n{exc}")
            return
        d = {
            "speeds": {k: getattr(cfg.speeds, k)
                       for k in ("VS1", "VA", "VB", "VC", "VD", "VF")},
            "wing_area_m2": cfg.wing_area_m2,
            "mean_chord_m": cfg.mean_chord_m,
            "CLalpha": cfg.CLalpha,
            "altitudes_m": cfg.altitudes_m,
            "gust_Ude_VC_fps": cfg.gust_Ude_VC_fps,
            "gust_Ude_VD_fps": cfg.gust_Ude_VD_fps,
            "weight_cg": [{"label": w.label, "weight_N": w.weight_N,
                           "cg_x": w.cg_x}
                          for w in cfg.weight_cg_conditions],
        }
        if cfg.nz_max_override:
            d["nz_max"] = cfg.nz_max_override
        self.load_dict(d)

    def _open_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "기체 설정 열기", "", "Config (*.yaml *.yml *.json)")
        if path:
            self.open_yaml_file(path)

    def open_yaml_file(self, path: str) -> None:
        """YAML/JSON 설정을 로드하고, bdf_model 참조가 있으면 BDF도 연다."""
        try:
            if path.endswith(".json"):
                import json

                with open(path) as f:
                    d = json.load(f)
            else:
                import yaml

                with open(path) as f:
                    d = yaml.safe_load(f)
            self.load_dict(d or {})
        except Exception as exc:
            QMessageBox.warning(self, "NastAero", f"열기 실패:\n{exc}")
            return

        # bdf_model 참조 → 해석 모델 자동 열기 (이미 열려 있으면 생략)
        ref = (d or {}).get("bdf_model", "")
        if not ref or not callable(self.open_bdf_requested):
            return
        bdf = resolve_bdf_ref(path, str(ref))
        if bdf is None:
            QMessageBox.warning(
                self, "NastAero",
                f"설정의 bdf_model을 찾을 수 없습니다:\n{ref}")
            return
        from pathlib import Path

        if self._bdf_path and Path(self._bdf_path).resolve() == Path(bdf):
            return  # 같은 모델이 이미 열려 있음
        logger.info("Opening BDF from config: %s", bdf)
        self.open_bdf_requested(bdf)

    def _save_yaml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "기체 설정 저장", "aircraft_config.yaml",
            "Config (*.yaml *.yml)")
        if not path:
            return
        try:
            import os
            import yaml

            d = self.to_dict()
            # 현재 열린 해석 모델을 YAML 위치 기준 상대경로로 기록
            if self._bdf_path:
                d["bdf_model"] = os.path.relpath(
                    self._bdf_path, os.path.dirname(os.path.abspath(path)))
            with open(path, "w") as f:
                yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            QMessageBox.warning(self, "NastAero", f"저장 실패:\n{exc}")
