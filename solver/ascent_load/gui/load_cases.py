# 하중해석 케이스 탭 — 서브케이스별 비행/트림 조건과 트림 결과·총 하중 6분력 합을 표로 표시
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("ascent_load.gui")

# 라디안 → 도 변환해 표시할 트림 변수
_ANGLE_LABELS = {"ANGLEA", "SIDES", "ELEV", "ARON", "AILERON", "RUD",
                 "RUDDER"}

LOAD_TYPES = ["combined", "aero", "inertial"]

# combined 잔차 허용치: 해당 케이스 aero 합 최대 성분 대비 비율
TOLERANCE_FRACTION = 0.01


def _fmt_trim_vars(variables) -> str:
    """트림 변수 (label, value) 목록을 'ANGLEA=13.2°, URDD3=9810' 형태로."""
    parts = []
    for label, val in variables:
        if label.upper() in _ANGLE_LABELS:
            parts.append(f"{label}={math.degrees(val):.2f}°")
        elif val == 0.0:
            parts.append(f"{label}=0")
        else:
            parts.append(f"{label}={val:.4g}")
    return ", ".join(parts)


def _case_accels(trim_card, rsc) -> tuple:
    """케이스의 총 가속도 문자열 ("nx, ny, nz [g]", "ṗ, q̇, ṙ [rad/s²]").

    입력(TRIM 카드 URDD2/3)과 해석 후 inertia relief 가속도
    (trim_balance의 relief_*/p_dot 등)를 합산한다. URDD3=0은 솔버
    규약대로 1g 수평비행으로 해석.
    """
    tv: Dict[str, float] = {}
    if trim_card is not None:
        tv.update({k.upper(): v for k, v in trim_card.variables})
    if rsc is not None and rsc.trim_variables:
        tv.update({k.upper(): v for k, v in rsc.trim_variables.items()})
    bal = (rsc.trim_balance or {}) if rsc is not None else {}

    urdd3 = tv.get("URDD3")
    nz = 1.0 if not urdd3 else urdd3          # 솔버 규약: 0/부재 = 1g
    nx = bal.get("relief_nx", 0.0)
    ny = tv.get("URDD2", 0.0) + bal.get("relief_ny", 0.0)
    nz += bal.get("relief_nz", 0.0)
    p = tv.get("URDD4", 0.0) + bal.get("p_dot", 0.0)
    q = tv.get("URDD5", 0.0) + bal.get("q_dot", 0.0)
    r = tv.get("URDD6", 0.0) + bal.get("r_dot", 0.0)

    def _fmt(v: float) -> str:
        return "0" if abs(v) < 1e-9 else f"{v:.3g}"   # 부동소수점 노이즈 제거

    lin = f"{_fmt(nx)}, {_fmt(ny)}, {_fmt(nz)}"
    ang = f"{_fmt(p)}, {_fmt(q)}, {_fmt(r)}"
    return lin, ang


def _force_sum(model, forces: Dict[int, np.ndarray]) -> Optional[List[float]]:
    """절점력 합산 — verify_trim_balance 재사용 (모멘트는 원점 기준)."""
    if not forces:
        return None
    from ..loads_analysis.trim_loads import verify_trim_balance

    bal = verify_trim_balance(model, forces, ref_point=np.zeros(3))
    return [bal[k] for k in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]


def summarize_cases(model, results, load_type: str = "combined") -> List[dict]:
    """서브케이스별 입력 조건 + 해석 결과 요약 행 목록 (표 표시용 순수 데이터).

    반환 행 키: subcase, selectors, mach, q, fixed, trim_result, sums,
    sums_note. sums는 [Fx,Fy,Fz,Mx,My,Mz] 또는 None.
    """
    rows: List[dict] = []

    # 입력측: BDFModel의 subcases/trims (VizModel에는 없음)
    subcase_meta: Dict[int, dict] = {}
    for sc in getattr(model, "subcases", []) or []:
        meta = {"selectors": [], "mach": None, "q": None, "fixed": "",
                "desc": getattr(sc, "label", "") or ""}
        for attr, tag in (("spc_id", "SPC"), ("load_id", "LOAD"),
                          ("method_id", "METHOD"), ("trim_id", "TRIM")):
            val = getattr(sc, attr, 0)
            if val:
                meta["selectors"].append(f"{tag}={val}")
        trim = (getattr(model, "trims", {}) or {}).get(
            getattr(sc, "trim_id", 0))
        if trim is not None:
            meta["mach"] = trim.mach
            meta["q"] = trim.q
            meta["fixed"] = _fmt_trim_vars(trim.variables)
        meta["trim_card"] = trim
        subcase_meta[sc.id] = meta

    # 결과측
    result_map = {}
    if results is not None:
        for rsc in results.subcases:
            result_map[rsc.subcase_id] = rsc

    all_ids = sorted(set(subcase_meta) | set(result_map))
    for sid in all_ids:
        meta = subcase_meta.get(sid, {"selectors": [], "mach": None,
                                      "q": None, "fixed": "", "desc": "",
                                      "trim_card": None})
        rsc = result_map.get(sid)
        # BDF SUBCASE가 없는 결과(착륙/지상 등) — 결과 객체에 실린
        # 메타(label/category/far_section/nz_cg)로 입력 컬럼을 채운다
        if sid not in subcase_meta and rsc is not None:
            if getattr(rsc, "label", ""):
                meta["desc"] = rsc.label
            far = getattr(rsc, "far_section", "")
            cat = getattr(rsc, "category", "")
            if far or cat:
                meta["selectors"] = [x for x in (cat, far) if x]
            nz = getattr(rsc, "nz_cg", None)
            if nz is not None:
                lift = getattr(rsc, "lift_factor", 0.0)
                fixed = f"nz={nz:.2f}"
                if lift:
                    fixed += f", 양력 {lift:.2f}W (§23.473(e))"
                wl = getattr(rsc, "weight_label", "")
                if wl:
                    fixed += f", {wl}"
                meta["fixed"] = fixed
        accel, ang_accel = _case_accels(meta.get("trim_card"), rsc)
        # 착륙/지상 결과의 가속도 표시는 조건의 nz를 우선
        if sid not in subcase_meta and rsc is not None \
                and getattr(rsc, "nz_cg", None) is not None:
            bal = rsc.trim_balance or {}
            nz_tot = rsc.nz_cg + bal.get("relief_nz", 0.0)
            accel = (f"{bal.get('relief_nx', 0.0):.3g}, "
                     f"{bal.get('relief_ny', 0.0):.3g}, {nz_tot:.3g}")
        row = {
            "subcase": sid,
            "desc": meta.get("desc", ""),
            "selectors": " ".join(meta["selectors"]),
            "mach": meta["mach"],
            "q": meta["q"],
            "accel": accel,
            "ang_accel": ang_accel,
            "fixed": meta["fixed"],
            "trim_result": "",
            "sums": None,
            "sums_note": "",
            "tol": None,
        }
        if rsc is not None:
            if rsc.trim_variables:
                row["trim_result"] = _fmt_trim_vars(
                    sorted(rsc.trim_variables.items()))
            forces = {
                "combined": rsc.nodal_combined_forces,
                "aero": rsc.nodal_aero_forces,
                "inertial": rsc.nodal_inertial_forces,
            }.get(load_type)
            if load_type == "combined" and rsc.trim_balance:
                # 솔버가 저장한 CG 기준 평형 (가장 정확)
                row["sums"] = [rsc.trim_balance.get(k, 0.0)
                               for k in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
                row["sums_note"] = "CG 기준 (솔버 trim_balance)"
            elif forces:
                row["sums"] = _force_sum(model, forces)
                row["sums_note"] = "원점 기준"
            # combined 잔차 허용치: aero 합 크기의 TOLERANCE_FRACTION
            if load_type == "combined" and row["sums"] is not None:
                aero_sums = _force_sum(model, rsc.nodal_aero_forces or {})
                if aero_sums is not None:
                    ref_f = max(abs(v) for v in aero_sums[:3])
                    ref_m = max(abs(v) for v in aero_sums[3:])
                    row["tol"] = (
                        ref_f * TOLERANCE_FRACTION if ref_f > 0 else None,
                        ref_m * TOLERANCE_FRACTION if ref_m > 0 else None,
                    )
        rows.append(row)
    return rows


class LoadCasesPanel:
    """summarize_cases() 결과를 QTableWidget으로 렌더링하는 탭 위젯."""

    HEADERS = ["Subcase", "케이스 설명", "선택 카드", "Mach", "Q",
               "가속도 nx,ny,nz (g)", "각가속도 ṗ,q̇,ṙ (rad/s²)",
               "고정 트림조건", "트림 결과", "ΣFx (N)", "ΣFy (N)",
               "ΣFz (N)", "ΣMx (N-mm)", "ΣMy (N-mm)", "ΣMz (N-mm)"]
    _SUM_COL0 = 9  # ΣFx 열 인덱스 (잔차 강조 기준)

    def __init__(self, parent=None) -> None:
        from qtpy.QtWidgets import (
            QComboBox, QHBoxLayout, QLabel, QTableWidget, QVBoxLayout,
            QWidget,
        )

        self.widget = QWidget(parent)
        self._model = None
        self._results = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("하중 합계"))
        self._load_type = QComboBox()
        self._load_type.addItems(LOAD_TYPES)
        self._load_type.currentIndexChanged.connect(self._refresh_table)
        controls.addWidget(self._load_type)
        self._note = QLabel("")
        controls.addWidget(self._note, 1)

        self._table = QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)

        layout = QVBoxLayout(self.widget)
        layout.addLayout(controls)
        layout.addWidget(self._table, 1)

    # ------------------------------------------------------------------
    def set_data(self, model, results) -> None:
        self._model = model
        self._results = results
        self._refresh_table()

    def _refresh_table(self, *_args) -> None:
        from qtpy.QtGui import QBrush, QColor, QFont
        from qtpy.QtWidgets import QTableWidgetItem

        self._table.setRowCount(0)
        if self._model is None:
            return
        load_type = self._load_type.currentText()
        try:
            rows = summarize_cases(self._model, self._results, load_type)
        except Exception:
            logger.exception("Load case summary failed")
            return
        self._table.setRowCount(len(rows))
        note = ""
        red = QBrush(QColor("#c62828"))
        bold = QFont()
        bold.setBold(True)
        s0 = self._SUM_COL0
        for r, row in enumerate(rows):
            cells = [
                str(row["subcase"]),
                row.get("desc") or "-",
                row["selectors"],
                f"{row['mach']:.3g}" if row["mach"] is not None else "-",
                f"{row['q']:.4g}" if row["q"] is not None else "-",
                row.get("accel") or "-",
                row.get("ang_accel") or "-",
                row["fixed"] or "-",
                row["trim_result"] or "-",
            ]
            if row["sums"] is not None:
                cells += [f"{v:.4g}" for v in row["sums"]]
                note = row["sums_note"]
            else:
                cells += ["-"] * 6
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                # combined 잔차가 허용치를 넘으면 빨간색 강조
                if row["tol"] is not None and s0 <= c < s0 + 6 \
                        and row["sums"] is not None:
                    tol = row["tol"][0] if c < s0 + 3 else row["tol"][1]
                    if tol is not None and abs(row["sums"][c - s0]) > tol:
                        item.setForeground(red)
                        item.setFont(bold)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        if note:
            pct = int(TOLERANCE_FRACTION * 100)
            self._note.setText(
                f"모멘트 {note} — 트림 평형이면 combined 6분력 ≈ 0, "
                f"빨간색: |잔차| > aero 합 최대성분의 {pct}%")
        else:
            self._note.setText("")
