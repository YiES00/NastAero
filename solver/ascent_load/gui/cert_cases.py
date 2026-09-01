# 인증 케이스 탭 — FAR §23 하중 케이스 매트릭스 나열 + SOL 144 TRIM 해석 BDF 생성
from __future__ import annotations

import logging
import math
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

logger = logging.getLogger("ascent_load.gui")


def detect_q_scale(model) -> float:
    """모델 단위 감지 — 인증 파이프라인 q(Pa)를 모델 단위로 변환하는 배율.

    BatchRunner._detect_q_scale과 동일 휴리스틱: AEROS.REFC > 100 → mm 모델
    → 1e-6 (Pa → N/mm²).
    """
    refc = 0.0
    if getattr(model, "aeros", None):
        refc = model.aeros.refc
    elif getattr(model, "aero", None):
        refc = model.aero.refc
    return 1e-6 if refc > 100 else 1.0


def strip_trim_cards(bulk_lines: List[str]) -> tuple:
    """벌크에서 기존 TRIM 카드(연속행 포함)를 제거한다.

    생성 덱은 케이스 컨트롤을 새로 쓰므로 원본 TRIM은 참조되지 않는
    죽은 카드이고, 남겨두면 생성 TRIM과 ID가 겹칠 수 있다.

    Returns
    -------
    (남은 행 목록, 제거된 TRIM ID 집합)
    """
    from .card_form import card_extent

    removed_ids: set = set()
    drop = [False] * len(bulk_lines)
    for i, line in enumerate(bulk_lines):
        if drop[i]:
            continue
        keyword = line.split(",")[0][:8].strip().upper().rstrip("*")
        if keyword != "TRIM":
            continue
        extent = card_extent(bulk_lines, i)
        if extent is None:
            continue
        start, end = extent
        from ..bdf.field_parser import nastran_int
        from .card_form import parse_card

        try:
            removed_ids.add(nastran_int(parse_card(
                bulk_lines[start:end])[1]))
        except Exception:
            pass
        for j in range(start, end):
            drop[j] = True
    return ([l for i, l in enumerate(bulk_lines) if not drop[i]],
            removed_ids)


def render_trim_bdf(master_text: str, cases: List, q_scale: float) -> tuple:
    """인증 케이스들을 해석 가능한 SOL 144 BDF 텍스트로 렌더링한다.

    원본 덱의 실행부(exec)와 케이스 컨트롤 전역부(첫 SUBCASE 이전)는
    유지하고, SUBCASE 블록을 케이스별로 새로 생성한다. 기존 TRIM
    카드는 벌크에서 제거하고, 생성 TRIM 카드는 **인증 케이스 ID를
    그대로 SUBCASE/TRIM 번호로** 사용해 표와 덱의 번호가 일치한다.

    Returns
    -------
    (BDF 텍스트, 제거된 원본 TRIM ID 집합)
    """
    from .card_form import format_card

    lines = master_text.split("\n")
    i_cend = next(i for i, l in enumerate(lines)
                  if l.strip().upper().startswith("CEND"))
    i_bulk = next(i for i, l in enumerate(lines)
                  if l.strip().upper().startswith("BEGIN")
                  and "BULK" in l.upper())

    exec_lines = lines[:i_cend + 1]
    cc_lines = lines[i_cend + 1:i_bulk]
    bulk_lines, removed_ids = strip_trim_cards(lines[i_bulk:])

    # 케이스 컨트롤 전역부(첫 SUBCASE 이전)만 유지
    cc_global = []
    for l in cc_lines:
        if l.strip().upper().startswith("SUBCASE"):
            break
        cc_global.append(l)

    new_cc = list(cc_global)
    trim_cards: List[str] = []
    for case in cases:
        tc = case.trim_condition
        tid = tc.case_id   # 인증 케이스 ID = SUBCASE = TRIM (혼선 방지)
        subtitle = (case.label or tc.label)[:64]
        new_cc.append(f"SUBCASE {tid}")
        # LABEL이 케이스 설명(파서 우선순위 높음), SUBTITLE이 분류 참고
        new_cc.append(f"  SUBTITLE ={case.far_section} {case.category}"[:72])
        new_cc.append(f"  LABEL    ={subtitle}")
        new_cc.append(f"  TRIM     = {tid}")

        fields = ["TRIM", str(tid), f"{tc.mach:.5f}",
                  f"{tc.q * q_scale:.6g}"]
        variables = list(tc.fixed_vars.items())
        if "URDD3" not in {k.upper() for k in tc.fixed_vars}:
            variables.append(("URDD3", tc.nz))
        for label, value in variables:
            fields += [label.upper(), f"{value:.6g}"]
        trim_cards.append(f"$ case {tc.case_id}: {tc.label}")
        trim_cards.extend(format_card(fields))

    # ENDDATA 앞에 TRIM 카드 삽입
    i_end = next((i for i, l in enumerate(bulk_lines)
                  if l.strip().upper().startswith("ENDDATA")),
                 len(bulk_lines))
    header = ["$ --- certification trim cases (generated) ---"]
    if removed_ids:
        header.append("$ original TRIM cards removed: "
                      + ", ".join(str(t) for t in sorted(removed_ids)))
    new_bulk = (bulk_lines[:i_end] + header + trim_cards
                + bulk_lines[i_end:])

    return "\n".join(exec_lines + new_cc + new_bulk), removed_ids


def resolve_vtol_factory(config_dict: dict):
    """설정 dict의 'vtol_factory'(점 표기 임포트 경로)를 콜러블로 해석.

    예: vtol_factory: ascent_load.models.ilc8.make_ilc8_vtol_config
    키가 없거나 해석 실패면 None — 재트림 버튼 비활성의 근거가 된다.
    """
    path = (config_dict or {}).get("vtol_factory")
    if not path or not isinstance(path, str):
        return None
    mod_name, _, attr = path.rpartition(".")
    if not mod_name:
        return None
    try:
        import importlib

        fn = getattr(importlib.import_module(mod_name), attr, None)
        return fn if callable(fn) else None
    except Exception:
        return None


class CertCasesPanel(QWidget):
    """인증 기준(FAR §23) 하중 케이스 나열 → 선택 → TRIM 해석 BDF 생성."""

    HEADERS = ["ID", "분류", "FAR §", "케이스", "V_EAS (m/s)", "Mach",
               "q (모델단위)", "nz", "중량", "고도 (m)", "해석"]

    def __init__(self, info_panel, parent=None) -> None:
        super().__init__(parent)
        self._info = info_panel
        self._model = None
        self._bdf_path: Optional[str] = None
        self._cases: List = []
        self.open_requested = None   # main window가 주입하는 콜백
        self.landing_results_ready = None  # (list[SubcaseResult]) 콜백
        self.retrim_results_ready = None   # (list[SubcaseResult]) 콜백
        info_panel.config_changed.connect(self._update_retrim_enabled)

        controls = QHBoxLayout()
        gen_btn = QPushButton("케이스 생성")
        gen_btn.clicked.connect(self.generate_cases)
        controls.addWidget(gen_btn)
        self._dynamic_chk = QCheckBox("동적 기동 포함 (6-DOF 시뮬, 느림)")
        controls.addWidget(self._dynamic_chk)
        self._export_btn = QPushButton("TRIM 해석 BDF 생성…")
        self._export_btn.clicked.connect(self.export_bdf)
        self._export_btn.setEnabled(False)
        controls.addWidget(self._export_btn)
        self._landing_btn = QPushButton("착륙 케이스 해석")
        self._landing_btn.setToolTip(
            "착륙/지상 케이스의 자기평형 절점하중을 계산해 결과 탭"
            "(Load Cases/VMT/Design Loads)에 합류시킵니다")
        self._landing_btn.clicked.connect(self.solve_landing_cases)
        self._landing_btn.setEnabled(False)
        controls.addWidget(self._landing_btn)
        self._retrim_btn = QPushButton("재트림 케이스 선별")
        self._retrim_btn.setToolTip(
            "(고장×재트림) 확장 사건 공간을 선형 기저로 전수 선별하고 "
            "P·C 판정으로 채택된 지배 패턴을 결과 탭에 합류시킵니다.\n"
            "기체 YAML에 vtol_factory 키(로터 구성 팩토리 임포트 경로)가 "
            "필요합니다")
        self._retrim_btn.clicked.connect(self.solve_retrim_cases)
        self._retrim_btn.setEnabled(False)
        controls.addWidget(self._retrim_btn)
        self._retrim_tr_chk = QCheckBox("천이 포함 (트림해 16회, ~1분)")
        self._retrim_tr_chk.setToolTip(
            "체크 시 천이 계열(q>0)의 (V,nz) 상태까지 아핀 트림 기저로 "
            "선별합니다. 해제 시 호버 계열만(수 초)")
        controls.addWidget(self._retrim_tr_chk)
        controls.addStretch()

        self._summary = QLabel(
            "기체 정보 탭 입력 후 [케이스 생성] — §23.331-341/441/473 기반")
        self._table = QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._summary)
        layout.addWidget(self._table, 1)

    # ------------------------------------------------------------------
    def set_context(self, model, bdf_path: Optional[str]) -> None:
        self._model = model
        self._bdf_path = bdf_path
        self._update_retrim_enabled()

    def _update_retrim_enabled(self) -> None:
        ok = False
        if self._model is not None and hasattr(self._model, "masses"):
            try:
                ok = resolve_vtol_factory(self._info.to_dict()) is not None
            except Exception:
                ok = False
        self._retrim_btn.setEnabled(ok)

    # ------------------------------------------------------------------
    def generate_cases(self) -> None:
        try:
            config = self._info.get_config()
        except Exception as exc:
            QMessageBox.warning(self, "ASCENT-Load", f"기체 설정 오류:\n{exc}")
            return
        if not config.weight_cg_conditions or config.speeds.VC <= 0:
            QMessageBox.information(
                self, "ASCENT-Load",
                "기체 정보 탭에서 설계 속도와 중량/CG 조건을 먼저 입력하세요")
            return
        from ..loads_analysis.certification.load_case_matrix import (
            LoadCaseMatrix,
        )

        include_dyn = self._dynamic_chk.isChecked()
        if include_dyn and self._model is None:
            QMessageBox.information(self, "ASCENT-Load",
                                    "동적 기동에는 .bdf 모델이 필요합니다")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            matrix = LoadCaseMatrix(config)
            matrix.generate_all(bdf_model=self._model,
                                include_dynamic=include_dyn)
        except Exception as exc:
            logger.exception("Case generation failed")
            QMessageBox.critical(self, "ASCENT-Load", f"케이스 생성 실패:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._cases = (list(matrix.flight_cases)
                       + list(matrix.dynamic_cases)
                       + list(matrix.landing_cases))
        counts = matrix.summary()
        self._populate_table()
        self._summary.setText(
            f"총 {len(self._cases)}개 케이스 — "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            + " | 선택 없이 내보내면 trim 형식 전체를 생성합니다")
        self._export_btn.setEnabled(True)
        self._landing_btn.setEnabled(
            any(getattr(c, "condition_type", None) is not None
                for c in self._cases))
        self._matrix = matrix

    def _populate_table(self) -> None:
        q_scale = detect_q_scale(self._model) if self._model else 1.0
        self._table.setRowCount(len(self._cases))
        for r, case in enumerate(self._cases):
            # 비행 케이스는 CertLoadCase, 착륙/지상은 LandingCondition
            tc = getattr(case, "trim_condition", None)
            vn_pt = getattr(case, "vn_point", None)
            v_eas = f"{vn_pt.V_eas:.1f}" if vn_pt is not None else ""
            category = getattr(case, "category", None) or getattr(
                getattr(case, "condition_type", None), "value", "landing")
            nz = tc.nz if tc else getattr(case, "nz_cg", None)
            wc = getattr(case, "weight_cg", None)
            cells = [
                str(case.case_id),
                category,
                case.far_section,
                case.label or (tc.label if tc else ""),
                v_eas,
                f"{tc.mach:.3f}" if tc else "-",
                f"{tc.q * q_scale:.4g}" if tc else "-",
                f"{nz:+.2f}" if nz is not None else "-",
                wc.label if wc else "-",
                f"{getattr(case, 'altitude_m', 0.0):.0f}",
                getattr(case, "solve_type", "static"),
            ]
            for c, text in enumerate(cells):
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def solve_landing_cases(self) -> None:
        """착륙/지상 케이스를 자기평형 절점하중으로 해석해 결과에 합류.

        BatchRunner._solve_landing_case와 동일 경로(기어 반력 + 관성 +
        §23.473(e) 양력 + 관성 릴리프)를 쓰므로 CLI 배치와 결과가
        일치한다. 산출된 절점하중은 SubcaseResult로 변환돼 Load
        Cases/VMT/Design Loads 탭이 트림 결과와 함께 소비한다.
        """
        from types import SimpleNamespace

        if self._model is None or not hasattr(self._model, "masses"):
            QMessageBox.information(
                self, "ASCENT-Load",
                "착륙 해석에는 질량 정보가 있는 .bdf 모델이 필요합니다"
                " (.aload 결과만 열린 상태에서는 실행할 수 없습니다)")
            return
        landing = [c for c in self._cases
                   if getattr(c, "condition_type", None) is not None]
        if not landing:
            QMessageBox.information(self, "ASCENT-Load",
                                    "생성된 착륙/지상 케이스가 없습니다")
            return

        from ..loads_analysis.certification.batch_runner import BatchRunner
        from ..output.result_data import SubcaseResult

        try:
            config = self._info.get_config()
            runner = BatchRunner(
                SimpleNamespace(config=config, flight_cases=[],
                                landing_cases=landing, dynamic_cases=[]),
                bdf_model=self._model)
            from ..loads_analysis.trim_loads import verify_trim_balance
            from ..solvers.sol144 import _compute_cg

            cg = _compute_cg(self._model)
            results = []
            for cond in landing:
                r = runner._solve_landing_case(cond)
                sc = SubcaseResult(subcase_id=r.case_id,
                                   nodal_combined_forces=r.nodal_forces)
                # Load Cases 표 표시용 메타 — BDF SUBCASE가 없는 결과라
                # 설명/분류/조건을 결과 객체에 직접 싣는다
                sc.label = r.label
                sc.category = r.category
                sc.far_section = getattr(cond, "far_section", "")
                sc.nz_cg = getattr(cond, "nz_cg", None)
                sc.lift_factor = getattr(cond, "lift_factor", 0.0)
                sc.weight_label = (cond.weight_cg.label
                                   if cond.weight_cg else "")
                # CG 기준 6분력 합 — 관성 릴리프 폐합 확인용
                sc.trim_balance = verify_trim_balance(
                    self._model, r.nodal_forces, ref_point=cg)
                if r.flight_state:
                    sc.trim_balance.update(
                        {k: v for k, v in r.flight_state.items()
                         if k.startswith(("relief_", "p_dot", "q_dot",
                                          "r_dot"))})
                results.append(sc)
        except Exception as exc:
            logger.exception("Landing case solve failed")
            QMessageBox.critical(self, "ASCENT-Load",
                                 f"착륙 케이스 해석 실패:\n{exc}")
            return

        self._summary.setText(
            f"착륙/지상 {len(results)}개 케이스 해석 완료 — 절점하중이 "
            "Load Cases·VMT·Design Loads 탭에 합류했습니다")
        if callable(self.landing_results_ready):
            self.landing_results_ready(results)

    # ------------------------------------------------------------------
    def solve_retrim_cases(self) -> None:
        """(고장×재트림) 확장 사건을 선별하고 채택 케이스를 결과에 합류.

        certification/retrim_events.RetrimScreen과 동일 경로(로터별
        선형 VMT 기저 전수 선별 + P·C 채택 판정)를 쓰므로 배치
        드라이버 8b 단계와 결과가 일치한다. 채택된 지배 패턴의
        자기평형 절점하중은 SubcaseResult로 변환돼 Load Cases/VMT/
        Design Loads 탭이 소비한다.
        """
        if self._model is None or not hasattr(self._model, "masses"):
            QMessageBox.information(
                self, "ASCENT-Load",
                "재트림 선별에는 질량 정보가 있는 .bdf 모델이 필요합니다")
            return
        try:
            config = self._info.get_config()
        except Exception as exc:
            QMessageBox.warning(self, "ASCENT-Load", f"기체 설정 오류:\n{exc}")
            return
        factory = resolve_vtol_factory(self._info.to_dict())
        if factory is None:
            QMessageBox.information(
                self, "ASCENT-Load",
                "기체 YAML에 vtol_factory 키(로터 구성 팩토리 임포트 "
                "경로)가 필요합니다.\n예: vtol_factory: "
                "ascent_load.models.ilc8.make_ilc8_vtol_config")
            return

        from ..loads_analysis.certification.retrim_events import (
            RetrimScreen,
        )
        from ..loads_analysis.trim_loads import verify_trim_balance
        from ..output.result_data import SubcaseResult
        from ..solvers.sol144 import _compute_cg

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            screen = RetrimScreen(
                self._model, factory(), config,
                transition=self._retrim_tr_chk.isChecked())
            events = screen.screen()
            cases = screen.realize(events, threshold_pct=1.0, top_n=8)
            cg = _compute_cg(self._model)
            results = []
            for r in cases:
                sc = SubcaseResult(subcase_id=r.case_id,
                                   nodal_combined_forces=r.nodal_forces)
                sc.label = r.label
                sc.category = r.category
                sc.far_section = r.far_section
                sc.nz_cg = r.nz
                sc.weight_label = r.weight_label
                sc.trim_balance = verify_trim_balance(
                    self._model, r.nodal_forces, ref_point=cg)
                if r.flight_state:
                    sc.trim_balance.update({
                        k: v for k, v in r.flight_state.items()
                        if k.startswith("retrim_")})
                results.append(sc)
        except Exception as exc:
            logger.exception("Retrim screen failed")
            QMessageBox.critical(self, "ASCENT-Load",
                                 f"재트림 선별 실패:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        n_pat = sum(e.n_patterns for e in events)
        self._summary.setText(
            f"(고장×재트림) 확장 {len(events)}사건/{n_pat:,}패턴 선별 → "
            f"채택 {len(results)}케이스가 Load Cases·VMT·Design Loads "
            "탭에 합류했습니다 (P·C 판정, C ≥ 1%)")
        if callable(self.retrim_results_ready):
            self.retrim_results_ready(results)

    def export_bdf(self) -> None:
        if self._bdf_path is None or self._model is None:
            QMessageBox.information(
                self, "ASCENT-Load",
                "TRIM BDF를 생성하려면 기준 .bdf 모델을 먼저 여세요")
            return
        # 선택된 행(없으면 전체)에서 trim 형식만
        sel_rows = sorted({i.row() for i in self._table.selectedItems()})
        pool = ([self._cases[r] for r in sel_rows] if sel_rows
                else self._cases)
        cases = [c for c in pool
                 if getattr(c, "solve_type", "") == "trim"
                 and getattr(c, "trim_condition", None) is not None]
        skipped = len(pool) - len(cases)
        if not cases:
            QMessageBox.information(self, "ASCENT-Load",
                                    "TRIM 형식 케이스가 없습니다")
            return

        from pathlib import Path

        # INCLUDE 상대경로가 살아 있도록 기본 저장 위치는 마스터 옆
        default = str(Path(self._bdf_path).parent / "cert_trim_cases.bdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "TRIM 해석 BDF 저장", default, "BDF (*.bdf)")
        if not path:
            return
        if Path(path).parent != Path(self._bdf_path).parent \
                and "INCLUDE" in Path(self._bdf_path).read_text(
                    errors="replace").upper():
            QMessageBox.warning(
                self, "ASCENT-Load",
                "마스터와 다른 폴더에 저장하면 INCLUDE 상대경로가 깨질 수 "
                "있습니다. 마스터 BDF와 같은 폴더를 권장합니다.")

        master_text = Path(self._bdf_path).read_text(errors="replace")
        q_scale = detect_q_scale(self._model)
        try:
            text, removed = render_trim_bdf(master_text, cases, q_scale)
        except Exception as exc:
            logger.exception("TRIM BDF rendering failed")
            QMessageBox.critical(self, "ASCENT-Load", f"BDF 생성 실패:\n{exc}")
            return
        # INCLUDE 파일에 정의된 TRIM은 텍스트 제거가 불가 — ID 충돌 확인
        include_trims = set(getattr(self._model, "trims", {}) or {}) - removed
        clash = include_trims & {c.trim_condition.case_id for c in cases}
        if clash:
            QMessageBox.critical(
                self, "ASCENT-Load",
                "INCLUDE 파일에 정의된 TRIM 카드와 케이스 ID가 겹칩니다: "
                f"{sorted(clash)}\nINCLUDE 쪽 TRIM을 정리한 뒤 다시 시도하세요.")
            return
        Path(path).write_text(text)

        note = f"\n(비-trim {skipped}건 제외)" if skipped else ""
        answer = QMessageBox.question(
            self, "ASCENT-Load",
            f"TRIM 케이스 {len(cases)}개를 담은 BDF를 저장했습니다:\n{path}"
            f"{note}\n\n지금 GUI에서 열까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes and callable(self.open_requested):
            self.open_requested(path)
