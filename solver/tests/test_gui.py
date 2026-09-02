# GUI 순수 로직 단위시험 — 솔버 커맨드 구성, 모델 요약, 구문 강조 규칙 (Qt 위젯 불필요)
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

VM_DIR = Path(__file__).parent / "validation" / "nastran_vm"


# ---------------------------------------------------------------------------
# run_panel.build_solver_command / aload_path_for — qtpy 필요 (임포트 시)
# ---------------------------------------------------------------------------
qtpy = pytest.importorskip("qtpy", reason="GUI deps not installed")

from ascent_load.gui.run_panel import build_solver_command, aload_path_for  # noqa: E402
from ascent_load.gui.model_tree import summarize_model  # noqa: E402
from ascent_load.gui.editor import HIGHLIGHT_RULES  # noqa: E402


class TestBuildSolverCommand:
    def test_defaults(self):
        cmd = build_solver_command("model.bdf")
        assert cmd[:3] == [sys.executable, "-m", "ascent_load"]
        assert "model.bdf" in cmd
        assert "--save" in cmd
        # 기본값은 CLI 기본과 같으므로 생략
        assert "--parallel" not in cmd
        assert "--spline-slope" not in cmd

    def test_parallel_and_spline(self):
        cmd = build_solver_command("m.bdf", parallel=-1,
                                   spline_slope="rotation",
                                   log_level="DEBUG")
        assert cmd[cmd.index("--parallel") + 1] == "-1"
        assert cmd[cmd.index("--spline-slope") + 1] == "rotation"
        assert cmd[cmd.index("--log-level") + 1] == "DEBUG"

    def test_aload_path(self):
        assert aload_path_for("/a/b/model.bdf") == "/a/b/model.aload"


class TestKillProcessTree:
    def test_kills_children_with_parent(self):
        import subprocess
        import sys
        import time

        from ascent_load.gui.run_panel import kill_process_tree, list_child_pids

        # 부모가 자식(워커 역할)을 하나 띄우고 대기하는 미니 트리
        parent = subprocess.Popen([
            sys.executable, "-c",
            "import subprocess, sys, time;"
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
            "time.sleep(60)",
        ])
        try:
            for _ in range(50):  # 자식 스폰 대기
                kids = list_child_pids(parent.pid)
                if kids:
                    break
                time.sleep(0.1)
            assert kids, "child never spawned"
            killed = kill_process_tree(parent.pid)
            assert killed >= 1
            time.sleep(0.3)
            import os
            for k in kids:
                try:
                    os.kill(k, 0)
                    assert False, f"child {k} survived"
                except OSError:
                    pass  # 종료 확인
            assert parent.poll() is not None or parent.wait(5) is not None
        finally:
            if parent.poll() is None:
                parent.kill()


class TestSummarizeModel:
    def test_vm1_counts(self):
        from ascent_load.bdf.parser import parse_bdf

        model = parse_bdf(str(VM_DIR / "vm1_rod_axial.bdf"))
        groups = dict(summarize_model(model))
        node_key = next(k for k in groups if k.startswith("Nodes"))
        assert f"({len(model.nodes)})" in node_key
        elem_key = next(k for k in groups if k.startswith("Elements"))
        assert f"({len(model.elements)})" in elem_key

    def test_item_cap(self):
        from ascent_load.bdf.parser import parse_bdf

        model = parse_bdf(str(VM_DIR / "vm1_rod_axial.bdf"))
        groups = dict(summarize_model(model, max_items=2))
        node_items = groups[next(k for k in groups if k.startswith("Nodes"))]
        if len(model.nodes) > 2:
            assert node_items[-1].endswith("more")
            assert len(node_items) == 3


class TestIncludeListing:
    @pytest.mark.skipif(
        not (Path(__file__).parent / "validation" / "GACOMP").is_dir(),
        reason="comparison-model data not present in this archive")
    def test_gacomp_includes(self):
        from ascent_load.gui.editor import list_bdf_files

        master = (Path(__file__).parent / "validation" / "GACOMP"
                  / "p400r3-free-trim.bdf")
        files = list_bdf_files(str(master))
        assert files[0] == str(master.resolve())
        assert len(files) > 1  # BULK/ INCLUDE 파일들
        assert all(Path(f).exists() for f in files)

    def test_no_includes(self):
        from ascent_load.gui.editor import list_bdf_files

        master = VM_DIR / "vm1_rod_axial.bdf"
        files = list_bdf_files(str(master))
        assert files == [str(master.resolve())]


class TestCardForm:
    def test_extent_single_line(self):
        from ascent_load.gui.card_form import card_extent

        lines = ["$ comment", "GRID    1               0.      0.      0.",
                 "GRID    2               1.      0.      0."]
        assert card_extent(lines, 1) == (1, 2)
        assert card_extent(lines, 0) is None  # 주석

    def test_extent_continuation(self):
        from ascent_load.gui.card_form import card_extent

        lines = [
            "CONM2   1       10              5.0",
            "+       1.0     0.0     2.0",
            "GRID    2               1.      0.      0.",
        ]
        # 카드 첫 행에서도, 연속행에서도 같은 범위
        assert card_extent(lines, 0) == (0, 2)
        assert card_extent(lines, 1) == (0, 2)
        assert card_extent(lines, 2) == (2, 3)

    def test_extent_skips_structural_lines(self):
        from ascent_load.gui.card_form import card_extent

        assert card_extent(["ENDDATA"], 0) is None
        assert card_extent(["INCLUDE 'BULK/x.bdf'"], 0) is None

    def test_fit8(self):
        from ascent_load.gui.card_form import fit8

        assert fit8("1.5") == "1.5"
        assert fit8("ABCDEFGHIJ") == "ABCDEFGH"
        long_float = fit8("0.2298472344348814")
        assert len(long_float) <= 8
        assert abs(float(long_float) - 0.2298472344348814) < 1e-5

    def test_format_parse_roundtrip(self):
        from ascent_load.gui.card_form import format_card, parse_card

        fields = ["GRID", "999", "", "500.", "0.", "99."]
        lines = format_card(fields)
        assert len(lines) == 1
        assert lines[0].startswith("GRID    999")
        assert parse_card(lines) == fields

    def test_format_continuation(self):
        from ascent_load.gui.card_form import format_card, parse_card

        # 데이터 8개 초과 → '+' 연속행 생성
        fields = ["CONM2", "1", "10", "0", "5.0", "0.", "0.", "0.", "",
                  "1.0", "0.", "2.0"]
        lines = format_card(fields)
        assert len(lines) == 2
        assert lines[1].startswith("+")
        assert parse_card(lines) == fields

    def test_format_trims_trailing_blanks(self):
        from ascent_load.gui.card_form import format_card

        lines = format_card(["GRID", "1", "", "0.", "0.", "0.", "", "", ""])
        assert lines == ["GRID    1               0.      0.      0."]


class TestDesignLoads:
    def _mini_results(self):
        import numpy as np
        from ascent_load.output.result_data import ResultData, SubcaseResult

        results = ResultData(title="t", subcases=[])
        for sid in (1, 2):
            sc = SubcaseResult(subcase_id=sid)
            sc.nodal_combined_forces = {1: np.ones(6) * sid}
            sc.trim_variables = {"ANGLEA": 0.1 * sid}
            results.subcases.append(sc)
        return results

    def test_batch_from_results(self):
        from ascent_load.gui.design_loads import batch_from_results

        batch = batch_from_results(self._mini_results())
        assert batch.n_converged == 2
        assert [cr.case_id for cr in batch.case_results] == [1, 2]
        cr = batch.case_results[0]
        assert cr.converged and cr.nodal_forces
        assert cr.label == "Subcase 1"

    def test_batch_carries_landing_metadata(self):
        # 착륙 SubcaseResult에 실린 라벨/분류/FAR/nz가 CaseResult로 승계
        import numpy as np
        from ascent_load.output.result_data import SubcaseResult
        from ascent_load.gui.design_loads import batch_from_results

        results = self._mini_results()
        sc = SubcaseResult(subcase_id=40)
        sc.nodal_combined_forces = {1: np.ones(6)}
        sc.label = "Level landing 3pt nz=3.08 MTOW"
        sc.category = "landing"
        sc.far_section = "§23.479"
        sc.nz_cg = 3.08
        results.subcases.append(sc)
        batch = batch_from_results(results)
        cr = {c.case_id: c for c in batch.case_results}[40]
        assert cr.label == "Level landing 3pt nz=3.08 MTOW"
        assert cr.category == "landing"
        assert cr.far_section == "§23.479"
        assert cr.nz == 3.08
        # 트림 결과는 기존 기본값 유지
        assert cr.converged
        assert {c.case_id: c for c in batch.case_results}[1].category == "trim"

    def test_batch_skips_missing_forces(self):
        from ascent_load.output.result_data import SubcaseResult
        from ascent_load.gui.design_loads import batch_from_results

        results = self._mini_results()
        results.subcases.append(SubcaseResult(subcase_id=3))  # 하중 없음
        batch = batch_from_results(results)
        assert 3 not in batch.completed_ids
        assert batch.n_converged == 2


class TestLoadCases:
    def _fake_model(self):
        import numpy as np
        from types import SimpleNamespace

        trim = SimpleNamespace(mach=0.3, q=0.0064,
                               variables=[("URDD3", 9810.0), ("PITCH", 0.0)])
        sc = SimpleNamespace(id=1, spc_id=0, load_id=0, method_id=0,
                             trim_id=10)
        nodes = {
            1: SimpleNamespace(xyz_global=np.array([0.0, 0.0, 0.0])),
            2: SimpleNamespace(xyz_global=np.array([1000.0, 0.0, 0.0])),
        }
        return SimpleNamespace(subcases=[sc], trims={10: trim}, nodes=nodes)

    def _fake_results(self, with_balance: bool):
        import numpy as np
        from ascent_load.output.result_data import ResultData, SubcaseResult

        sc = SubcaseResult(subcase_id=1)
        sc.trim_variables = {"ANGLEA": 0.1}
        # 절점 1에 +Fz, 절점 2에 -Fz → ΣFz=0, 원점 기준 ΣMy=+1000*100
        sc.nodal_combined_forces = {
            1: np.array([0., 0., 100., 0., 0., 0.]),
            2: np.array([0., 0., -100., 0., 0., 0.]),
        }
        sc.nodal_aero_forces = {1: np.array([0., 0., 100., 0., 0., 0.])}
        if with_balance:
            sc.trim_balance = {"Fx": 0.0, "Fy": 0.0, "Fz": 1e-8,
                               "Mx": 0.0, "My": 2e-8, "Mz": 0.0}
        return ResultData(title="t", subcases=[sc])

    def test_input_conditions(self):
        from ascent_load.gui.load_cases import summarize_cases

        rows = summarize_cases(self._fake_model(), None)
        assert len(rows) == 1
        row = rows[0]
        assert row["mach"] == 0.3
        assert "TRIM=10" in row["selectors"]
        assert "URDD3=9810" in row["fixed"]
        assert row["sums"] is None

    def test_case_description_from_subtitle(self):
        from ascent_load.gui.load_cases import summarize_cases

        model = self._fake_model()
        model.subcases[0].label = "MACH 0.3 - 1G LEVEL FLIGHT"
        rows = summarize_cases(model, None)
        assert rows[0]["desc"] == "MACH 0.3 - 1G LEVEL FLIGHT"
        # label 없는 서브케이스는 빈 문자열
        del model.subcases[0].label
        assert summarize_cases(model, None)[0]["desc"] == ""

    def test_trim_balance_preferred_for_combined(self):
        from ascent_load.gui.load_cases import summarize_cases

        rows = summarize_cases(self._fake_model(),
                               self._fake_results(with_balance=True),
                               load_type="combined")
        assert rows[0]["sums"][2] == 1e-8  # 솔버 trim_balance 사용
        assert "CG" in rows[0]["sums_note"]
        assert "ANGLEA=5.73°" in rows[0]["trim_result"]

    def test_computed_sums_fallback(self):
        from ascent_load.gui.load_cases import summarize_cases

        rows = summarize_cases(self._fake_model(),
                               self._fake_results(with_balance=False),
                               load_type="combined")
        fx, fy, fz, mx, my, mz = rows[0]["sums"]
        assert fz == 0.0  # +100 -100
        assert my == 100000.0  # x=1000mm 위치의 -100N → +My
        assert "원점" in rows[0]["sums_note"]

    def test_aero_sum(self):
        from ascent_load.gui.load_cases import summarize_cases

        rows = summarize_cases(self._fake_model(),
                               self._fake_results(with_balance=True),
                               load_type="aero")
        assert rows[0]["sums"][2] == 100.0  # aero 총 양력

    def test_tolerance_from_aero_sums(self):
        from ascent_load.gui.load_cases import summarize_cases, TOLERANCE_FRACTION

        rows = summarize_cases(self._fake_model(),
                               self._fake_results(with_balance=True),
                               load_type="combined")
        tol_f, tol_m = rows[0]["tol"]
        # aero: 절점1(원점)에 Fz=100 → ref_F=100, 모멘트 합 0 → tol_m None
        assert tol_f == 100.0 * TOLERANCE_FRACTION
        assert tol_m is None

    def test_tolerance_absent_for_aero_view(self):
        from ascent_load.gui.load_cases import summarize_cases

        rows = summarize_cases(self._fake_model(),
                               self._fake_results(with_balance=True),
                               load_type="aero")
        assert rows[0]["tol"] is None

    def test_acceleration_columns(self):
        from ascent_load.gui.load_cases import summarize_cases

        model = self._fake_model()
        # URDD3=9810이 nz로 그대로 (0이 아니므로), relief 가속도 합산
        results = self._fake_results(with_balance=True)
        results.subcases[0].trim_balance.update({
            "relief_nx": 0.0, "relief_ny": 0.01, "relief_nz": 0.0,
            "p_dot": 8.68, "q_dot": 0.0, "r_dot": 0.44,
        })
        row = summarize_cases(model, results)[0]
        assert row["accel"] == "0, 0.01, 9.81e+03"
        assert row["ang_accel"] == "8.68, 0, 0.44"

    def test_acceleration_default_1g(self):
        from ascent_load.gui.load_cases import summarize_cases
        from types import SimpleNamespace

        # URDD3 부재 → 솔버 규약상 1g
        model = self._fake_model()
        model.trims[10] = SimpleNamespace(mach=0.3, q=0.0064, variables=[])
        row = summarize_cases(model, None)[0]
        assert row["accel"] == "0, 0, 1"
        assert row["ang_accel"] == "0, 0, 0"

    def test_vizmodel_without_subcases(self):
        from ascent_load.gui.load_cases import summarize_cases
        from types import SimpleNamespace
        import numpy as np

        viz = SimpleNamespace(nodes={1: SimpleNamespace(
            xyz_global=np.array([0., 0., 0.]))})
        rows = summarize_cases(viz, self._fake_results(with_balance=True))
        assert rows[0]["mach"] is None
        assert rows[0]["sums"] is not None


class TestResolveBdfRef:
    def test_relative_to_yaml(self, tmp_path):
        from ascent_load.gui.cert_setup import resolve_bdf_ref

        (tmp_path / "model.bdf").write_text("CEND\n")
        yaml_path = str(tmp_path / "cfg.yaml")
        assert resolve_bdf_ref(yaml_path, "model.bdf") == str(
            (tmp_path / "model.bdf").resolve())
        assert resolve_bdf_ref(yaml_path, "missing.bdf") is None
        assert resolve_bdf_ref(yaml_path, "") is None

    def test_absolute(self, tmp_path):
        from ascent_load.gui.cert_setup import resolve_bdf_ref

        p = tmp_path / "abs.bdf"
        p.write_text("CEND\n")
        assert resolve_bdf_ref("/elsewhere/cfg.yaml", str(p)) == str(
            p.resolve())


class TestCertBdfRender:
    MASTER = "\n".join([
        "SOL 144",
        "CEND",
        "TITLE = ORIGINAL",
        "SUBCASE 1",
        "  TRIM = 1",
        "BEGIN BULK",
        "AEROS   0       0       1500.   10000.  1.7E7",
        "TRIM    1       0.30000 0.006   URDD3   1.0",
        "ENDDATA",
    ])

    def _cases(self):
        from types import SimpleNamespace

        tc = SimpleNamespace(case_id=5, mach=0.25, q=3828.0, nz=3.8,
                             fixed_vars={"ROLL": 0.0}, label="VA pull-up")
        return [SimpleNamespace(trim_condition=tc, label="VA pull-up",
                                far_section="§23.337", category="symmetric",
                                solve_type="trim")]

    def test_detect_q_scale_mm(self):
        from ascent_load.gui.cert_cases import detect_q_scale
        from types import SimpleNamespace

        mm = SimpleNamespace(aeros=SimpleNamespace(refc=1500.0), aero=None)
        m = SimpleNamespace(aeros=SimpleNamespace(refc=1.5), aero=None)
        assert detect_q_scale(mm) == 1e-6
        assert detect_q_scale(m) == 1.0

    def test_render_roundtrip(self):
        from ascent_load.gui.cert_cases import render_trim_bdf

        text, removed = render_trim_bdf(self.MASTER, self._cases(),
                                        q_scale=1e-6)
        lines = text.split("\n")
        # 전역 케이스컨트롤 유지, 원본 SUBCASE 제거, 새 SUBCASE 생성
        assert "TITLE = ORIGINAL" in text
        # 케이스 ID(5)가 그대로 SUBCASE/TRIM 번호
        assert "SUBCASE 5" in text
        assert not any(l.strip() == "SUBCASE 1" for l in lines)
        assert "  TRIM     = 5" in text
        # TRIM 카드: q 변환(3828 Pa → 0.003828), URDD3=nz 자동 부가
        i = next(i for i, l in enumerate(lines) if l.startswith("TRIM    5"))
        assert "0.003828" in lines[i]
        assert "URDD3" in lines[i]
        # 원본 벌크 TRIM 1은 제거됨 (죽은 카드 + ID 충돌 방지)
        assert removed == {1}
        assert not any(l.startswith("TRIM    1") for l in lines)
        assert lines[-1].strip().upper().startswith("ENDDATA")

    def test_rendered_bdf_parses(self, tmp_path):
        from ascent_load.gui.cert_cases import render_trim_bdf
        from ascent_load.bdf.parser import parse_bdf

        text, _removed = render_trim_bdf(self.MASTER, self._cases(),
                                         q_scale=1e-6)
        p = tmp_path / "gen.bdf"
        p.write_text(text)
        model = parse_bdf(str(p))
        assert set(model.trims) == {5}   # 원본 TRIM 1 제거, 케이스 ID 5만
        trim = model.trims[5]
        assert abs(trim.q - 0.003828) < 1e-9
        assert dict(trim.variables)["URDD3"] == 3.8
        assert [sc.id for sc in model.subcases] == [5]
        assert model.subcases[0].trim_id == 5


class TestHighlightRules:
    """강조 규칙은 (이름, 정규식, 색, bold) 순수 데이터 — Qt 없이 검증한다."""

    def _match(self, name: str, text: str):
        pattern = next(p for n, p, _c, _b in HIGHLIGHT_RULES if n == name)
        return re.compile(pattern).match(text)

    def test_keyword(self):
        m = self._match("keyword", "GRID    1       0       0.0")
        assert m and m.group() == "GRID"
        m = self._match("keyword", "CQUAD4  10      1       1")
        assert m and m.group() == "CQUAD4"
        m = self._match("keyword", "GRID*   1")
        assert m and m.group() == "GRID*"

    def test_comment(self):
        m = self._match("comment", "$ this is a comment")
        assert m and m.group().startswith("$")
        assert self._match("comment", "GRID 1") is None

    def test_continuation(self):
        assert self._match("continuation", "+CONT1  2.0")
        assert self._match("continuation", "GRID") is None


class TestDescribeItem:
    """트리 항목 상세 정보 — BDFModel과 VizModel(.aload) 양쪽에서 예외 없이 동작."""

    @staticmethod
    def _exercise(model, results=None):
        from ascent_load.gui.model_tree import describe_item

        for group, labels in summarize_model(model, results):
            flat = []
            for label in labels:
                if isinstance(label, tuple):   # (타입 행, [개별 요소 행])
                    flat.append(label[0])
                    flat.extend(label[1])
                else:
                    flat.append(label)
            for label in flat:
                text = describe_item(model, results, group, label)
                assert isinstance(text, str)

    def test_single_element_describe_and_eids(self):
        ilc8 = (Path(__file__).parent / "validation" / "ILC8" / "ilc8.bdf")
        from ascent_load.bdf.parser import parse_bdf
        from ascent_load.gui.model_tree import ModelTreeWidget, describe_item

        m = parse_bdf(str(ilc8))
        eid = next(e for e, o in m.elements.items() if o.type == "CQUAD4")
        text = describe_item(m, None, "Elements (3872)", f"CQUAD4 {eid}")
        assert f"CQUAD4 {eid}" in text and "GRID" in text
        t = ModelTreeWidget.__new__(ModelTreeWidget)
        t._model = m
        assert t._eids_for("Elements (3872)", f"CQUAD4 {eid}") == [eid]

    def test_bdf_model_all_items(self):
        from ascent_load.bdf.parser import parse_bdf

        model = parse_bdf(str(VM_DIR / "vm1_rod_axial.bdf"))
        self._exercise(model)

    def test_viz_model_all_items(self):
        archive = (Path(__file__).parent / "validation" / "ILC8"
                   / "ilc8.aload")
        if not archive.exists():
            pytest.skip("ilc8.aload not generated")
        from ascent_load.output.result_io import load_results

        results, viz = load_results(str(archive))
        self._exercise(viz, results)

    def test_elem_type_uses_dict_keys(self):
        """VizElement에는 eid 속성이 없다 — dict 키로 EID 범위를 뽑아야 함."""
        from ascent_load.gui.model_tree import _describe_elem_type
        from ascent_load.output.result_io import VizElement

        class M:
            elements = {7: VizElement(type="CQUAD4", pid=1,
                                      node_ids=[1, 2, 3, 4]),
                        9: VizElement(type="CQUAD4", pid=1,
                                      node_ids=[2, 3, 4, 5])}
            properties = {}

        text = _describe_elem_type(M(), "CQUAD4")
        assert "7 … 9" in text


class TestElementOverlay:
    """build_element_overlay — 셸은 면, 보는 선으로 변환 (렌더러 불필요)."""

    def test_ilc8_mixed_types(self):
        ilc8 = (Path(__file__).parent / "validation" / "ILC8" / "ilc8.bdf")
        from ascent_load.bdf.parser import parse_bdf
        from ascent_load.gui.scene import build_element_overlay

        m = parse_bdf(str(ilc8))
        quads = [eid for eid, e in m.elements.items()
                 if e.type == "CQUAD4"][:20]
        bars = [eid for eid, e in m.elements.items()
                if e.type == "CBAR"][:10]
        poly = build_element_overlay(m, quads + bars)
        assert poly is not None
        assert poly.n_faces_strict == 20
        assert poly.n_lines == 10

    def test_empty(self):
        from ascent_load.gui.scene import build_element_overlay

        assert build_element_overlay(None, [1]) is None

    def test_eids_for_property(self):
        ilc8 = (Path(__file__).parent / "validation" / "ILC8" / "ilc8.bdf")
        from ascent_load.bdf.parser import parse_bdf
        from ascent_load.gui.model_tree import ModelTreeWidget

        m = parse_bdf(str(ilc8))
        t = ModelTreeWidget.__new__(ModelTreeWidget)  # Qt 초기화 없이
        t._model = m
        n_shell = len(t._eids_for("Properties (15)", "PSHELL 1101"))
        assert n_shell == sum(1 for e in m.elements.values()
                              if getattr(e, "pid", None) == 1101)
        n_type = len(t._eids_for("Elements (3872)", "CBAR: 480"))
        assert n_type == sum(1 for e in m.elements.values()
                             if e.type == "CBAR")


class TestLandingGearForm:
    """Aircraft 탭 착륙장치 폼 — ID 파싱 (Qt 초기화 없이)."""

    def test_parse_ids(self):
        from ascent_load.gui.cert_setup import AircraftInfoPanel

        parse = AircraftInfoPanel._parse_ids
        assert parse("101720, 101718") == [101720, 101718]
        assert parse("100  200 abc 300,") == [100, 200, 300]
        assert parse("") == []


class TestLandingRowsInLoadCases:
    """BDF SUBCASE 없는 착륙 결과 행 — 결과 메타로 표 컬럼 채움."""

    def test_summarize_uses_result_meta(self):
        import numpy as np
        from ascent_load.bdf.parser import parse_bdf
        from ascent_load.gui.load_cases import summarize_cases
        from ascent_load.output.result_data import ResultData, SubcaseResult

        model = parse_bdf(str(VM_DIR / "vm1_rod_axial.bdf"))
        sc = SubcaseResult(subcase_id=9001)
        sc.nodal_combined_forces = {1: np.zeros(6)}
        sc.label = "Level landing 3pt nz=3.08 MTOW"
        sc.category = "landing"
        sc.far_section = "§23.479"
        sc.nz_cg = 3.08
        sc.lift_factor = 2 / 3
        sc.weight_label = "MTOW"
        sc.trim_balance = {k: 0.0 for k in
                           ("Fx", "Fy", "Fz", "Mx", "My", "Mz")}
        rows = summarize_cases(model, ResultData(subcases=[sc]))
        row = [r for r in rows if r["subcase"] == 9001][0]
        assert row["desc"] == "Level landing 3pt nz=3.08 MTOW"
        assert "§23.479" in row["selectors"]
        assert "nz=3.08" in row["fixed"] and "MTOW" in row["fixed"]
        assert row["accel"].endswith("3.08")
        assert row["sums"] == [0.0] * 6


class TestElementTypeFilter:
    """3D 뷰 종류별 표시 — mesh_builder element_types 필터 검증."""

    def _model(self, tmp_path):
        from ascent_load.bdf.parser import parse_bdf

        deck = (
            "SOL 101\nCEND\nBEGIN BULK\n"
            "GRID    1               0.      0.      0.\n"
            "GRID    2               1.      0.      0.\n"
            "GRID    3               1.      1.      0.\n"
            "GRID    4               0.      1.      0.\n"
            "GRID    5               2.      0.      0.\n"
            "CQUAD4  10      1       1       2       3       4\n"
            "CTRIA3  11      1       2       5       3\n"
            "CROD    12      2       2       5\n"
            "ENDDATA\n")
        p = tmp_path / "m.bdf"
        p.write_text(deck)
        return parse_bdf(str(p))

    def test_type_filter(self, tmp_path):
        pv = pytest.importorskip("pyvista")  # noqa: F841
        from ascent_load.visualization.mesh_builder import build_structural_mesh

        m = self._model(tmp_path)
        assert build_structural_mesh(m).n_cells == 3
        only_quad = build_structural_mesh(m, element_types={"CQUAD4"})
        assert only_quad.n_cells == 1
        # 필터가 주어지면 include_beams보다 멤버십이 우선한다
        rod = build_structural_mesh(m, include_beams=False,
                                    element_types={"CROD"})
        assert rod.n_cells == 1
        none = build_structural_mesh(m, element_types=set())
        assert none.n_cells == 0

    def test_beam_tube_type_filter(self, tmp_path):
        pv = pytest.importorskip("pyvista")  # noqa: F841
        from ascent_load.visualization.mesh_builder import build_beam_tubes

        m = self._model(tmp_path)
        assert build_beam_tubes(m) is not None
        assert build_beam_tubes(m, element_types=set()) is None


class TestPload4Available:
    """Design Loads 탭 PLOAD4 버튼 활성 조건."""

    def _results(self, with_aero=True):
        import numpy as np
        from ascent_load.output.result_data import ResultData, SubcaseResult

        sc = SubcaseResult(subcase_id=1)
        if with_aero:
            sc.aero_forces = np.ones((4, 3))
        return ResultData(title="t", subcases=[sc])

    def test_requires_caero_and_aero_forces(self):
        from types import SimpleNamespace
        from ascent_load.gui.design_loads import pload4_available

        model = SimpleNamespace(caero_panels={1001: object()})
        assert pload4_available(model, self._results(True))
        assert not pload4_available(model, self._results(False))
        assert not pload4_available(SimpleNamespace(caero_panels={}),
                                    self._results(True))
        assert not pload4_available(None, self._results(True))
        assert not pload4_available(model, None)


class TestResolveVtolFactory:
    """재트림 버튼의 vtol_factory 훅 — 점 표기 임포트 경로 해석."""

    def test_valid_path_resolves_to_callable(self):
        from ascent_load.gui.cert_cases import resolve_vtol_factory

        fn = resolve_vtol_factory(
            {"vtol_factory": "ascent_load.models.ilc8.make_ilc8_vtol_config"})
        assert callable(fn)
        vc = fn()
        assert len(vc.hover_rotors) == 8

    def test_missing_or_bad_key_returns_none(self):
        from ascent_load.gui.cert_cases import resolve_vtol_factory

        assert resolve_vtol_factory({}) is None
        assert resolve_vtol_factory(None) is None
        assert resolve_vtol_factory({"vtol_factory": ""}) is None
        assert resolve_vtol_factory({"vtol_factory": "no_dot"}) is None
        assert resolve_vtol_factory(
            {"vtol_factory": "ascent_load.models.ilc8.nope"}) is None
        assert resolve_vtol_factory(
            {"vtol_factory": "not.a.module.fn"}) is None

    def test_ilc8_cert_yaml_carries_factory(self):
        import os

        import yaml

        path = os.path.join(os.path.dirname(__file__), "validation",
                            "ILC8", "ilc8_cert_config.yaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        from ascent_load.gui.cert_cases import resolve_vtol_factory

        assert callable(resolve_vtol_factory(d))
