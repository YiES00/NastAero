# SUPORT 카드 파싱과 자유-자유 트림 기준(마운트) 선정을 검증하는 시험
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nastaero.bdf.cards.constraints import SUPORT
from nastaero.bdf.parser import parse_bdf
from nastaero.solvers.sol144 import _suport_mount_idx

DECK = (Path(__file__).parent / "validation" / "ILC8"
        / "ilc8_msc_sol144_v8_shellbend.bdf")


class TestSuportCard:
    def test_parses_grid_component_pairs(self):
        s = SUPORT.from_fields(
            ["SUPORT", "101519", "123", "101719", "23", "101501", "3"])
        assert s.entries == [(101519, "123"), (101719, "23"), (101501, "3")]

    def test_dof_count_is_sum_of_components(self):
        s = SUPORT.from_fields(
            ["SUPORT", "101519", "123", "101719", "23", "101501", "3"])
        assert s.dof_count == 6

    def test_blank_component_defaults_to_all_six(self):
        s = SUPORT.from_fields(["SUPORT", "42", ""])
        assert s.entries == [(42, "123456")]
        assert s.dof_count == 6

    def test_trailing_blank_pairs_ignored(self):
        s = SUPORT.from_fields(["SUPORT", "7", "123", "", "", "", ""])
        assert s.entries == [(7, "123")]

    def test_non_dof_characters_not_counted(self):
        # 0 은 유효 성분이 아니다 (성분은 1~6)
        s = SUPORT.from_fields(["SUPORT", "7", "1203"])
        assert s.dof_count == 3

    def test_type_label(self):
        assert SUPORT().type == "SUPORT"


class TestSuportParsedFromDeck:
    def test_deck_suport_is_collected(self):
        model = parse_bdf(str(DECK))
        assert len(model.suports) == 1
        assert model.suports[0].entries == [
            (101519, "123"), (101719, "23"), (101501, "3")]

    def test_deck_suport_fixes_six_rigid_body_dofs(self):
        model = parse_bdf(str(DECK))
        assert sum(s.dof_count for s in model.suports) == 6


class _StubDofMgr:
    """절점·성분을 유일한 정수 자유도로 사상하는 최소 스텁."""

    def get_dof(self, nid: int, comp: int) -> int:
        return nid * 10 + comp


class TestSuportMountSelection:
    def _model_with(self, entries):
        model = parse_bdf(str(DECK))
        sup = SUPORT()
        sup.entries = entries
        model.suports = [sup]
        return model

    def test_returns_none_without_suport(self):
        model = parse_bdf(str(DECK))
        model.suports = []
        assert _suport_mount_idx(model, _StubDofMgr(), {}) is None

    def test_maps_deck_suport_to_free_set_indices(self):
        model = self._model_with([(1, "123"), (2, "23"), (3, "3")])
        mgr = _StubDofMgr()
        dofs = [mgr.get_dof(1, c) for c in (1, 2, 3)]
        dofs += [mgr.get_dof(2, c) for c in (2, 3)]
        dofs += [mgr.get_dof(3, 3)]
        f_index = {d: i for i, d in enumerate(dofs)}
        idx = _suport_mount_idx(model, mgr, f_index)
        assert idx is not None
        assert sorted(idx.tolist()) == list(range(6))

    def test_skips_dofs_outside_the_free_set(self):
        model = self._model_with([(1, "123")])
        mgr = _StubDofMgr()
        # T3 만 자유 — 나머지는 이미 구속된 상황
        f_index = {mgr.get_dof(1, 3): 0}
        idx = _suport_mount_idx(model, mgr, f_index)
        assert idx is not None
        assert idx.tolist() == [0]

    def test_returns_none_when_no_suport_dof_is_free(self):
        model = self._model_with([(1, "123")])
        assert _suport_mount_idx(model, _StubDofMgr(), {999: 0}) is None

    def test_duplicate_dofs_are_not_repeated(self):
        model = self._model_with([(1, "3"), (1, "3")])
        mgr = _StubDofMgr()
        f_index = {mgr.get_dof(1, 3): 0}
        idx = _suport_mount_idx(model, mgr, f_index)
        assert idx.tolist() == [0]


class TestSuportChangesTheReferenceFrame:
    """마운트 기준이 바뀌면 상대 변위가 바뀐다 — SUPORT 구현의 요점."""

    @pytest.mark.slow
    def test_deck_suport_is_used_for_the_trim_mount(self, caplog):
        pytest.importorskip("scipy")
        from nastaero.solvers.sol144 import solve_trim
        model = parse_bdf(str(DECK))
        with caplog.at_level("INFO"):
            solve_trim(model)
        assert any("Deck SUPORT mount" in r.message for r in caplog.records)
