# BDF 파서 규약 시험 (2026-08 감사)
"""MSC 입력 형식 규약 준수 검증.

감사에서 확인된 결함.
- 인라인 '$' 주석을 제거하지 않아 자유장 카드가 통째로 버려지고
  고정장 필드가 오염됐다.
- 탭 문자를 전개하지 않아 탭 구분 카드가 조용히 사라졌다.
- SPC1/RBE2의 THRU를 선두에서 한 번만 처리해 중간 범위의 내부
  ID가 통째로 빠졌다.
- EIGRL NORM 기본값이 MAX였다(MSC는 MASS).
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest
from ascent_load.bdf.parser import parse_bdf
from ascent_load.bdf.cards.constraints import SPC1
from ascent_load.bdf.cards.sets import SET1
from ascent_load.bdf.cards.rbe import RBE2
from ascent_load.bdf.cards.eigrl import EIGRL
from ascent_load.bdf.field_parser import expand_thru


def _parse(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
    f.write(text)
    f.close()
    try:
        return parse_bdf(f.name)
    finally:
        os.unlink(f.name)


class TestInlineComments:
    def test_free_field_card_with_trailing_comment(self):
        model = _parse("""SOL 101
CEND
BEGIN BULK
GRID,1,,0.,0.,0.  $ 뒤따르는 주석
GRID,2,,10.,0.,0.
ENDDATA
""")
        assert sorted(model.nodes) == [1, 2]
        np.testing.assert_allclose(model.nodes[1].xyz, [0.0, 0.0, 0.0])

    def test_full_line_comment_still_skipped(self):
        model = _parse("""SOL 101
CEND
BEGIN BULK
$ 전체 주석 줄
GRID    1               0.      0.      0.
ENDDATA
""")
        assert sorted(model.nodes) == [1]


class TestTabExpansion:
    def test_tab_delimited_card_is_parsed(self):
        model = _parse("SOL 101\nCEND\nBEGIN BULK\n"
                       "GRID\t1\t\t0.\t0.\t0.\n"
                       "GRID\t2\t\t20.\t0.\t0.\nENDDATA\n")
        assert sorted(model.nodes) == [1, 2]
        np.testing.assert_allclose(model.nodes[2].xyz, [20.0, 0.0, 0.0])


class TestThruExpansion:
    def test_helper_handles_interior_ranges(self):
        assert expand_thru(["10", "THRU", "14", "20", "30", "THRU", "32"]) == \
            [10, 11, 12, 13, 14, 20, 30, 31, 32]

    def test_spc1_interior_thru(self):
        spc = SPC1.from_fields(
            ["SPC1", "1", "123", "10", "THRU", "14", "20", "30", "THRU", "32"])
        assert spc.node_ids == [10, 11, 12, 13, 14, 20, 30, 31, 32]

    def test_set1_thru(self):
        assert SET1.from_fields(["SET1", "1", "5", "THRU", "8", "11"]).ids == \
            [5, 6, 7, 8, 11]

    def test_rbe2_thru(self):
        rbe = RBE2.from_fields(
            ["RBE2", "1", "100", "123456", "201", "THRU", "204"])
        assert rbe.dependent_nodes == [201, 202, 203, 204]


class TestEIGRLDefaults:
    def test_norm_defaults_to_mass(self):
        assert EIGRL.from_fields(["EIGRL", "1", "", "", "10"]).norm == "MASS"

    def test_explicit_norm_is_kept(self):
        e = EIGRL.from_fields(
            ["EIGRL", "1", "", "", "10", "", "", "", "MAX"])
        assert e.norm == "MAX"
