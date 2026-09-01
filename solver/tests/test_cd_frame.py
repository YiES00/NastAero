# CD(변위 좌표계) 처리 일관성 시험 (2026-08 감사)
"""CD != 0 절점이 관여할 때의 좌표계 규약 검증.

감사에서 확인된 결함 3건.
- CBAR/CBEAM의 X1X2X3는 OFFT 첫 글자가 'B'가 아닌 한 GA의 변위
  좌표계 성분인데(기본값 GGG) 기본좌표계로 해석하고 있었다.
- FORCE/MOMENT/GRAV의 방향 벡터를 정규화하고(MSC는 f = F*N),
  CID를 적용하지 않았으며, 조립된 CD 성분 자유도로 옮기지도 않았다.
- RBE2/MPC 구속 계수를 기본좌표계로 쓰면서 CD 성분 자유도에
  적용해 프레임이 섞였다.
GACOMP 비교 모델은 CD != 0 절점이 72개이고 그중 17개가 RBE2
종속이다.
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest
from ascent_load.bdf.parser import parse_bdf
from ascent_load.solvers.sol101 import solve_static

# CORD2R 7: ez = 기본 y, ex = 기본 x  =>  CD 성분 (0,0,1) = 기본 (0,1,0)
_CORD = """CORD2R  7               0.      0.      0.      0.      1.      0.      +
+       1.      0.      0."""


def _solve(deck_text, node):
    f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
    f.write(deck_text)
    f.close()
    try:
        model = parse_bdf(f.name)
        res = solve_static(model)
        return np.linalg.norm(res.subcases[0].displacements[node][:3])
    finally:
        os.unlink(f.name)


def _cantilever(cd, x, offt=""):
    """I1=1e3, I2=1e5인 캔틸레버 — 굽힘 평면이 100배를 가른다."""
    return f"""SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
{_CORD}
GRID    1               0.      0.      0.      {cd}
GRID    2               100.    0.      0.      {cd}
CBAR    10      1       1       2       {x[0]}      {x[1]}      {x[2]}      {offt}
PBAR    1       1       10.     1000.   100000. 100.
MAT1    1       70000.          0.33    2.7-9
SPC1    1       123456  1
FORCE   1       2               1000.   0.      0.      1.
ENDDATA
"""


class TestBarOrientationInCD:
    # PL^3/(3 E I): I1=1e3 -> 4.7619, I2=1e5 -> 0.047619
    SOFT = 1000.0 * 100.0**3 / (3 * 70000.0 * 1000.0)
    STIFF = 1000.0 * 100.0**3 / (3 * 70000.0 * 100000.0)

    def test_basic_frame_reference(self):
        assert _solve(_cantilever(0, ("0.", "0.", "1.")), 2) == \
            pytest.approx(self.SOFT, rel=1e-6)

    def test_cd_components_rotate_orientation(self):
        """CD=7에서 X=(0,0,1)은 기본 (0,1,0)이므로 굽힘 평면이 바뀐다."""
        assert _solve(_cantilever(7, ("0.", "0.", "1.")), 2) == \
            pytest.approx(self.STIFF, rel=1e-6)

    def test_offt_b_keeps_basic_frame(self):
        """OFFT 첫 글자 B는 방향 벡터를 기본좌표계로 고정한다."""
        assert _solve(_cantilever(7, ("0.", "0.", "1."), "BGG"), 2) == \
            pytest.approx(self.SOFT, rel=1e-6)


class TestLoadFrames:
    def test_force_direction_not_normalized(self):
        """MSC 규약 f = F*N — 방향 벡터를 정규화하지 않는다."""
        base = _cantilever(0, ("0.", "0.", "1."))
        doubled = base.replace(
            "FORCE   1       2               1000.   0.      0.      1.",
            "FORCE   1       2               1000.   0.      0.      2.")
        assert _solve(doubled, 2) == pytest.approx(2.0 * _solve(base, 2),
                                                   rel=1e-9)

    def test_force_cid_is_applied(self):
        """CID가 지정되면 그 좌표계 성분으로 해석해야 한다."""
        basic_y = _cantilever(0, ("0.", "0.", "1.")).replace(
            "FORCE   1       2               1000.   0.      0.      1.",
            "FORCE   1       2               1000.   0.      1.      0.")
        cid_z = _cantilever(0, ("0.", "0.", "1.")).replace(
            "FORCE   1       2               1000.   0.      0.      1.",
            "FORCE   1       2       7       1000.   0.      0.      1.")
        # CID=7 성분 (0,0,1) = 기본 (0,1,0)
        assert _solve(cid_z, 2) == pytest.approx(_solve(basic_y, 2), rel=1e-9)


class TestRBE2InCD:
    def _frame(self, cd):
        return f"""SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
{_CORD}
GRID    1               0.      0.      0.      0
GRID    2               100.    0.      0.      {cd}
GRID    3               100.    50.     0.      {cd}
GRID    4               200.    50.     0.      0
CBAR    10      1       1       2       0.      0.      1.      BGG
CBAR    11      1       3       4       0.      0.      1.      BGG
RBE2    20      2       123456  3
PBAR    1       1       10.     1000.   100000. 100.
MAT1    1       70000.          0.33    2.7-9
SPC1    1       123456  1
FORCE   1       4               1000.   0.      0.      1.
ENDDATA
"""

    def test_rbe2_result_is_cd_invariant(self):
        """RBE2가 걸린 절점의 CD를 바꿔도 물리 결과는 같아야 한다."""
        u0 = _solve(self._frame(0), 4)
        u7 = _solve(self._frame(7), 4)
        assert u7 == pytest.approx(u0, rel=1e-9), (
            f"CD=0 {u0:.6f} vs CD=7 {u7:.6f}")
