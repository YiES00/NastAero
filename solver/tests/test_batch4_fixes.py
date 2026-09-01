# PBEAM·ZOFFS·회전경로 theta_y 부호 회귀 시험 (2026-08 감사 4차)
"""감사 잔여 결함 3건의 회귀 시험.

- PBEAM(표 형식 보 물성)이 미지원이라 CBEAM+PBEAM 덱의 요소가
  통째로 빠졌다. MSC 필드 순서는 PBAR와 달라 I12가 J보다 앞에 온다.
- 쉘 ZOFFS가 파싱만 되고 강성/질량에 반영되지 않아 오프셋된 중면의
  막-굽힘 결합이 사라졌다.
- 회전/보-스플라인 경로의 theta_y 부호가 반대였다. u = theta x r
  에서 u_z = -theta_y*dx 이므로 워시 dz/dx = -theta_y다. 이 경로는
  SPLINE2·공선 SET1(보-스틱 모델)·스플라인 없는 대체 경로에서
  주 경로로 쓰인다(Goland 검증 덱이 해당).
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest
from ascent_load.bdf.parser import parse_bdf
from ascent_load.solvers.sol101 import solve_static

MAT = "MAT1    1       70000.          0.33    2.7-9"


def _solve(deck_text):
    f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
    f.write(deck_text)
    f.close()
    try:
        model = parse_bdf(f.name)
        return model, solve_static(model)
    finally:
        os.unlink(f.name)


class TestPBEAM:
    DECK = f"""SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               100.    0.      0.
CBEAM   10      1       1       2       0.      0.      1.
PBEAM   1       1       10.     1000.   100000. 0.      100.
{MAT}
SPC1    1       123456  1
FORCE   1       2               1000.   0.      0.      1.
ENDDATA
"""

    def test_field_order_puts_j_after_i12(self):
        """MSC PBEAM은 PID MID A I1 I2 I12 J NSM 순서다."""
        model, _ = _solve(self.DECK)
        p = model.properties[1]
        assert p.A == pytest.approx(10.0)
        assert p.I1 == pytest.approx(1000.0)
        assert p.I2 == pytest.approx(100000.0)
        assert p.I12 == pytest.approx(0.0)
        assert p.J == pytest.approx(100.0), "J가 I12 자리에서 읽히면 안 된다"

    def test_element_is_assembled(self):
        """CBEAM+PBEAM 덱의 요소가 조립되어 이론 처짐을 내야 한다."""
        _, res = _solve(self.DECK)
        uz = res.subcases[0].displacements[2][2]
        E, I1, L, P = 70000.0, 1000.0, 100.0, 1000.0
        assert uz == pytest.approx(P * L**3 / (3 * E * I1), rel=1e-6)


class TestShellZOFFS:
    def _deck(self, zoffs):
        return f"""SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               0.      10.     0.
GRID    3               100.    10.     0.
GRID    4               100.    0.      0.
CQUAD4  10      1       1       2       3       4       0.      {zoffs}
PSHELL  1       1       2.      1
{MAT}
SPC1    1       123456  1       2
FORCE   1       3               500.    1.      0.      0.
FORCE   1       4               500.    1.      0.      0.
ENDDATA
"""

    def test_zero_offset_has_no_membrane_bending_coupling(self):
        """오프셋이 없으면 면내 하중이 면외 변위를 만들지 않는다."""
        _, res = _solve(self._deck("0."))
        assert abs(res.subcases[0].displacements[3][2]) < 1e-9

    def test_offset_creates_membrane_bending_coupling(self):
        """오프셋된 중면에서는 면내 하중이 굽힘을 유발한다."""
        _, res = _solve(self._deck("20."))
        d = res.subcases[0].displacements[3]
        assert abs(d[2]) > 1.0, f"면외 변위가 생겨야 한다: uz={d[2]}"
        assert abs(d[4]) > 1e-3, f"회전이 생겨야 한다: ry={d[4]}"

    def test_offset_sign_flips_bending_direction(self):
        """오프셋 부호가 뒤집히면 굽힘 방향도 뒤집힌다."""
        _, r_pos = _solve(self._deck("20."))
        _, r_neg = _solve(self._deck("-20."))
        up = r_pos.subcases[0].displacements[3][2]
        un = r_neg.subcases[0].displacements[3][2]
        assert up * un < 0, f"부호가 반대여야 한다: {up} vs {un}"


class TestBeamPathThetaYSign:
    """보-스플라인(회전) 경로의 theta_y 부호.

    Goland 덱은 SET1이 공선이라 이 경로를 주 경로로 사용한다.
    """

    def test_goland_trim_balances_weight(self):
        from ascent_load.solvers.sol144 import solve_trim
        deck = os.path.join(os.path.dirname(__file__), "validation",
                            "goland_wing", "goland_static.bdf")
        model = parse_bdf(deck)
        res = solve_trim(model, n_workers=0)
        sc = res.subcases[0]
        total_lift = float(np.sum(np.real(sc.aero_forces[:, 2])))
        assert total_lift > 0, "양력이 양수여야 한다"
        alpha = sc.trim_variables["ANGLEA"]
        assert 0.0 < np.degrees(alpha) < 15.0, (
            f"1g 트림 받음각이 물리적 범위여야 한다: {np.degrees(alpha):.3f} deg")

    def test_wash_sign_matches_rigid_pitch_kinematics(self):
        """u = theta x r 에서 u_z = -theta_y*dx 이므로 워시는 -theta_y다.

        보-스플라인 경로의 G_w 열 부호가 이 운동학과 맞는지 본다.
        """
        from ascent_load.solvers.sol144 import _fill_geff
        from ascent_load.fem.dof_manager import DOFManager

        n_box, n_node = 1, 1
        dof_mgr = DOFManager([7])
        f_dofs = list(dof_mgr.get_node_dofs(7))
        f_dof_index = {d: i for i, d in enumerate(f_dofs)}
        G_w = np.zeros((n_box, len(f_dofs)))
        G_d = np.zeros((n_box, len(f_dofs)))
        _fill_geff(G_w, G_d,
                   np.ones((n_box, n_node)), np.ones((n_box, n_node)),
                   [0], [7],
                   force_pts=np.array([[100.0, 0.0, 0.0]]),
                   struct_xyz=np.array([[0.0, 0.0, 0.0]]),
                   dof_mgr=dof_mgr, f_dof_index=f_dof_index)
        ry_col = f_dof_index[dof_mgr.get_dof(7, 5)]
        assert G_w[0, ry_col] < 0, "워시 = -theta_y 여야 한다"
        assert G_d[0, ry_col] < 0, "힘 작용점 z = -theta_y*dx 여야 한다"
