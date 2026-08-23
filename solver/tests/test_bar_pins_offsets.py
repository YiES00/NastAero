# CBAR/CBEAM 핀 플래그와 단부 오프셋 시험 (2026-08 감사)
"""핀(PA/PB)과 단부 오프셋(W1A~W3B)의 파싱과 강성 반영 검증.

감사에서 확인된 결함: 두 기능 모두 파싱조차 되지 않거나(W 필드)
파싱만 되고 조립에서 무시되어(PA/PB), 핀 조인트가 모멘트를
전달하고 오프셋된 부재가 절점 위에 놓인 것처럼 계산됐다.
GACOMP 비교 모델은 핀이 걸린 CBAR가 14개다.
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest
from nastaero.bdf.parser import parse_bdf
from nastaero.elements.bar import CBarElement
from nastaero.solvers.sol101 import solve_static

E, I1, L = 70000.0, 1000.0, 100.0
NU = 0.33
G = E / (2 * (1 + NU))
J = 100.0
_KW = dict(E=E, G=G, A=10.0, I1=I1, I2=100000.0, J=J)
_N1 = np.zeros(3)
_N2 = np.array([L, 0.0, 0.0])
_V = np.array([0.0, 0.0, 1.0])


def _solve(deck_text, node):
    f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
    f.write(deck_text)
    f.close()
    try:
        model = parse_bdf(f.name)
        res = solve_static(model)
        return model, res.subcases[0].displacements[node]
    finally:
        os.unlink(f.name)


class TestPinFlags:
    def test_far_end_moment_release_softens_to_3EI(self):
        """B단 회전을 풀면 횡강성이 12EI/L^3에서 3EI/L^3로 떨어진다."""
        plain = CBarElement(_N1, _N2, _V, rho=0.0, nsm=0.0, **_KW)
        pinned = CBarElement(_N1, _N2, _V, rho=0.0, nsm=0.0, pb=6, **_KW)
        k0 = plain._local_stiffness()
        kp = pinned._apply_pins(plain._local_stiffness())
        assert k0[7, 7] == pytest.approx(12 * E * I1 / L**3, rel=1e-9)
        assert kp[7, 7] == pytest.approx(3 * E * I1 / L**3, rel=1e-9)

    def test_released_dof_carries_no_force(self):
        pinned = CBarElement(_N1, _N2, _V, rho=0.0, nsm=0.0, pb=6, **_KW)
        kp = pinned._apply_pins(pinned._local_stiffness())
        np.testing.assert_allclose(kp[11, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(kp[:, 11], 0.0, atol=1e-12)

    def test_rigid_body_modes_survive_pin(self):
        """핀을 넣어도 강체 운동은 무응력이어야 한다."""
        pinned = CBarElement(_N1, _N2, _V, rho=0.0, nsm=0.0, pb=6, **_KW)
        k = pinned.stiffness_matrix()
        for comp in range(3):
            rb = np.zeros(12)
            rb[comp] = 1.0
            rb[6 + comp] = 1.0
            assert np.linalg.norm(k @ rb) < 1e-6

    def test_pin_flags_are_parsed(self):
        # PA/PB는 연속행 필드다 (첫 행은 OFFT까지 9개 필드)
        deck = """SOL 101
CEND
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               100.    0.      0.
CBAR    10      1       1       2       0.      0.      1.      GGG     +
+       6       5
PBAR    1       1       10.     1000.   100000. 100.
MAT1    1       70000.          0.33    2.7-9
ENDDATA
"""
        f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
        f.write(deck)
        f.close()
        try:
            model = parse_bdf(f.name)
            assert model.elements[10].pa == 6
            assert model.elements[10].pb == 5
        finally:
            os.unlink(f.name)


class TestEndOffsets:
    DECK = """SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               100.    0.      0.
CBAR    10      1       1       2       0.      0.      1.      GGG     +
+                       0.      50.     0.      0.      50.     0.
PBAR    1       1       10.     1000.   100000. 100.
MAT1    1       70000.          0.33    2.7-9
SPC1    1       123456  1
FORCE   1       2               1000.   0.      0.      1.
ENDDATA
"""

    def test_offsets_are_parsed(self):
        model, _ = _solve(self.DECK, 2)
        np.testing.assert_allclose(model.elements[10].wa, [0.0, 50.0, 0.0])
        np.testing.assert_allclose(model.elements[10].wb, [0.0, 50.0, 0.0])

    def test_rigid_arm_produces_torsion(self):
        """절점에 건 하중이 강체팔을 통해 부재에 비틀림을 만든다."""
        _, d = _solve(self.DECK, 2)
        expected = 1000.0 * 50.0 * L / (G * J)
        assert abs(d[3]) == pytest.approx(expected, rel=1e-6)

    def test_arm_contributes_to_tip_deflection(self):
        """팁 처짐 = 굽힘 + 팔 회전 기여(w * rx)."""
        _, d = _solve(self.DECK, 2)
        bending = 1000.0 * L**3 / (3 * E * I1)
        assert abs(d[2]) == pytest.approx(bending + 50.0 * abs(d[3]),
                                           rel=1e-6)

    def test_zero_offset_matches_plain_bar(self):
        plain = CBarElement(_N1, _N2, _V, rho=0.0, nsm=0.0, **_KW)
        zero = CBarElement(_N1, _N2, _V, rho=0.0, nsm=0.0,
                           wa=[0.0, 0.0, 0.0], wb=[0.0, 0.0, 0.0], **_KW)
        np.testing.assert_allclose(zero.stiffness_matrix(),
                                    plain.stiffness_matrix(), atol=1e-9)
