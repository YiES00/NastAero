# PSHELL 필드 반영과 CTRIA6 전단 적분 시험 (2026-08 감사)
"""쉘 물성 필드와 2차 삼각형 요소의 전단 랭크 검증.

감사에서 확인된 결함.
- PSHELL 12I/T^3(굽힘 관성비)과 NSM(단위면적당 비구조 질량)이
  파싱만 되고 강성·질량 어느 경로에도 반영되지 않았다.
- CTRIA6의 횡전단을 1점 축약으로 적분해 판 부분에 스퓨리어스
  영에너지 모드가 4개 남았다(조립하면 계가 특이해진다).
- PAERO1이 구조 물성과 ID 공간을 공유해 PID가 겹치면 구조 물성을
  덮어쓰고 그 요소가 조용히 조립에서 빠졌다.
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pytest
from nastaero.bdf.parser import parse_bdf
from nastaero.elements.quad4 import CQuad4Element
from nastaero.elements.tria6 import CTria6Element
from nastaero.solvers.sol101 import solve_static

_QUAD = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                  [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]])


class TestPSHELLBendingRatio:
    def test_ratio_scales_bending_stiffness(self):
        """12I/T^3는 굽힘 구성행렬에 곱해져야 한다."""
        # K(r12) = K_막전단 + r12 * K_굽힘 이므로 r12에 대해 아핀이다.
        # (회전 대각에는 횡전단도 섞여 있어 단순 배율 비교는 안 된다.)
        def k(r12):
            return CQuad4Element(_QUAD, 70000.0, 0.3, 2.0, 2.7e-9,
                                 r12=r12).stiffness_matrix()
        k0, k1, k8 = k(0.0), k(1.0), k(8.0)
        np.testing.assert_allclose(k8 - k1, 7.0 * (k1 - k0), rtol=1e-10,
                                    atol=1e-9)
        assert np.linalg.norm(k1 - k0) > 0.0, "굽힘 항이 존재해야 한다"

    def test_default_ratio_preserves_results(self):
        plain = CQuad4Element(_QUAD, 70000.0, 0.3, 2.0, 2.7e-9)
        explicit = CQuad4Element(_QUAD, 70000.0, 0.3, 2.0, 2.7e-9, r12=1.0)
        np.testing.assert_allclose(explicit.stiffness_matrix(),
                                    plain.stiffness_matrix(), atol=1e-12)

    def test_ratio_changes_solved_deflection(self):
        """전 경로(파싱->조립->해석)에서 반영되는지 확인한다."""
        def deck(r12):
            return f"""SOL 101
CEND
SPC = 1
LOAD = 1
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               0.      10.     0.
GRID    3               50.     10.     0.
GRID    4               50.     0.      0.
GRID    5               100.    10.     0.
GRID    6               100.    0.      0.
CQUAD4  10      1       1       2       3       4
CQUAD4  11      1       4       3       5       6
PSHELL  1       1       2.0     1       {r12:8s}1
MAT1    1       70000.          0.33    2.7-9
SPC1    1       123456  1       2
FORCE   1       5               500.    0.      0.      1.
FORCE   1       6               500.    0.      0.      1.
ENDDATA
"""
        def run(r12):
            f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
            f.write(deck(r12))
            f.close()
            try:
                res = solve_static(parse_bdf(f.name))
                return abs(res.subcases[0].displacements[5][2])
            finally:
                os.unlink(f.name)
        assert run("") / run("8.0") > 5.0


class TestPSHELLNSM:
    def test_nsm_adds_to_element_mass(self):
        plain = CQuad4Element(_QUAD, 70000.0, 0.3, 2.0, 2.7e-9)
        heavy = CQuad4Element(_QUAD, 70000.0, 0.3, 2.0, 2.7e-9, nsm=1.0e-6)
        assert np.trace(heavy.mass_matrix()) > np.trace(plain.mass_matrix())

    def test_nsm_reaches_trim_node_masses(self):
        from nastaero.loads_analysis.trim_loads import compute_node_masses
        deck = """SOL 101
CEND
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               0.      10.     0.
GRID    3               10.     10.     0.
GRID    4               10.     0.      0.
CQUAD4  10      1       1       2       3       4
PSHELL  1       1       2.0     1               1               {nsm}
MAT1    1       70000.          0.33    2.7-9
ENDDATA
"""
        def total(nsm):
            f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
            f.write(deck.format(nsm=nsm))
            f.close()
            try:
                return sum(compute_node_masses(parse_bdf(f.name)).values())
            finally:
                os.unlink(f.name)
        base = total("")
        with_nsm = total("1.0-6")
        # 면적 100 mm^2 x 1e-6 = 1e-4 추가
        assert with_nsm == pytest.approx(base + 1.0e-4, rel=1e-6)


class TestCTRIA6ShearRank:
    def _element(self):
        v = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        mid = np.array([(v[0] + v[1]) / 2, (v[1] + v[2]) / 2,
                        (v[2] + v[0]) / 2])
        return CTria6Element(np.vstack([v, mid]), 70000.0, 0.3, 2.0, 2.7e-9)

    def test_plate_block_has_no_spurious_mechanism(self):
        """판 부분(w, rx, ry) 18x18의 랭크는 15여야 한다 (강체 3개만 영)."""
        k = self._element().stiffness_matrix()
        k = (k + k.T) / 2
        plate = [6 * i + c for i in range(6) for c in (2, 3, 4)]
        kp = k[np.ix_(plate, plate)]
        w = np.linalg.eigvalsh((kp + kp.T) / 2)
        rank = int(np.sum(np.abs(w) > np.max(np.abs(kp)) * 1e-12))
        assert rank == 15, f"판 랭크 {rank} (스퓨리어스 기구 {15 - rank}개)"

    def test_only_rigid_body_modes_are_zero(self):
        k = self._element().stiffness_matrix()
        k = (k + k.T) / 2
        w = np.linalg.eigvalsh(k)
        n_zero = int(np.sum(np.abs(w) < np.max(np.abs(k)) * 1e-12))
        assert n_zero == 6, f"영에너지 모드 {n_zero}개 (강체 6개여야 함)"


class TestPAEROIdNamespace:
    def test_paero_does_not_clobber_structural_property(self):
        deck = """SOL 101
CEND
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               0.      10.     0.
GRID    3               10.     10.     0.
GRID    4               10.     0.      0.
CQUAD4  10      1       1       2       3       4
PSHELL  1       1       2.0     1
MAT1    1       70000.          0.33    2.7-9
PAERO1  1
ENDDATA
"""
        f = tempfile.NamedTemporaryFile("w", suffix=".bdf", delete=False)
        f.write(deck)
        f.close()
        try:
            model = parse_bdf(f.name)
            assert type(model.properties[1]).__name__ == "PSHELL"
            assert 1 in model.paeros
        finally:
            os.unlink(f.name)
