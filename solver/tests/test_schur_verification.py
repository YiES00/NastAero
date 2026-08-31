# Schur 축약 트림해의 시스템 수준 검증(r3 MC7): 직접 전체해 대조와 모델 평행이동 불변성
from __future__ import annotations
import os
import numpy as np
import pytest

from nastaero.bdf.parser import BDFParser
from nastaero.solvers import sol144
from nastaero.solvers.sol144 import solve_trim

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
GOLAND_BDF = os.path.join(VALIDATION_DIR, "goland_wing", "goland_static.bdf")


def _parse(path):
    return BDFParser().parse(path)


def _first_subcase(results):
    return results.subcases[0]


class TestDirectVsSchur:
    """같은 모델을 조밀 전체계(모놀리식 lstsq)와 Schur 축약 경로로
    각각 풀어 트림 변수와 변위장이 일치하는지 본다."""

    def test_goland_dense_equals_schur(self, monkeypatch):
        model_a = _parse(GOLAND_BDF)
        res_dense = solve_trim(model_a)
        sc_d = _first_subcase(res_dense)

        monkeypatch.setattr(sol144, "DENSE_DOF_LIMIT", 0)
        model_b = _parse(GOLAND_BDF)
        res_schur = solve_trim(model_b)
        sc_s = _first_subcase(res_schur)

        for name, v in sc_d.trim_variables.items():
            assert sc_s.trim_variables[name] == pytest.approx(
                v, rel=1e-6, abs=1e-12), name

        nids = sorted(sc_d.displacements)
        u_d = np.array([sc_d.displacements[n] for n in nids])
        u_s = np.array([sc_s.displacements[n] for n in nids])
        scale = max(np.abs(u_d).max(), 1e-12)
        # 관측 편차는 상대 2e-8 수준(lstsq vs LU 반올림) — 1e-6이면
        # 정식화 오류는 걸러지고 산술 반올림은 통과한다.
        assert np.max(np.abs(u_s - u_d)) / scale < 1e-6


class TestTranslationInvariance:
    """모델 전체(GRID + CAERO 기하)를 상수 벡터로 평행이동해도
    트림 변수와 변위장은 동일해야 한다. 기준점·모멘트 팔 계산이
    절대좌표에 은닉 의존하면 여기서 깨진다."""

    @staticmethod
    def _translate(model, t):
        for node in model.nodes.values():
            node.xyz = node.xyz + t
            node.xyz_global = node.xyz_global + t
        for caero in model.caero_panels.values():
            caero.p1 = caero.p1 + t
            caero.p4 = caero.p4 + t
        return model

    def test_goland_translation(self):
        base = solve_trim(_parse(GOLAND_BDF))
        sc0 = _first_subcase(base)

        t = np.array([5000.0, 300.0, -200.0])
        model = self._translate(_parse(GOLAND_BDF), t)
        moved = solve_trim(model)
        sc1 = _first_subcase(moved)

        for name, v in sc0.trim_variables.items():
            assert sc1.trim_variables[name] == pytest.approx(
                v, rel=1e-6, abs=1e-10), name

        nids = sorted(sc0.displacements)
        u0 = np.array([sc0.displacements[n] for n in nids])
        u1 = np.array([sc1.displacements[n] for n in nids])
        scale = max(np.abs(u0).max(), 1e-12)
        assert np.max(np.abs(u1 - u0)) / scale < 1e-6
