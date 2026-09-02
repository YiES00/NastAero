# 회전 컴플라이언스 수 Π 진단 헬퍼의 단위 시험 (T2-E7)
"""Unit tests for the rotation-compliance screen (_rotation_compliance_qdiv).

Covers:
  * scalar closed form — one box, one rotation DOF: q_hat = k / (A g_w g_d)
  * no rotation columns → inf (surface-construction G_w)
  * nonpositive diagonal entries are excluded from the surrogate
  * FSW ha144a integration smoke: rotation slope mode runs the screen
    without exception and logs it
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from ascent_load.solvers.sol144 import _rotation_compliance_qdiv

HERE = os.path.dirname(os.path.abspath(__file__))
FSW_BDF = os.path.join(HERE, "validation", "FSW", "ha144a.bdf")


def test_scalar_closed_form():
    """단일 박스·단일 회전 자유도: q_hat = k / (A * gw * gd)."""
    k = 250.0
    gw = np.array([[0.0, 0.8]])      # col 0 = translation(0), col 1 = rotation
    gd = np.array([[0.5, 1.2]])
    A = np.array([[3.0]])
    q_hat = _rotation_compliance_qdiv(gw, gd, np.array([1e9, k]), A)
    assert q_hat == pytest.approx(k / (3.0 * 0.8 * 1.2), rel=1e-12)


def test_no_rotation_columns_is_inf():
    """표면 구성처럼 G_w가 전부 0열이면 대용이 정의되지 않는다 → inf."""
    gw = np.zeros((3, 4))
    gd = np.random.default_rng(1).normal(size=(3, 4))
    A = np.eye(3)
    assert _rotation_compliance_qdiv(gw, gd, np.ones(4), A) == np.inf


def test_nonpositive_diagonal_excluded():
    """k<=0 자유도는 대용에서 제외된다(무한 강성으로 처리)."""
    gw = np.array([[1.0, 1.0]])
    gd = np.array([[1.0, 1.0]])
    A = np.array([[2.0]])
    # 두 번째 자유도의 k<=0 → 첫 자유도만 남아 q_hat = k1/(A*1*1)
    q_hat = _rotation_compliance_qdiv(gw, gd, np.array([10.0, -5.0]), A)
    assert q_hat == pytest.approx(10.0 / 2.0, rel=1e-12)


def test_diagonal_surrogate_is_not_one_sided():
    """r5 심사 반례(2x2 SPD): 대각 대용은 일반적 단방향 경계가 아니다.

    K=[[1,.9],[.9,1]] (SPD), g_w=g_d=[1,1], A=1 에서 참 루프 이득은
    g K^-1 g^T = 2/1.9 -> q_div = 0.95 이지만 대각 대용은
    g diag(1/k) g^T = 2 -> q_hat = 0.5 < q_div. 즉 q=0.6 에서
    Pi = 1.2 >= 1 인데 실제로는 아직 발산 전 — 거짓 양성.
    Pi 는 경험적 지표이며 무오경보가 아님을 이 시험이 고정한다."""
    K = np.array([[1.0, 0.9], [0.9, 1.0]])
    g = np.array([[1.0, 1.0]])
    A = np.array([[1.0]])
    q_hat = _rotation_compliance_qdiv(g, g, np.diag(K), A)
    q_div_true = 1.0 / float((g @ np.linalg.inv(K) @ g.T)[0, 0])
    assert q_hat == pytest.approx(0.5, rel=1e-12)
    assert q_div_true == pytest.approx(0.95, rel=1e-12)
    assert q_hat < q_div_true          # 대용이 과소평가 → 거짓 양성 가능


def test_parallel_paths_stack():
    """동일 강성 회전 자유도 n개의 병렬 루프는 이득을 n배로 쌓는다."""
    n = 4
    gw = np.ones((1, n))
    gd = np.ones((1, n))
    A = np.array([[1.0]])
    k = 8.0
    q_hat = _rotation_compliance_qdiv(gw, gd, np.full(n, k), A)
    assert q_hat == pytest.approx(k / n, rel=1e-12)


@pytest.mark.skipif(not os.path.exists(FSW_BDF), reason="FSW deck missing")
def test_fsw_rotation_mode_runs_screen(caplog):
    """FSW ha144a에서 rotation 모드 shared-data 구축이 스크린을 수행한다."""
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.solvers.sol144 import _build_shared_data

    model = parse_bdf(FSW_BDF)
    with caplog.at_level(logging.DEBUG, logger="ascent_load"):
        shared = _build_shared_data(model, spline_slope_method="rotation")
    assert shared is not None
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "rotation-compliance" in joined
