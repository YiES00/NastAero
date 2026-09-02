# In-line 발산 근접 추정기의 단위 시험: 전형 단면 폐형해와 시프트 관계 (T4-E11)
"""Unit tests for _estimate_divergence_proximity.

The 2-DOF typical section gives the classic closed form
    q_div = k_theta / (S a e),
and the shift relation mu = q / (q_div - q), margin = 1 + 1/mu is
checked on both sides of the crossing, plus nearest-crossing selection
and the failure path.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascent_load.solvers.sol144 import (_estimate_divergence_proximity,
                                        _fundamental_divergence_margin)

K_H, K_TH = 1.0e5, 200.0
S_A, ECC = 2.0, 0.5              # S*a and eccentricity e
Q_DIV = K_TH / (S_A * ECC)       # = 200.0

G_SP = np.array([[0.0, 1.0]])    # wash = theta
G_D = np.array([[1.0, ECC]])     # z at force point = h + e*theta


def _solve_fn_at(q):
    A_jj = np.array([[q * S_A]])
    Q_q = G_D.T @ A_jj @ G_SP
    K_eff = np.diag([K_H, K_TH]) - Q_q
    return (lambda rhs: np.linalg.solve(K_eff, rhs)), A_jj


def test_typical_section_pre_divergence():
    """q = 50 < q_div = 200: margin 4.0, mu = 1/3."""
    q = 50.0
    solve_fn, A_jj = _solve_fn_at(q)
    est = _estimate_divergence_proximity(solve_fn, G_SP, G_D, A_jj, q)
    assert est is not None and not est['complex_dominant']
    assert est['mu'] == pytest.approx(q / (Q_DIV - q), rel=1e-10)
    assert est['margin'] == pytest.approx(Q_DIV / q, rel=1e-10)
    assert est['q_div'] == pytest.approx(Q_DIV, rel=1e-10)


def test_typical_section_post_divergence():
    """q = 400 > q_div: mu = -2, margin = 0.5 — 후발산 플래그 가지."""
    q = 400.0
    solve_fn, A_jj = _solve_fn_at(q)
    est = _estimate_divergence_proximity(solve_fn, G_SP, G_D, A_jj, q)
    assert est['mu'] == pytest.approx(-2.0, rel=1e-10)
    assert est['margin'] == pytest.approx(0.5, rel=1e-10)
    assert est['margin'] < 1.0


def test_nearest_crossing_selected():
    """독립 2-교차(100, 1000) 사이 q=150에서 최근접(100)을 고른다."""
    k = np.array([200.0, 2000.0])
    a = np.array([2.0, 2.0])          # q_div_i = k_i / a_i = 100, 1000
    q = 150.0
    G_sp = np.eye(2)
    G_d = np.eye(2)
    A_jj = np.diag(q * a)
    K_eff = np.diag(k) - G_d.T @ A_jj @ G_sp
    est = _estimate_divergence_proximity(
        lambda rhs: np.linalg.solve(K_eff, rhs), G_sp, G_d, A_jj, q)
    # mu candidates: 150/(100-150) = -3 (nearest), 150/(1000-150) ~ 0.18
    assert est['mu'] == pytest.approx(-3.0, rel=1e-10)
    assert est['q_div'] == pytest.approx(100.0, rel=1e-10)
    assert est['margin'] == pytest.approx(100.0 / 150.0, rel=1e-10)


def test_solver_failure_returns_none():
    def bad(rhs):
        raise RuntimeError("factorization gone")
    est = _estimate_divergence_proximity(bad, G_SP, G_D,
                                         np.array([[1.0]]), 1.0)
    assert est is None


def test_solve_count_reported():
    q = 50.0
    solve_fn, A_jj = _solve_fn_at(q)
    est = _estimate_divergence_proximity(solve_fn, G_SP, G_D, A_jj, q)
    assert est['n_solves'] >= 1


# ---- 기본(최소) 발산 여유: 비시프트 구조-분해 경로 ----

def _solve_k_fn():
    K = np.diag([K_H, K_TH])
    return lambda rhs: np.linalg.solve(K, rhs)


def test_fundamental_typical_section():
    """전형 단면: 기본 여유 = q_div/q, 작동점 양측에서 동일 공식."""
    for q in (50.0, 400.0):
        A_jj = np.array([[q * S_A]])
        est = _fundamental_divergence_margin(_solve_k_fn(), G_SP, G_D,
                                             A_jj, q)
        assert est['margin'] == pytest.approx(Q_DIV / q, rel=1e-10)
        assert est['q_div'] == pytest.approx(Q_DIV, rel=1e-10)


def test_fundamental_picks_smallest_crossing():
    """교차 2개(100, 1000) 중 기본 지표는 최소(100)를 고른다 —
    최근접 교차 지표와 의미론이 갈라지는 지점."""
    k = np.array([200.0, 2000.0])
    a = np.array([2.0, 2.0])          # q_div_i = 100, 1000
    q = 150.0                          # 100을 이미 지난 작동점
    G_sp = np.eye(2)
    G_d = np.eye(2)
    A_jj = np.diag(q * a)
    K = np.diag(k)
    est = _fundamental_divergence_margin(
        lambda rhs: np.linalg.solve(K, rhs), G_sp, G_d, A_jj, q)
    assert est['q_div'] == pytest.approx(100.0, rel=1e-10)
    assert est['margin'] == pytest.approx(100.0 / 150.0, rel=1e-10)
    assert est['margin'] < 1.0        # 기본 여유는 후발산을 정확히 플래그


def test_fundamental_failure_returns_none():
    def bad(rhs):
        raise RuntimeError("no factorization")
    assert _fundamental_divergence_margin(bad, G_SP, G_D,
                                          np.array([[1.0]]), 1.0) is None
