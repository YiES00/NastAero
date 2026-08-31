# BEMT 축각 규약 회귀 시험 (2026-08 검토 P0)
"""forward_flight의 alpha_shaft 규약과 호출부 좌표 변환 검증.

규약: alpha_shaft는 자유류와 로터 디스크면 사이 각이다.
  V_axial = V sin(alpha), V_plane = V cos(alpha)
  alpha = 90° → 순수 축류(축방향 상승), alpha = 0 → 순수 면내류.

검토에서 확인된 결함: 하중 케이스 생성기가 축각을 여각(90°−σ)으로
전달해 축류와 면내류가 뒤바뀌었다. 수직 리프트 로터의 전진비행이
축방향 상승으로, 순항 프로펠러(σ=90°)가 면내류로 풀려 토크가
조건에 따라 +55%~10배 어긋났다.
"""
from __future__ import annotations
import numpy as np
import pytest
from nastaero.models.ilc8 import make_ilc8_vtol_config
from nastaero.rotor.forward_flight import ForwardFlightBEMT

RHO = 1.225


@pytest.fixture(scope="module")
def ff():
    cfg = make_ilc8_vtol_config()
    return ForwardFlightBEMT(cfg.hover_rotors[0].blade), cfg.hover_rotors[0]


class TestAxisDotProductConvention:
    """축 벡터 내적으로 독립 유도한 축류 성분과 규약의 일치."""

    @pytest.mark.parametrize("sigma_deg", [0.0, 30.0, 60.0, 90.0])
    def test_tilt_axis_alpha_equals_sigma(self, sigma_deg):
        """축 a(σ)=[sinσ,0,cosσ], V=[V,0,0]이면 V·a = V·sinσ 이므로
        규약(V_axial = V sin α)에서 α = σ 다."""
        s = np.radians(sigma_deg)
        axis = np.array([np.sin(s), 0.0, np.cos(s)])
        V = 25.0
        v_axial_geom = V * float(np.dot(np.array([1.0, 0.0, 0.0]), axis))
        v_axial_conv = V * np.sin(s)   # alpha = sigma
        assert v_axial_geom == pytest.approx(v_axial_conv, abs=1e-12)
        # 여각(90-σ)이었다면 면내 성분이 나온다
        assert V * np.sin(np.pi / 2 - s) == pytest.approx(
            V * np.cos(s), abs=1e-12)

    def test_vertical_rotor_forward_flight_is_edgewise(self, ff):
        """수직축 로터 + 수평 전진 = 면내류(α=0). 축류(α=90°)로 풀면
        전진속도 전체가 상승류로 오인되어 토크가 크게 어긋난다."""
        solver, rotor = ff
        edge = solver.solve_for_thrust(2330.0, rotor.rpm_hover, 20.0,
                                       alpha_shaft=0.0, rho=RHO)
        axial = solver.solve_for_thrust(2330.0, rotor.rpm_hover, 20.0,
                                        alpha_shaft=np.pi / 2, rho=RHO)
        assert edge.torque < axial.torque * 0.7, (
            f"면내류 토크({edge.torque:.1f})가 축류 오인 토크"
            f"({axial.torque:.1f})보다 훨씬 작아야 한다")

    def test_axial_limit_matches_axial_solver(self, ff):
        """α=90°는 축류 BEMT와 일치해야 한다(μ<0.05 폴백 경유)."""
        solver, rotor = ff
        res = solver.solve(rotor.rpm_hover, 10.0, np.pi / 2, RHO, 0.05)
        res_ax = solver._axial_solver.solve(rotor.rpm_hover, 10.0, RHO, 0.05)
        assert res.thrust == pytest.approx(res_ax.thrust, rel=1e-9)


class TestClosureConsistency:
    """CT–λ 폐합: 유입 가정과 블레이드 적분이 자기일관해야 한다."""

    @pytest.mark.parametrize("V,alpha_deg", [(15.0, 0.0), (25.0, 30.0)])
    def test_glauert_inflow_closed_with_blade_CT(self, ff, V, alpha_deg):
        solver, rotor = ff
        a = np.radians(alpha_deg)
        res = solver.solve(rotor.rpm_hover, V, a, RHO, 0.08)
        omega = rotor.rpm_hover * 2 * np.pi / 60.0
        R = solver.blade.radius
        mu = V * np.cos(a) / (omega * R)
        lam_c = V * np.sin(a) / (omega * R)
        lam = solver._glauert_inflow(mu, lam_c, res.CT)
        cached = getattr(solver, "_lambda_cache", None)
        assert cached is not None
        assert lam == pytest.approx(cached[1], rel=2e-2), (
            "블레이드 적분 CT로 되계산한 Glauert 유입이 사용된 유입과 "
            "수 % 이내로 폐합해야 한다")


class TestCallSiteAngles:
    """호출부가 실제로 정정된 각을 넘기는지 소스 수준으로 고정."""

    def test_no_complement_angles_in_generators(self):
        import inspect
        from nastaero.loads_analysis.certification import (
            retrim_events, vtol_load_case_matrix)
        src = (inspect.getsource(vtol_load_case_matrix)
               + inspect.getsource(retrim_events))
        assert "pi / 2 - s_r" not in src
        assert "pi / 2 - s_s" not in src
        assert "pi / 2 - a_r" not in src
        assert "alpha_shaft=np.pi / 2,  # Vertical shaft" not in src
