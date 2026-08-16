# 로터 지령 실현 가능성을 수치적 수렴과 분리해 기록하는지 검증하는 시험
from __future__ import annotations

import numpy as np
import pytest

from nastaero.rotor.bemt_solver import RotorLoads
from nastaero.loads_analysis.certification.vtol_load_case_matrix import (
    VTOLLoadCaseMatrix,
)
from nastaero.loads_analysis.certification.batch_runner import CaseResult
from nastaero.loads_analysis.certification.load_case_matrix import CertLoadCase


class TestRotorLoadsSaturationState:
    """BEMT가 지령 추력에 도달하지 못한 사실이 결과에 남아야 한다.

    도달 실패 시에도 달성 추력으로 하중이 조립되고 관성 릴리프가
    잔차를 닫으므로, 전역 평형만 보면 정상 케이스와 구분되지 않는다.
    """

    def test_unsaturated_default(self):
        r = RotorLoads(thrust=2328.0)
        assert r.thrust_saturated is False
        assert r.thrust_target_N is None
        assert r.thrust_shortfall_frac == 0.0

    def test_shortfall_fraction(self):
        r = RotorLoads(thrust=2869.2, thrust_target_N=3903.2,
                       thrust_saturated=True)
        assert r.thrust_shortfall_frac == pytest.approx(0.2649, abs=1e-3)

    def test_shortfall_zero_without_target(self):
        # 고정 콜렉티브 solve()는 지령이 없으므로 부족률도 없다
        r = RotorLoads(thrust=100.0, thrust_saturated=True)
        assert r.thrust_shortfall_frac == 0.0


class TestSaturationSummary:
    def test_flags_worst_rotor(self):
        lm = {
            0: RotorLoads(thrust=2869.2, thrust_target_N=3903.2,
                          thrust_saturated=True),
            1: RotorLoads(thrust=2880.7, thrust_target_N=3250.6,
                          thrust_saturated=True),
            2: RotorLoads(thrust=2328.0),
        }
        feasible, worst = VTOLLoadCaseMatrix._saturation_summary(lm)
        assert feasible is False
        assert worst == pytest.approx(0.2649, abs=1e-3)   # 최악 로터 기준

    def test_all_feasible(self):
        lm = {0: RotorLoads(thrust=2328.0), 1: RotorLoads(thrust=2330.0)}
        feasible, worst = VTOLLoadCaseMatrix._saturation_summary(lm)
        assert feasible is True and worst == 0.0

    def test_empty_map_is_feasible(self):
        assert VTOLLoadCaseMatrix._saturation_summary(None) == (True, 0.0)
        assert VTOLLoadCaseMatrix._saturation_summary({}) == (True, 0.0)


class TestFeasibilityIsSeparateFromConvergence:
    """``converged``와 ``rotor_command_feasible``는 서로 다른 축이다."""

    def test_case_result_defaults(self):
        cr = CaseResult(case_id=1)
        assert cr.rotor_command_feasible is True
        assert cr.rotor_thrust_shortfall == 0.0

    def test_converged_but_infeasible_is_representable(self):
        cr = CaseResult(case_id=1, converged=True,
                        rotor_command_feasible=False,
                        rotor_thrust_shortfall=0.265)
        # 수치적으로는 성공했으나 지령 비행상태는 실현되지 않았다
        assert cr.converged is True
        assert cr.rotor_command_feasible is False
        assert cr.rotor_thrust_shortfall > 0.25

    def test_cert_case_carries_the_flags(self):
        c = CertLoadCase(rotor_command_feasible=False,
                         rotor_thrust_shortfall=0.1)
        assert c.rotor_command_feasible is False
        assert c.rotor_thrust_shortfall == pytest.approx(0.1)
