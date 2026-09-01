# 로터 포화(추진계 한계) 케이스의 포락선·설계하중 전파 시험 (r3 MC2)
from __future__ import annotations
import numpy as np
import pytest

from ascent_load.loads_analysis.certification.batch_runner import (
    BatchResult, CaseResult,
)
from ascent_load.loads_analysis.certification import envelope as env_mod
from ascent_load.loads_analysis.certification.envelope import (
    select_critical_design_loads, design_load_table,
)
from ascent_load.loads_analysis.certification.force_export import _build_header


STATIONS = np.array([0.0, 2500.0, 5000.0])


def _curve(v, m, t):
    return {"stations": STATIONS,
            "shear": np.asarray(v, dtype=float),
            "bending": np.asarray(m, dtype=float),
            "torsion": np.asarray(t, dtype=float)}


def _batch(infeasible_exceeds: bool):
    br = BatchResult()
    br.case_results = [
        CaseResult(case_id=1, category="symmetric", far_section="23.337",
                   converged=True, nz=3.8, label="nz +3.8",
                   nodal_forces={1: np.zeros(6)}),
        CaseResult(case_id=2, category="symmetric", far_section="23.337",
                   converged=True, nz=-1.5, label="nz -1.5",
                   nodal_forces={1: np.zeros(6)}),
        CaseResult(case_id=6, category="vtol_transition",
                   far_section="SC-VTOL.2215", converged=True, nz=1.0,
                   label="saturated climb",
                   nodal_forces={1: np.zeros(6)},
                   rotor_command_feasible=False,
                   rotor_thrust_shortfall=0.268),
    ]
    br.completed_ids = {1, 2, 6}
    t6_root = 3.0e6 if infeasible_exceeds else 5.0e5
    vmt = {
        1: {"Wing": _curve([50e3, 30e3, 0], [0, 8e7, 1e8], [1e6, 5e5, 0])},
        2: {"Wing": _curve([-20e3, -12e3, 0], [0, -3e7, -4e7],
                           [-5e5, -3e5, 0])},
        6: {"Wing": _curve([40e3, 25e3, 0], [0, 6e7, 8e7],
                           [t6_root, 4e5, 0])},
    }
    return br, vmt


def _run(monkeypatch, br, vmt, **kw):
    import ascent_load.loads_analysis.certification.vmt_bridge as vb
    monkeypatch.setattr(vb, "compute_vmt_for_batch",
                        lambda *a, **k: vmt)
    return select_critical_design_loads(
        model=None, batch_result=br, include_3d=False, **kw)


class TestSeparatePolicy:
    def test_exceeding_saturated_case_is_flagged_not_pooled(self, monkeypatch):
        br, vmt = _batch(infeasible_exceeds=True)
        res = _run(monkeypatch, br, vmt)
        # 실현가능 포락선의 비틀림 최대는 케이스 1이어야 한다(6이 아님)
        env = res["processor"].get_envelope("Wing")
        assert env.envelopes[0].T_max_case_id == 1
        # 포화 케이스는 추진계 한계 클래스로 별도 편입
        pl = res["propulsion_limit"]
        assert pl["n_infeasible"] == 1
        assert pl["n_appended_design_cases"] == 1
        assert any(e["case_id"] == 6 and e["quantity"] == "T"
                   for e in pl["exceedances"])
        flagged = [d for d in res["design_cases"] if d.case_id == 6]
        assert len(flagged) == 1
        assert flagged[0].rotor_command_feasible is False
        assert flagged[0].basis == "propulsion-limit"
        assert flagged[0].rotor_thrust_shortfall == pytest.approx(0.268)
        assert "propulsion-limit" in flagged[0].why()

    def test_non_exceeding_saturated_case_reported_only(self, monkeypatch):
        br, vmt = _batch(infeasible_exceeds=False)
        res = _run(monkeypatch, br, vmt)
        pl = res["propulsion_limit"]
        assert pl["n_infeasible"] == 1
        assert pl["exceedances"] == []
        assert pl["n_appended_design_cases"] == 0
        assert all(d.case_id != 6 for d in res["design_cases"])

    def test_feasible_cases_keep_flag_true(self, monkeypatch):
        br, vmt = _batch(infeasible_exceeds=True)
        res = _run(monkeypatch, br, vmt)
        for d in res["design_cases"]:
            if d.case_id != 6:
                assert d.rotor_command_feasible is True

    def test_table_and_csv_carry_flag(self, monkeypatch, tmp_path):
        br, vmt = _batch(infeasible_exceeds=True)
        res = _run(monkeypatch, br, vmt)
        rows = design_load_table(res["design_cases"])
        r6 = [r for r in rows if r["case_id"] == 6][0]
        assert r6["rotor_command_feasible"] is False
        csv_path = env_mod.write_design_load_summary_csv(
            res["design_cases"], str(tmp_path / "d.csv"))
        text = open(csv_path).read()
        assert "rotor_command_feasible" in text


class TestLegacyPolicies:
    def test_include_reproduces_pooled_envelope(self, monkeypatch):
        br, vmt = _batch(infeasible_exceeds=True)
        res = _run(monkeypatch, br, vmt, infeasible_policy="include")
        env = res["processor"].get_envelope("Wing")
        # 병합 정책에서는 포화 케이스가 그대로 비틀림 최대를 차지
        assert env.envelopes[0].T_max_case_id == 6
        assert res["propulsion_limit"]["n_appended_design_cases"] == 0

    def test_exclude_drops_saturated_entirely(self, monkeypatch):
        br, vmt = _batch(infeasible_exceeds=True)
        res = _run(monkeypatch, br, vmt, infeasible_policy="exclude")
        assert all(d.case_id != 6 for d in res["design_cases"])
        assert res["propulsion_limit"]["exceedances"] == []

    def test_bad_policy_raises(self, monkeypatch):
        br, vmt = _batch(infeasible_exceeds=True)
        with pytest.raises(ValueError):
            _run(monkeypatch, br, vmt, infeasible_policy="whatever")


class TestForceExportHeader:
    def test_header_marks_infeasible(self):
        cr = CaseResult(case_id=6, category="vtol_transition",
                        far_section="SC-VTOL.2215", converged=True,
                        nz=1.0, label="saturated climb",
                        rotor_command_feasible=False,
                        rotor_thrust_shortfall=0.268)
        h = _build_header(cr, [])
        assert "ROTOR COMMAND INFEASIBLE" in h
        assert "26.8" in h
        assert "PROPULSION-LIMIT CASE" in h

    def test_header_clean_for_feasible(self):
        cr = CaseResult(case_id=1, category="symmetric",
                        far_section="23.337", converged=True, nz=3.8)
        assert "INFEASIBLE" not in _build_header(cr, [])
