# ILC-8T 틸트 변형 시험 — 덱 생성, 틸트 배분 평형, 변환 회랑, 매트릭스 통합
"""Tests for the ILC-8T tilt variant (Phase A)."""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest
import yaml

W = 18633.0
S = 16.2
CL = 1.0
RHO = 1.225


class TestTiltAllocation:
    """제어법칙 없는 결정적 배분 — Fx/Fz 2식 평형."""

    def test_force_balance(self):
        from nastaero.loads_analysis.certification.vtol_conditions import (
            tilt_allocation, tilt_drag_estimate,
        )
        for V, sg, nz in ((15.0, 30.0, 1.0), (25.0, 60.0, 1.5),
                          (35.0, 90.0, 1.0)):
            F, A, L, ok = tilt_allocation(V, sg, nz, W, S, CL, RHO)
            D = tilt_drag_estimate(V, W)
            # Fx: 전열 총추력의 수평 성분 = 항력
            assert F * math.sin(math.radians(sg)) == pytest.approx(D)
            # Fz: 수직 성분 합 = nz*W (능력 내에서)
            if ok and A > 0:
                assert (L + F * math.cos(math.radians(sg)) + A
                        == pytest.approx(nz * W, rel=1e-9))

    def test_wing_capability_cap(self):
        from nastaero.loads_analysis.certification.vtol_conditions import (
            tilt_allocation,
        )
        _F, _A, L, _ok = tilt_allocation(35.0, 80.0, 1.0, W, S, CL, RHO)
        L_cap = CL * 0.5 * RHO * 35.0 ** 2 * S
        assert L <= L_cap + 1e-9

    def test_front_thrust_cap_sets_lower_sigma_bound(self):
        """과소 틸트에서 F = D/sin(σ)가 전열 한계를 넘어 비가용."""
        from nastaero.loads_analysis.certification.vtol_conditions import (
            tilt_allocation,
        )
        _F, _A, _L, ok = tilt_allocation(35.0, 2.0, 1.0, W, S, CL, RHO)
        assert not ok


class TestTiltCorridor:
    def _conds(self):
        from nastaero.loads_analysis.certification.vtol_conditions import (
            generate_tilt_transition_conditions,
        )
        return generate_tilt_transition_conditions(
            15.0, 35.0, [0.0], wing_area_m2=S, CL_transition=CL,
            weight_N=W)

    def test_corridor_nonempty_all_speeds(self):
        conds = self._conds()
        speeds = sorted({round(c.V_eas, 1) for c in conds})
        assert len(speeds) == 8
        assert all(c.phase.value == "tilt_transition" for c in conds)

    def test_corridor_edges_physical(self):
        """저속에서 σ 상한 제한(후열 수직 지지), 고속으로 갈수록
        하한 σ_lo 증가(D/sinσ 전열 한계)."""
        conds = self._conds()
        by_v = {}
        for c in conds:
            by_v.setdefault(round(c.V_eas, 1), []).append(c.tilt_deg)
        speeds = sorted(by_v)
        lo = [min(by_v[v]) for v in speeds]
        assert lo[-1] > lo[0]          # σ_lo가 속도와 함께 증가
        assert max(by_v[speeds[-1]]) == pytest.approx(90.0, abs=2.5)

    def test_case_count_reasonable(self):
        conds = self._conds()
        # 8속도 x (중심+양끝 <=3 σ) x nz{1.0,1.5} 이하
        assert 16 <= len(conds) <= 48


@pytest.fixture(scope="module")
def ilc8t_model(tmp_path_factory):
    from nastaero.bdf.parser import parse_bdf
    from nastaero.models.ilc8t import build_ilc8t

    out = tmp_path_factory.mktemp("ilc8t")
    return parse_bdf(build_ilc8t(str(out)))


class TestIlc8tDeck:
    def test_mass_cg_matches_lc(self, ilc8t_model):
        """동일 MTOW, CG 오차 < 10 mm — 등중량 비교 성립."""
        from nastaero.loads_analysis.trim_loads import compute_node_masses

        nm = compute_node_masses(ilc8t_model)
        mt = sum(nm.values()) * 1000
        cg = sum(m * ilc8t_model.nodes[n].xyz_global[0]
                 for n, m in nm.items()) / sum(nm.values())
        assert mt == pytest.approx(1900.0, abs=0.5)
        assert abs(cg - 4450.0) < 10.0

    def test_pusher_removed_hubs_present(self, ilc8t_model):
        assert 990201 not in ilc8t_model.nodes
        for nid in range(990101, 990109):
            assert nid in ilc8t_model.nodes


class TestTiltMatrix:
    @pytest.fixture(scope="class")
    def matrix_cases(self):
        from nastaero.loads_analysis.certification.aircraft_config import (
            AircraftConfig,
        )
        from nastaero.loads_analysis.certification.vtol_load_case_matrix import (
            VTOLLoadCaseMatrix,
        )
        from nastaero.models.ilc8t import make_ilc8t_vtol_config

        cfg_path = os.path.join(os.path.dirname(__file__),
                                "validation/ILC8/ilc8_cert_config.yaml")
        with open(cfg_path) as f:
            cfg = AircraftConfig.from_dict(yaml.safe_load(f))
        mx = VTOLLoadCaseMatrix(make_ilc8t_vtol_config(), cfg)
        return mx, mx.generate_all()

    def test_categories(self, matrix_cases):
        _mx, cases = matrix_cases
        cats = {c.category for c in cases}
        assert "vtol_tilt_transition" in cats
        assert "vtol_transition" not in cats      # L+C 스케줄 대체
        assert {"vtol_hover", "vtol_oei",
                "vtol_rotor_jam"} <= cats

    def test_tilt_force_direction_and_fx(self, matrix_cases):
        """전열 허브력이 축 [sinσ,0,cosσ] 방향이고 ΣFx ≈ D
        (BEMT 포화 없는 케이스 기준)."""
        from nastaero.loads_analysis.certification.vtol_conditions import (
            tilt_drag_estimate,
        )
        from nastaero.models.ilc8t import make_ilc8t_vtol_config

        _mx, cases = matrix_cases
        vc = make_ilc8t_vtol_config()
        front_ids = {r.hub_node_id for r in vc.rotors
                     if r.rotor_type.value == "tilt"}
        checked = 0
        for c in cases:
            if c.category != "vtol_tilt_transition" or not c.rotor_forces:
                continue
            lab = c.trim_condition.label
            sg = float(lab.split("s=")[1].split("deg")[0])
            V = float(lab.split("V=")[1].split("m/s")[0])
            if sg < 30.0 or sg > 80.0:
                # 저틸트: 콜렉티브 상한 포화 가능 / 근사 축류(>80°):
                # 미소 추력 요청이 콜렉티브 하한에 걸려 달성치가 큼
                # — 두 경우 모두 달성값 관례(relief 폐합)로 유효
                continue
            Fx = sum(v[0] for nid, v in c.rotor_forces.items()
                     if nid in front_ids)
            for nid, v in c.rotor_forces.items():
                if nid not in front_ids:
                    continue
                f = np.array(v[:3])
                axis = np.array([math.sin(math.radians(sg)), 0.0,
                                 math.cos(math.radians(sg))])
                if np.linalg.norm(f) > 1.0:
                    cosang = float(f @ axis / np.linalg.norm(f))
                    assert cosang > 0.999
            D = tilt_drag_estimate(V, W)
            if Fx > 1.0:
                assert Fx == pytest.approx(D, rel=0.35)  # BEMT 달성치
                checked += 1
        assert checked > 0

    def test_sigma90_no_aft_lift_when_wing_sufficient(self, matrix_cases):
        """σ=90° 케이스: 전열 힘은 순수 +x (수직 성분 ≈ 0)."""
        from nastaero.models.ilc8t import make_ilc8t_vtol_config

        _mx, cases = matrix_cases
        vc = make_ilc8t_vtol_config()
        front_ids = {r.hub_node_id for r in vc.rotors
                     if r.rotor_type.value == "tilt"}
        found = False
        for c in cases:
            if c.category != "vtol_tilt_transition" or not c.rotor_forces:
                continue
            m = re.search(r"s=(\d+(?:\.\d+)?)deg",
                          str(c.trim_condition.label))
            if not m or abs(float(m.group(1)) - 90.0) > 1e-6:
                continue
            for nid, v in c.rotor_forces.items():
                if nid in front_ids and abs(v[0]) > 1.0:
                    assert abs(v[2]) < 0.01 * abs(v[0])
                    found = True
        assert found


class TestTiltStuck:
    """틸트 액추에이터 고착(M6) — 틸트로터 고유 고장 모드."""

    def _conds(self):
        from nastaero.loads_analysis.certification.vtol_conditions import (
            generate_tilt_stuck_conditions,
        )
        return generate_tilt_stuck_conditions(
            15.0, 35.0, [0.0], [1, 2, 3, 4], wing_area_m2=S,
            CL_transition=CL, weight_N=W)

    def test_census_and_fields(self):
        conds = self._conds()
        assert len(conds) > 0
        for c in conds:
            assert c.phase.value == "tilt_stuck"
            assert c.failed_rotor_id in (1, 2, 3, 4)
            assert c.stuck_tilt_deg in (0.0, 90.0)
            assert abs(c.stuck_tilt_deg - c.tilt_deg) >= 5.0

    def test_matrix_stuck_cases_and_asymmetry(self):
        """매트릭스 통합 + 고착 비대칭이 롤 모멘트를 만드는지."""
        from nastaero.loads_analysis.certification.aircraft_config import (
            AircraftConfig,
        )
        from nastaero.loads_analysis.certification.vtol_load_case_matrix import (
            VTOLLoadCaseMatrix,
        )
        from nastaero.models.ilc8t import make_ilc8t_vtol_config

        cfg_path = os.path.join(os.path.dirname(__file__),
                                "validation/ILC8/ilc8_cert_config.yaml")
        with open(cfg_path) as f:
            cfg = AircraftConfig.from_dict(yaml.safe_load(f))
        vc = make_ilc8t_vtol_config()
        mx = VTOLLoadCaseMatrix(vc, cfg)
        cases = mx.generate_all()
        stuck = [c for c in cases if c.category == "vtol_tilt_stuck"]
        assert len(stuck) > 0
        pos = {r.hub_node_id: np.array(r.hub_position)
               for r in vc.rotors}
        found_asym = False
        for c in stuck:
            if not c.rotor_forces:
                continue
            Mx = sum(pos[nid][1] * v[2]
                     for nid, v in c.rotor_forces.items()
                     if nid in pos)
            if abs(Mx) > 1e5:      # N·mm — 비대칭 롤 모멘트 존재
                found_asym = True
        assert found_asym

    def test_stuck_rotor_force_along_stuck_axis(self):
        from nastaero.loads_analysis.certification.aircraft_config import (
            AircraftConfig,
        )
        from nastaero.loads_analysis.certification.vtol_conditions import (
            VTOLCondition, VTOLFlightPhase,
        )
        from nastaero.loads_analysis.certification.vtol_load_case_matrix import (
            VTOLLoadCaseMatrix,
        )
        from nastaero.models.ilc8t import make_ilc8t_vtol_config

        cfg_path = os.path.join(os.path.dirname(__file__),
                                "validation/ILC8/ilc8_cert_config.yaml")
        with open(cfg_path) as f:
            cfg = AircraftConfig.from_dict(yaml.safe_load(f))
        vc = make_ilc8t_vtol_config()
        mx = VTOLLoadCaseMatrix(vc, cfg)
        wc = cfg.weight_cg_conditions[0]
        cond = VTOLCondition(
            label="t", phase=VTOLFlightPhase.TILT_STUCK, V_eas=25.0,
            nz=1.0, tilt_deg=50.0, stuck_tilt_deg=0.0,
            failed_rotor_id=1)
        forces = mx._compute_rotor_forces_tilt(cond, wc)
        stuck_nid = [r.hub_node_id for r in vc.rotors
                     if r.rotor_id == 1][0]
        f = np.array(forces[stuck_nid][:3])
        # 고착각 0° = 수직 축: Fx ≈ 0, Fz > 0
        assert abs(f[0]) < 0.01 * abs(f[2])
        assert f[2] > 0


class TestConversionSweep:
    """변환 기동 스윕 — 준정적 스케줄 + 자이로 항 정량화."""

    @pytest.fixture(scope="class")
    def sweep(self, tmp_path_factory):
        import sys

        import yaml as _yaml

        from nastaero.bdf.parser import parse_bdf
        from nastaero.loads_analysis.certification.aircraft_config import (
            AircraftConfig,
        )
        from nastaero.loads_analysis.certification.vtol_transient_loads import (
            VTOLTransientLoadsRunner,
        )
        from nastaero.models.ilc8t import build_ilc8t, make_ilc8t_vtol_config

        out = tmp_path_factory.mktemp("ilc8t_sweep")
        model = parse_bdf(build_ilc8t(str(out)))
        cfg_path = os.path.join(os.path.dirname(__file__),
                                "validation/ILC8T/ilc8t_cert_config.yaml")
        with open(cfg_path) as f:
            cfg = AircraftConfig.from_dict(_yaml.safe_load(f))
        runner = VTOLTransientLoadsRunner(
            model, make_ilc8t_vtol_config(), cfg)
        return runner.run_tilt_conversion_sweep(dt=0.5)

    def test_schedule_monotonic_and_within_corridor(self, sweep):
        sg = sweep["sigma_deg"]
        assert np.all(np.diff(sg) > 0)
        assert 0.0 < sg[0] < sg[-1] <= 90.0

    def test_peak_bounded_by_corridor_statics(self, sweep):
        """변환 기동 피크가 회랑 중심 정적 대비 완만(≤1.2×) —
        회랑 경계 케이스가 이미 포락함을 정량화."""
        assert 0.95 <= sweep["peak_over_static"] <= 1.2

    def test_gyro_negligible_for_this_rotor_class(self, sweep):
        """자이로 허브 모멘트 손계산 일치 + 단면하중 기여 < 1%
        (CW/CCW 쌍의 루트 상쇄 + 소형 로터 관성)."""
        import math as m

        I_p = 1.44 * 0.75 ** 2 / 3.0
        expect = I_p * (2400 * 2 * m.pi / 60) * m.radians(10.0) * 1e3
        assert sweep["gyro_hub_moment_Nmm"] == pytest.approx(expect,
                                                             rel=1e-6)
        assert sweep["gyro_share"] < 0.01
