# 호버 돌풍 과도 피크 케이스화 시험 — 집계 유입류 경로와 배치 합류 규약
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gust_results(tmp_path_factory):
    """ILC-8 생성 모델로 축약 파라미터 돌풍 과도 2건(상/하) 실행."""
    import yaml

    from nastaero.bdf.parser import parse_bdf
    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig,
    )
    from nastaero.loads_analysis.certification.vtol_transient_loads import (
        VTOLTransientLoadsRunner,
    )
    from nastaero.models.ilc8 import build_ilc8, make_ilc8_vtol_config
    import os

    out = tmp_path_factory.mktemp("ilc8_gust")
    build_ilc8(str(out))
    model = parse_bdf(str(out / "ilc8.bdf"))
    cfg_path = os.path.join(os.path.dirname(__file__),
                            "validation/ILC8/ilc8_cert_config.yaml")
    with open(cfg_path) as f:
        acfg = AircraftConfig.from_dict(yaml.safe_load(f))
    runner = VTOLTransientLoadsRunner(model, make_ilc8_vtol_config(), acfg)
    return runner.run_all_hover_gust(t_sim=2.0, dt=0.01, dt_loads=0.05)


class TestHoverGustTransient:
    def test_two_directions(self, gust_results):
        assert len(gust_results) == 2
        labels = [r.failed_rotor_label for r in gust_results]
        assert any("up" in s for s in labels)
        assert any("down" in s for s in labels)
        for r in gust_results:
            assert r.event_type == "gust"

    def test_peak_loads_and_snapshot(self, gust_results):
        for r in gust_results:
            assert r.peak_wing_Mx > 0
            assert r.qs_wing_Mx > 0
            # 케이스화용 피크 절점하중 스냅샷: 로터 허브 + 질량 절점
            assert r.peak_nodal_forces
            assert len(r.peak_nodal_forces) > 100
            assert len(r.rotor_thrusts) == 8   # 양력 로터 8기 이력

    def test_up_gust_attenuated_below_no_lag_bound(self, gust_results):
        # 논문 2의 핵심: 도달 시차+유입류 지연으로 피크가 무시차
        # 준정적 상한 아래로 감쇠한다 (상방 돌풍)
        up = next(r for r in gust_results if "up" in r.failed_rotor_label)
        assert up.daf_wing_Mx < 1.05

    def test_down_gust_rebound_exceeds_static_equiv(self, gust_results):
        # 하방 돌풍은 돌풍 종료 후 유입류 리바운드 오버슈트가 준정적
        # 등가를 초과 — 정적 ±Δnz 케이스가 놓치는 과도 임계
        down = next(r for r in gust_results
                    if "down" in r.failed_rotor_label)
        assert down.daf_wing_Mx > 1.0
        assert down.peak_wing_Mx_time > 0.5   # 돌풍(T_g=0.5) 이후 발생

    def test_batch_case_conversion_contract(self, gust_results):
        # 배치 러너 합류 규약: category/far/nodal_forces 형식 재현
        from nastaero.loads_analysis.certification.batch_runner import (
            CaseResult,
        )

        tr = gust_results[0]
        far = {"oei": "SC-VTOL.2140",
               "gust": "SC-VTOL.2135"}.get(tr.event_type, "SC-VTOL.2150")
        cr = CaseResult(case_id=30000,
                        category=f"vtol_{tr.event_type}_transient",
                        far_section=far,
                        nodal_forces=tr.peak_nodal_forces,
                        converged=True, label=tr.failed_rotor_label)
        assert cr.category == "vtol_gust_transient"
        assert cr.far_section == "SC-VTOL.2135"
        assert cr.nodal_forces is tr.peak_nodal_forces
        total_fz = sum(f[2] for f in cr.nodal_forces.values())
        # 피크 시점 절점하중은 로터 상향력-관성 하향력의 순힘 (유한값)
        assert np.isfinite(total_fz)


@pytest.fixture(scope="module")
def tr_gust_results(tmp_path_factory):
    """천이 과도 돌풍 — 대표 속도점 1개(상/하), 축약 파라미터."""
    import os

    import yaml

    from nastaero.bdf.parser import parse_bdf
    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig,
    )
    from nastaero.loads_analysis.certification.vtol_transient_loads import (
        VTOLTransientLoadsRunner,
    )
    from nastaero.models.ilc8 import build_ilc8, make_ilc8_vtol_config

    out = tmp_path_factory.mktemp("ilc8_trgust")
    build_ilc8(str(out))
    model = parse_bdf(str(out / "ilc8.bdf"))
    cfg_path = os.path.join(os.path.dirname(__file__),
                            "validation/ILC8/ilc8_cert_config.yaml")
    with open(cfg_path) as f:
        acfg = AircraftConfig.from_dict(yaml.safe_load(f))
    runner = VTOLTransientLoadsRunner(model, make_ilc8_vtol_config(),
                                      acfg)
    return runner.run_all_transition_gust(
        n_speeds=1, t_sim=2.0, dt=0.01, dt_loads=0.05)


class TestForwardFlightInflow:
    """Pitt-Peters 전진비 일반화 — 호버 중첩과 질량유량 스케일."""

    def test_hover_nesting_exact(self):
        from nastaero.rotor.dynamic_inflow import PittPetersInflow

        p0 = PittPetersInflow(rotor_radius=1.2, T_steady=2300.0)
        pf = PittPetersInflow(rotor_radius=1.2, T_steady=2300.0,
                              V_forward=0.0)
        assert pf.nu_steady == pytest.approx(p0.nu_steady, abs=1e-12)
        assert pf.tau_nu == pytest.approx(p0.tau_nu, abs=1e-12)
        assert pf.thrust_slope == pytest.approx(
            2.0 * 1.225 * p0.A * p0.nu_steady)

    def test_implicit_momentum_solution(self):
        """T = 2 rho A U nu_s가 수치적으로 만족되는지."""
        import math

        from nastaero.rotor.dynamic_inflow import PittPetersInflow

        p = PittPetersInflow(rotor_radius=1.2, T_steady=2300.0,
                             V_forward=20.0)
        U = math.sqrt(20.0 ** 2 + p.nu_steady ** 2)
        assert 2.0 * 1.225 * p.A * U * p.nu_steady == pytest.approx(
            2300.0, rel=1e-9)
        assert p.nu_steady < PittPetersInflow(
            rotor_radius=1.2, T_steady=2300.0).nu_steady
        assert p.tau_nu < PittPetersInflow(
            rotor_radius=1.2, T_steady=2300.0).tau_nu


class TestTransitionGustTransient:
    def test_two_directions_with_labels(self, tr_gust_results):
        assert len(tr_gust_results) == 2
        labels = [r.failed_rotor_label for r in tr_gust_results]
        assert any("up" in s for s in labels)
        assert any("down" in s for s in labels)
        assert all(s.startswith("TR V=") for s in labels)

    def test_peak_snapshot_self_equilibrated(self, tr_gust_results):
        """아핀 조립 + relief 폐합 — 피크 스냅샷 합력 0."""
        for r in tr_gust_results:
            assert r.peak_nodal_forces
            tot = np.zeros(3)
            for f in r.peak_nodal_forces.values():
                tot += f[:3]
            assert np.abs(tot).max() < 1e-3

    def test_up_gust_attenuated_below_no_lag_bound(self, tr_gust_results):
        """상승 돌풍: 시차·유입류 지연 감쇠로 무시차 상한 이하."""
        up = next(r for r in tr_gust_results
                  if "up" in r.failed_rotor_label)
        assert up.daf_wing_Mx < 1.05

    def test_down_gust_transient_exceeds_qs(self, tr_gust_results):
        """하강 돌풍: 준정적 등가(거의 상쇄)를 넘는 과도 피크 —
        정적 회랑 돌풍 케이스가 원리적으로 못 보는 하중."""
        dn = next(r for r in tr_gust_results
                  if "down" in r.failed_rotor_label)
        assert dn.peak_wing_Mx > dn.qs_wing_Mx

    def test_nz_history_sane(self, tr_gust_results):
        for r in tr_gust_results:
            assert -1.0 < float(r.nz_history.min())
            assert float(r.nz_history.max()) < 3.0
