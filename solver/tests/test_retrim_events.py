# (고장×재트림) 확장 사건 선별기 테스트 — 기저 선형성, 사건 공간, 채택 판정, 케이스 실현
"""Tests for the (failure x re-trim) event-space extension."""
from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

ILC8 = os.path.join(os.path.dirname(__file__), "validation", "ILC8")


@pytest.fixture(scope="module")
def screen():
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.loads_analysis.certification.aircraft_config import (
        AircraftConfig,
    )
    from ascent_load.loads_analysis.certification.retrim_events import (
        RetrimScreen,
    )
    from ascent_load.loads_analysis.component_id import (
        identify_components_manual,
    )
    from ascent_load.models.ilc8 import make_ilc8_vtol_config

    model = parse_bdf(os.path.join(ILC8, "ilc8.bdf"))
    with open(os.path.join(ILC8, "ilc8_cert_config.yaml")) as f:
        config = AircraftConfig.from_dict(yaml.safe_load(f))
    vtol_config = make_ilc8_vtol_config()

    def _nids(lo, hi, extra=()):
        ids = [n for n in model.nodes if lo <= n <= hi]
        return ids + [n for n in extra if n in model.nodes]

    WING = dict(span_axis=1, shear_axis=2, bending_axis=0,
                torsion_axis=1)
    comps = identify_components_manual(model, [
        dict(name="Right Wing", integration_sign=1.0, color="blue",
             node_ids=_nids(400000, 499999)
             + _nids(730000, 749999,
                     (990103, 990104, 990107, 990108)), **WING),
        dict(name="Left Wing", integration_sign=-1.0,
             color="dodgerblue",
             node_ids=_nids(300000, 399999)
             + _nids(710000, 729999,
                     (990101, 990102, 990105, 990106)), **WING),
        dict(name="Fuselage", integration_sign=-1.0, color="gray",
             node_ids=_nids(100000, 299999, (990201,)),
             span_axis=0, shear_axis=2, bending_axis=1,
             torsion_axis=0),
    ])
    return RetrimScreen(model, vtol_config, config, components=comps,
                        fuselage_cg_x=4450.0)


@pytest.fixture(scope="module")
def events(screen):
    return screen.screen()


class TestBasis:
    def test_linearity_reconstructs_uniform_hover(self, screen):
        """전 로터 1.0 패턴의 기저 합성 == 직접 조립한 호버 VMT."""
        from ascent_load.loads_analysis.certification.batch_runner import (
            BatchResult, CaseResult,
        )
        from ascent_load.loads_analysis.certification.vmt_bridge import (
            compute_vmt_for_batch,
        )

        pat = np.ones(screen.n)
        forces, nz = screen._pattern_forces(pat)
        batch = BatchResult()
        batch.case_results.append(CaseResult(
            case_id=1, category="t", converged=True,
            nodal_forces=forces, nz=nz, label="hover"))
        batch.completed_ids.add(1)
        vmt = compute_vmt_for_batch(screen.model, batch,
                                    components=screen.components,
                                    fuselage_cg_x=screen._fus_cg_x)
        ref = screen._vmt_mat(vmt[1])
        rec = screen._B.sum(axis=0)
        assert np.abs(rec - ref).max() <= 1e-6 * np.abs(ref).max()

    def test_free_fall_is_load_free(self, screen):
        """추력 0 패턴은 자유낙하 — relief 폐합 후 잔여 6분력 0."""
        forces, nz = screen._pattern_forces(np.zeros(screen.n))
        tot = np.zeros(6)
        for f in forces.values():
            tot[:3] += f[:3]
        assert nz == 0.0
        assert np.abs(tot[:3]).max() < 1e-6


class TestEventSpace:
    def test_event_count_and_modes(self, events):
        """호버 계열: M1/M4/M5 각 8 + M2 거울쌍 4 + M3 같은쪽쌍 ≥2."""
        by_mode = {}
        for e in events:
            if e.phase == "HV":
                by_mode.setdefault(e.mode, []).append(e)
        assert len(by_mode["M1"]) == 8
        assert len(by_mode["M4"]) == 8
        assert len(by_mode["M5"]) == 8
        assert len(by_mode["M2"]) == 4
        assert len(by_mode["M3"]) >= 2

    def test_failed_rotor_pinned_per_mode(self, screen, events):
        """호버 고장 로터 레벨: M1/M2/M3=0, M4=1.0, M5=1.5."""
        pin = {"M1": 0.0, "M2": 0.0, "M3": 0.0, "M4": 1.0, "M5": 1.5}
        idx = {r.rotor_id: k for k, r in enumerate(screen.rotors)}
        for e in events:
            if e.phase != "HV":
                continue
            for rid in e.failed_ids:
                assert e.pattern[idx[rid]] == pytest.approx(pin[e.mode])

    def test_survivors_stay_in_command_band(self, screen, events):
        """생존 로터는 정상 지령 대역 — 추가 비정상 로터 없음
        (P가 부모 사건 그대로 승계되는 근거)."""
        idx = {r.rotor_id: k for k, r in enumerate(screen.rotors)}
        for e in events:
            if e.phase != "HV":
                continue
            failed = {idx[rid] for rid in e.failed_ids}
            for k, li in enumerate(e.pattern):
                if k not in failed:
                    assert 0.7 <= li <= 1.3

    def test_probability_inheritance(self, screen, events):
        """P = (고장 로터 수/n) x P_phase x P_mode."""
        for e in events:
            p_phase = 0.10 if e.phase == "HV" else 0.025
            expect = (len(e.failed_ids) / screen.n) * p_phase * \
                screen.p_mode[e.mode]
            assert e.P == pytest.approx(expect)

    def test_retrim_space_adds_content(self, events):
        """재트림 공간이 이산 격자 밖 하중을 실제로 만든다 —
        반례가 예고한 대로 C가 유의미하게 큼."""
        assert max(e.consequence for e in events) > 0.05
        assert all(e.consequence >= 0.0 for e in events)

    def test_ranking_by_risk(self, events):
        risks = [e.risk for e in events]
        assert risks == sorted(risks, reverse=True)


class TestRealize:
    def test_adoption_threshold_and_cases(self, screen, events):
        cases = screen.realize(events, threshold_pct=1.0, top_n=5)
        assert len(cases) == 5
        for c in cases:
            assert c.category == "vtol_retrim"
            assert c.converged
            assert c.nodal_forces
            assert c.flight_state["retrim_C_pct"] >= 1.0
            # 자기평형: relief 폐합 후 합력 == 0
            tot = np.zeros(3)
            for f in c.nodal_forces.values():
                tot += f[:3]
            assert np.abs(tot).max() < 1e-4

    def test_threshold_filters(self, screen, events):
        none = screen.realize(events, threshold_pct=1e6)
        assert none == []


class TestTransition:
    def test_tr_events_present(self, events):
        """천이 계열 사건이 (V, nz) 해상도로 생성된다."""
        tr = [e for e in events if e.phase == "TR"]
        assert len(tr) > 0
        assert all(e.V_eas > 0 for e in tr)
        assert {e.nz_cond for e in tr} >= {1.0}

    def test_tr_pin_and_schedule_band(self, screen, events):
        """천이: 고장 로터는 {0, tf(고착), 1.5(폭주)} 고정, 생존
        로터는 스케줄 tf x [0.7, 1.3] 대역."""
        idx = {r.rotor_id: k for k, r in enumerate(screen.rotors)}
        checked = 0
        for e in events:
            if e.phase != "TR":
                continue
            tf = screen._entry_for(e)["tf"]
            pin = {"M1": 0.0, "M2": 0.0, "M3": 0.0,
                   "M4": tf, "M5": 1.5}[e.mode]
            failed = {idx[rid] for rid in e.failed_ids}
            for k, li in enumerate(e.pattern):
                if k in failed:
                    assert li == pytest.approx(pin)
                else:
                    assert (tf * 0.7 - 1e-9 <= li
                            <= tf * 1.3 + 1e-9)
            checked += 1
        assert checked > 0

    def test_tr_nz_eff_within_bounds(self, screen, events):
        """nz_eff = (nz_cond + 돌풍) - 로터 분담이 [-0.5, 실속 상한] 안."""
        for e in events:
            if e.phase != "TR":
                continue
            nz_eff = (e.nz_cond + e.gust_dn
                      - e.pattern.sum() / screen.n)
            assert nz_eff >= -0.5 - 1e-9
            assert nz_eff <= screen._nz_eff_max(e.V_eas) + 1e-9

    def test_tr_affine_prediction_matches_assembly(self, screen, events):
        """기저 아핀 예측 VMT == 실제 힘 조립 + 적분 VMT (선형 정합)."""
        from ascent_load.loads_analysis.certification.batch_runner import (
            BatchResult, CaseResult,
        )
        from ascent_load.loads_analysis.certification.vmt_bridge import (
            compute_vmt_for_batch,
        )

        e = next(ev for ev in events if ev.phase == "TR"
                 and ev.consequence > 0.01)
        entry = screen._entry_for(e)
        nz_g = e.nz_cond + e.gust_dn
        pred, _ = screen._tr_pattern_rows(entry, e.pattern[None, :],
                                          nz_g=nz_g)
        forces, nz = screen._tr_pattern_forces(entry, e.pattern,
                                               nz_g=nz_g)
        batch = BatchResult()
        batch.case_results.append(CaseResult(
            case_id=1, category="t", converged=True,
            nodal_forces=forces, nz=nz, label="tr"))
        batch.completed_ids.add(1)
        vmt = compute_vmt_for_batch(screen.model, batch,
                                    components=screen.components,
                                    fuselage_cg_x=screen._fus_cg_x)
        actual = screen._vmt_mat(vmt[1])
        scale = np.abs(actual).max()
        assert np.abs(pred[0] - actual).max() <= 1e-6 * scale

    def test_tr_gust_axis(self, screen, events):
        """돌풍 축: gust_dn은 조건별 {−dn, 0, +dn} 중 하나이고,
        일부 사건에서 돌풍 환경이 지배(비영) — ①과 동일 혼합 모델."""
        seen_nonzero = False
        for e in events:
            if e.phase != "TR":
                continue
            dn = screen._entry_for(e)["dn"]
            assert dn > 0.0
            assert any(abs(e.gust_dn - v) < 1e-9
                       for v in (-dn, 0.0, dn))
            if e.gust_dn != 0.0:
                seen_nonzero = True
        assert seen_nonzero

    def test_tr_realize_self_equilibrated(self, screen, events):
        """천이 재트림 케이스도 relief 폐합 후 합력 0."""
        e = next(ev for ev in events if ev.phase == "TR")
        cases = screen.realize([e], threshold_pct=0.0, top_n=1)
        assert len(cases) == 1
        c = cases[0]
        assert c.flight_state["retrim_phase"] == "TR"
        assert c.mach > 0.0
        tot = np.zeros(3)
        for f in c.nodal_forces.values():
            tot += f[:3]
        assert np.abs(tot).max() < 1e-3


ILC8T = os.path.join(os.path.dirname(__file__), "validation", "ILC8T")


@pytest.fixture(scope="module")
def tilt_screen():
    """ILC-8T (틸트) 선별기 — 속도 2점으로 축소한 틸트 기저."""
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.loads_analysis.certification.aircraft_config import (
        AircraftConfig,
    )
    from ascent_load.loads_analysis.certification.retrim_events import (
        RetrimScreen,
    )
    from ascent_load.loads_analysis.component_id import (
        identify_components_manual,
    )
    from ascent_load.models.ilc8t import make_ilc8t_vtol_config

    model = parse_bdf(os.path.join(ILC8T, "ilc8t.bdf"))
    with open(os.path.join(ILC8T, "ilc8t_cert_config.yaml")) as f:
        config = AircraftConfig.from_dict(yaml.safe_load(f))
    vtol_config = make_ilc8t_vtol_config()

    def _nids(lo, hi, extra=()):
        ids = [n for n in model.nodes if lo <= n <= hi]
        return ids + [n for n in extra if n in model.nodes]

    WING = dict(span_axis=1, shear_axis=2, bending_axis=0,
                torsion_axis=1)
    comps = identify_components_manual(model, [
        dict(name="Right Wing", integration_sign=1.0, color="blue",
             node_ids=_nids(400000, 499999)
             + _nids(730000, 749999,
                     (990103, 990104, 990107, 990108)), **WING),
        dict(name="Left Wing", integration_sign=-1.0,
             color="dodgerblue",
             node_ids=_nids(300000, 399999)
             + _nids(710000, 729999,
                     (990101, 990102, 990105, 990106)), **WING),
        dict(name="Fuselage", integration_sign=-1.0, color="gray",
             node_ids=_nids(100000, 299999, (990201,)),
             span_axis=0, shear_axis=2, bending_axis=1,
             torsion_axis=0),
    ])

    class _FastTilt(RetrimScreen):
        def _build_tilt_bases(self, n_speeds: int = 2):
            return super()._build_tilt_bases(n_speeds=2)

    return _FastTilt(model, vtol_config, config, components=comps,
                     fuselage_cg_x=4450.0)


@pytest.fixture(scope="module")
def tilt_events(tilt_screen):
    return tilt_screen.screen()


class TestTiltFamily:
    def test_tilt_family_replaces_tr(self, tilt_screen, tilt_events):
        """틸트 기체: TILT 계열이 천이(TR) 계열을 대체하고 M1~M6이
        모두 나타나며, 호버 계열은 그대로 유지된다."""
        assert tilt_screen._has_tilt
        assert tilt_screen._tr == []
        assert len(tilt_screen._tilt) > 0
        phases = {e.phase for e in tilt_events}
        assert "TILT" in phases and "HV" in phases
        assert "TR" not in phases
        modes = {e.mode for e in tilt_events if e.phase == "TILT"}
        assert modes == {"M1", "M2", "M3", "M4", "M5", "M6"}

    def test_tilt_state_membership(self, tilt_screen, tilt_events):
        """모든 TILT 사건은 실제로 구축된 (V, σ) 기저 상태에 속한다."""
        states = {(round(en["V"], 3), round(en["sigma"], 3))
                  for en in tilt_screen._tilt}
        for e in tilt_events:
            if e.phase == "TILT":
                assert (round(e.V_eas, 3),
                        round(e.tilt_deg, 3)) in states

    def test_tilt_m6_stuck_fields_and_p(self, tilt_screen,
                                        tilt_events):
        """M6: 고착각 {0,90}, 스케줄각과 5° 미만 상태는 제외, 고착
        로터는 틸트열(can-tilt)만, P=(1/n)·P_tr·P_M6."""
        from ascent_load.rotor.rotor_config import RotorType

        tilt_ids = {r.rotor_id for r in tilt_screen.rotors
                    if r.rotor_type == RotorType.TILT}
        m6 = [e for e in tilt_events if e.mode == "M6"]
        assert len(m6) > 0
        p_ref = ((1.0 / tilt_screen.n) * tilt_screen.p_phase_tr
                 * tilt_screen.p_mode_m6)
        for e in m6:
            assert e.phase == "TILT"
            assert e.stuck_rotor_id in tilt_ids
            assert e.stuck_deg in (0.0, 90.0)
            assert abs(e.stuck_deg - e.tilt_deg) >= 5.0
            assert e.P == pytest.approx(p_ref)

    def test_tilt_p_inheritance(self, tilt_screen, tilt_events):
        """M1~M5 P = (고장수/n)·P_tr·P_mode (천이 위상확률 상속)."""
        for e in tilt_events:
            if e.phase != "TILT" or e.mode == "M6":
                continue
            p_ref = ((len(e.failed_ids) / tilt_screen.n)
                     * tilt_screen.p_phase_tr
                     * tilt_screen.p_mode[e.mode])
            assert e.P == pytest.approx(p_ref)

    def test_tilt_gust_axis(self, tilt_screen, tilt_events):
        """돌풍 환경축: gust_dn ∈ {−dn, 0, +dn} (상태별 혼합 dn)."""
        for e in tilt_events:
            if e.phase != "TILT":
                continue
            dn = tilt_screen._tilt_entry_for(e)["dn"]
            assert dn > 0.0
            assert any(abs(e.gust_dn - v) < 1e-9
                       for v in (-dn, 0.0, dn))

    def test_tilt_nz_eff_within_bounds(self, tilt_screen,
                                       tilt_events):
        """지배 패턴의 nz_eff가 선별 필터 대역 안에 있다."""
        for e in tilt_events:
            if e.phase != "TILT":
                continue
            entry = tilt_screen._tilt_entry_for(e)
            stuck = None
            if e.stuck_rotor_id is not None:
                k = next(i for i, r in
                         enumerate(tilt_screen.rotors)
                         if r.rotor_id == e.stuck_rotor_id)
                stuck = (k, e.stuck_deg)
            _, nz_eff = tilt_screen._tilt_rows(
                entry, e.pattern[None, :], stuck=stuck,
                nz_g=1.0 + e.gust_dn)
            hi = tilt_screen._nz_eff_max(e.V_eas)
            assert -0.5 - 1e-9 <= nz_eff[0] <= hi + 1e-9

    def _assert_affine_matches(self, tilt_screen, e):
        from ascent_load.loads_analysis.certification.batch_runner import (
            BatchResult, CaseResult,
        )
        from ascent_load.loads_analysis.certification.vmt_bridge import (
            compute_vmt_for_batch,
        )

        entry = tilt_screen._tilt_entry_for(e)
        stuck = None
        if e.stuck_rotor_id is not None:
            k = next(i for i, r in enumerate(tilt_screen.rotors)
                     if r.rotor_id == e.stuck_rotor_id)
            stuck = (k, e.stuck_deg)
        pred, _ = tilt_screen._tilt_rows(entry, e.pattern[None, :],
                                         stuck=stuck,
                                         nz_g=1.0 + e.gust_dn)
        forces, nz = tilt_screen._tilt_pattern_forces(entry, e)
        batch = BatchResult()
        batch.case_results.append(CaseResult(
            case_id=1, category="t", converged=True,
            nodal_forces=forces, nz=nz, label="tilt"))
        batch.completed_ids.add(1)
        vmt = compute_vmt_for_batch(
            tilt_screen.model, batch,
            components=tilt_screen.components,
            fuselage_cg_x=tilt_screen._fus_cg_x)
        actual = tilt_screen._vmt_mat(vmt[1])
        scale = np.abs(actual).max()
        assert np.abs(pred[0] - actual).max() <= 1e-6 * scale

    def test_tilt_affine_prediction_matches_assembly(
            self, tilt_screen, tilt_events):
        """틸트 아핀 예측 VMT == 실제 힘 조립 + 적분 VMT."""
        e = next(ev for ev in tilt_events if ev.phase == "TILT"
                 and ev.mode != "M6" and ev.consequence > 0.01)
        self._assert_affine_matches(tilt_screen, e)

    def test_tilt_stuck_prediction_matches_assembly(
            self, tilt_screen, tilt_events):
        """M6 고착 기저 치환(Bstuck/Fstuck)도 조립과 정합."""
        e = next(ev for ev in tilt_events if ev.mode == "M6")
        self._assert_affine_matches(tilt_screen, e)

    def test_tilt_realize_self_equilibrated(self, tilt_screen,
                                            tilt_events):
        """틸트 재트림 케이스도 relief 폐합 후 합력 0."""
        e = next(ev for ev in tilt_events if ev.phase == "TILT")
        cases = tilt_screen.realize([e], threshold_pct=0.0, top_n=1)
        assert len(cases) == 1
        c = cases[0]
        assert c.flight_state["retrim_phase"] == "TILT"
        assert c.category == "vtol_retrim"
        assert c.mach > 0.0
        tot = np.zeros(3)
        for f in c.nodal_forces.values():
            tot += f[:3]
        assert np.abs(tot).max() < 1e-3
