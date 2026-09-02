# 국부 6분력이 포락선·임계선정·출력 경로까지 전달되는지 확인 (r4 MC3)
from __future__ import annotations
import numpy as np
import pytest
from types import SimpleNamespace

from ascent_load.loads_analysis.component_id import ComponentDef, ComponentSet
from ascent_load.loads_analysis.certification.batch_runner import (
    BatchResult, CaseResult,
)
from ascent_load.loads_analysis.certification.vmt_bridge import (
    compute_vmt_for_batch,
)
from ascent_load.loads_analysis.certification.envelope import (
    EnvelopeProcessor, LOCAL_QUANTITIES, select_critical_design_loads,
)


def _canted_model(gamma_deg=40.0, n_span=15, n_chord=3, span=1000.0,
                  chord=200.0):
    """40도 경사면 — 전역축 투영이 단면 전단을 과소평가하는 형상."""
    g = np.deg2rad(gamma_deg)
    e1 = np.array([0.0, np.cos(g), np.sin(g)])
    coords = []
    for u in np.linspace(0.0, span, n_span):
        for x in np.linspace(0.0, chord, n_chord):
            coords.append(np.array([x, 0.0, 0.0]) + u * e1)
    nodes = {i + 1: SimpleNamespace(xyz_global=np.asarray(c, float))
             for i, c in enumerate(coords)}
    return SimpleNamespace(nodes=nodes), g


def _comp(nids):
    return ComponentDef(name="Canted Fin", node_ids=list(nids), span_axis=1,
                        shear_axis=2, bending_axis=0, torsion_axis=1,
                        integration_sign=1.0)


def _batch(model, g, n_cases=3):
    """면법선 하중 크기만 다른 케이스들 — 국부 Vz가 케이스를 가른다."""
    n_hat = np.array([0.0, -np.sin(g), np.cos(g)])
    br = BatchResult()
    for k in range(n_cases):
        f = 50.0 * (k + 1)
        forces = {nid: np.concatenate([f * n_hat, np.zeros(3)])
                  for nid in model.nodes}
        br.case_results.append(CaseResult(
            case_id=k + 1, category="symmetric", far_section="23.337",
            converged=True, nz=1.0 + k, label=f"case {k+1}",
            nodal_forces=forces))
    br.completed_ids = {c.case_id for c in br.case_results}
    return br


@pytest.fixture
def setup():
    model, g = _canted_model()
    comps = ComponentSet(components=[_comp(model.nodes)])
    return model, g, comps, _batch(model, g)


class TestBridgeCarriesLocal:
    def test_bridge_keeps_local_arrays(self, setup):
        model, g, comps, br = setup
        data = compute_vmt_for_batch(model, br, components=comps,
                                     n_stations=8)
        entry = data[1]["Canted Fin"]
        # 전역 3키는 그대로 (기존 소비자 무회귀)
        for k in ("stations", "shear", "bending", "torsion"):
            assert k in entry
        # 국부 6분력이 함께 전달된다
        assert entry["local_stations"] is not None
        for q in LOCAL_QUANTITIES:
            assert q in entry and entry[q] is not None
        assert entry["local_frame"].shape == (3, 3)
        assert entry["local_cut_points"].shape[1] == 3

    def test_local_captures_what_global_misses(self, setup):
        model, g, comps, br = setup
        data = compute_vmt_for_batch(model, br, components=comps,
                                     n_stations=8)
        e = data[3]["Canted Fin"]
        # 전역 수직 전단은 면법선 전단의 cos(gamma)만 포착
        assert abs(e["Vz"][0]) > abs(e["shear"][0])
        assert e["shear"][0] == pytest.approx(
            e["Vz"][0] * np.cos(g), rel=1e-6)


class TestEnvelopeAndSelection:
    def test_envelope_accumulates_local_extremes(self, setup):
        model, g, comps, br = setup
        data = compute_vmt_for_batch(model, br, components=comps,
                                     n_stations=8)
        proc = EnvelopeProcessor(br, data)
        proc.compute_envelopes()
        env = proc.get_envelope("Canted Fin")
        se = env.envelopes[0]
        assert set(se.local) == set(LOCAL_QUANTITIES)
        # 하중이 가장 큰 케이스 3이 국부 Vz 극값을 지배
        lo, hi = se.local["Vz"]
        assert hi > lo
        assert 3 in se.local_case["Vz"]

    def test_local_critical_cases_recorded(self, setup):
        model, g, comps, br = setup
        data = compute_vmt_for_batch(model, br, components=comps,
                                     n_stations=8)
        proc = EnvelopeProcessor(br, data)
        proc.compute_envelopes()
        added = proc.identify_local_critical_cases()
        assert added
        qtys = {c.quantity for c in added}
        assert qtys <= set(LOCAL_QUANTITIES)
        assert "Vz" in qtys
        # 전역 기록과 이름이 겹치지 않는다
        assert not (qtys & {"V", "M", "T"})

    def test_selection_flag_changes_design_set(self, setup, monkeypatch):
        model, g, comps, br = setup
        data = compute_vmt_for_batch(model, br, components=comps,
                                     n_stations=8)
        import ascent_load.loads_analysis.certification.vmt_bridge as vb
        monkeypatch.setattr(vb, "compute_vmt_for_batch",
                            lambda *a, **k: data)
        glob = select_critical_design_loads(
            model, br, include_3d=False, vmt_data=data,
            components=comps, include_local6=False)
        loc = select_critical_design_loads(
            model, br, include_3d=False, vmt_data=data,
            components=comps, include_local6=True)
        n_glob = sum(1 for d in glob["design_cases"]
                     for gq in d.governs if gq[2] in ("V", "M", "T"))
        n_loc = sum(1 for d in loc["design_cases"]
                    for gq in d.governs if gq[2] in LOCAL_QUANTITIES)
        assert n_glob > 0
        assert n_loc > 0        # 국부 기록이 선정에 실제로 들어간다
        assert loc["n_critical"] > glob["n_critical"]
