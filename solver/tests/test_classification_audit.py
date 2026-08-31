# 구성품 분류 감사·fail-closed 게이트(r3 MC8) 시험
from __future__ import annotations
import numpy as np
import pytest
from types import SimpleNamespace

from nastaero.loads_analysis.component_id import (
    ComponentDef, ComponentSet, ClassificationAudit,
    ComponentClassificationError, audit_classification,
    assert_classification_complete,
)


def _model(coords):
    nodes = {i + 1: SimpleNamespace(xyz_global=np.asarray(c, dtype=float))
             for i, c in enumerate(coords)}
    return SimpleNamespace(nodes=nodes)


def _comp(name, nids):
    return ComponentDef(name=name, node_ids=list(nids), span_axis=1,
                        shear_axis=2, bending_axis=0, torsion_axis=1,
                        integration_sign=1.0)


@pytest.fixture
def setup():
    # 절점 1-4 = 날개, 5-6 = 붐(미분류), 7 = 중복 분류 후보
    coords = [[0, y, 0] for y in (100, 200, 300, 400)] \
        + [[500, 150, 0], [800, 150, 0]] + [[0, 500, 0]]
    m = _model(coords)
    comps = ComponentSet(components=[_comp('Wing', [1, 2, 3, 4])])
    return m, comps


class TestAudit:
    def test_clean_audit(self, setup):
        m, comps = setup
        comps.components[0].node_ids = list(m.nodes)  # 전 절점 분류
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0]) for nid in m.nodes}
        a = audit_classification(m, comps, forces)
        assert a.ok()
        assert a.loaded_unclassified == []
        assert np.linalg.norm(a.force_residual) < 1e-12
        assert np.linalg.norm(a.moment_residual) < 1e-9

    def test_loaded_unclassified_detected(self, setup):
        m, comps = setup
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0]) for nid in m.nodes}
        a = audit_classification(m, comps, forces)
        assert not a.ok()
        bad = {t[0] for t in a.loaded_unclassified}
        assert bad == {5, 6, 7}
        # 잔차 = 미분류 절점의 하중 합
        assert a.force_residual[2] == pytest.approx(300.0)

    def test_unloaded_unclassified_is_reported_not_fatal(self, setup):
        m, comps = setup
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0])
                  for nid in (1, 2, 3, 4)}
        a = audit_classification(m, comps, forces)
        assert a.unclassified_nids == [5, 6, 7]
        assert a.loaded_unclassified == []
        assert np.linalg.norm(a.force_residual) < 1e-12

    def test_multi_assignment_detected(self, setup):
        m, comps = setup
        comps.components.append(_comp('Boom', [4, 5, 6, 7]))  # 4 중복
        a = audit_classification(m, comps)
        assert 4 in a.multi_assigned
        assert set(a.multi_assigned[4]) == {'Wing', 'Boom'}


class TestFailClosed:
    def test_raises_on_loaded_unclassified(self, setup):
        m, comps = setup
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0]) for nid in m.nodes}
        with pytest.raises(ComponentClassificationError,
                           match="not.*assigned"):
            assert_classification_complete(m, comps, forces)

    def test_raises_on_double_count(self, setup):
        m, comps = setup
        comps.components.append(_comp('Boom', [4, 5, 6, 7]))
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0]) for nid in m.nodes}
        with pytest.raises(ComponentClassificationError,
                           match="more than"):
            assert_classification_complete(m, comps, forces)

    def test_passes_clean(self, setup):
        m, comps = setup
        comps.components.append(_comp('Boom', [5, 6, 7]))
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0]) for nid in m.nodes}
        a = assert_classification_complete(m, comps, forces)
        assert isinstance(a, ClassificationAudit)
        assert a.ok()


class TestBridgeStrictMode:
    def test_bridge_gate(self, setup):
        from nastaero.loads_analysis.certification.batch_runner import (
            BatchResult, CaseResult)
        from nastaero.loads_analysis.certification.vmt_bridge import (
            compute_vmt_for_batch)
        m, comps = setup
        forces = {nid: np.array([0, 0, 100.0, 0, 0, 0]) for nid in m.nodes}
        br = BatchResult()
        br.case_results = [CaseResult(case_id=1, converged=True,
                                      nodal_forces=forces)]
        with pytest.raises(ComponentClassificationError):
            compute_vmt_for_batch(m, br, components=comps,
                                  strict_classification=True)
        # 기본값은 종전과 동일하게 통과(경고 없이 적분)
        out = compute_vmt_for_batch(m, br, components=comps)
        assert 1 in out
