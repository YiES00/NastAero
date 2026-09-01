# RBE3 가중 최소자승 보간 조립 시험 (2026-08 감사)
"""RBE3가 조립에 반영되는지, 강체 운동을 정확히 재현하는지 검증.

감사에서 확인된 결함: RBE3는 파싱만 되고 조립 단계에서 무시되어
REFGRID가 아무 구속도 받지 않았다(경고도 없음). GACOMP 비교 모델은
RBE3 30개 중 26개의 REFGRID에 요소가 붙어 있지 않고 25개에 CONM2
질량이 달려 있어, 그 질량의 관성 하중이 구조로 전달되지 않았다.
"""
from __future__ import annotations
import numpy as np
import pytest
from ascent_load.bdf.model import BDFModel
from ascent_load.bdf.cards.grid import GRID
from ascent_load.bdf.cards.rbe import RBE3
from ascent_load.fem.dof_manager import DOFManager
from ascent_load.fem.assembly import _build_rbe3_slave_deps


def _model(ref_xyz, indep):
    model = BDFModel()
    pts = dict(indep)
    pts[99] = ref_xyz
    for nid, p in pts.items():
        g = GRID()
        g.nid = nid
        g.xyz = np.array(p, dtype=float)
        g.xyz_global = g.xyz.copy()
        model.nodes[nid] = g
    r = RBE3()
    r.eid = 1
    r.refgrid = 99
    r.refc = "123456"
    r.weight_sets = [(1.0, "123", sorted(indep))]
    model.rigids[1] = r
    return model, r


def _predict(deps, dof_mgr, u):
    pred = np.zeros(6)
    for c in range(6):
        ds = dof_mgr.get_dof(99, c + 1)
        pred[c] = sum(coef * u[md] for md, coef in deps.get(ds, []))
    return pred


class TestRBE3Interpolation:
    def _run(self, indep, ref_xyz, t, th):
        model, r = _model(ref_xyz, indep)
        dof_mgr = DOFManager(sorted(model.nodes))
        deps = {}
        _build_rbe3_slave_deps(r, model, dof_mgr, deps)
        u = np.zeros(dof_mgr.total_dof)
        for nid, p in indep.items():
            d = np.array(t) + np.cross(th, np.array(p, dtype=float))
            for c in range(3):
                u[dof_mgr.get_dof(nid, c + 1)] = d[c]
        exact = np.concatenate([
            np.array(t) + np.cross(th, np.array(ref_xyz, dtype=float)),
            np.array(th)])
        return _predict(deps, dof_mgr, u), exact, deps

    def test_rigid_motion_reproduced_exactly(self):
        """독립 절점이 강체 운동하면 REFGRID도 같은 강체 운동이어야 한다."""
        indep = {1: [0, 0, 0], 2: [10, 0, 0], 3: [10, 10, 0], 4: [0, 10, 0]}
        pred, exact, deps = self._run(indep, [5.0, 5.0, 3.0],
                                      [0.3, -0.2, 0.7], [0.01, -0.02, 0.015])
        assert len(deps) == 6, "REFC 123456이면 종속 자유도가 6개여야 한다"
        np.testing.assert_allclose(pred, exact, atol=1e-10)

    def test_pure_translation(self):
        indep = {1: [0, 0, 0], 2: [4, 0, 0], 3: [0, 6, 0]}
        pred, exact, _ = self._run(indep, [1.0, 2.0, 0.5],
                                   [1.0, 2.0, -3.0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pred, exact, atol=1e-10)

    def test_collinear_independents_do_not_blow_up(self):
        """공선 배치에서도 유사역행렬로 유한한 계수를 내야 한다."""
        indep = {1: [0, 0, 0], 2: [5, 0, 0], 3: [10, 0, 0]}
        model, r = _model([5.0, 0.0, 2.0], indep)
        dof_mgr = DOFManager(sorted(model.nodes))
        deps = {}
        _build_rbe3_slave_deps(r, model, dof_mgr, deps)
        for terms in deps.values():
            for _, c in terms:
                assert np.isfinite(c)

    @pytest.mark.parametrize("weights", [(1.0, 1.0), (3.0, 1.0)])
    def test_linear_field_reproduced_regardless_of_weights(self, weights):
        """선형 변위장은 가중치와 무관하게 정확히 재현되어야 한다.

        z가 x에 선형인 장을 두 절점에 주면 최소자승 강체 적합이
        그 기울기를 정확히 잡아내므로, 중간 지점 REFGRID는 선형
        보간값을 가진다. 가중 평균만으로 계산하면 틀린다.
        """
        model, r = _model([5.0, 0.0, 0.0], {1: [0, 0, 0], 2: [10, 0, 0]})
        r.weight_sets = [(weights[0], "123", [1]), (weights[1], "123", [2])]
        dof_mgr = DOFManager(sorted(model.nodes))
        deps = {}
        _build_rbe3_slave_deps(r, model, dof_mgr, deps)
        u = np.zeros(dof_mgr.total_dof)
        u[dof_mgr.get_dof(1, 3)] = 0.0
        u[dof_mgr.get_dof(2, 3)] = 4.0
        pred = _predict(deps, dof_mgr, u)
        assert pred[2] == pytest.approx(2.0, abs=1e-9)   # x=5의 선형 보간
        assert pred[4] == pytest.approx(-0.4, abs=1e-9)  # theta_y = -dz/dx


class TestRBE3Assembly:
    def test_rbe3_is_assembled(self):
        """RBE3가 소거 대상 종속 자유도로 실제 조립되어야 한다."""
        from ascent_load.fem.assembly import assemble_global_matrices
        from ascent_load.bdf.cards.elements import CELAS2
        indep = {1: [0, 0, 0], 2: [10, 0, 0], 3: [10, 10, 0], 4: [0, 10, 0]}
        model, _ = _model([5.0, 5.0, 0.0], indep)
        for i, nid in enumerate(sorted(indep)):
            sp = CELAS2()
            sp.eid = 100 + i
            sp.k = 1000.0
            sp.g1 = nid
            sp.c1 = 3
            sp.g2 = 0
            sp.c2 = 0
            model.springs[100 + i] = sp
        model.cross_reference()
        dof_mgr = DOFManager(sorted(model.nodes))
        K, _, _ = assemble_global_matrices(model, dof_mgr)
        # REFGRID의 z 자유도는 소거되어 독립 절점으로 옮겨간다
        ref_z = dof_mgr.get_dof(99, 3)
        assert abs(K.tocsr()[ref_z, ref_z]) < 1e-9
