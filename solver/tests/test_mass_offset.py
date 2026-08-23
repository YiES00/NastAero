# CONM2 오프셋 질량의 CG·관성하중·관성해제 일관성 시험 (2026-08 감사)
"""집중질량이 절점에서 떨어져 있을 때의 질량 부기 일관성 검증.

감사에서 확인된 결함: 트림 경로(compute_node_masses)가 CONM2
오프셋을 무시하고 질량을 절점에 그대로 집중시켜, CG와 관성하중의
모멘트 팔이 실제 질량중심과 어긋났다. GACOMP 비교 모델은 CONM2
18,406장이 전부 CID=-1(절대좌표)이라 CG가 (-17.3, 0, +21.4) mm
어긋나 있었고, 이는 기준 코드 1350 mm의 1.3%다.
"""
from __future__ import annotations
import numpy as np
import pytest
from nastaero.bdf.model import BDFModel
from nastaero.bdf.cards.grid import GRID
from nastaero.bdf.cards.mass import CONM2
from nastaero.loads_analysis.trim_loads import (
    apply_inertia_relief, compute_node_masses, compute_node_mass_centroids,
    compute_nodal_inertial_forces)


def _cloud(n=8, seed=7, cid=0):
    rng = np.random.default_rng(seed)
    model = BDFModel()
    for i in range(1, n + 1):
        g = GRID()
        g.nid = i
        g.xyz = rng.uniform(-500, 500, 3)
        g.xyz_global = g.xyz.copy()
        model.nodes[i] = g
    for i in range(1, n + 1):
        c = CONM2()
        c.eid = i
        c.node_id = i
        c.cid = cid
        c.mass = float(rng.uniform(0.1, 2.0))
        off = rng.uniform(-80, 80, 3)
        c.offset = (model.nodes[i].xyz_global + off) if cid == -1 else off
        model.masses[i] = c
    return model


class TestMassCentroids:
    def test_offset_shifts_centroid(self):
        model = _cloud()
        cen = compute_node_mass_centroids(model)
        for nid, c in model.masses.items():
            expected = model.nodes[nid].xyz_global + c.offset
            np.testing.assert_allclose(cen[nid], expected, atol=1e-9)

    def test_cid_minus_one_is_absolute(self):
        """CID=-1이면 X1~X3가 곧 질량중심 절대좌표다."""
        model = _cloud(cid=-1)
        cen = compute_node_mass_centroids(model)
        for nid, c in model.masses.items():
            np.testing.assert_allclose(cen[nid], c.offset, atol=1e-9)

    def test_no_offset_gives_node_position(self):
        model = _cloud()
        for c in model.masses.values():
            c.offset = np.zeros(3)
        cen = compute_node_mass_centroids(model)
        for nid in model.nodes:
            np.testing.assert_allclose(cen[nid],
                                       model.nodes[nid].xyz_global, atol=1e-9)


class TestInertialLoadOffsetMoment:
    def test_offset_mass_carries_moment(self):
        """질량중심에 작용하는 힘을 절점으로 옮기면 팔 모멘트가 생긴다."""
        model = _cloud()
        nf = compute_nodal_inertial_forces(model, nz=1.0, g=9810.0)
        cen = compute_node_mass_centroids(model)
        for nid, f in nf.items():
            d = cen[nid] - model.nodes[nid].xyz_global
            np.testing.assert_allclose(f[3:6], np.cross(d, f[:3]), atol=1e-9)

    def test_total_moment_matches_mass_centroids(self):
        """절점 하중의 총 모멘트가 질량중심 기준 결과와 같아야 한다."""
        model = _cloud()
        nf = compute_nodal_inertial_forces(model, nz=1.0, g=9810.0)
        cen = compute_node_mass_centroids(model)
        M_nodal = np.zeros(3)
        M_direct = np.zeros(3)
        for nid, f in nf.items():
            M_nodal += np.cross(model.nodes[nid].xyz_global, f[:3]) + f[3:6]
            M_direct += np.cross(cen[nid], f[:3])
        np.testing.assert_allclose(M_nodal, M_direct, atol=1e-6)


class TestReliefClosure:
    @pytest.mark.parametrize("cid", [0, -1])
    def test_closure_exact_with_offsets(self, cid):
        """오프셋 질량이 있어도 관성해제 후 6분력이 정확히 0이어야 한다."""
        rng = np.random.default_rng(11)
        model = _cloud(cid=cid)
        nm = compute_node_masses(model)
        cen = compute_node_mass_centroids(model)
        total = sum(nm.values())
        cg = sum(nm[n] * cen[n] for n in nm) / total

        combined = {n: np.concatenate([rng.uniform(-50, 50, 3), np.zeros(3)])
                    for n in model.nodes}
        inertial = {n: np.zeros(6) for n in model.nodes}
        apply_inertia_relief(model, inertial, combined, cg, 9810.0)

        F = np.zeros(3)
        M = np.zeros(3)
        for nid, f in combined.items():
            r = model.nodes[nid].xyz_global - cg
            F += f[:3]
            M += np.cross(r, f[:3]) + f[3:6]
        assert np.linalg.norm(F) < 1e-9, f"잔여 합력 {F}"
        assert np.linalg.norm(M) < 1e-6, f"잔여 합모멘트 {M}"
