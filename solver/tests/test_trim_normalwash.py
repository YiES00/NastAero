# 트림변수 노멀워시 법선 투영 + 트림 질량(중량/CG)-관성력 일관성 단위시험
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ascent_load.solvers.sol144 import (
    _compute_cg_x, _compute_total_weight, _detect_gravity,
    _trim_variable_normalwash,
)

VM_DIR = Path(__file__).parent / "validation" / "nastran_vm"


def _boxes():
    """수평 패널 1개(법선 +Z) + 수직 패널 1개(법선 +Y, VTP류)."""
    horiz = SimpleNamespace(box_id=1, control_point=np.array([1000., 500., 0.]),
                            normal=np.array([0., 0., 1.]))
    vert = SimpleNamespace(box_id=2, control_point=np.array([8000., 0., 1500.]),
                           normal=np.array([0., 1., 0.]))
    return [horiz, vert]


def _model():
    return SimpleNamespace(
        aeros=SimpleNamespace(refb=10000.0, refc=1500.0),
        aesurfs={}, aelists={}, aelinks=[], caero_panels={},
        nodes={}, masses={},
    )


class TestNormalwashProjection:
    def test_anglea_zero_on_vertical_panel(self):
        w = _trim_variable_normalwash("ANGLEA", _boxes(), {1: 0, 2: 1},
                                      _model())
        assert w[0] == -1.0   # 수평 패널: 전량
        assert w[1] == 0.0    # 수직 패널: 받음각 워시 없음

    def test_anglea_scales_with_dihedral(self):
        boxes = _boxes()
        boxes[0].normal = np.array([0., -0.5, np.sqrt(0.75)])  # 상반각 30도
        w = _trim_variable_normalwash("ANGLEA", boxes, {1: 0, 2: 1}, _model())
        assert abs(w[0] - (-np.sqrt(0.75))) < 1e-12

    def test_sides_full_on_vertical_panel(self):
        w = _trim_variable_normalwash("SIDES", _boxes(), {1: 0, 2: 1},
                                      _model())
        assert w[0] == 0.0    # 수평 패널: 옆미끄럼 워시 없음
        assert w[1] == -1.0   # 수직 패널(VTP): 전량

    def test_roll_zero_on_vertical_panel(self):
        w = _trim_variable_normalwash("ROLL", _boxes(), {1: 0, 2: 1}, _model())
        assert w[0] == -2.0 * 500.0 / 10000.0
        assert w[1] == 0.0

    def test_urdd5_projection(self):
        w = _trim_variable_normalwash("URDD5", _boxes(), {1: 0, 2: 1},
                                      _model())
        assert w[0] == -1000.0
        assert w[1] == 0.0


class TestControlSurfaceHingeSigns:
    """두 면 AESURF의 면별 워시 부호 — 에일러론 차동(+/-), 승강타 대칭(+/+)."""

    def _model(self):
        left_T = np.eye(3)
        left_T[1, 1] = -1.0  # 왼쪽 힌지: y축 반전 (차동)
        coords = {
            11: SimpleNamespace(transform=np.eye(3)),
            21: SimpleNamespace(transform=left_T),
            12: SimpleNamespace(transform=np.eye(3)),
            22: SimpleNamespace(transform=np.eye(3)),
        }
        aesurfs = {
            1: SimpleNamespace(label="ARON", eff=1.0,
                               cid1=11, alid1=1, cid2=21, alid2=2),
            2: SimpleNamespace(label="ELEV", eff=1.0,
                               cid1=12, alid1=3, cid2=22, alid2=4),
        }
        aelists = {
            1: SimpleNamespace(elements=[101]),  # 우 에일러론 박스
            2: SimpleNamespace(elements=[102]),  # 좌 에일러론 박스
            3: SimpleNamespace(elements=[103]),
            4: SimpleNamespace(elements=[104]),
        }
        return SimpleNamespace(aesurfs=aesurfs, aelists=aelists,
                               coords=coords, aelinks=[])

    def test_aileron_differential(self):
        from ascent_load.solvers.sol144 import _get_control_surface_boxes

        idx = {101: 0, 102: 1, 103: 2, 104: 3}
        signs, eff = _get_control_surface_boxes("ARON", self._model(), idx)
        assert signs[0] == 1.0 and signs[1] == -1.0

    def test_elevator_symmetric(self):
        from ascent_load.solvers.sol144 import _get_control_surface_boxes

        idx = {101: 0, 102: 1, 103: 2, 104: 3}
        signs, eff = _get_control_surface_boxes("ELEV", self._model(), idx)
        assert signs[2] == 1.0 and signs[3] == 1.0

    def test_aileron_wash_antisymmetric(self):
        boxes = [SimpleNamespace(box_id=b, control_point=np.zeros(3),
                                 normal=np.array([0., 0., 1.]))
                 for b in (101, 102, 103, 104)]
        idx = {101: 0, 102: 1, 103: 2, 104: 3}
        w = _trim_variable_normalwash("ARON", boxes, idx, self._model())
        assert w[0] == -1.0 and w[1] == 1.0  # 차동
        assert w[2] == 0.0 and w[3] == 0.0


class TestInertiaReliefClosure:
    """잔여 6분력이 강체 관성하중으로 정확히 닫히는지 (자기평형화)."""

    def _model(self):
        nodes = {
            1: SimpleNamespace(xyz_global=np.array([1000.0, 0.0, 0.0])),
            2: SimpleNamespace(xyz_global=np.array([-1000.0, 0.0, 0.0])),
            3: SimpleNamespace(xyz_global=np.array([0.0, 1000.0, 0.0])),
            4: SimpleNamespace(xyz_global=np.array([0.0, -1000.0, 0.0])),
        }
        masses = {i: SimpleNamespace(node_id=i, mass=2.0) for i in nodes}
        return SimpleNamespace(nodes=nodes, elements={}, masses=masses)

    def test_residual_closes_exactly(self):
        from ascent_load.loads_analysis.trim_loads import (
            apply_inertia_relief, verify_trim_balance,
        )

        model = self._model()
        cg = np.zeros(3)
        # 절점 1에만 상방 100 N → F_res=(0,0,100), M_res=(0,-1e5,0)
        combined = {i: np.zeros(6) for i in model.nodes}
        combined[1][2] = 100.0
        inertial = {i: np.zeros(6) for i in model.nodes}

        relief = apply_inertia_relief(model, inertial, combined,
                                      cg=cg, g=9810.0)
        assert abs(relief["relief_nz"] - 100.0 / 8.0 / 9810.0) < 1e-12
        assert relief["q_dot"] != 0.0

        bal = verify_trim_balance(model, combined, ref_point=cg)
        for k, val in bal.items():
            assert abs(val) < 1e-6, f"{k}={val}"

    def test_zero_residual_noop(self):
        from ascent_load.loads_analysis.trim_loads import apply_inertia_relief

        model = self._model()
        combined = {i: np.zeros(6) for i in model.nodes}
        inertial = {i: np.zeros(6) for i in model.nodes}
        relief = apply_inertia_relief(model, inertial, combined,
                                      cg=np.zeros(3), g=9810.0)
        assert all(abs(v) < 1e-12 for v in relief.values())
        assert all(np.allclose(f, 0) for f in combined.values())


class TestTrimMassConsistency:
    """트림 제약식의 중량/CG가 관성력 질량 분포와 정확히 일치해야 한다.

    불일치하면 combined 잔차가 영구적으로 남는다:
    Fz 잔차 = nz*(W_trim - W_inertial), My 잔차 = W * (cg_trim - cg_inertial).
    """

    def _model(self):
        from ascent_load.bdf.parser import parse_bdf

        model = parse_bdf(str(VM_DIR / "vm6_fixed_fixed_beam.bdf"))
        model.cross_reference()
        return model

    def test_weight_matches_inertial_sum(self):
        from ascent_load.loads_analysis.trim_loads import (
            compute_nodal_inertial_forces,
        )

        model = self._model()
        g = _detect_gravity(model)
        inertial = compute_nodal_inertial_forces(model, nz=1.0, g=g)
        total_fz = sum(f[2] for f in inertial.values())
        assert abs(_compute_total_weight(model) + total_fz) < 1e-9 * abs(total_fz)

    def test_cg_matches_inertial_centroid(self):
        from ascent_load.loads_analysis.trim_loads import (
            compute_nodal_inertial_forces,
        )

        model = self._model()
        inertial = compute_nodal_inertial_forces(model, nz=1.0, g=9810.0)
        w_sum = sum(f[2] for f in inertial.values())
        x_centroid = sum(model.nodes[nid].xyz_global[0] * f[2]
                         for nid, f in inertial.items()) / w_sum
        assert abs(_compute_cg_x(model) - x_centroid) < 1e-9 * max(
            1.0, abs(x_centroid))
