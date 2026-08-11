# 3D 장면 컨트롤러 — QtInteractor를 소유하고 표시 모드별로 mesh_builder 산출물을 렌더링
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("nastaero.gui")

_COMPONENT_SCALARS = {
    "Magnitude": "Displacement_Magnitude",
    "T1": "T1", "T2": "T2", "T3": "T3",
    "R1": "R1", "R2": "R2", "R3": "R3",
}


def auto_deform_scale(model, displacements: Dict[int, np.ndarray],
                      target_fraction: float = 0.05) -> float:
    """최대 변위가 모델 대각선의 target_fraction이 되는 배율을 반환한다."""
    if not displacements:
        return 1.0
    max_disp = max(
        (float(np.linalg.norm(d[:3])) for d in displacements.values()),
        default=0.0,
    )
    if max_disp <= 0.0:
        return 1.0
    coords = np.array([n.xyz_global for n in model.nodes.values()])
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    if diag <= 0.0:
        return 1.0
    return diag * target_fraction / max_disp


def build_element_overlay(model, eids):
    """요소 ID 목록 → 하이라이트용 PolyData (셸=면, 2절점 요소=선).

    렌더러 없이 테스트 가능한 순수 함수. 표시할 것이 없으면 None.
    """
    if model is None or not eids:
        return None
    import pyvista as pv

    nodes = getattr(model, "nodes", {}) or {}
    elements = getattr(model, "elements", {}) or {}
    pts: list = []
    index: Dict[int, int] = {}

    def idx(nid: int) -> int:
        if nid not in index:
            index[nid] = len(pts)
            pts.append(nodes[nid].xyz_global)
        return index[nid]

    faces: list = []
    lines: list = []
    for eid in eids:
        e = elements.get(eid)
        nids = [n for n in (getattr(e, "node_ids", None) or ())
                if n in nodes]
        if len(nids) >= 3:
            faces.extend([len(nids), *(idx(n) for n in nids)])
        elif len(nids) == 2:
            lines.extend([2, idx(nids[0]), idx(nids[1])])
    if not pts:
        return None
    poly = pv.PolyData(np.asarray(pts, dtype=float))
    if faces:
        poly.faces = np.asarray(faces)
    if lines:
        poly.lines = np.asarray(lines)
    return poly


class SceneController:
    """중앙 3D 뷰포트. mesh_builder의 build_* 함수를 직접 호출해 렌더링한다."""

    def __init__(self, parent) -> None:
        from pyvistaqt import QtInteractor

        self.plotter = QtInteractor(parent)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.model = None
        self.results = None
        self._hl_label = None    # 하이라이트 절점 라벨 액터
        # 보기 토글 상태
        self.show_structure = True       # 구조 FEM 전체 (마스터)
        self.hidden_types: set = set()   # 숨긴 FEM 요소 종류
        self.show_beams = True
        self.show_aero = True
        self.show_rbe = True
        self.show_edges = True
        self.zoom_on_highlight = False   # 절점 하이라이트 시 카메라 줌인
        self.isolate_mode = False        # 클릭 항목만 표시(격리)

    # ------------------------------------------------------------------
    # 모델/결과 주입
    # ------------------------------------------------------------------
    def set_model(self, model, results=None) -> None:
        self.model = model
        self.results = results
        self.display_model(reset_camera=True)

    def clear(self) -> None:
        self.plotter.clear()
        self.plotter.add_axes()
        self._hl_label = None   # plotter.clear()가 액터를 함께 지움

    # ------------------------------------------------------------------
    # 절점 하이라이트 (모델 트리 클릭 연동)
    # ------------------------------------------------------------------
    def highlight_node(self, nid: int) -> None:
        """절점을 구 마커 + 라벨로 강조한다. 재클릭 시 마커가 이동."""
        node = (getattr(self.model, "nodes", {}) or {}).get(nid)
        if node is None:
            return
        import pyvista as pv

        pos = np.asarray(node.xyz_global, dtype=float)
        radius = max(self._model_diag() * 0.004, 1e-6)
        # name 지정 → 같은 이름의 이전 마커를 자동 교체
        self.plotter.add_mesh(
            pv.Sphere(radius=radius, center=pos), color="red",
            name="node_highlight", reset_camera=False)
        if self._hl_label is not None:
            self.plotter.remove_actor(self._hl_label)
            self._hl_label = None
        try:
            self._hl_label = self.plotter.add_point_labels(
                [pos], [f"GRID {nid}"], font_size=12, text_color="red",
                point_size=1, shape_opacity=0.6)
        except Exception:
            logger.debug("Node label failed", exc_info=True)
        if self.zoom_on_highlight:
            self._zoom_to(pos)
        self.plotter.render()

    def _zoom_to(self, pos: np.ndarray) -> None:
        """절점이 보이는 각도로 카메라를 돌리며 접근한다.

        모델 중심 → 절점의 바깥 방향에서 바라보면 절점이 항상 구조물의
        앞면에 놓여 가려지지 않는다 (중심 근처 절점은 기존 시선 유지).
        """
        cam = self.plotter.camera
        outward = pos - self._model_center()
        norm = float(np.linalg.norm(outward))
        if norm < self._model_diag() * 0.01:
            outward = np.asarray(cam.position) - np.asarray(cam.focal_point)
            norm = float(np.linalg.norm(outward))
            if norm <= 0:
                return
        outward = outward / norm
        dist = self._model_diag() * 0.12
        cam.focal_point = pos.tolist()
        cam.position = (pos + outward * dist).tolist()
        # 시선과 평행하지 않은 업 벡터 선택
        cam.up = (0.0, 1.0, 0.0) if abs(outward[2]) > 0.9 else (0.0, 0.0, 1.0)

    def highlight_elements(self, eids, title: str = "") -> None:
        """요소 집합 강조 — 기본은 주황 오버레이, 격리 모드면 단독 표시."""
        poly = build_element_overlay(self.model, eids)
        if poly is None:
            return
        if self.isolate_mode:
            self.clear()
            self.plotter.add_mesh(
                poly, color="lightsteelblue", show_edges=self.show_edges,
                edge_color="gray", line_width=3, name="isolated",
                reset_camera=False)
            if title:
                self.plotter.add_text(f"격리 표시: {title}",
                                      font_size=10, color="black")
        else:
            self.plotter.add_mesh(poly, color="orange", opacity=0.9,
                                  line_width=4, name="elem_highlight",
                                  reset_camera=False)
        # 단일 요소 클릭이면 절점 줌인과 동일하게 도심으로 접근
        if self.zoom_on_highlight and len(eids) == 1:
            self._zoom_to(np.asarray(poly.points).mean(axis=0))
        self.plotter.render()

    def _model_center(self) -> np.ndarray:
        nodes = getattr(self.model, "nodes", {}) or {}
        if not nodes:
            return np.zeros(3)
        coords = np.array([n.xyz_global for n in nodes.values()])
        return (coords.max(axis=0) + coords.min(axis=0)) / 2.0

    def _model_diag(self) -> float:
        nodes = getattr(self.model, "nodes", {}) or {}
        if not nodes:
            return 1.0
        coords = np.array([n.xyz_global for n in nodes.values()])
        return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))

    # ------------------------------------------------------------------
    # 표시 모드
    # ------------------------------------------------------------------
    #: 3D 뷰에 실제로 그려지는 FEM 요소 종류
    DRAWABLE_TYPES = ("CQUAD4", "CQUAD8", "CTRIA3", "CTRIA6", "CBAR", "CROD")

    def present_element_types(self) -> list:
        """모델에 존재하는 렌더링 대상 FEM 요소 종류 (보기 메뉴 구성용)."""
        if self.model is None:
            return []
        present = {e.type for e in
                   (getattr(self.model, "elements", {}) or {}).values()}
        return [t for t in self.DRAWABLE_TYPES if t in present]

    def _visible_types(self) -> set:
        return set(self.present_element_types()) - self.hidden_types

    def display_model(self, reset_camera: bool = False) -> None:
        """미변형 모델: 구조 메시 + 보 튜브 + 공력 패널 + RBE 라인."""
        if self.model is None:
            return
        from ..visualization.mesh_builder import build_structural_mesh

        self.clear()
        if self.show_structure:
            vis = self._visible_types()
            mesh_types = vis & {"CQUAD4", "CQUAD8", "CTRIA3", "CTRIA6"}
            if not self.show_beams:   # 튜브 꺼짐 → 보/로드를 라인으로
                mesh_types |= vis & {"CBAR", "CROD"}
            grid = build_structural_mesh(self.model,
                                         element_types=mesh_types)
            if grid.n_cells > 0:
                self.plotter.add_mesh(
                    grid, color="lightsteelblue", show_edges=self.show_edges,
                    edge_color="gray", label="Structure",
                )
        self._add_common_overlays()
        self._finish(reset_camera)

    def display_displacement(self, subcase_idx: int, component: str = "Magnitude",
                             scale: Optional[float] = None) -> None:
        """변형 형상 + 성분 컨투어. scale=None이면 자동 배율."""
        sc = self._subcase(subcase_idx)
        if sc is None or not sc.displacements:
            logger.warning("No displacements in subcase index %d", subcase_idx)
            return
        from ..visualization.mesh_builder import (
            add_displacement_data, build_deformed_beam_tubes, build_deformed_mesh,
        )

        if scale is None:
            scale = auto_deform_scale(self.model, sc.displacements)
        self.clear()
        grid = build_deformed_mesh(self.model, sc.displacements, scale=scale)
        add_displacement_data(grid, self.model, sc.displacements)
        scalars = _COMPONENT_SCALARS.get(component, "Displacement_Magnitude")
        if grid.n_cells > 0:
            self.plotter.add_mesh(
                grid, scalars=scalars, cmap="jet", show_edges=self.show_edges,
                scalar_bar_args={"title": scalars},
            )
        if self.show_beams:
            tubes = build_deformed_beam_tubes(self.model, sc.displacements, scale=scale)
            if tubes is not None:
                self.plotter.add_mesh(tubes, color="steelblue")
        self.plotter.add_text(f"Subcase {sc.subcase_id}  scale={scale:.3g}",
                              font_size=10, color="black")
        self._finish(False)

    def display_mode(self, subcase_idx: int, mode_idx: int,
                     scale: Optional[float] = None) -> None:
        """모드 형상 (mode_idx는 0-기준)."""
        sc = self._subcase(subcase_idx)
        if sc is None or not sc.mode_shapes:
            logger.warning("No mode shapes in subcase index %d", subcase_idx)
            return
        mode_idx = max(0, min(mode_idx, len(sc.mode_shapes) - 1))
        shape = sc.mode_shapes[mode_idx]
        from ..visualization.mesh_builder import build_mode_shape_mesh

        if scale is None:
            scale = auto_deform_scale(self.model, shape)
        self.clear()
        grid = build_mode_shape_mesh(self.model, shape, scale=scale)
        if grid.n_cells > 0:
            self.plotter.add_mesh(
                grid, scalars="Mode_Magnitude", cmap="jet",
                show_edges=self.show_edges,
                scalar_bar_args={"title": "Mode_Magnitude"},
            )
        freq = ""
        if sc.frequencies is not None and mode_idx < len(sc.frequencies):
            freq = f"  f = {sc.frequencies[mode_idx]:.3f} Hz"
        self.plotter.add_text(f"Mode {mode_idx + 1}{freq}  scale={scale:.3g}",
                              font_size=10, color="black")
        self._finish(False)

    def display_pressure(self, subcase_idx: int) -> None:
        """공력 압력 계수(Cp) 분포."""
        sc = self._subcase(subcase_idx)
        if sc is None or sc.aero_pressures is None or sc.aero_boxes is None:
            logger.warning("No aero pressures in subcase index %d", subcase_idx)
            return
        from ..visualization.mesh_builder import build_aero_pressure_mesh

        self.clear()
        mesh = build_aero_pressure_mesh(
            sc.aero_boxes, sc.aero_pressures,
            bdf_model=self.model, trim_variables=sc.trim_variables,
        )
        self.plotter.add_mesh(
            mesh, scalars="Pressure" if "Pressure" in mesh.array_names else None,
            cmap="coolwarm", show_edges=True,
            scalar_bar_args={"title": "Pressure"},
        )
        self.plotter.add_text(f"Subcase {sc.subcase_id}  aero pressure",
                              font_size=10, color="black")
        self._add_structure_faint()
        self._finish(False)

    def display_forces(self, subcase_idx: int, load_type: str = "combined") -> None:
        """절점 하중 화살표 (aero / inertial / combined)."""
        sc = self._subcase(subcase_idx)
        if sc is None:
            return
        forces = {
            "aero": sc.nodal_aero_forces,
            "inertial": sc.nodal_inertial_forces,
            "combined": sc.nodal_combined_forces,
        }.get(load_type)
        if not forces:
            logger.warning("No %s nodal forces in subcase index %d",
                           load_type, subcase_idx)
            return
        from ..visualization.mesh_builder import build_nodal_force_arrows

        self.clear()
        self._add_structure_faint()
        arrows = build_nodal_force_arrows(self.model, forces)
        color = {"aero": "blue", "inertial": "red", "combined": "black"}[load_type]
        if arrows is not None:
            self.plotter.add_mesh(arrows, color=color,
                                  label=f"{load_type.title()} Forces")
        self.plotter.add_text(f"Subcase {sc.subcase_id}  {load_type} forces",
                              font_size=10, color="black")
        self._finish(False)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _subcase(self, idx: int):
        if self.results is None or not self.results.subcases:
            return None
        if not (0 <= idx < len(self.results.subcases)):
            return None
        return self.results.subcases[idx]

    def _add_structure_faint(self) -> None:
        from ..visualization.mesh_builder import build_structural_mesh

        grid = build_structural_mesh(self.model)
        if grid.n_cells > 0:
            self.plotter.add_mesh(grid, color="lightgray", opacity=0.3)

    def _add_common_overlays(self) -> None:
        from ..visualization.mesh_builder import (
            build_beam_tubes, build_rbe_lines,
        )

        if self.show_structure and self.show_beams:
            beam_types = self._visible_types() & {"CBAR", "CROD"}
            tubes = build_beam_tubes(self.model, element_types=beam_types) \
                if beam_types else None
            if tubes is not None:
                self.plotter.add_mesh(tubes, color="steelblue", label="Beams")
        if self.show_structure and self.show_rbe:
            rbe = build_rbe_lines(self.model)
            if rbe is not None:
                self.plotter.add_mesh(rbe, color="magenta", line_width=1,
                                      opacity=0.6, label="RBE")
        if self.show_aero and getattr(self.model, "caero_panels", None):
            try:
                from ..aero.panel import generate_all_panels
                from ..visualization.mesh_builder import build_aero_mesh

                boxes = generate_all_panels(self.model)
                if boxes:
                    mesh = build_aero_mesh(boxes, bdf_model=self.model)
                    self.plotter.add_mesh(
                        mesh, color="cyan", show_edges=True,
                        edge_color="darkblue", line_width=1, opacity=0.35,
                        label="Aero Panels",
                    )
            except Exception:
                logger.debug("Aero panel meshing failed", exc_info=True)

    def _finish(self, reset_camera: bool) -> None:
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def close(self) -> None:
        self.plotter.close()
