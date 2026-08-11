# 모델 트리 위젯 — 절점/요소/물성/재료/공력/서브케이스를 그룹별 상한을 두고 나열
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

MAX_ITEMS_PER_GROUP = 500

GroupData = Tuple[str, List[str]]  # (그룹 라벨, 항목 라벨 목록)


def _cap(labels: List[str], total: int) -> List[str]:
    if total > len(labels):
        labels.append(f"… {total - len(labels)} more")
    return labels


def summarize_model(model, results=None,
                    max_items: int = MAX_ITEMS_PER_GROUP) -> List[GroupData]:
    """모델(BDFModel 또는 VizModel)을 트리 표시용 순수 데이터로 요약한다."""
    groups: List[GroupData] = []

    nodes = getattr(model, "nodes", {}) or {}
    labels = []
    for nid in sorted(nodes)[:max_items]:
        xyz = nodes[nid].xyz_global
        labels.append(f"GRID {nid}  ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})")
    groups.append((f"Nodes ({len(nodes)})", _cap(labels, len(nodes))))

    # Elements: 타입별 하위 그룹 + 개별 요소(상한 max_items) 2단 구조.
    # 항목이 (라벨, [하위 라벨]) 튜플이면 populate가 한 단계 더 만든다.
    elements = getattr(model, "elements", {}) or {}
    by_type: Dict[str, List[int]] = {}
    for eid in sorted(elements):
        by_type.setdefault(elements[eid].type, []).append(eid)
    labels = []
    for etype, eids in sorted(by_type.items()):
        subs = [f"{etype} {eid}" for eid in eids[:max_items]]
        labels.append((f"{etype}: {len(eids)}", _cap(subs, len(eids))))
    groups.append((f"Elements ({len(elements)})", labels))

    properties = getattr(model, "properties", {}) or {}
    labels = [f"{getattr(properties[pid], 'type', 'PROP')} {pid}"
              for pid in sorted(properties)[:max_items]]
    groups.append((f"Properties ({len(properties)})", _cap(labels, len(properties))))

    materials = getattr(model, "materials", {}) or {}
    if materials:
        labels = [f"{getattr(materials[mid], 'type', 'MAT')} {mid}"
                  for mid in sorted(materials)[:max_items]]
        groups.append((f"Materials ({len(materials)})", _cap(labels, len(materials))))

    rigids = getattr(model, "rigids", {}) or {}
    if rigids:
        type_counts = Counter(getattr(r, "type", "RBE") for r in rigids.values())
        labels = [f"{rtype}: {count}" for rtype, count in sorted(type_counts.items())]
        groups.append((f"Rigid Elements ({len(rigids)})", labels))

    caero = getattr(model, "caero_panels", {}) or {}
    if caero:
        aero_labels = [f"CAERO1 {eid}" for eid in sorted(caero)[:max_items]]
        aero_labels = _cap(aero_labels, len(caero))
        for surf_id, surf in sorted((getattr(model, "aesurfs", {}) or {}).items()):
            name = getattr(surf, "label", "") or getattr(surf, "name", "")
            aero_labels.append(f"AESURF {surf_id} {name}")
        splines = getattr(model, "splines", {}) or {}
        if splines:
            aero_labels.append(f"SPLINE cards: {len(splines)}")
        trims = getattr(model, "trims", {}) or {}
        for tid in sorted(trims):
            aero_labels.append(f"TRIM {tid}")
        groups.append((f"Aero ({len(caero)} panels)", aero_labels))

    subcases = getattr(model, "subcases", []) or []
    if subcases:
        labels = []
        for sc in subcases:
            sels = []
            for attr, tag in (("spc_id", "SPC"), ("load_id", "LOAD"),
                              ("method_id", "METHOD"), ("trim_id", "TRIM")):
                val = getattr(sc, attr, 0)
                if val:
                    sels.append(f"{tag}={val}")
            labels.append(f"SUBCASE {sc.id}  " + " ".join(sels))
        groups.append((f"Subcases ({len(subcases)})", labels))

    if results is not None and getattr(results, "subcases", None):
        labels = []
        for sc in results.subcases:
            parts = []
            if sc.displacements:
                parts.append("disp")
            if getattr(sc, "frequencies", None) is not None:
                parts.append(f"{len(sc.frequencies)} modes")
            if getattr(sc, "trim_variables", None):
                parts.append("trim")
            if getattr(sc, "aero_pressures", None) is not None:
                parts.append("Cp")
            labels.append(f"Subcase {sc.subcase_id}: " + ", ".join(parts or ["-"]))
        groups.append((f"Results ({len(results.subcases)} subcases)", labels))

    return groups


# ---------------------------------------------------------------------------
# 항목 상세 정보 (클릭 → 텍스트)
# ---------------------------------------------------------------------------
_SKIP_ATTRS = ("node_refs", "property_ref", "material_ref", "coord_ref")


def _fmt_val(v) -> str:
    import numpy as np

    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, np.ndarray):
        if v.size <= 3:
            return "(" + ", ".join(f"{x:g}" for x in v) + ")"
        return f"array{v.shape}"
    if isinstance(v, (list, tuple)) and len(v) > 12:
        return f"[{len(v)} items]"
    return str(v)


def _dump_attrs(obj, title: str) -> str:
    """카드 객체의 공개 속성을 '필드 = 값' 표로 덤프한다."""
    lines = [title]
    attrs = {k: v for k, v in vars(obj).items()
             if not k.startswith("_") and k not in _SKIP_ATTRS
             and not callable(v)}
    width = max((len(k) for k in attrs), default=0)
    for k, v in attrs.items():
        lines.append(f"  {k:<{width}} = {_fmt_val(v)}")
    return "\n".join(lines)


def _describe_node(model, nid: int) -> str:
    node = model.nodes.get(nid)
    if node is None:
        return f"GRID {nid}: 모델에 없음"
    lines = [_dump_attrs(node, f"GRID {nid}")]
    elements = getattr(model, "elements", {}) or {}
    attached = Counter(e.type for e in elements.values()
                       if nid in (getattr(e, "node_ids", None) or ()))
    if attached:
        lines.append("연결 요소: " + ", ".join(
            f"{t}×{c}" for t, c in sorted(attached.items())))
    conm = [m for m in (getattr(model, "masses", {}) or {}).values()
            if getattr(m, "node_id", None) == nid]
    if conm:
        tot = sum(m.mass for m in conm)
        lines.append(f"CONM2 질량: {tot * 1000:.3f} kg ({len(conm)}개 카드)")
    return "\n".join(lines)


def _describe_elem_type(model, etype: str) -> str:
    # 요소 ID는 dict 키 사용 — VizElement(.naero 프록시)에는 eid 속성이 없다
    elements = getattr(model, "elements", {}) or {}
    of_type = [(eid, e) for eid, e in elements.items() if e.type == etype]
    if not of_type:
        return f"{etype}: 없음"
    pids = Counter(getattr(e, "pid", None) for _, e in of_type)
    eids = sorted(eid for eid, _ in of_type)
    lines = [f"{etype}: {len(of_type)}개",
             f"  EID 범위 = {eids[0]} … {eids[-1]}",
             "  물성 분포:"]
    props = getattr(model, "properties", {}) or {}
    for pid, cnt in pids.most_common(8):
        ptype = getattr(props.get(pid), "type", "?")
        lines.append(f"    {ptype} {pid}: {cnt}개")
    if len(pids) > 8:
        lines.append(f"    … 외 {len(pids) - 8}개 물성")
    return "\n".join(lines)


def _describe_element(model, eid: int) -> str:
    elem = (getattr(model, "elements", {}) or {}).get(eid)
    if elem is None:
        return f"요소 {eid}: 모델에 없음"
    lines = [_dump_attrs(elem, f"{elem.type} {eid}")]
    props = getattr(model, "properties", {}) or {}
    prop = props.get(getattr(elem, "pid", None))
    if prop is not None:
        ptype = getattr(prop, "type", None) or type(prop).__name__
        lines.append(f"물성: {ptype} {elem.pid}")
    nodes = getattr(model, "nodes", {}) or {}
    for nid in (getattr(elem, "node_ids", None) or ())[:8]:
        n = nodes.get(nid)
        if n is not None:
            x, y, z = n.xyz_global
            lines.append(f"  GRID {nid}  ({x:.1f}, {y:.1f}, {z:.1f})")
    return "\n".join(lines)


def _describe_property(model, pid: int) -> str:
    prop = (getattr(model, "properties", {}) or {}).get(pid)
    if prop is None:
        return f"PROP {pid}: 모델에 없음"
    title = f"{getattr(prop, 'type', None) or type(prop).__name__} {pid}"
    lines = [_dump_attrs(prop, title)]
    elements = getattr(model, "elements", {}) or {}
    used = Counter(e.type for e in elements.values()
                   if getattr(e, "pid", None) == pid)
    if used:
        lines.append("사용 요소: " + ", ".join(
            f"{t}×{c}" for t, c in sorted(used.items())))
    return "\n".join(lines)


def _describe_material(model, mid: int) -> str:
    mat = (getattr(model, "materials", {}) or {}).get(mid)
    if mat is None:
        return f"MAT {mid}: 모델에 없음"
    mtitle = getattr(mat, "type", None) or type(mat).__name__
    lines = [_dump_attrs(mat, f"{mtitle} {mid}")]
    props = getattr(model, "properties", {}) or {}
    used = [pid for pid, p in props.items()
            if mid in (getattr(p, "mid", None), getattr(p, "mid2", None),
                       getattr(p, "mid3", None))]
    if used:
        lines.append(f"참조 물성: {sorted(used)}")
    return "\n".join(lines)


def _describe_caero(model, eid: int) -> str:
    panel = (getattr(model, "caero_panels", {}) or {}).get(eid)
    if panel is None:
        return f"CAERO1 {eid}: 모델에 없음"
    lines = [_dump_attrs(panel, f"CAERO1 {eid}")]
    n = panel.nspan * panel.nchord
    lines.append(f"박스: {n}개 ({panel.nspan} 스팬 × {panel.nchord} 시위), "
                 f"ID {eid} … {eid + n - 1}")
    for sp in (getattr(model, "splines", {}) or {}).values():
        if getattr(sp, "caero", None) == eid:
            lines.append(f"SPLINE1 {sp.eid}: SET1 {sp.setg}, "
                         f"박스 {sp.box1}–{sp.box2} ({sp.method})")
    return "\n".join(lines)


def _describe_results_subcase(results, scid: int) -> str:
    import math

    import numpy as np

    sc = next((s for s in results.subcases if s.subcase_id == scid), None)
    if sc is None:
        return f"Subcase {scid}: 결과 없음"
    lines = [f"결과 Subcase {scid}"]
    tv = getattr(sc, "trim_variables", None)
    if tv:
        lines.append("  트림 변수:")
        for k, v in tv.items():
            # URDD*는 가속도 성분 — 각도 환산 표시는 각도/각속도류만
            deg = ("" if k.startswith("URDD") or abs(v) >= 1.6
                   else f" ({math.degrees(v):+.3f}°)")
            lines.append(f"    {k:<8} = {v:+.6f}{deg}")
    tb = getattr(sc, "trim_balance", None)
    if tb:
        lines.append("  하중 합력 (combined):")
        for k in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
            if k in tb:
                lines.append(f"    {k} = {tb[k]:+.4g}")
    if sc.displacements:
        arr = np.array([d[:3] for d in sc.displacements.values()])
        mags = np.linalg.norm(arr, axis=1)
        nid = list(sc.displacements)[int(np.argmax(mags))]
        lines.append(f"  최대 변위: {mags.max():.3f} mm @ GRID {nid}")
    freqs = getattr(sc, "frequencies", None)
    if freqs is not None and len(freqs):
        shown = ", ".join(f"{f:.2f}" for f in freqs[:10])
        lines.append(f"  고유진동수 (Hz): {shown}"
                     + (" …" if len(freqs) > 10 else ""))
    return "\n".join(lines)


def describe_item(model, results, group_label: str, label: str) -> str:
    """트리 항목 라벨 → 상세 정보 텍스트. summarize_model 라벨 형식과 짝."""
    if model is None or label.startswith("…"):
        return ""
    tok = label.split()
    try:
        if group_label.startswith("Nodes") and tok[0] == "GRID":
            return _describe_node(model, int(tok[1]))
        if group_label.startswith("Elements"):
            # "CQUAD4 17" = 개별 요소, "CQUAD4: 3392" = 타입 행 (콜론 유무)
            if (len(tok) == 2 and tok[1].isdigit()
                    and not tok[0].endswith(":")):
                return _describe_element(model, int(tok[1]))
            return _describe_elem_type(model, label.split(":")[0])
        if group_label.startswith("Properties"):
            return _describe_property(model, int(tok[1]))
        if group_label.startswith("Materials"):
            return _describe_material(model, int(tok[1]))
        if group_label.startswith("Rigid"):
            rtype = label.split(":")[0]
            rigids = getattr(model, "rigids", {}) or {}
            ids = sorted(r for r, o in rigids.items()
                         if getattr(o, "type", "RBE") == rtype)
            return (f"{rtype}: {len(ids)}개\n"
                    f"  EID 범위 = {ids[0]} … {ids[-1]}" if ids else label)
        if group_label.startswith("Aero"):
            if tok[0] == "CAERO1":
                return _describe_caero(model, int(tok[1]))
            if tok[0] == "AESURF":
                surf = (getattr(model, "aesurfs", {}) or {}).get(int(tok[1]))
                return _dump_attrs(surf, label) if surf else label
            if tok[0] == "TRIM":
                trim = (getattr(model, "trims", {}) or {}).get(int(tok[1]))
                return _dump_attrs(trim, label) if trim else label
            if tok[0] == "SPLINE":
                lines = []
                for sp in (getattr(model, "splines", {}) or {}).values():
                    lines.append(f"SPLINE1 {sp.eid}: CAERO {sp.caero}, "
                                 f"SET1 {sp.setg}, 박스 {sp.box1}–{sp.box2}")
                return "\n".join(lines)
        if group_label.startswith("Subcases") and tok[0] == "SUBCASE":
            sc = next((s for s in getattr(model, "subcases", [])
                       if s.id == int(tok[1])), None)
            return _dump_attrs(sc, label) if sc else label
        if group_label.startswith("Results") and results is not None:
            return _describe_results_subcase(
                results, int(tok[1].rstrip(":")))
    except (ValueError, IndexError):
        pass
    return label


class ModelTreeWidget:
    """summarize_model() 트리 + 클릭 항목 상세 정보 패널."""

    def __init__(self, parent=None) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QFontDatabase
        from qtpy.QtWidgets import QPlainTextEdit, QSplitter, QTreeWidget

        self.widget = QTreeWidget(parent)
        self.widget.setHeaderLabel("Model")
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setPlaceholderText("트리 항목을 클릭하면 상세 정보가 표시됩니다")
        self.info.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.container = QSplitter(Qt.Vertical)
        self.container.addWidget(self.widget)
        self.container.addWidget(self.info)
        self.container.setStretchFactor(0, 3)
        self.container.setStretchFactor(1, 2)
        self._model = None
        self._results = None
        self.node_clicked = None   # main window가 주입 (nid → 3D 하이라이트)
        self.elements_clicked = None   # (eid 목록 → 3D 하이라이트)
        self.widget.itemClicked.connect(self._on_item_clicked)

    def populate(self, model, results=None) -> None:
        from qtpy.QtWidgets import QTreeWidgetItem

        self._model = model
        self._results = results
        self.widget.clear()
        self.info.clear()
        if model is None:
            return
        for group_label, item_labels in summarize_model(model, results):
            top = QTreeWidgetItem([group_label])
            for label in item_labels:
                if isinstance(label, tuple):   # (하위 그룹, [하위 항목])
                    sub = QTreeWidgetItem([label[0]])
                    for s in label[1]:
                        sub.addChild(QTreeWidgetItem([s]))
                    top.addChild(sub)
                else:
                    top.addChild(QTreeWidgetItem([label]))
            self.widget.addTopLevelItem(top)
        sol = getattr(model, "sol", None)
        if sol:
            self.widget.setHeaderLabel(f"Model (SOL {sol})")

    def _on_item_clicked(self, item, _col) -> None:
        parent = item.parent()
        label = item.text(0)
        if parent is None:
            self.info.setPlainText(label)
            return
        # 그룹 문맥은 최상위 조상 기준 (요소 리프의 부모는 타입 행)
        top = item
        while top.parent() is not None:
            top = top.parent()
        group = top.text(0)
        self.info.setPlainText(
            describe_item(self._model, self._results, group, label))
        tok = label.split()
        if (group.startswith("Nodes") and tok and tok[0] == "GRID"
                and callable(self.node_clicked)):
            try:
                self.node_clicked(int(tok[1]))
            except (ValueError, IndexError):
                pass
        elif callable(self.elements_clicked) and not label.startswith("…"):
            eids = self._eids_for(group, label)
            if eids:
                self.elements_clicked(eids, label)

    def _eids_for(self, group: str, label: str):
        """요소 타입 행/물성 행 → 해당 요소 ID 목록."""
        elements = getattr(self._model, "elements", {}) or {}
        try:
            if group.startswith("Elements"):
                tok = label.split()
                if (len(tok) == 2 and tok[1].isdigit()
                        and not tok[0].endswith(":")):   # 개별 요소
                    eid = int(tok[1])
                    return [eid] if eid in elements else []
                etype = label.split(":")[0]
                return [eid for eid, e in elements.items()
                        if e.type == etype]
            if group.startswith("Properties"):
                pid = int(label.split()[1])
                return [eid for eid, e in elements.items()
                        if getattr(e, "pid", None) == pid]
        except (ValueError, IndexError):
            pass
        return []
