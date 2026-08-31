# (고장×재트림) 확장 사건 공간 — 선형 패턴 전수 선별(C) + 확률 채택 판정(P)의 구현
"""(failure x re-trim state) event-space extension.

The DEP failure taxonomy (rotor x mode x phase) discretizes an event
as the instantaneous thrust state of the failure; the states in which
the FCS *holds* the aircraft afterwards (surviving rotors re-trimmed
inside the normal command band) are outside that grid, and the ILC-8
counterexample showed such a state can leave the planar-selected
design set by >5 % in a combined (V, M, T) direction.

This module implements the extension for the hover family, where the
map from the per-rotor thrusts to the station loads is exactly linear
(point hub loads + inertia + inertia-relief closure; no stiffness
solve). Per failure event:

  1. the failed rotor(s) are pinned at the mode's thrust level
     (M1/M2/M3: 0, M4 stuck: 1.0, M5 runaway: 1.5 x T_hover);
  2. the surviving rotors sweep the normal command band (default
     levels 0.7/1.0/1.3) -- the FCS re-trim space;
  3. every pattern is evaluated through an 8-rotor VMT basis and
     scored by its directional exceedance beyond the convex hull of
     the DISCRETE-taxonomy states (per station, axis-normalized) --
     the consequence C of the re-trim space, i.e. the load content
     the discrete grid does not cover;
  4. the probability side P adjudicates adoption: each extended event
     inherits its parent failure event's P (rotor share x phase x
     mode split) -- re-trim commands add no failure, so no extra
     abnormal rotors enter, and multi-failure patterns are excluded
     by construction. Events are ranked by P x C and the governing
     pattern of each selected event is realized as a pipeline case.

Usage:
    screen = RetrimScreen(model, vtol_config, aircraft_config,
                          components=components, fuselage_cg_x=...)
    events = screen.screen()                       # ranked, all modes
    cases  = screen.realize(events, threshold_pct=1.0, top_n=8)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..trim_loads import (
    apply_inertia_relief, compute_nodal_inertial_forces,
    compute_node_masses,
)
from .batch_runner import BatchResult, CaseResult
from .vmt_bridge import compute_vmt_for_batch
from .vtol_conditions import VTOLCondition, VTOLFlightPhase
from .vtol_load_case_matrix import VTOLLoadCaseMatrix

logger = logging.getLogger(__name__)

# ch6 분류 상수와 동일한 모드 분할·호버 단계 확률 (상대 우선순위)
P_MODE_DEFAULT = {"M1": 0.50, "M2": 0.05, "M3": 0.05,
                  "M4": 0.20, "M5": 0.20}
P_PHASE_HOVER = 0.10
RUNAWAY_FACTOR = 1.5
COMMAND_LEVELS = (0.7, 1.0, 1.3)     # FCS 정상 지령 대역의 이산화
RETRIM_CASE_ID_START = 30000

# 모드별 far section (기존 러너의 OEI/잼 매핑과 정합)
_FAR = {"M1": "SC-VTOL.2140", "M2": "SC-VTOL.2140",
        "M3": "SC-VTOL.2140", "M4": "SC-VTOL.2150",
        "M5": "SC-VTOL.2150", "M6": "SC-VTOL.2150"}


P_PHASE_TRANSITION = 0.025


@dataclass
class RetrimEvent:
    """확장 사건 하나 = (고장 사건) x (지배 재트림 패턴)."""
    mode: str                     # M1..M5
    rotor_label: str              # 대표(1차) 고장 로터
    failed_ids: Tuple[int, ...]   # 고장 로터 rotor_id 집합
    P: float                      # 부모 고장 사건의 발생 가중치
    pattern: np.ndarray           # 로터별 추력 배율 (x T_hover)
    consequence: float            # C = 이산 격자 헐 대비 방향 초과율
    worst_component: str = ""
    worst_station: float = 0.0
    n_patterns: int = 0           # 이 사건에서 선별한 패턴 수
    phase: str = "HV"             # HV(호버) | TR(천이) | TILT(틸트 회랑)
    V_eas: float = 0.0            # 천이 사건의 비행 속도 (m/s EAS)
    nz_cond: float = 1.0          # 천이 사건의 조건 하중배수
    gust_dn: float = 0.0          # 지배 패턴의 돌풍 증분 (±dn 또는 0)
    tilt_deg: float = 0.0         # 틸트 사건의 회랑 스케줄 각
    stuck_rotor_id: Optional[int] = None  # M6 고착 로터
    stuck_deg: float = 0.0        # M6 고착각

    @property
    def risk(self) -> float:
        return self.P * self.consequence

    @property
    def label(self) -> str:
        pat = "/".join(f"{v:g}" for v in self.pattern)
        head = ("Retrim" if self.phase == "HV"
                else f"Retrim TR V={self.V_eas:.1f} nz={self.nz_cond:g}"
                if self.phase == "TR"
                else f"Retrim TILT V={self.V_eas:.1f} "
                     f"s={self.tilt_deg:.0f}deg"
                + (f" stuck R{self.stuck_rotor_id}"
                   f"@{self.stuck_deg:.0f}"
                   if self.stuck_rotor_id else ""))
        head = head + (f" g={self.gust_dn:+.2f}" if self.gust_dn
                       else "")
        return (f"{head} {self.mode} {self.rotor_label} "
                f"l={pat} C={self.consequence*100:.1f}%")


def _mirror_idx(rotors, i: int) -> int:
    """같은 |y| 반대편 로터 — 동률이면 같은 열(x)이 우선."""
    yi = rotors[i].hub_position[1]
    xi = rotors[i].hub_position[0]
    best, dbest = i, 1e18
    for j, r in enumerate(rotors):
        if j == i:
            continue
        d = (abs(r.hub_position[1] + yi)
             + 1e-6 * abs(r.hub_position[0] - xi))
        if d < dbest:
            best, dbest = j, d
    return best


def _sameside_idx(rotors, i: int) -> int:
    yi = rotors[i].hub_position[1]
    best, dbest = i, 1e18
    for j, r in enumerate(rotors):
        yj = r.hub_position[1]
        if j == i or yi * yj <= 0:
            continue
        d = abs(yj - yi)
        if d < dbest:
            best, dbest = j, d
    return best


def solve_transition_trim_affine(model, config, wc,
                                 speeds: Sequence[float]):
    """속도별 대칭 트림 아핀쌍 {V: (C0f, C1f)} — 고정 q에서 트림
    절점하중이 nz에 정확히 아핀임을 이용해 nz=1, 2 두 해로 절편과
    기울기를 구한다. 재트림 스크린과 천이 과도 돌풍 러너가 공유하는
    공용 헬퍼이다."""
    from types import SimpleNamespace

    from ..case_generator import TrimCondition
    from .aircraft_config import dynamic_pressure_from_eas, eas_to_mach
    from .load_case_matrix import CertLoadCase

    from .batch_runner import BatchRunner

    cases = []
    for k, V in enumerate(speeds):
        for m, nz in enumerate((1.0, 2.0)):
            tc = TrimCondition(
                case_id=980000 + 10 * k + m,
                mach=eas_to_mach(V, 0.0),
                q=dynamic_pressure_from_eas(V), nz=nz,
                fixed_vars={"ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0,
                            "URDD4": 0.0, "URDD6": 0.0},
                free_vars=["ANGLEA", "ELEV"],
                label=f"retrim TR basis V={V:.1f} nz={nz:g}")
            cases.append(CertLoadCase(
                trim_condition=tc, category="retrim_basis",
                weight_cg=wc))
    runner = BatchRunner(
        SimpleNamespace(config=config, flight_cases=cases,
                        landing_cases=[], dynamic_cases=[]),
        bdf_model=model)
    result = runner.run()
    by_id = {r.case_id: r for r in result.case_results
             if r.converged and r.nodal_forces}
    out = {}
    for k, V in enumerate(speeds):
        a = by_id.get(980000 + 10 * k)
        b = by_id.get(980000 + 10 * k + 1)
        if a is None or b is None:
            logger.warning("transition basis trim failed at V=%.1f", V)
            continue
        C1 = {n: np.array(b.nodal_forces[n])
              - np.array(a.nodal_forces.get(n, np.zeros(6)))
              for n in b.nodal_forces}
        for n, f in a.nodal_forces.items():
            if n not in C1:
                C1[n] = -np.array(f)
        C0 = {}
        for n in set(a.nodal_forces) | set(C1):
            C0[n] = (np.array(a.nodal_forces.get(n, np.zeros(6)))
                     - C1.get(n, np.zeros(6)))
        out[round(V, 3)] = (C0, C1)
    return out


class RetrimScreen:
    """호버 계열 (고장×재트림) 사건 공간의 선형 전수 선별기."""

    def __init__(self, bdf_model, vtol_config, aircraft_config,
                 components=None, fuselage_cg_x: Optional[float] = None,
                 p_mode: Optional[Dict[str, float]] = None,
                 p_phase: float = P_PHASE_HOVER,
                 transition: bool = True,
                 p_phase_tr: float = P_PHASE_TRANSITION,
                 p_mode_m6: float = 0.20):
        self.model = bdf_model
        self.vtol_config = vtol_config
        self.config = aircraft_config
        self.components = components
        self.p_mode = dict(p_mode or P_MODE_DEFAULT)
        self.p_phase = p_phase
        self.transition = transition
        self.p_phase_tr = p_phase_tr
        # M6(틸트 액추에이터 고착)의 모드 분할 가중 — M4(모터 고착)와
        # 같은 고착 계열로 두는 가정값 (분류 확장, 상대 우선순위)
        self.p_mode_m6 = p_mode_m6

        self.rotors = [r for r in vtol_config.hover_rotors]
        self.n = len(self.rotors)
        self.wc = aircraft_config.weight_cg_conditions[0]

        nm = compute_node_masses(bdf_model)
        self._total_mass = sum(nm.values())
        cg = np.zeros(3)
        for nid, m in nm.items():
            cg += m * bdf_model.nodes[nid].xyz_global
        self._cg = cg / self._total_mass
        self._g = 9806.65                          # mm/s^2 (N-mm-s)
        if fuselage_cg_x is None:
            fuselage_cg_x = float(self._cg[0])
        self._fus_cg_x = fuselage_cg_x

        self._hub_vecs = self._nominal_hub_vecs()
        self._B, self._stations = self._build_basis()
        from ...rotor.rotor_config import RotorType
        self._has_tilt = any(r.rotor_type == RotorType.TILT
                             for r in vtol_config.rotors)
        if transition and self._has_tilt:
            # 틸트 구성: L+C 천이 계열 대신 틸트 회랑 계열을 선별
            self._tr = []
            self._tilt = self._build_tilt_bases()
        else:
            self._tr = (self._build_transition_bases()
                        if transition else [])
            self._tilt = []

    # ---------------------------------------------------------------
    def _nominal_hub_vecs(self) -> List[Tuple[int, np.ndarray]]:
        """호버 1g 명목 상태의 로터별 허브 6분력 (파이프라인 경로)."""
        matrix = VTOLLoadCaseMatrix(self.vtol_config, self.config)
        cond = VTOLCondition(
            label="retrim basis hover", phase=VTOLFlightPhase.HOVER,
            V_eas=0.0, nz=1.0, altitude_m=0.0, thrust_fraction=1.0)
        forces = matrix._compute_rotor_forces_hover(cond, self.wc)
        return [(r.hub_node_id, np.array(forces[r.hub_node_id], float))
                for r in self.rotors]

    def _pattern_forces(self, pattern: np.ndarray) -> Tuple[Dict, float]:
        """추력 패턴 -> 자기평형 절점하중 (관성 + relief 폐합)."""
        nz = float(np.mean(pattern))
        forces: Dict[int, np.ndarray] = {}
        for (nid, vec), li in zip(self._hub_vecs, pattern):
            if abs(li) < 1e-12:
                continue
            forces[nid] = forces.get(nid, np.zeros(6)) + li * vec
        inertial = compute_nodal_inertial_forces(self.model, nz, self._g)
        for nid, f in inertial.items():
            forces[nid] = forces.get(nid, np.zeros(6)) + f
        apply_inertia_relief(self.model, {}, forces,
                             cg=self._cg, g=self._g)
        return forces, nz

    def _build_basis(self):
        """로터별 단위 기저 VMT — 사상 전체가 추력에 선형이므로
        임의 패턴의 VMT = sum(l_i * B_i)."""
        batch = BatchResult()
        for i in range(self.n):
            pat = np.zeros(self.n)
            pat[i] = 1.0
            forces, nz = self._pattern_forces(pat)
            batch.case_results.append(CaseResult(
                case_id=990000 + i, category="retrim_basis",
                converged=True, nodal_forces=forces, nz=nz,
                label=f"retrim basis R{i+1}"))
            batch.completed_ids.add(990000 + i)
        vmt = compute_vmt_for_batch(self.model, batch,
                                    components=self.components,
                                    fuselage_cg_x=self._fus_cg_x)
        comps = sorted(next(iter(vmt.values())).keys())
        self._comps = comps
        stations = []
        for comp in comps:
            for s in vmt[990000][comp]["stations"]:
                stations.append((comp, float(s)))
        B = np.stack([self._vmt_mat(vmt[990000 + i]) for i in
                      range(self.n)])                  # (n, S, 3)
        return B, stations

    def _vmt_mat(self, vmt_case) -> np.ndarray:
        rows = []
        for comp in self._comps:
            d = vmt_case[comp]
            rows.append(np.column_stack([d["shear"], d["bending"],
                                         d["torsion"]]))
        return np.vstack(rows)

    # ---------------------------------------------------------------
    # 천이 계열 (q>0) — 고정 q에서 트림 하중이 nz_eff에 정확히
    # 아핀이므로(선형 공탄성) 속도점당 트림해 2회 + 로터 허브 기저로
    # 임의 재트림 패턴의 VMT가 닫힌형으로 평가된다:
    #   VMT(l; V, nz) = C0(V) + nz_eff*C1(V) + sum l_i*R_i(V) + Ap(V)
    #   nz_eff = nz - sum(l_i)/n      (관성 기저는 relief로 정확 소거)
    # ---------------------------------------------------------------
    def _transition_conditions(self):
        from .vtol_conditions import generate_transition_conditions

        conds = generate_transition_conditions(
            self.vtol_config.v_mca, self.vtol_config.v_transition_end,
            [0.0],
            wing_area_m2=getattr(self.config, "wing_area_m2", 0.0),
            CL_transition=1.0, weight_N=self.wc.weight_N)
        return [c for c in conds if c.thrust_fraction > 0.05]

    def _solve_trim_pair(self, speeds: Sequence[float]):
        """속도별 대칭 트림해 (nz=1, nz=2) — 아핀 기저 C0/C1의 원료."""
        return solve_transition_trim_affine(self.model, self.config,
                                            self.wc, speeds)

    def _build_transition_bases(self):
        """조건별 (V, nz, tf) 기저: 트림 아핀쌍 + 로터/푸셔 허브 기저."""
        from .vtol_conditions import VTOLFlightPhase

        conds = self._transition_conditions()
        if not conds:
            return []
        matrix = VTOLLoadCaseMatrix(self.vtol_config, self.config)
        speeds = sorted({round(c.V_eas, 3) for c in conds})
        trim_pairs = self._solve_trim_pair(speeds)

        T_hover = self.wc.weight_N / self.n
        hub_ids = {r.hub_node_id for r in self.rotors}
        entries = []
        force_sets = []           # (tag, forces) -> VMT 일괄 계산
        for c in conds:
            key = round(c.V_eas, 3)
            if key not in trim_pairs:
                continue
            forces = matrix._compute_rotor_forces_transition(c, self.wc)
            tf = c.thrust_fraction
            per_rotor_T = tf * T_hover          # 조건 명목 로터 추력
            if per_rotor_T <= 0:
                continue
            Rf = []
            for r in self.rotors:
                vec = np.array(forces.get(r.hub_node_id, np.zeros(6)))
                Rf.append({r.hub_node_id: vec / tf})   # 단위 T_hover당
            Apf = {n: np.array(v) for n, v in forces.items()
                   if n not in hub_ids}                 # 푸셔 등
            C0f, C1f = trim_pairs[key]
            # 준정적 돌풍 증분 — 날개 Pratt + 로터 분담 혼합 (①과
            # 동일 모델); 재트림 공간의 환경 자유도로 스윕된다
            from .vtol_conditions import transition_gust_delta_nz
            share = tf / c.nz if c.nz > 0 else tf
            dn = transition_gust_delta_nz(
                c.V_eas, share, self.wc.weight_N,
                getattr(self.config, "wing_area_m2", 0.0),
                getattr(self.config, "CLalpha", 0.0),
                getattr(self.config, "mean_chord_m", 1.0))
            entry = {"cond": c, "V": c.V_eas, "nz": c.nz, "tf": tf,
                     "dn": dn,
                     "C0f": C0f, "C1f": C1f, "Rf": Rf, "Apf": Apf}
            entries.append(entry)
            base = len(force_sets)
            force_sets.append(("C0", C0f))
            force_sets.append(("C1", C1f))
            force_sets.append(("Ap", Apf))
            for rf in Rf:
                force_sets.append(("R", rf))
            entry["_slice"] = (base, len(force_sets))

        # 기저 힘 세트 전부의 VMT (각각 relief 폐합; 선형이라 합성 유효)
        batch = BatchResult()
        for k, (tag, fdict) in enumerate(force_sets):
            fd = {n: np.array(v, float).copy() for n, v in fdict.items()}
            if tag in ("R", "Ap"):
                apply_inertia_relief(self.model, {}, fd,
                                     cg=self._cg, g=self._g)
            batch.case_results.append(CaseResult(
                case_id=970000 + k, category="retrim_basis",
                converged=True, nodal_forces=fd, nz=0.0,
                label=f"tr basis {tag}{k}"))
            batch.completed_ids.add(970000 + k)
        if not batch.case_results:
            return []
        vmt = compute_vmt_for_batch(self.model, batch,
                                    components=self.components,
                                    fuselage_cg_x=self._fus_cg_x)
        for entry in entries:
            i0, i1 = entry.pop("_slice")
            mats = [self._vmt_mat(vmt[970000 + k]) for k in range(i0, i1)]
            entry["C0"] = mats[0]
            entry["C1"] = mats[1]
            entry["Ap"] = mats[2]
            entry["R"] = np.stack(mats[3:])          # (n, S, 3)
        logger.info("transition bases: %d conditions, %d trim solves",
                    len(entries), 2 * len(speeds))
        return entries

    def _nz_eff_max(self, V_eas: float) -> float:
        """날개 실속 상한: nz_eff <= CL_max * q * S / W."""
        q = 0.5 * 1.225 * V_eas ** 2
        S = getattr(self.config, "wing_area_m2", 0.0)
        if S <= 0 or self.wc.weight_N <= 0:
            return 2.0
        return min(2.0, 1.0 * q * S / self.wc.weight_N)

    def _tr_pattern_rows(self, entry, P_pat: np.ndarray,
                         nz_g: Optional[float] = None) -> np.ndarray:
        """천이 패턴 행렬 -> VMT 행렬 (N, S, 3).

        nz_g가 주어지면 조건 하중배수 대신 사용한다(돌풍 증분:
        nz_g = nz_cond ± dn — 아핀 기저라 같은 식으로 평가됨)."""
        nz_rotor = P_pat.sum(axis=1) / self.n
        nz_eff = (entry["nz"] if nz_g is None else nz_g) - nz_rotor
        rows = (entry["C0"][None, :, :]
                + nz_eff[:, None, None] * entry["C1"][None, :, :]
                + entry["Ap"][None, :, :]
                + np.tensordot(P_pat, entry["R"], axes=(1, 0)))
        return rows, nz_eff

    def _tr_discrete_patterns(self, tf: float) -> List[np.ndarray]:
        """천이 조건의 이산 격자 상태 (T_hover 배수) — 명목 스케줄 +
        모드별 명목 재분배."""
        pats = [np.full(self.n, tf)]
        for mode in ("M1", "M2", "M3", "M4", "M5"):
            for i in range(self.n):
                failed = self._failed_set(mode, i)
                if failed is None:
                    continue
                pin = self._tr_pin_level(mode, tf)
                pat = np.full(self.n, tf)
                surv = [k for k in range(self.n) if k not in failed]
                deficit = sum(tf - pin for _ in failed)
                for k in failed:
                    pat[k] = pin
                if surv:
                    for k in surv:
                        pat[k] = tf + deficit / len(surv)
                pats.append(pat)
        return pats

    def _tr_pin_level(self, mode: str, tf: float) -> float:
        """천이 고장 로터 고정 레벨 (T_hover 배수): 정지 0, 고착은
        직전 지령(스케줄 tf), 폭주는 절대 최대 1.5*T_hover."""
        return {"M1": 0.0, "M2": 0.0, "M3": 0.0,
                "M4": tf, "M5": RUNAWAY_FACTOR}[mode]

    # ---------------------------------------------------------------
    def _discrete_patterns(self) -> List[np.ndarray]:
        """이산 분류 격자가 이미 포괄하는 추력 상태들 — 명목 호버
        (0.7/1.0/1.15/1.3) + 각 고장 사건의 명목 재분배 상태."""
        pats = [np.full(self.n, tf) for tf in (0.7, 1.0, 1.15, 1.3)]
        for mode in ("M1", "M2", "M3", "M4", "M5"):
            for i in range(self.n):
                failed = self._failed_set(mode, i)
                if failed is None:
                    continue
                pat = np.full(self.n, 1.0)
                pin = self._pin_level(mode)
                surv = [k for k in range(self.n) if k not in failed]
                # 명목 재분배: 생존 로터가 1g 양력을 균등 분담
                lift_deficit = sum(1.0 - pin for _ in failed)
                for k in failed:
                    pat[k] = pin
                if surv:
                    for k in surv:
                        pat[k] = 1.0 + lift_deficit / len(surv)
                pats.append(pat)
        return pats

    def _pin_level(self, mode: str) -> float:
        return {"M1": 0.0, "M2": 0.0, "M3": 0.0,
                "M4": 1.0, "M5": RUNAWAY_FACTOR}[mode]

    def _failed_set(self, mode: str, i: int):
        """모드별 고장 로터 인덱스 집합 (없으면 None: 중복 사건)."""
        if mode in ("M1", "M4", "M5"):
            return frozenset([i])
        if mode == "M2":
            j = _mirror_idx(self.rotors, i)
            return frozenset([i, j]) if i < j else None
        if mode == "M3":
            j = _sameside_idx(self.rotors, i)
            if j == i:
                return None
            return frozenset([i, j]) if i < j else None
        return None

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # 틸트 회랑 계열 (C단계) — (V, σ) 상태별 방향 기저 + M6 고착
    # ---------------------------------------------------------------
    def _tilt_unit_forces(self, matrix, rotor, V, angle_deg, rho):
        """로터 1기의 단위(T_hover당) 6분력 — 축 [sinα,0,cosα],
        BEMT 달성 추력·토크로 방향/토크비를 잡고 T_hover로 정규화."""
        import math

        from ...rotor.rotor_config import RotationDir

        T_ref = self.wc.weight_N / self.n
        a_r = math.radians(angle_deg)
        axis = np.array([math.sin(a_r), 0.0, math.cos(a_r)])
        if V > 1.0:
            solver = matrix._get_ff_solver(rotor)
            loads = solver.solve_for_thrust(
                T_ref, rotor.rpm_hover, V,
                alpha_shaft=a_r, rho=rho)
        else:
            solver = matrix._get_bemt_solver(rotor)
            loads = solver.solve_for_thrust(T_ref, rotor.rpm_hover,
                                            rho)
        scale = T_ref / max(loads.thrust, 1e-6)
        rot_sign = (1.0 if rotor.rotation_dir == RotationDir.CW
                    else -1.0)
        fvec = np.zeros(6)
        fvec[:3] = loads.thrust * scale * axis
        fvec[3:] = -rot_sign * loads.torque * scale * axis
        return {rotor.hub_node_id: fvec}

    def _build_tilt_bases(self, n_speeds: int = 4):
        """틸트 회랑 (V, σ) 상태별 기저 — 트림 아핀쌍 + 로터·각도별
        단위 방향 기저 (M6 고착각 {0, 90} 변형 포함)."""
        from ...rotor.rotor_config import RotorType
        from .vtol_conditions import (
            tilt_allocation, transition_gust_delta_nz,
        )

        matrix = VTOLLoadCaseMatrix(self.vtol_config, self.config)
        rho = 1.225
        W = self.wc.weight_N
        S = getattr(self.config, "wing_area_m2", 0.0)
        tilt_idx = [k for k, r in enumerate(self.rotors)
                    if r.rotor_type == RotorType.TILT]
        aft_idx = [k for k in range(self.n) if k not in tilt_idx]
        v0 = max(1.0, self.vtol_config.v_mca * 0.5)
        speeds = np.unique(np.round(np.linspace(
            v0, self.vtol_config.v_transition_end, n_speeds), 3))
        from .vtol_conditions import TILT_GRID_DEG
        grid = np.arange(TILT_GRID_DEG, 90.0 + 1e-9,
                         TILT_GRID_DEG)

        # 상태 목록: (V, σ_lo/σ_c/σ_hi) + 상태별 명목 배분
        states = []
        for V in speeds:
            feas = [sg for sg in grid
                    if tilt_allocation(float(V), float(sg), 1.0, W, S,
                                       1.0, rho, len(tilt_idx),
                                       len(aft_idx))[3]]
            if not feas:
                continue
            picks = sorted({feas[0], 0.5 * (feas[0] + feas[-1]),
                            feas[-1]})
            for sg in picks:
                F, A, L, _ok = tilt_allocation(
                    float(V), float(sg), 1.0, W, S, 1.0, rho,
                    len(tilt_idx), len(aft_idx))
                share_rot = (F * np.cos(np.radians(sg)) + A) / W
                dn = transition_gust_delta_nz(
                    float(V), share_rot, W, S,
                    getattr(self.config, "CLalpha", 0.0),
                    getattr(self.config, "mean_chord_m", 1.0))
                states.append({"V": float(V), "sigma": float(sg),
                               "F": F, "A": A, "dn": dn})

        trim_pairs = solve_transition_trim_affine(
            self.model, self.config, self.wc,
            sorted({st["V"] for st in states}))

        # 로터·각도별 단위 힘 세트 → VMT 기저 일괄 계산
        T_hover = W / self.n
        stuck_angles = (0.0, 90.0)
        force_sets = {}                     # (V, k, angle) -> fdict
        for st in states:
            V = st["V"]
            for k in tilt_idx:
                for ang in {st["sigma"], *stuck_angles}:
                    key = (V, k, round(float(ang), 1))
                    if key not in force_sets:
                        force_sets[key] = self._tilt_unit_forces(
                            matrix, self.rotors[k], V, float(ang),
                            rho)
            for k in aft_idx:
                key = (V, k, 0.0)
                if key not in force_sets:
                    force_sets[key] = self._tilt_unit_forces(
                        matrix, self.rotors[k], V, 0.0, rho)

        batch = BatchResult()
        keys = sorted(force_sets)
        for i, key in enumerate(keys):
            fd = {n: np.array(v, float).copy()
                  for n, v in force_sets[key].items()}
            apply_inertia_relief(self.model, {}, fd, cg=self._cg,
                                 g=self._g)
            batch.case_results.append(CaseResult(
                case_id=960000 + i, category="retrim_basis",
                converged=True, nodal_forces=fd, nz=0.0,
                label=f"tilt basis {key}"))
            batch.completed_ids.add(960000 + i)
        vmt = compute_vmt_for_batch(self.model, batch,
                                    components=self.components,
                                    fuselage_cg_x=self._fus_cg_x)
        Bmap = {key: self._vmt_mat(vmt[960000 + i])
                for i, key in enumerate(keys)}

        entries = []
        for st in states:
            V, sg = st["V"], st["sigma"]
            key0 = round(V, 3)
            if key0 not in trim_pairs:
                continue
            C0f, C1f = trim_pairs[key0]
            angle_of = {}
            for k in range(self.n):
                angle_of[k] = sg if k in tilt_idx else 0.0
            B = np.stack([Bmap[(V, k, round(angle_of[k], 1))]
                          for k in range(self.n)])
            # M6 고착각 변형 기저 (틸트 로터만)
            Bstuck = {(k, a): Bmap[(V, k, a)]
                      for k in tilt_idx for a in stuck_angles}
            l_nom = np.empty(self.n)
            for k in range(self.n):
                l_nom[k] = ((st["F"] / len(tilt_idx))
                            if k in tilt_idx
                            else (st["A"] / max(len(aft_idx), 1))) \
                    / T_hover
            Fkey = {k: force_sets[(V, k, round(angle_of[k], 1))]
                    for k in range(self.n)}
            Fstuck = {(k, a): force_sets[(V, k, a)]
                      for k in tilt_idx for a in stuck_angles}
            fz = np.array([sum(v[2] for v in Fkey[k].values())
                           for k in range(self.n)])
            fz_stuck = {ka: sum(v[2] for v in fd.values())
                        for ka, fd in Fstuck.items()}
            entries.append({
                "V": V, "sigma": sg, "dn": st["dn"],
                "C0f": C0f, "C1f": C1f,
                "C0": None, "C1": None,       # 아래에서 채움
                "B": B, "Bstuck": Bstuck, "l_nom": l_nom,
                "tilt_idx": tilt_idx, "aft_idx": aft_idx,
                "Fkey": Fkey, "Fstuck": Fstuck,
                "_fz": fz, "_fz_stuck": fz_stuck,
            })

        # 트림 아핀쌍의 VMT 행렬 (자유낙하 상쇄로 관성 기저 불필요)
        pair_batch = BatchResult()
        vmap = {}
        i = 0
        for V in sorted({st["V"] for st in states}):
            key0 = round(V, 3)
            if key0 not in trim_pairs:
                continue
            for tag, fd0 in zip(("C0", "C1"), trim_pairs[key0]):
                fd = {n: np.array(v, float).copy()
                      for n, v in fd0.items()}
                pair_batch.case_results.append(CaseResult(
                    case_id=955000 + i, category="retrim_basis",
                    converged=True, nodal_forces=fd, nz=0.0,
                    label=f"tilt {tag} V={V}"))
                pair_batch.completed_ids.add(955000 + i)
                vmap[(key0, tag)] = 955000 + i
                i += 1
        vmt2 = compute_vmt_for_batch(self.model, pair_batch,
                                     components=self.components,
                                     fuselage_cg_x=self._fus_cg_x)
        for entry in entries:
            key0 = round(entry["V"], 3)
            entry["C0"] = self._vmt_mat(vmt2[vmap[(key0, "C0")]])
            entry["C1"] = self._vmt_mat(vmt2[vmap[(key0, "C1")]])
        logger.info("tilt bases: %d states, %d unit force sets",
                    len(entries), len(keys))
        return entries

    def _tilt_rows(self, entry, P_pat: np.ndarray,
                   stuck: Optional[Tuple[int, float]] = None,
                   nz_g: float = 1.0):
        """틸트 패턴 -> VMT 행렬 (N, S, 3).

        stuck=(rotor_idx, angle)이면 그 로터의 방향 기저와 수직
        성분을 고착각 변형으로 치환한다(M6)."""
        B = entry["B"]
        fz = entry["_fz"].copy()
        contrib = np.zeros((len(P_pat),) + B.shape[1:])
        for k in range(self.n):
            if stuck is not None and k == stuck[0]:
                Bk = entry["Bstuck"][(k, stuck[1])]
                fz_k = entry["_fz_stuck"][(k, stuck[1])]
            else:
                Bk = B[k]
                fz_k = fz[k]
            contrib += P_pat[:, k, None, None] * Bk[None, :, :]
            if k == 0:
                vert = P_pat[:, k] * fz_k
            else:
                vert = vert + P_pat[:, k] * fz_k
        nz_eff = nz_g - vert / self.wc.weight_N
        rows = (entry["C0"][None, :, :]
                + nz_eff[:, None, None] * entry["C1"][None, :, :]
                + contrib)
        return rows, nz_eff

    def screen(self, levels: Sequence[float] = COMMAND_LEVELS,
               nz_bounds: Tuple[float, float] = (0.5, 1.5),
               ) -> List[RetrimEvent]:
        """모든 (고장×재트림) 사건을 선별하고 P·C 위험도로 정렬.

        이산 격자 헐(S0)은 호버 + 천이 계열의 분류 상태 전부로
        구성되며, 호버·천이 재트림 패턴 모두 같은 헐에 대해
        채점된다 (스테이션별 축 정규화)."""
        from scipy.spatial import ConvexHull

        rows = [np.stack([np.tensordot(p, self._B, axes=(0, 0))
                          for p in self._discrete_patterns()])]
        for entry in self._tr:
            pats = np.array(self._tr_discrete_patterns(entry["tf"]))
            r, _ = self._tr_pattern_rows(entry, pats)
            rows.append(r)
        for entry in self._tilt:
            # 이산 격자 = 상태 명목 + 회랑 돌풍 명목(nz=1±dn) + M6
            # 고착 명목 (매트릭스의 실제 이산 케이스들)
            from .vtol_conditions import tilt_allocation
            r, _ = self._tilt_rows(entry, entry["l_nom"][None, :])
            rows.append(r)
            W = self.wc.weight_N
            S_w = getattr(self.config, "wing_area_m2", 0.0)
            T_hover = W / self.n
            ti, ai = entry["tilt_idx"], entry["aft_idx"]
            for g in (-1.0, 1.0):
                nz_g = 1.0 + g * entry["dn"]
                F, A, _L, _ok = tilt_allocation(
                    entry["V"], entry["sigma"], max(nz_g, 0.05), W,
                    S_w, 1.0, 1.225, len(ti), len(ai))
                l_g = np.empty(self.n)
                for k in range(self.n):
                    l_g[k] = ((F / len(ti)) if k in ti
                              else (A / max(len(ai), 1))) / T_hover
                r, _ = self._tilt_rows(entry, l_g[None, :],
                                       nz_g=nz_g)
                rows.append(r)
            for (k, a) in entry["Bstuck"]:
                r, _ = self._tilt_rows(entry,
                                       entry["l_nom"][None, :],
                                       stuck=(k, a))
                rows.append(r)
        S0 = np.concatenate(rows, axis=0)             # (m, S, 3)
        n_sta = self._B.shape[1]

        hulls = []
        for si in range(n_sta):
            pts = S0[:, si, :]
            lo = pts.min(axis=0)
            span = np.ptp(pts, axis=0)
            span[span == 0] = 1.0
            q = (pts - lo) / span
            try:
                hulls.append((lo, span, ConvexHull(q).equations))
            except Exception:
                hulls.append(None)

        def _score(V):
            worst = np.zeros(len(V))
            worst_sta = np.zeros(len(V), int)
            for si in range(n_sta):
                if hulls[si] is None:
                    continue
                lo, span, eq = hulls[si]
                qp = (V[:, si, :] - lo) / span
                viol = (qp @ eq[:, :3].T + eq[:, 3]).max(axis=1)
                upd = viol > worst
                worst_sta[upd] = si
                worst = np.maximum(worst, viol)
            return worst, worst_sta

        events: List[RetrimEvent] = []

        # ── 호버 계열 ──
        seen = set()
        for mode in ("M1", "M2", "M3", "M4", "M5"):
            for i in range(self.n):
                failed = self._failed_set(mode, i)
                if failed is None or (mode, failed) in seen:
                    continue
                if not all(self.rotors[k].can_fail for k in failed):
                    continue
                seen.add((mode, failed))
                pin = self._pin_level(mode)
                surv = [k for k in range(self.n) if k not in failed]
                # 재트림 공간: 생존 로터만 지령 대역 스윕 (추가
                # 비정상 로터 없음 — P는 부모 사건 그대로)
                combos = np.array(list(itertools.product(
                    levels, repeat=len(surv))))
                P_pat = np.full((len(combos), self.n), pin)
                P_pat[:, surv] = combos
                nz = P_pat.mean(axis=1)
                P_pat = P_pat[(nz >= nz_bounds[0]) & (nz <= nz_bounds[1])]
                if not len(P_pat):
                    continue
                V = np.tensordot(P_pat, self._B, axes=(1, 0))
                worst, worst_sta = _score(V)
                k = int(np.argmax(worst))
                comp_w, sta_w = self._stations[worst_sta[k]]
                P_event = ((len(failed) / self.n) * self.p_phase
                           * self.p_mode[mode])
                events.append(RetrimEvent(
                    mode=mode, rotor_label=self.rotors[i].label,
                    failed_ids=tuple(sorted(
                        self.rotors[k2].rotor_id for k2 in failed)),
                    P=P_event, pattern=P_pat[k].copy(),
                    consequence=float(max(worst[k], 0.0)),
                    worst_component=comp_w, worst_station=sta_w,
                    n_patterns=len(P_pat)))

        # ── 천이 계열 — 생존 로터가 스케줄 tf 주변의 지령 대역을
        #    스윕, 날개가 잔여 nz_eff를 트림 (실속 상한으로 제한) ──
        for entry in self._tr:
            tf = entry["tf"]
            nz_hi = self._nz_eff_max(entry["V"])
            seen = set()
            for mode in ("M1", "M2", "M3", "M4", "M5"):
                for i in range(self.n):
                    failed = self._failed_set(mode, i)
                    if failed is None or (mode, failed) in seen:
                        continue
                    if not all(self.rotors[k].can_fail for k in failed):
                        continue
                    seen.add((mode, failed))
                    pin = self._tr_pin_level(mode, tf)
                    surv = [k for k in range(self.n)
                            if k not in failed]
                    combos = np.array(list(itertools.product(
                        [tf * v for v in levels], repeat=len(surv))))
                    P_all = np.full((len(combos), self.n), pin)
                    P_all[:, surv] = combos
                    # 돌풍 축: 같은 사건을 nz_cond ± dn 환경에서도
                    # 평가 — 아핀 기저 재사용이라 추가 해석 비용 없음
                    best = None
                    n_eval = 0
                    for g in (-1.0, 0.0, 1.0):
                        nz_g = entry["nz"] + g * entry["dn"]
                        V, nz_eff = self._tr_pattern_rows(
                            entry, P_all, nz_g=nz_g)
                        ok = (nz_eff >= -0.5) & (nz_eff <= nz_hi)
                        P_pat, V = P_all[ok], V[ok]
                        if not len(P_pat):
                            continue
                        n_eval += len(P_pat)
                        worst, worst_sta = _score(V)
                        k = int(np.argmax(worst))
                        if best is None or worst[k] > best[0]:
                            best = (worst[k], worst_sta[k],
                                    P_pat[k].copy(), g * entry["dn"])
                    if best is None:
                        continue
                    cbest, sta_k, pat_k, gdn = best
                    comp_w, sta_w = self._stations[sta_k]
                    P_event = ((len(failed) / self.n) * self.p_phase_tr
                               * self.p_mode[mode])
                    events.append(RetrimEvent(
                        mode=mode, rotor_label=self.rotors[i].label,
                        failed_ids=tuple(sorted(
                            self.rotors[k2].rotor_id for k2 in failed)),
                        P=P_event, pattern=pat_k,
                        consequence=float(max(cbest, 0.0)),
                        worst_component=comp_w, worst_station=sta_w,
                        n_patterns=n_eval, phase="TR",
                        V_eas=entry["V"], nz_cond=entry["nz"],
                        gust_dn=float(gdn)))
        # ── 틸트 회랑 계열 — M1~M5 + M6(고착), 돌풍 환경축 ──
        for entry in self._tilt:
            l_nom = entry["l_nom"]
            nz_hi = self._nz_eff_max(entry["V"])

            def _eval_tilt(P_all, stuck=None):
                best, n_eval = None, 0
                for g in (-1.0, 0.0, 1.0):
                    nz_g = 1.0 + g * entry["dn"]
                    Vr, nz_eff = self._tilt_rows(entry, P_all,
                                                 stuck=stuck,
                                                 nz_g=nz_g)
                    ok = (nz_eff >= -0.5) & (nz_eff <= nz_hi)
                    Pp, Vr = P_all[ok], Vr[ok]
                    if not len(Pp):
                        continue
                    n_eval += len(Pp)
                    worst, worst_sta = _score(Vr)
                    k = int(np.argmax(worst))
                    if best is None or worst[k] > best[0]:
                        best = (worst[k], worst_sta[k], Pp[k].copy(),
                                g * entry["dn"])
                return best, n_eval

            seen = set()
            for mode in ("M1", "M2", "M3", "M4", "M5"):
                for i in range(self.n):
                    failed = self._failed_set(mode, i)
                    if failed is None or (mode, failed) in seen:
                        continue
                    if not all(self.rotors[k].can_fail
                               for k in failed):
                        continue
                    seen.add((mode, failed))
                    surv = [k for k in range(self.n)
                            if k not in failed]
                    combos = np.array(list(itertools.product(
                        levels, repeat=len(surv))))
                    P_all = np.empty((len(combos), self.n))
                    for k in failed:
                        P_all[:, k] = (0.0 if mode in ("M1", "M2",
                                                       "M3")
                                       else l_nom[k] if mode == "M4"
                                       else RUNAWAY_FACTOR)
                    P_all[:, surv] = combos * l_nom[surv]
                    best, n_eval = _eval_tilt(P_all)
                    if best is None:
                        continue
                    cb, sta_k, pat_k, gdn = best
                    comp_w, sta_w = self._stations[sta_k]
                    P_event = ((len(failed) / self.n)
                               * self.p_phase_tr
                               * self.p_mode[mode])
                    events.append(RetrimEvent(
                        mode=mode,
                        rotor_label=self.rotors[i].label,
                        failed_ids=tuple(sorted(
                            self.rotors[k2].rotor_id
                            for k2 in failed)),
                        P=P_event, pattern=pat_k,
                        consequence=float(max(cb, 0.0)),
                        worst_component=comp_w, worst_station=sta_w,
                        n_patterns=n_eval, phase="TILT",
                        V_eas=entry["V"], tilt_deg=entry["sigma"],
                        gust_dn=float(gdn)))
            # M6 — 틸트 고착: 방향 기저 치환, 추력은 전 로터 대역
            for (k, a) in sorted(entry["Bstuck"]):
                if abs(a - entry["sigma"]) < 5.0:
                    continue
                combos = np.array(list(itertools.product(
                    levels, repeat=self.n)))
                P_all = combos * l_nom[None, :]
                best, n_eval = _eval_tilt(P_all, stuck=(k, a))
                if best is None:
                    continue
                cb, sta_k, pat_k, gdn = best
                comp_w, sta_w = self._stations[sta_k]
                P_event = ((1.0 / self.n) * self.p_phase_tr
                           * self.p_mode_m6)
                events.append(RetrimEvent(
                    mode="M6", rotor_label=self.rotors[k].label,
                    failed_ids=(self.rotors[k].rotor_id,),
                    P=P_event, pattern=pat_k,
                    consequence=float(max(cb, 0.0)),
                    worst_component=comp_w, worst_station=sta_w,
                    n_patterns=n_eval, phase="TILT",
                    V_eas=entry["V"], tilt_deg=entry["sigma"],
                    stuck_rotor_id=self.rotors[k].rotor_id,
                    stuck_deg=float(a), gust_dn=float(gdn)))

        events.sort(key=lambda e: -e.risk)
        return events

    def _tr_pattern_forces(self, entry, pattern: np.ndarray,
                           nz_g: Optional[float] = None):
        """천이 패턴 -> 절점하중: 트림 아핀 공력 + 로터/푸셔 +
        로터 분담분 관성 + relief 폐합. nz_g는 돌풍 증분 반영값."""
        nz_cond = entry["nz"] if nz_g is None else nz_g
        nz_eff = float(nz_cond - pattern.sum() / self.n)
        forces: Dict[int, np.ndarray] = {}

        def _acc(fdict, scale=1.0):
            for nid, v in fdict.items():
                forces[nid] = (forces.get(nid, np.zeros(6))
                               + scale * np.asarray(v, float))

        _acc(entry["C0f"])
        _acc(entry["C1f"], nz_eff)
        _acc(entry["Apf"])
        for rf, li in zip(entry["Rf"], pattern):
            if abs(li) > 1e-12:
                _acc(rf, li)
        inertial = compute_nodal_inertial_forces(
            self.model, nz_cond - nz_eff, self._g)
        _acc(inertial)
        apply_inertia_relief(self.model, {}, forces,
                             cg=self._cg, g=self._g)
        return forces, nz_cond

    def _tilt_entry_for(self, e: RetrimEvent):
        for entry in self._tilt:
            if (abs(entry["V"] - e.V_eas) < 1e-6
                    and abs(entry["sigma"] - e.tilt_deg) < 1e-6):
                return entry
        raise KeyError(f"no tilt basis for V={e.V_eas} "
                       f"s={e.tilt_deg}")

    def _tilt_pattern_forces(self, entry, e: RetrimEvent):
        """틸트 패턴 -> 절점하중: 아핀 공력 + 방향 기저 힘 세트 +
        잔여 관성 + relief (M6 고착 로터는 고착각 힘 세트)."""
        nz_g = 1.0 + e.gust_dn
        stuck_k = None
        if e.stuck_rotor_id is not None:
            stuck_k = next(k for k, r in enumerate(self.rotors)
                           if r.rotor_id == e.stuck_rotor_id)
        vert = 0.0
        forces: Dict[int, np.ndarray] = {}

        def _acc(fdict, scale=1.0):
            for nid, v in fdict.items():
                forces[nid] = (forces.get(nid, np.zeros(6))
                               + scale * np.asarray(v, float))

        for k in range(self.n):
            li = e.pattern[k]
            if abs(li) < 1e-12:
                continue
            if stuck_k is not None and k == stuck_k:
                fd = entry["Fstuck"][(k, e.stuck_deg)]
                fz = entry["_fz_stuck"][(k, e.stuck_deg)]
            else:
                fd = entry["Fkey"][k]
                fz = entry["_fz"][k]
            _acc(fd, li)
            vert += li * fz
        nz_eff = nz_g - vert / self.wc.weight_N
        _acc(entry["C0f"])
        _acc(entry["C1f"], nz_eff)
        inertial = compute_nodal_inertial_forces(
            self.model, nz_g - nz_eff, self._g)
        _acc(inertial)
        apply_inertia_relief(self.model, {}, forces,
                             cg=self._cg, g=self._g)
        return forces, nz_g

    def _entry_for(self, e: RetrimEvent):
        for entry in self._tr:
            if (abs(entry["V"] - e.V_eas) < 1e-6
                    and abs(entry["nz"] - e.nz_cond) < 1e-6):
                return entry
        raise KeyError(f"no transition basis for V={e.V_eas}")

    # ---------------------------------------------------------------
    def realize(self, events: List[RetrimEvent],
                threshold_pct: float = 1.0,
                top_n: Optional[int] = None) -> List[CaseResult]:
        """채택 판정(P·C 순위 + C 문턱) 후 지배 패턴을 케이스로 실현."""
        adopt = [e for e in events
                 if e.consequence * 100.0 >= threshold_pct]
        if top_n is not None:
            adopt = adopt[:top_n]
        cases = []
        for m, e in enumerate(adopt):
            if e.phase == "TILT":
                entry = self._tilt_entry_for(e)
                forces, nz = self._tilt_pattern_forces(entry, e)
                mach = e.V_eas / 340.3
            elif e.phase == "TR":
                entry = self._entry_for(e)
                forces, nz = self._tr_pattern_forces(
                    entry, e.pattern, nz_g=e.nz_cond + e.gust_dn)
                mach = entry["cond"].V_eas / 340.3
            else:
                forces, nz = self._pattern_forces(e.pattern)
                mach = 0.0
            cases.append(CaseResult(
                case_id=RETRIM_CASE_ID_START + m,
                category="vtol_retrim",
                far_section=_FAR[e.mode],
                converged=True, nodal_forces=forces, nz=nz,
                weight_label=self.wc.label, mach=mach,
                label=e.label,
                flight_state={"retrim_mode": e.mode,
                              "retrim_phase": e.phase,
                              "retrim_V_eas": e.V_eas,
                              "retrim_P": e.P,
                              "retrim_C_pct": e.consequence * 100.0}))
        logger.info("retrim events: %d screened, %d adopted (C >= "
                    "%.1f%%)", len(events), len(cases), threshold_pct)
        return cases
