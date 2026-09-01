"""VTOL batch runner for combined conventional + VTOL load cases.

Wraps the existing BatchRunner and handles VTOL-specific cases:
- Hover cases (q≈0): static analysis with rotor forces only
- Transition cases: SOL 144 trim with injected rotor forces
- OEI/jam cases: handled via VTOLSimRunner (6-DOF dynamic)

The VTOL batch runner produces CaseResult objects compatible with
the existing VMT and deduplication pipeline.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
import time

import numpy as np

from .batch_runner import BatchRunner, BatchResult, CaseResult, _build_flight_state
from .load_case_matrix import CertLoadCase, LoadCaseMatrix
from .vtol_load_case_matrix import VTOLLoadCaseMatrix
from .vtol_sim_runner import VTOLSimRunner
from .vtol_conditions import VTOLFlightPhase
from .aircraft_config import AircraftConfig, WeightCGCondition
from .flight_sim import AircraftParams
from ...rotor.rotor_config import VTOLConfig

logger = logging.getLogger(__name__)


class VTOLBatchRunner:
    """Combined batch runner for conventional + VTOL load cases.

    Parameters
    ----------
    conv_matrix : LoadCaseMatrix
        Conventional flight case matrix.
    vtol_matrix : VTOLLoadCaseMatrix
        VTOL-specific case matrix.
    bdf_model : BDFModel
        The base BDF model.
    vtol_config : VTOLConfig
        VTOL rotor configuration.
    n_workers : int
        Number of parallel workers for conventional cases.
    """

    def __init__(self, conv_matrix: LoadCaseMatrix,
                 vtol_matrix: VTOLLoadCaseMatrix,
                 bdf_model=None,
                 vtol_config: Optional[VTOLConfig] = None,
                 n_workers: int = 0,
                 airfoil_config=None,
                 include_transient: bool = True):
        self.conv_matrix = conv_matrix
        self.vtol_matrix = vtol_matrix
        self.bdf_model = bdf_model
        self.vtol_config = vtol_config
        self.n_workers = n_workers
        self.airfoil_config = airfoil_config
        self.include_transient = include_transient
        self._results: List[CaseResult] = []

    def run(self) -> BatchResult:
        """Execute all load cases (conventional + VTOL).

        Returns
        -------
        BatchResult
            Combined results from all cases.
        """
        t0 = time.time()

        # 1. Run conventional cases through standard BatchRunner
        logger.info("=== Phase 1: Conventional load cases ===")
        conv_runner = BatchRunner(
            self.conv_matrix, self.bdf_model,
            n_workers=self.n_workers,
            airfoil_config=self.airfoil_config)
        conv_result = conv_runner.run()
        self._results.extend(conv_result.case_results)

        # 2. Run VTOL trim cases (transition with q > 0)
        logger.info("=== Phase 2: VTOL trim cases ===")
        vtol_trim_results = self._run_vtol_trim_cases()
        self._results.extend(vtol_trim_results)

        # 3. Run VTOL static cases (hover, q=0)
        logger.info("=== Phase 3: VTOL hover (static) cases ===")
        vtol_static_results = self._run_vtol_static_cases()
        self._results.extend(vtol_static_results)

        # 4. VTOL dynamic transient cases (OEI/jam → structural peak loads)
        logger.info("=== Phase 4: VTOL dynamic transient cases ===")
        self._transient_results = []
        if (self.include_transient and self.vtol_config is not None
                and self.bdf_model is not None):
            try:
                from .vtol_transient_loads import (
                    VTOLTransientLoadsRunner, summarize_transient_results,
                )
                transient_runner = VTOLTransientLoadsRunner(
                    self.bdf_model, self.vtol_config,
                    self.conv_matrix.config,
                    airfoil_config=self.airfoil_config,
                    n_workers=self.n_workers,
                )
                oei_results = transient_runner.run_all_oei(
                    t_sim=5.0, dt=0.005,
                    t_recognition_list=[0.15, 0.5, 1.0],
                )
                jam_results = transient_runner.run_all_jam(
                    t_sim=3.0, dt=0.005,
                )
                # 호버 1-cos 돌풍 과도 피크 (논문 2 통합 유입류 경로)
                gust_results = transient_runner.run_all_hover_gust()
                # 천이 회랑 과도 돌풍 (전진비 유입류 + 날개 준정적)
                tr_gust_results = (
                    transient_runner.run_all_transition_gust())
                self._transient_results = (oei_results + jam_results
                                           + gust_results
                                           + tr_gust_results)
                logger.info(summarize_transient_results(self._transient_results))

                # Add peak critical cases to the batch result
                case_id = 30000
                for tr in self._transient_results:
                    # Create a CaseResult for the peak transient condition
                    label_parts = [tr.event_type.upper(), tr.failed_rotor_label]
                    if tr.with_recovery:
                        label_parts.append(f"t_rec={tr.t_recognition:.2f}s")
                    label_parts.append(f"DAF={tr.daf_wing_Mx:.2f}")
                    far = {"oei": "SC-VTOL.2140",
                           "gust": "SC-VTOL.2215"}.get(tr.event_type,
                                                       "SC-VTOL.2150")
                    peak_result = CaseResult(
                        case_id=case_id,
                        category=f"vtol_{tr.event_type}_transient",
                        far_section=far,
                        # 피크 절점하중 스냅샷이 있으면 케이스로 합류
                        # (VMT/포락선/설계하중 선정에 참여)
                        nodal_forces=tr.peak_nodal_forces,
                        converged=True,
                        weight_label="MTOW",
                        altitude_m=0.0,
                        nz=1.0,
                        mach=0.0,
                        label=" ".join(label_parts),
                        flight_state={
                            "peak_wing_Mx": tr.peak_wing_Mx,
                            "peak_time": tr.peak_wing_Mx_time,
                            "daf_wing_Mx": tr.daf_wing_Mx,
                            "daf_wing_Vy": tr.daf_wing_Vy,
                            "daf_boom_Mx": tr.daf_boom_Mx,
                            "event_type": tr.event_type,
                            "failed_rotor": tr.failed_rotor_label,
                            "with_recovery": tr.with_recovery,
                            "t_recognition": tr.t_recognition,
                        },
                    )
                    self._results.append(peak_result)
                    case_id += 1
            except Exception as e:
                logger.warning("Phase 4 transient loads failed: %s", e)
                import traceback
                traceback.print_exc()

        wall_time = time.time() - t0
        logger.info("VTOL batch complete: %d total cases in %.1fs",
                     len(self._results), wall_time)

        result = BatchResult(
            case_results=self._results,
            completed_ids={r.case_id for r in self._results},
            config=self.conv_matrix.config,
            wall_time_s=wall_time,
        )
        return result

    # ------------------------------------------------------------------
    # 공통 헬퍼 — 질량 CG와 중력가속도 (관성/밸런싱 기준)
    # ------------------------------------------------------------------
    def _mass_refs(self):
        from ...solvers.sol144 import _compute_cg, _detect_gravity

        cg = _compute_cg(self.bdf_model)
        g = _detect_gravity(self.bdf_model)
        return cg, g

    @staticmethod
    def _merge_forces(base: dict, extra: dict) -> dict:
        out = {nid: f.copy() for nid, f in base.items()}
        for nid, f in extra.items():
            if nid in out:
                out[nid] = out[nid] + f
            else:
                out[nid] = f.copy()
        return out

    def _run_vtol_trim_cases(self) -> List[CaseResult]:
        """Run VTOL cases that have q > 0 (transition with rotors).

        로터가 양력의 일부를 분담하므로, 날개 트림은 잔여 하중배수
        nz_eff = (nz·W − ΣFz_rotor)/W 로 풀고, 결과 절점하중에
        로터 허브하중과 잔여 관성분 Δnz = nz − nz_eff 를 합산한다.
        구성상 ΣFz가 정확히 닫히고, 로터 모멘트 잔차는 inertia
        relief로 각가속도에 흡수돼 자기평형 하중이 된다.
        """
        from types import SimpleNamespace

        from ..trim_loads import (
            apply_inertia_relief, compute_nodal_inertial_forces,
        )

        trim_cases = [c for c in self.vtol_matrix.cases
                      if c.solve_type == "trim"
                      and c.trim_condition is not None]
        if not trim_cases:
            return []

        cg, g = self._mass_refs()

        # 로터 분담을 뺀 유효 하중배수로 트림 조건 재구성
        import copy

        eff_cases = []
        delta_nz = {}
        for case in trim_cases:
            W = case.weight_cg.weight_N if case.weight_cg else 0.0
            rotor_fz = sum(f[2] for f in (case.rotor_forces or {}).values())
            nz_eff = case.trim_condition.nz - (rotor_fz / W if W else 0.0)
            delta_nz[case.case_id] = case.trim_condition.nz - nz_eff
            c2 = copy.copy(case)
            c2.trim_condition = copy.copy(case.trim_condition)
            c2.trim_condition.nz = nz_eff
            eff_cases.append(c2)
            logger.info("  VTOL trim %s: nz=%.2f -> wing nz_eff=%.3f "
                        "(rotor Fz=%.0f N)", case.label,
                        case.trim_condition.nz, nz_eff, rotor_fz)

        stub = SimpleNamespace(flight_cases=eff_cases, landing_cases=[],
                               dynamic_cases=[],
                               config=self.conv_matrix.config)
        runner = BatchRunner(stub, self.bdf_model,
                             n_workers=self.n_workers,
                             airfoil_config=self.airfoil_config)
        eff_result = runner.run()

        by_id = {c.case_id: c for c in trim_cases}
        results = []
        for r in eff_result.case_results:
            case = by_id.get(r.case_id)
            if case is None:
                continue
            if r.converged and r.nodal_forces:
                nodal = self._merge_forces(r.nodal_forces,
                                           case.rotor_forces or {})
                dnz = delta_nz.get(case.case_id, 0.0)
                if abs(dnz) > 1e-9:
                    extra = compute_nodal_inertial_forces(
                        self.bdf_model, nz=dnz, g=g)
                    nodal = self._merge_forces(nodal, extra)
                # inertial 인자는 더미 — 같은 dict를 두 번 주면 이중 가산됨
                relief = apply_inertia_relief(
                    self.bdf_model, {}, nodal, cg=cg, g=g)
                r.nodal_forces = nodal
                r.nz = case.trim_condition.nz   # 보고는 원래 nz로
                if r.flight_state is not None:
                    r.flight_state.update(relief)
            results.append(r)

        logger.info("VTOL trim cases: %d (%d converged)",
                    len(results), sum(1 for r in results if r.converged))
        return results

    def _run_vtol_static_cases(self) -> List[CaseResult]:
        """Run VTOL hover cases (q=0, rotor + inertial loads).

        In hover, there's no aerodynamic force from the wing/tail.
        절점하중 = 로터 허브하중 + 관성(nz·g) 분포. OEI/로터잼처럼
        의도적으로 불평형인 케이스는 inertia relief로 잔차를 강체
        가속도에 흡수시켜 자기평형 하중으로 만든다.
        """
        from ..trim_loads import (
            apply_inertia_relief, compute_nodal_inertial_forces,
        )

        cg, g = self._mass_refs()
        results = []

        for case in self.vtol_matrix.cases:
            if case.solve_type != "static_rotor":
                continue

            nz = case.trim_condition.nz if case.trim_condition else 1.0
            inertial = compute_nodal_inertial_forces(self.bdf_model, nz, g)
            nodal_forces = self._merge_forces(inertial,
                                              case.rotor_forces or {})
            relief = apply_inertia_relief(
                self.bdf_model, {}, nodal_forces, cg=cg, g=g)
            fs = _build_flight_state(case)
            fs.update(relief)

            result = CaseResult(
                case_id=case.case_id,
                category=case.category,
                far_section=case.far_section,
                converged=True,
                nodal_forces=nodal_forces,
                weight_label=case.weight_cg.label if case.weight_cg else "",
                altitude_m=case.altitude_m,
                nz=nz,
                mach=0.0,
                label=case.label,
                flight_state=fs,
            )
            results.append(result)

        logger.info("VTOL static (hover) cases: %d", len(results))
        return results
