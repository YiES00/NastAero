#!/usr/bin/env python3
# ILC-8 (1,900kg 리프트+크루즈) 인증 하중해석 드라이버 — 쉘 FE 전기체 모델
"""Run VTOL certification loads analysis on the ILC-8 Lift+Cruise model.

Model: nastaero.models.ilc8 procedural shell FE (tests/validation/ILC8/ilc8.bdf)
  - 8 lift rotors (R=0.75 m, boom-mounted) + 1 pusher (R=0.65 m)
  - MTOW 1,900 kg, CG x=4450 mm

Pipeline mirrors run_vtol_cert_analysis.py (GACOMP TR-12):
  conventional Part 23 matrix + VTOL matrix -> VTOLBatchRunner
  -> rotor hub loads -> VMT -> envelopes/potato -> critical cases
  -> FORCE card export.
"""
import os
import time
from datetime import datetime
import numpy as np

from nastaero.bdf.parser import parse_bdf
from nastaero.config import setup_logging


def _gear_nodes(model, x_target: float, n: int = 2):
    """동체 하면 절점 중 착륙장치 부착점 n개.

    n=2(메인 기어)는 반드시 좌/우 대칭쌍으로 고른다 — 같은 쪽 절점
    2개를 고르면 반력이 롤 모멘트를 만들어 관성 릴리프가 비대칭
    하중을 유발한다. n=1(노즈)은 중심선(y≈0) 절점.
    """
    # 동체 + 기어 레그(992xxx) 절점 — 최저 z가 레그 끝이면 그곳이 부착점
    fus = [(nid, nd.xyz_global) for nid, nd in model.nodes.items()
           if 100000 <= nid < 300000 or 992000 <= nid < 993000]
    zmin = min(p[2] for _, p in fus)
    belly = [(nid, p) for nid, p in fus if p[2] < zmin + 150.0]
    if n == 1:
        cand = [t for t in belly if abs(t[1][1]) < 50.0]
        cand.sort(key=lambda t: abs(t[1][0] - x_target))
        return [cand[0][0]]
    right = [t for t in belly if t[1][1] > 50.0]
    right.sort(key=lambda t: abs(t[1][0] - x_target))
    r_nid, r_pos = right[0]
    # 우현 절점의 y-미러 절점을 좌현에서 탐색
    left = [t for t in belly if t[1][1] < -50.0]
    l_nid = min(left, key=lambda t: (abs(t[1][0] - r_pos[0])
                                     + abs(t[1][1] + r_pos[1])))[0]
    return [r_nid, l_nid]


def main():
    setup_logging("WARNING")
    print("=" * 70)
    print("ILC-8 Lift+Cruise VTOL Certification Loads Analysis")
    print("=" * 70)
    t0 = time.time()

    ts = datetime.now()
    output_dir = f"ilc8_cert_results_{ts.strftime('%Y%m%d_%H%M%S')}"
    timestamp_label = ts.strftime("Analysis: %Y-%m-%d %H:%M:%S")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}/\n")

    # ---- 1. Parse ILC-8 BDF ----
    model = parse_bdf("tests/validation/ILC8/ilc8.bdf")
    print(f"[1] BDF parsed: {len(model.nodes)} nodes, "
          f"{len(model.elements)} elements  ({time.time()-t0:.1f}s)")

    # ---- 2. Aircraft Config ----
    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig, SpeedSchedule, WeightCGCondition,
        ControlSurfaceLimits, LandingGearConfig,
    )
    from nastaero.aero.dlm import compute_rigid_clalpha

    # CONM2 + 구조(요소) 질량 전부 럼핑 — model.masses(CONM2)만 합치면
    # 쉘 구조 질량 ~0.43 t이 빠져 로터 추력 목표가 과소 산정된다
    from nastaero.loads_analysis.trim_loads import compute_node_masses

    total_mass_kg = sum(compute_node_masses(model).values()) * 1000
    weight_N = total_mass_kg * 9.80665
    print(f"    Total mass: {total_mass_kg:.1f} kg | weight: {weight_N:.0f} N")

    wing_area_mm2 = 16.2e6
    clalpha_vlm = compute_rigid_clalpha(model, mach=80.0 / 340.3,
                                        ref_area=wing_area_mm2)
    print(f"    CLa (VLM, M=0.235): {clalpha_vlm:.3f} /rad")

    main_gear = _gear_nodes(model, 4700.0, n=2)
    nose_gear = _gear_nodes(model, 1500.0, n=1)
    print(f"    Gear nodes: main={main_gear}, nose={nose_gear}")

    config = AircraftConfig(
        speeds=SpeedSchedule(VS1=36.0, VA=71.0, VB=0.0, VC=80.0,
                             VD=100.0, VF=50.0),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW", weight_N=weight_N, cg_x=4450.0),
        ],
        altitudes_m=[0.0],
        wing_area_m2=16.2,
        CLalpha=clalpha_vlm,
        mean_chord_m=1.35,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0, rudder_max_deg=25.0, elevator_max_deg=25.0,
        ),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=main_gear,
            nose_gear_node_ids=nose_gear,
            main_gear_x=4700.0, nose_gear_x=1500.0,
            strut_efficiency=0.7, stroke=0.25, sink_rate_fps=10.0,
            tire_deflection=0.03, tire_efficiency=0.3,
        ),
        gust_Ude_VC_fps=50.0,
        gust_Ude_VD_fps=25.0,
    )

    # ---- 3. VTOL config (ILC-8: 8 lift + pusher) ----
    from nastaero.models.ilc8 import make_ilc8_vtol_config

    vtol_config = make_ilc8_vtol_config()
    config.vtol_config = vtol_config
    print(f"\n[2] VTOL Configuration: {vtol_config.config_type}")
    print(f"    Lift rotors: {vtol_config.n_lift_rotors} | "
          f"Cruise rotors: {len(vtol_config.cruise_rotors)} | "
          f"rotor mass: {vtol_config.total_rotor_mass_kg:.0f} kg")

    # ---- 4. BEMT hover sanity ----
    from nastaero.rotor.bemt_solver import BEMTSolver
    from nastaero.loads_analysis.case_generator import isa_atmosphere

    rho_sl, _, _ = isa_atmosphere(0.0)
    test_rotor = vtol_config.hover_rotors[0]
    target = weight_N / vtol_config.n_hover_rotors
    hv = BEMTSolver(test_rotor.blade, test_rotor.n_blades).solve_for_thrust(
        target, test_rotor.rpm_hover, rho_sl)
    print(f"\n[3] BEMT hover: T={hv.thrust:.0f} N/rotor (target {target:.0f}), "
          f"P={hv.power/745.7:.1f} hp, coll={np.degrees(hv.collective_rad):.1f} deg")

    # ---- 5. Load case matrices ----
    from nastaero.loads_analysis.certification.load_case_matrix import (
        LoadCaseMatrix,
    )
    from nastaero.loads_analysis.certification.vtol_load_case_matrix import (
        VTOLLoadCaseMatrix,
    )

    conv_matrix = LoadCaseMatrix(config)
    conv_matrix.generate_all()
    print(f"\n[4] Conventional load cases: {conv_matrix.total_cases}")
    for cat, count in sorted(conv_matrix.summary().items()):
        print(f"      {cat:15s}: {count:3d}")

    vtol_matrix = VTOLLoadCaseMatrix(vtol_config, config)
    vtol_cases = vtol_matrix.generate_all()
    print(f"\n[5] VTOL load cases: {len(vtol_cases)}")
    for cat, count in sorted(vtol_matrix.summary().items()):
        print(f"      {cat:15s}: {count:3d}")

    from types import SimpleNamespace
    conv_only = SimpleNamespace(
        flight_cases=list(conv_matrix.flight_cases),
        landing_cases=list(conv_matrix.landing_cases),
        dynamic_cases=[],
        config=config,
    )
    conv_matrix.merge_vtol_cases(vtol_cases)
    conv_matrix.to_csv(os.path.join(output_dir, "ilc8_case_matrix.csv"))
    print(f"\n[6] Combined matrix: {conv_matrix.total_cases} cases "
          f"-> {output_dir}/ilc8_case_matrix.csv")

    # ---- 6. Batch solve ----
    from nastaero.loads_analysis.certification.vtol_batch_runner import (
        VTOLBatchRunner,
    )

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    print(f"\n[7] Batch solver ({n_cpus} workers)...")
    t_solve = time.time()
    runner = VTOLBatchRunner(
        conv_only, vtol_matrix, bdf_model=model, vtol_config=vtol_config,
        n_workers=n_cpus, include_transient=False)
    batch_result = runner.run()
    solve_time = time.time() - t_solve
    print(f"    Completed in {solve_time:.1f}s | converged "
          f"{batch_result.n_converged}/{batch_result.n_total}")

    by_cat = {}
    for r in batch_result.case_results:
        d = by_cat.setdefault(r.category, [0, 0])
        d[0] += 1
        d[1] += 1 if r.converged else 0
    for cat, (tot, conv) in sorted(by_cat.items()):
        print(f"      {cat:18s}: {conv}/{tot} converged")

    # ---- 7. Rotor hub loads table ----
    import csv
    hub_rows = []
    for case in vtol_cases:
        if not case.rotor_forces:
            continue
        for rotor in vtol_config.rotors:
            fv = case.rotor_forces.get(rotor.hub_node_id)
            if fv is None:
                continue
            hub_rows.append({
                "rotor_label": rotor.label, "rotor_id": rotor.rotor_id,
                "condition": case.label, "category": case.category,
                "case_id": case.case_id,
                "Fx": fv[0], "Fy": fv[1], "Fz": fv[2],
                "Mx": fv[3], "My": fv[4], "Mz": fv[5],
            })
    if hub_rows:
        hub_csv = os.path.join(output_dir, "rotor_hub_loads.csv")
        with open(hub_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hub_rows[0].keys()))
            w.writeheader()
            w.writerows(hub_rows)
        fz_max = max(hub_rows, key=lambda r: abs(r["Fz"]))
        print(f"\n[8] Hub loads: {len(hub_rows)} entries -> {hub_csv}")
        print(f"    max |Fz| = {fz_max['Fz']:.0f} N "
              f"({fz_max['rotor_label']}, {fz_max['condition']})")

    # ---- 8. VMT ----
    # 컴포넌트는 생성기 노드 대역으로 수동 정의 — 기하 휴리스틱이
    # ILC-8의 원통 동체 상면을 VTP로 오분류하고, 붐/허브(로터 하중
    # 작용점)를 어느 컴포넌트에도 넣지 않아 날개 VMT에서 로터 하중이
    # 누락되기 때문. 붐+허브는 엔진 파일런 관례대로 날개에 귀속한다.
    from nastaero.loads_analysis.certification.vmt_bridge import (
        compute_vmt_for_batch,
    )
    from nastaero.loads_analysis.component_id import (
        identify_components_manual,
    )

    def _nids(lo, hi, extra=()):
        ids = [n for n in model.nodes if lo <= n <= hi]
        return ids + [n for n in extra if n in model.nodes]

    WING = dict(span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1)
    hubs_r = (990103, 990104, 990107, 990108)
    hubs_l = (990101, 990102, 990105, 990106)
    components = identify_components_manual(model, [
        dict(name="Right Wing", integration_sign=1.0, color="blue",
             node_ids=_nids(400000, 499999)
             + _nids(730000, 749999, hubs_r), **WING),
        dict(name="Left Wing", integration_sign=-1.0, color="dodgerblue",
             node_ids=_nids(300000, 399999)
             + _nids(710000, 729999, hubs_l), **WING),
        # V-tail (40° 상반각): y-스팬 투영으로 적분 — z-전단/굽힘이
        # 패널 수직하중의 cos(40°) 성분을 담는 공학적 근사
        dict(name="Right V-Tail", integration_sign=1.0, color="red",
             node_ids=_nids(600000, 699999), **WING),
        dict(name="Left V-Tail", integration_sign=-1.0, color="salmon",
             node_ids=_nids(500000, 599999), **WING),
        dict(name="Fuselage", integration_sign=-1.0, color="gray",
             node_ids=_nids(100000, 299999, (990201,)),
             span_axis=0, shear_axis=2, bending_axis=1, torsion_axis=0),
    ])
    # ---- 8b. (고장×재트림) 확장 사건 — 선형 패턴 전수 선별 후
    #      지배 패턴을 케이스로 채택 (P·C 판정) ----
    from nastaero.loads_analysis.certification.retrim_events import (
        RetrimScreen,
    )

    rs = RetrimScreen(model, vtol_config, config,
                      components=components, fuselage_cg_x=4450.0)
    retrim_events = rs.screen()
    retrim_cases = rs.realize(retrim_events, threshold_pct=1.0,
                              top_n=8)
    for rc in retrim_cases:
        batch_result.case_results.append(rc)
        batch_result.completed_ids.add(rc.case_id)
    print(f"\n[8b] (failure x re-trim) extension: "
          f"{len(retrim_events)} events / "
          f"{sum(e.n_patterns for e in retrim_events)} patterns "
          f"screened -> {len(retrim_cases)} adopted (P*C, C >= 1%)")
    for e in retrim_events[:3]:
        print(f"     top: {e.mode} {e.rotor_label} C={e.consequence*100:.1f}% "
              f"@{e.worst_component}")

    print(f"\n[9] VMT internal loads... (manual components: "
          f"{', '.join(c.name for c in components.components)})")
    t_vmt = time.time()
    vmt_data = compute_vmt_for_batch(model, batch_result,
                                     components=components,
                                     fuselage_cg_x=4450.0)
    print(f"    VMT for {len(vmt_data)} cases in {time.time()-t_vmt:.1f}s")
    comp_names = list(next(iter(vmt_data.values())).keys()) if vmt_data else []
    print(f"    Components: {', '.join(comp_names)}")

    # ---- 9. Envelope + critical cases + design-set selection ----
    # select_critical_design_loads가 실현가능/추진계 한계 분리 정책
    # (r3 MC2)까지 처리한다. 포화 지령 케이스는 실현가능 포락선
    # 산정에서 제외되고, 그 포락선을 초과하는 경우에만 플래그된
    # 추진계 한계 설계 케이스로 별도 편입된다.
    from nastaero.loads_analysis.certification.envelope import (
        select_critical_design_loads,
    )

    force_dir = os.path.join(output_dir, "force_cards")
    sel = select_critical_design_loads(
        model, batch_result, output_dir=force_dir,
        fuselage_cg_x=4450.0, infeasible_policy="separate",
        components=components, vmt_data=vmt_data,
    )
    proc = sel["processor"]
    all_critical = proc.get_critical_cases()
    pl = sel["propulsion_limit"]
    print(f"    Design set: {sel['n_design_cases']} cases "
          f"({sel['compression']:.1f}:1), critical records "
          f"{sel['n_critical']}")
    print(f"    Propulsion-limit screening: {pl['n_infeasible']} "
          f"saturated case(s), {len(pl['exceedances'])} envelope "
          f"exceedance(s), {pl['n_appended_design_cases']} appended")
    n_plim_design = sum(1 for d in sel["design_cases"]
                        if not d.rotor_command_feasible)
    print(f"    Saturated cases in design set: {n_plim_design}")

    # 임계 케이스 표 (요약)
    print(f"    {'Component':14s} {'Qty':8s} {'Ext':6s} "
          f"{'Value':>14s} {'Station':>9s} {'Case':>6s} {'Category':16s}")
    for cc in all_critical:
        print(f"    {cc.component[:14]:14s} {cc.quantity[:8]:8s} "
              f"{cc.extreme[:6]:6s} {cc.value:14,.0f} {cc.station:9.1f} "
              f"{cc.case_id:6d} {cc.category[:16]:16s}")

    # ---- 10. Plots (envelope + potato + critical frequency) ----
    from nastaero.visualization.cert_plot import (
        plot_vmt_envelope, plot_potato, plot_critical_frequency,
    )
    from nastaero.loads_analysis.certification.monitoring_stations import (
        identify_monitoring_stations,
    )

    stations = identify_monitoring_stations(
        model, config=config, components=components,
        mass_threshold_kg=5.0, offset_mm=50.0, vtol_config=vtol_config)
    n_plots = 0
    for comp in comp_names:
        env = proc.get_envelope(comp)
        if not env:
            continue
        safe = comp.replace(" ", "_").lower()
        cg = 4450.0 if "fuselage" in comp.lower() else None
        plot_vmt_envelope(env, output_path=os.path.join(
            output_dir, f"03_vmt_envelope_{safe}.png"),
            timestamp=timestamp_label, cg_x=cg)
        n_plots += 1
        for idx, ms in enumerate(stations.get(comp, [])):
            potato = proc.compute_potato(comp, station=ms.position)
            if potato and potato.n_points >= 3:
                pdir = os.path.join(output_dir, f"potato_{safe}")
                os.makedirs(pdir, exist_ok=True)
                lab = ms.label.replace(" ", "_").replace("/", "_")
                plot_potato(potato, output_path=os.path.join(
                    pdir, f"04_potato_{safe}_{idx:02d}_{lab}.png"),
                    timestamp=timestamp_label)
                n_plots += 1
    freq = proc.critical_case_frequency()
    if freq:
        plot_critical_frequency(freq, batch_result=batch_result,
            output_path=os.path.join(output_dir, "05_critical_frequency.png"),
            timestamp=timestamp_label)
        n_plots += 1
    print(f"\n[11] Plots saved: {n_plots} -> {output_dir}/")

    # ---- 11. FORCE card export (9단계 선정에서 함께 수행) ----
    exp = sel["export"]
    print(f"\n[12] FORCE export: {exp['n_cases']} cases, "
          f"{exp['n_force_cards']:,} FORCE / {exp['n_moment_cards']:,} MOMENT")
    print(f"    Master BDF: {exp['master_bdf']}")
    print(f"    Design summary CSV: {sel.get('design_summary_csv')}")

    # ---- 12. Summary ----
    print(f"\n{'='*70}")
    print(f"ILC-8 Certification Loads Analysis Complete")
    print(f"  Cases: {conv_matrix.total_cases} | converged "
          f"{batch_result.n_converged}/{batch_result.n_total} | "
          f"critical {len(all_critical)}")
    print(f"  Wall time: {time.time()-t0:.1f}s | output: {output_dir}/")
    print(f"{'='*70}")

    # 후속 분석 스크립트(r3 응력 보존·국부 6분력 비교)가 같은 실행
    # 결과를 재사용할 수 있게 핵심 객체를 반환한다.
    return {
        "model": model, "config": config, "vtol_config": vtol_config,
        "batch_result": batch_result, "components": components,
        "vmt_data": vmt_data, "selection": sel, "processor": proc,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main()
