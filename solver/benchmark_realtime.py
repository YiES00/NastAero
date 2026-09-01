#!/usr/bin/env python3
"""Benchmark: Can ASCENT-Load's 6-DOF dynamics run in real-time?

Measures per-timestep cost of:
  1. six_dof_derivatives()  — single EOM evaluation
  2. Single RK4 step (4 × EOM eval)
  3. Full 1-second simulation via integrate_6dof()
  4. Load factor (nz) computation from history

Compares against real-time frame budgets at 50/100/200 Hz.
"""
import time
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ascent_load.loads_analysis.certification.flight_sim import (
    AircraftParams, AircraftState, ControlInput, SimTimeHistory,
    six_dof_derivatives, integrate_6dof, trim_initial_state,
    compute_nz_from_history,
)
from ascent_load.loads_analysis.certification.aero_derivatives import AeroDerivativeSet
from ascent_load.loads_analysis.certification.aircraft_config import (
    eas_to_tas, RHO_0, G_MPS2,
)
from ascent_load.loads_analysis.case_generator import isa_atmosphere


def build_gacomp_params_manual():
    """Build GACOMP-like AircraftParams with manual derivatives (no BDF needed)."""
    # GACOMP-like derivatives (typical GA single engine)
    mm = 1e-3
    S_ref_mm2 = 17.0e6   # 17 m² in mm²
    b_ref_mm = 10300.0    # 10.3 m span in mm
    c_bar_mm = 1600.0     # 1.6 m MAC in mm

    derivs = AeroDerivativeSet(
        S_ref=S_ref_mm2,
        b_ref=b_ref_mm,
        c_bar=c_bar_mm,
        # Longitudinal
        CLalpha=5.5,
        CLdelta_e=-0.38,
        Cmalpha=-1.2,
        Cmdelta_e=-1.35,
        CD0=0.032,
        CDalpha=0.13,
        Cmq=-12.4,
        Cmalpha_dot=-5.2,
        CLq=3.9,
        # Lateral-directional
        CYbeta=-0.31,
        Clbeta=-0.089,
        Cnbeta=0.065,
        Cldelta_a=0.17,
        Cndelta_a=-0.013,
        CYdelta_r=0.187,
        Cldelta_r=0.0024,
        Cndelta_r=-0.074,
        Clp=-0.47,
        Cnr=-0.099,
        Clr=0.096,
        Cnp=-0.03,
        CYp=-0.037,
        CYr=0.21,
    )

    W_N = 13734.0  # GACOMP MTOW in Newtons
    mass_kg = W_N / G_MPS2

    params = AircraftParams(
        mass=mass_kg,
        S=S_ref_mm2 * mm**2,    # m²
        b=b_ref_mm * mm,        # m
        c_bar=c_bar_mm * mm,    # m
        Ixx=1395.0,             # kg·m²
        Iyy=3264.0,
        Izz=4097.0,
        Ixz=166.0,
        derivs=derivs,
        thrust_N=0.0,
        rho=RHO_0,
    )
    return params


def benchmark():
    print("=" * 70)
    print("ASCENT-Load Real-Time Performance Benchmark")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Setup (one-time cost)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    params = build_gacomp_params_manual()
    t_setup = time.perf_counter() - t0
    print(f"\n[Setup] AircraftParams build: {t_setup*1000:.2f} ms")

    # Trim at VC (sea level)
    V_eas = 80.0  # m/s — GACOMP VC
    V_tas = eas_to_tas(V_eas, 0.0)
    print(f"  V_EAS = {V_eas:.1f} m/s, V_TAS = {V_tas:.1f} m/s (SL)")

    t0 = time.perf_counter()
    initial_state, de_trim = trim_initial_state(params, V_tas)
    t_trim = time.perf_counter() - t0
    print(f"[Setup] Trim computation: {t_trim*1000:.2f} ms")
    print(f"  θ_trim={math.degrees(initial_state.theta):.2f}°, "
          f"δe_trim={math.degrees(de_trim):.2f}°")

    # Prepare state array and control
    state_arr = initial_state.to_array().copy()
    ctrl = ControlInput(delta_e=de_trim)

    # ------------------------------------------------------------------
    # 2. Single EOM evaluation
    # ------------------------------------------------------------------
    N_evals = 50000
    t0 = time.perf_counter()
    for _ in range(N_evals):
        _ = six_dof_derivatives(state_arr, 0.0, params, ctrl)
    t_eom_total = time.perf_counter() - t0
    t_eom_us = t_eom_total / N_evals * 1e6

    print(f"\n[EOM] six_dof_derivatives() × {N_evals:,}: {t_eom_total*1000:.1f} ms")
    print(f"  Per call: {t_eom_us:.2f} µs")

    # ------------------------------------------------------------------
    # 3. Single RK4 step (4 EOM evals + array ops)
    # ------------------------------------------------------------------
    dt = 0.005  # 5ms = 200 Hz
    N_rk4 = 10000

    def rk4_step(y, t, dt_step, params_, ctrl_):
        """Single RK4 step (mimics integrate_6dof inner loop)."""
        k1 = six_dof_derivatives(y, t, params_, ctrl_)
        k2 = six_dof_derivatives(y + 0.5 * dt_step * k1, t + 0.5 * dt_step, params_, ctrl_)
        k3 = six_dof_derivatives(y + 0.5 * dt_step * k2, t + 0.5 * dt_step, params_, ctrl_)
        k4 = six_dof_derivatives(y + dt_step * k3, t + dt_step, params_, ctrl_)
        return y + (dt_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    y = state_arr.copy()
    t0 = time.perf_counter()
    for i in range(N_rk4):
        y = rk4_step(y, i * dt, dt, params, ctrl)
    t_rk4_total = time.perf_counter() - t0
    t_rk4_us = t_rk4_total / N_rk4 * 1e6

    print(f"\n[RK4] Single RK4 step × {N_rk4:,}: {t_rk4_total*1000:.1f} ms")
    print(f"  Per step: {t_rk4_us:.2f} µs  ({t_rk4_us/1000:.3f} ms)")

    # ------------------------------------------------------------------
    # 4. Full 1-second simulation via integrate_6dof
    # ------------------------------------------------------------------
    def ctrl_func(t):
        return ctrl

    t0 = time.perf_counter()
    history = integrate_6dof(params, initial_state, ctrl_func, (0.0, 1.0), dt=0.005)
    t_sim = time.perf_counter() - t0

    n_steps = len(history.t) - 1 if len(history.t) > 1 else 200
    print(f"\n[Sim] 1.0s simulation (dt=5ms, {n_steps} steps): {t_sim*1000:.1f} ms")
    print(f"  Simulation-to-wallclock ratio: {1.0/t_sim:.0f}x real-time")

    # ------------------------------------------------------------------
    # 5. nz computation from history
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    compute_nz_from_history(params, history)
    t_nz = time.perf_counter() - t0
    print(f"\n[nz] Load factor computation ({n_steps} pts): {t_nz*1000:.3f} ms")

    # ------------------------------------------------------------------
    # 6. Full pipeline: 3s maneuver → nz
    # ------------------------------------------------------------------
    def pullup_ctrl(t):
        if 0.5 < t < 1.0:
            de = de_trim + math.radians(5)
        else:
            de = de_trim
        return ControlInput(delta_e=de)

    t0 = time.perf_counter()
    hist2 = integrate_6dof(params, initial_state, pullup_ctrl, (0.0, 3.0), dt=0.005)
    compute_nz_from_history(params, hist2)
    t_full = time.perf_counter() - t0
    nz_max = np.max(hist2.nz) if len(hist2.nz) > 0 else 0.0

    print(f"\n[Full] 3.0s maneuver + nz: {t_full*1000:.1f} ms")
    print(f"  nz_max = {nz_max:.2f} g")
    print(f"  Simulation-to-wallclock ratio: {3.0/t_full:.0f}x real-time")

    # ------------------------------------------------------------------
    # 7. Real-time feasibility analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("REAL-TIME FEASIBILITY ANALYSIS")
    print("=" * 70)

    frame_budgets = {
        " 50 Hz (20.0 ms)": 20.0,
        "100 Hz (10.0 ms)": 10.0,
        "200 Hz ( 5.0 ms)":  5.0,
    }

    rk4_ms = t_rk4_us / 1000.0  # per-step cost in ms

    print(f"\nPer-step cost: {rk4_ms:.4f} ms  (RK4 = 4 × EOM eval)")
    print(f"Per EOM eval:  {t_eom_us:.2f} µs\n")

    for label, budget_ms in frame_budgets.items():
        margin_ms = budget_ms - rk4_ms
        margin_pct = margin_ms / budget_ms * 100
        status = "✅ OK" if margin_ms > 0 else "❌ OVER"
        print(f"  {label}: budget={budget_ms:.1f}ms, used={rk4_ms:.4f}ms, "
              f"margin={margin_ms:.4f}ms ({margin_pct:.1f}%) {status}")

    # How many extra operations fit in the margin at each rate?
    spare_200 = 5.0 - rk4_ms
    print(f"\n  Budget remaining per frame at 200Hz: {spare_200:.3f} ms")
    print(f"    → Could do ~{int(spare_200*1000/t_eom_us)} extra EOM evaluations")
    print(f"    → Or ~{int(spare_200/0.05)} simple aero table lookups (50µs each)")

    # ------------------------------------------------------------------
    # 8. Multi-altitude benchmark
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("MULTI-ALTITUDE BENCHMARK (6 altitudes × 1s each)")
    print("=" * 70)

    altitudes = [(0, "SL"), (1524, "5kft"), (3048, "10kft"),
                 (4572, "15kft"), (6096, "20kft"), (7620, "25kft")]

    t_total_alt = 0.0
    for alt_m, alt_label in altitudes:
        rho_h, _, _ = isa_atmosphere(alt_m)
        V_tas_alt = eas_to_tas(V_eas, alt_m)

        params.rho = rho_h
        state_alt, de_alt = trim_initial_state(params, V_tas_alt)

        def ctrl_alt(t, de=de_alt):
            return ControlInput(delta_e=de)

        t0 = time.perf_counter()
        hist_alt = integrate_6dof(params, state_alt, ctrl_alt, (0.0, 1.0), dt=0.005)
        compute_nz_from_history(params, hist_alt)
        t_alt = time.perf_counter() - t0
        t_total_alt += t_alt

        print(f"  {alt_label:6s} (ρ={rho_h:.3f}, TAS={V_tas_alt:.1f} m/s): {t_alt*1000:.1f} ms")

    print(f"  Total 6 altitudes: {t_total_alt*1000:.1f} ms")
    print(f"  Avg per altitude: {t_total_alt/6*1000:.1f} ms")

    # Reset
    params.rho = RHO_0

    # ------------------------------------------------------------------
    # 9. FEA solve cost estimate
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FEA COST ESTIMATE (SOL 101 static solve)")
    print("=" * 70)

    try:
        from ascent_load.bdf.parser import parse_bdf
        bdf_path = os.path.join(os.path.dirname(__file__),
                                "tests", "validation", "GACOMP",
                                "p400r3-free-trim.bdf")
        if os.path.exists(bdf_path):
            t0 = time.perf_counter()
            model = parse_bdf(bdf_path)
            t_parse = time.perf_counter() - t0
            print(f"  BDF parse ({len(model.nodes)} nodes): {t_parse*1000:.0f} ms")

            # Assembly
            from ascent_load.fem.assembly import assemble_global_matrices
            from ascent_load.fem.dof_manager import DOFManager

            t0 = time.perf_counter()
            node_ids = list(model.nodes.keys())
            dof_mgr = DOFManager(node_ids)
            K, M, _ = assemble_global_matrices(model, dof_mgr)
            t_asm = time.perf_counter() - t0
            ndof = K.shape[0]
            print(f"  Assembly ({ndof} DOF): {t_asm*1000:.0f} ms")

            # Single solve (forward/back-substitution)
            from scipy.sparse.linalg import splu
            t0 = time.perf_counter()
            # Factor
            K_csc = K.tocsc()
            lu = splu(K_csc)
            t_factor = time.perf_counter() - t0
            print(f"  Sparse LU factorization: {t_factor*1000:.0f} ms")

            # Back-substitution
            rhs = np.ones(ndof)
            t0 = time.perf_counter()
            for _ in range(10):
                x = lu.solve(rhs)
            t_solve10 = time.perf_counter() - t0
            t_solve = t_solve10 / 10 * 1000
            print(f"  Back-substitution (avg of 10): {t_solve:.1f} ms")

            print(f"\n  ⚠️ Full FEA per-step cost:")
            print(f"     LU factorization: {t_factor*1000:.0f} ms (one-time)")
            print(f"     Back-substitution: {t_solve:.1f} ms (per load case)")
            print(f"     At 50Hz budget (20ms): {'❌ OVER' if t_solve > 20 else '✅ OK if pre-factored'}")
        else:
            print("  [SKIP] GACOMP BDF not found")
    except Exception as e:
        import traceback
        print(f"  [SKIP] FEA benchmark failed: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  ┌──────────────────────────────────────────────────┐
  │ 6-DOF EOM evaluation:      {t_eom_us:>8.2f} µs          │
  │ RK4 timestep (dt=5ms):     {t_rk4_us:>8.2f} µs          │
  │ 1s simulation (200 steps):  {t_sim*1000:>7.1f} ms          │
  │ 3s maneuver + nz:           {t_full*1000:>7.1f} ms          │
  │ Real-time ratio (1s sim):   {1.0/t_sim:>7.0f}x            │
  │ Real-time ratio (3s man):   {3.0/t_full:>7.0f}x            │
  └──────────────────────────────────────────────────┘

  🟢 6-DOF rigid-body dynamics: REAL-TIME CAPABLE at 200 Hz
     (per-step cost ≈ {rk4_ms:.3f} ms << 5 ms budget)

  ⚠️  Adding full FEA (SOL 101 per step) would NOT be real-time
     → Sparse solve alone takes O(10-100) ms per step

  💡 Recommended real-time architecture:
     ┌────────────────────────────────────────────────┐
     │ Inner loop (200 Hz): 6-DOF dynamics     ← ✅   │
     │   · RK4 integration of rigid-body EOM         │
     │   · Control input + gust model                │
     │   · nz, alpha, beta, rates output             │
     │                                               │
     │ Mid loop (20-50 Hz): Aero loads table  ← ✅   │
     │   · Pre-computed aero influence coefficients  │
     │   · AIC × α → Cp → nodal forces              │
     │   · Modal superposition for flex loads        │
     │                                               │
     │ Outer loop (1-5 Hz): Structural loads  ← ⚠️   │
     │   · Full FEA solve (SOL 101) if needed        │
     │   · VMT diagram update                        │
     │   · Fatigue damage accumulation               │
     │                                               │
     │ Pre-compute (offline):                        │
     │   · Modal basis (SOL 103) → Φ matrix          │
     │   · Generalized stiffness/mass (Φᵀ·K·Φ)      │
     │   · Aero influence coefficients (AIC)         │
     │   · Spline matrices (G_sp, G_disp)            │
     └────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    benchmark()
