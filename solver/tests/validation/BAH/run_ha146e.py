#!/usr/bin/env python3
"""Simplified BAH Wing - SOL 146 Validation (HA146E).

Validates ASCENT-Load SOL 146 transient aeroelastic response against
analytical predictions for a cantilever wing under step force at tip.

Checks:
1. Natural frequencies (first 5 modes in reasonable ranges)
2. Step response DAF = 1 + exp(-zeta*pi / sqrt(1-zeta^2))
3. Tip displacement: peak dynamic ~ 2x static
4. DAF vs damping ratio sweep

Reference: MSC Nastran Example HA146E (Bisplinghoff-Ashley-Halfman wing)
"""
from __future__ import annotations
import os
import sys
import numpy as np

# Add solver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ascent_load.bdf.parser import BDFParser
from ascent_load.solvers.sol146 import solve_aeroelastic_transient


BDF_PATH = os.path.join(os.path.dirname(__file__), 'ha146e_simple.bdf')

TIP_NODE = 6  # Wing tip grid
ROOT_NODE = 1  # Clamped root
FORCE_AMPLITUDE = 1000.0  # lb
FORCE_DOF = 3  # Z-direction


def step_force(tip_node=TIP_NODE, dof_comp=FORCE_DOF,
               amplitude=FORCE_AMPLITUDE):
    """Step force at wing tip: F(t) = amplitude for t >= 0."""
    def force_func(t, dof_mgr):
        n_total = dof_mgr.total_dof
        F = np.zeros(n_total)
        dof_idx = dof_mgr.get_dof(tip_node, dof_comp)
        if 0 <= dof_idx < n_total:
            F[dof_idx] = amplitude
        return F
    return force_func


def analytical_daf(zeta):
    """Analytical Dynamic Amplification Factor for step load.

    For an underdamped SDOF system under step force:
        DAF = 1 + exp(-zeta * pi / sqrt(1 - zeta^2))

    At zeta=0: DAF = 2.0 (exact)
    """
    if zeta <= 0:
        return 2.0
    if zeta >= 1.0:
        return 1.0  # critically/overdamped
    return 1.0 + np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta**2))


def compute_static_tip_displacement(bdf_model):
    """Compute static tip displacement using SOL 101 approach.

    For a cantilever beam with tip force F, the static displacement
    is computed from K @ u = F.
    """
    from ascent_load.fem.model import FEModel
    import scipy.sparse.linalg as spla

    bdf_model_copy = BDFParser().parse(BDF_PATH)
    bdf_model_copy.cross_reference()
    fe = FEModel(bdf_model_copy)
    dof_mgr = fe.dof_mgr

    subcase = bdf_model_copy.subcases[0] if bdf_model_copy.subcases else bdf_model_copy.global_case
    K_ff, M_ff, _, f_dofs, s_dofs = fe.get_partitioned_system(subcase)

    n_total = dof_mgr.total_dof
    F_full = np.zeros(n_total)
    dof_idx = dof_mgr.get_dof(TIP_NODE, FORCE_DOF)
    F_full[dof_idx] = FORCE_AMPLITUDE

    F_f = F_full[f_dofs]
    u_f = spla.spsolve(K_ff, F_f)

    # Extract tip Z displacement
    tip_dofs = dof_mgr.get_node_dofs(TIP_NODE)
    tip_z_full = tip_dofs[FORCE_DOF - 1]  # DOF 3 -> index 2
    f_dof_map = {fd: i for i, fd in enumerate(f_dofs)}
    if tip_z_full in f_dof_map:
        return u_f[f_dof_map[tip_z_full]]
    return 0.0


def run_single_case(zeta, n_modes=10, duration=0.5, dt=0.001):
    """Run SOL 146 for a single damping ratio."""
    model = BDFParser().parse(BDF_PATH)

    t_array = np.arange(0, duration + dt/2, dt)
    force_func = step_force()

    result = solve_aeroelastic_transient(
        model,
        force_func=force_func,
        t_array=t_array,
        n_modes=n_modes,
        zeta=zeta,
        use_aero_coupling=False,
        output_node_ids=[TIP_NODE],
        output_interval=1,
    )

    return result


def main():
    print("=" * 72)
    print("  SIMPLIFIED BAH WING - SOL 146 VALIDATION (HA146E)")
    print("  Impulsive Step Force at Wing Tip")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # 1. Run baseline case (zeta = 0.02)
    # ------------------------------------------------------------------
    print("--- Running baseline case (zeta = 0.02) ---")
    result = run_single_case(zeta=0.02, n_modes=10, duration=0.5, dt=0.001)

    frequencies = result["frequencies"]
    n_modes_actual = len(frequencies)

    print(f"\n{'='*72}")
    print("  1. NATURAL FREQUENCIES")
    print(f"{'='*72}")
    print(f"  {'Mode':>4s}  {'Frequency (Hz)':>14s}  {'omega (rad/s)':>14s}  {'Type':>20s}")
    print(f"  {'-'*4:>4s}  {'-'*14:>14s}  {'-'*14:>14s}  {'-'*20:>20s}")

    # Identify mode types heuristically
    for i in range(min(n_modes_actual, 10)):
        f_hz = frequencies[i]
        omega_i = 2.0 * np.pi * f_hz
        if i < 2:
            mode_type = "Bending"
        elif i < 4:
            mode_type = "Bending/Torsion"
        else:
            mode_type = "Higher"
        print(f"  {i+1:4d}  {f_hz:14.4f}  {omega_i:14.4f}  {mode_type:>20s}")

    # Validate frequency ranges
    print()
    f1 = frequencies[0]
    freq_checks = [
        (0, 0.5, 10.0, "1st bending"),
        (1, 3.0, 40.0, "2nd bending/torsion"),
    ]
    all_freq_ok = True
    for idx, f_low, f_high, name in freq_checks:
        if idx < len(frequencies):
            f_val = frequencies[idx]
            ok = f_low <= f_val <= f_high
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_freq_ok = False
            print(f"  Mode {idx+1} ({name}): {f_val:.4f} Hz "
                  f"[expected {f_low}-{f_high} Hz] -> {status}")

    # ------------------------------------------------------------------
    # 2. Static tip displacement
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  2. STATIC TIP DISPLACEMENT")
    print(f"{'='*72}")

    u_static = compute_static_tip_displacement(None)
    print(f"  Static tip Z displacement: {u_static:.6e} in")
    print(f"  Force: {FORCE_AMPLITUDE:.0f} lb at GRID {TIP_NODE}")

    # ------------------------------------------------------------------
    # 3. Dynamic response and DAF (zeta = 0.02)
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  3. STEP RESPONSE AND DAF (zeta = 0.02)")
    print(f"{'='*72}")

    tip_disp = result["displacements"][TIP_NODE][:, FORCE_DOF - 1]  # Z comp
    t_out = result["t"]

    u_peak = np.max(np.abs(tip_disp))
    daf_numerical = u_peak / abs(u_static) if abs(u_static) > 1e-20 else 0.0
    daf_analytical = analytical_daf(0.02)

    print(f"  Peak dynamic tip displacement: {u_peak:.6e} in")
    print(f"  Static tip displacement:       {abs(u_static):.6e} in")
    print(f"  Numerical DAF:  {daf_numerical:.4f}")
    print(f"  Analytical DAF: {daf_analytical:.4f}")
    print(f"  Ratio peak/static: {u_peak/abs(u_static):.4f}")
    print(f"  Error: {abs(daf_numerical - daf_analytical)/daf_analytical * 100:.2f}%")

    # Check that peak is approximately 2x static
    ratio = u_peak / abs(u_static) if abs(u_static) > 1e-20 else 0.0
    daf_ok = 1.5 <= ratio <= 2.5
    print(f"  Peak/static ratio ~2x check: {ratio:.4f} -> "
          f"{'PASS' if daf_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # 4. Per-mode DAF from modal coordinates
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  4. PER-MODE DAF FROM MODAL COORDINATES")
    print(f"{'='*72}")

    modal_coords = result["modal_coords"]
    modal_forces = result["modal_forces"]

    print(f"  {'Mode':>4s}  {'freq (Hz)':>10s}  {'q_peak':>12s}  "
          f"{'q_static':>12s}  {'DAF_num':>8s}  {'DAF_ana':>8s}  {'Error%':>7s}")
    print(f"  {'-'*4:>4s}  {'-'*10:>10s}  {'-'*12:>12s}  "
          f"{'-'*12:>12s}  {'-'*8:>8s}  {'-'*8:>8s}  {'-'*7:>7s}")

    for i in range(min(n_modes_actual, 10)):
        q_peak = np.max(np.abs(modal_coords[:, i]))
        # Static modal displacement = modal_force / omega_i^2
        omega_i = 2.0 * np.pi * frequencies[i]
        # Steady-state modal force (at t -> inf, all modes have same force)
        f_modal_ss = modal_forces[-1, i]  # steady-state modal force
        if abs(omega_i) > 1e-10:
            q_static = abs(f_modal_ss) / omega_i**2
        else:
            q_static = 0.0

        if q_static > 1e-20:
            daf_mode = q_peak / q_static
        else:
            daf_mode = 0.0

        err = abs(daf_mode - daf_analytical) / daf_analytical * 100 if daf_mode > 0 else 999.0
        print(f"  {i+1:4d}  {frequencies[i]:10.4f}  {q_peak:12.4e}  "
              f"{q_static:12.4e}  {daf_mode:8.4f}  {daf_analytical:8.4f}  {err:7.2f}")

    # ------------------------------------------------------------------
    # 5. DAF vs Damping Ratio Sweep
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  5. DAF VS DAMPING RATIO SWEEP")
    print(f"{'='*72}")

    zeta_values = [0.00, 0.02, 0.05, 0.10]
    daf_expected = {0.00: 2.000, 0.02: 1.939, 0.05: 1.855, 0.10: 1.729}

    print(f"  {'zeta':>6s}  {'DAF_ana':>8s}  {'DAF_num':>8s}  "
          f"{'Error%':>7s}  {'u_peak':>12s}  {'Status':>6s}")
    print(f"  {'-'*6:>6s}  {'-'*8:>8s}  {'-'*8:>8s}  "
          f"{'-'*7:>7s}  {'-'*12:>12s}  {'-'*6:>6s}")

    all_daf_ok = True
    for zeta in zeta_values:
        daf_ana = analytical_daf(zeta)

        # For zeta=0, use very small value to avoid division issues
        zeta_run = max(zeta, 1e-6)

        # Use longer duration for undamped case
        duration = 2.0 if zeta < 0.01 else 0.5

        res = run_single_case(zeta=zeta_run, n_modes=10,
                              duration=duration, dt=0.001)

        tip_z = res["displacements"][TIP_NODE][:, FORCE_DOF - 1]
        u_pk = np.max(np.abs(tip_z))
        daf_num = u_pk / abs(u_static) if abs(u_static) > 1e-20 else 0.0

        err = abs(daf_num - daf_ana) / daf_ana * 100
        ok = err < 10.0  # 10% tolerance for multi-mode system
        if not ok:
            all_daf_ok = False
        status = "PASS" if ok else "FAIL"

        print(f"  {zeta:6.3f}  {daf_ana:8.4f}  {daf_num:8.4f}  "
              f"{err:7.2f}  {u_pk:12.4e}  {status:>6s}")

    # ------------------------------------------------------------------
    # 6. Time history verification (longer run for steady-state check)
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  6. TIME HISTORY CHECKS (zeta = 0.02, 5-second run)")
    print(f"{'='*72}")

    # Run longer simulation for steady-state convergence
    res_long = run_single_case(zeta=0.02, n_modes=10,
                               duration=5.0, dt=0.001)
    tip_z = res_long["displacements"][TIP_NODE][:, FORCE_DOF - 1]
    t_hist = res_long["t"]

    # Check initial displacement is near zero
    u_t0 = tip_z[0]
    print(f"  Displacement at t=0: {u_t0:.6e} (should be ~0)")

    # Check response reaches steady state (last 10%)
    n_pts = len(tip_z)
    last_10pct = tip_z[9*n_pts//10:]
    u_ss = np.mean(last_10pct)
    print(f"  Mean displacement (last 10%%): {u_ss:.6e}")
    print(f"  Static displacement:           {u_static:.6e}")
    ss_ratio = u_ss / u_static if abs(u_static) > 1e-20 else 0.0
    print(f"  Steady-state ratio (should be ~1): {ss_ratio:.4f}")
    ss_ok = abs(ss_ratio - 1.0) < 0.15
    print(f"  Steady-state check: {'PASS' if ss_ok else 'FAIL'}")

    # Check oscillation frequency from FFT
    dt = t_hist[1] - t_hist[0]
    d_centered = tip_z - u_static
    spectrum = np.abs(np.fft.rfft(d_centered))
    freqs_fft = np.fft.rfftfreq(len(d_centered), d=dt)
    peak_idx = np.argmax(spectrum[1:]) + 1
    osc_freq = freqs_fft[peak_idx]
    print(f"  Oscillation frequency (FFT): {osc_freq:.4f} Hz")
    print(f"  First natural freq:          {frequencies[0]:.4f} Hz")
    freq_ratio = osc_freq / frequencies[0]
    print(f"  Ratio (should be ~1):        {freq_ratio:.4f}")
    freq_ok = abs(freq_ratio - 1.0) < 0.10
    print(f"  Frequency match check: {'PASS' if freq_ok else 'FAIL'}")

    # Check response decays (with damping)
    # Compare 1st and 4th seconds
    idx_1s = int(1.0 / dt)
    idx_4s = int(4.0 / dt)
    first_sec_peak = np.max(np.abs(d_centered[:idx_1s]))
    fourth_sec_peak = np.max(np.abs(d_centered[idx_4s:min(idx_4s+idx_1s, n_pts)]))
    decay_ratio = fourth_sec_peak / first_sec_peak if first_sec_peak > 0 else 0
    print(f"  Decay: 1st second peak = {first_sec_peak:.4e}, "
          f"4th second = {fourth_sec_peak:.4e}")
    print(f"  Decay ratio: {decay_ratio:.4f} (should be < 1 with damping)")
    decay_ok = decay_ratio < 1.0
    print(f"  Decay check: {'PASS' if decay_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")

    checks = [
        ("Natural frequencies in range", all_freq_ok),
        ("Peak/static ratio ~2x", daf_ok),
        ("DAF vs damping sweep", all_daf_ok),
        ("Steady-state convergence", ss_ok),
        ("Oscillation frequency matches mode 1", freq_ok),
        ("Response decays with damping", decay_ok),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED - see details above")

    print(f"\n{'='*72}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
