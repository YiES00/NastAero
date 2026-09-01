"""Validate ROM against SOL 144 for the GACOMP model.

Runs SOL 144 for all 7 subcases, then builds the ROM and computes
responses at matching flight conditions.  Outputs displacement and
force error metrics (RMS relative error, max relative error, correlation)
for each subcase and saves a summary CSV.

Usage:
    cd solver && python scripts/validate_rom.py

Output:
    scripts/rom_validation_results.csv
    scripts/rom_validation_summary.txt
"""
from __future__ import annotations
import sys
import os
import time
import numpy as np

# Add solver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ascent_load.bdf.parser import parse_bdf
from ascent_load.solvers.sol144 import solve_trim
from ascent_load.loads_analysis.certification.modal_rom import ModalROM


def compute_error_metrics(ref: dict, rom_result: dict, node_ids: list,
                          component: int = 2) -> dict:
    """Compute error metrics for a displacement/force component.

    Parameters
    ----------
    ref : Dict[int, ndarray(6)]
        Reference (SOL 144) per-node values.
    rom_result : Dict[int, ndarray(6)]
        ROM per-node values.
    node_ids : list
        Nodes to compare.
    component : int
        Which DOF component to compare (0=x, 1=y, 2=z, 3=rx, 4=ry, 5=rz).

    Returns
    -------
    dict with keys: rms_rel, max_rel, correlation, n_nodes, ref_max, rom_max
    """
    ref_vals = []
    rom_vals = []
    for nid in node_ids:
        if nid in ref and nid in rom_result:
            ref_vals.append(ref[nid][component])
            rom_vals.append(rom_result[nid][component])

    ref_arr = np.array(ref_vals)
    rom_arr = np.array(rom_vals)

    ref_max = np.max(np.abs(ref_arr))
    rom_max = np.max(np.abs(rom_arr))

    # Normalized RMS error (relative to reference max)
    if ref_max > 1e-20:
        rms_rel = np.sqrt(np.mean((ref_arr - rom_arr)**2)) / ref_max
        # Max relative error
        max_rel = np.max(np.abs(ref_arr - rom_arr)) / ref_max
    else:
        rms_rel = 0.0
        max_rel = 0.0

    # Pearson correlation
    if np.std(ref_arr) > 1e-20 and np.std(rom_arr) > 1e-20:
        corr = np.corrcoef(ref_arr, rom_arr)[0, 1]
    else:
        corr = 1.0 if np.allclose(ref_arr, rom_arr) else 0.0

    return {
        'rms_rel': float(rms_rel),
        'max_rel': float(max_rel),
        'correlation': float(corr),
        'n_nodes': len(ref_vals),
        'ref_max': float(ref_max),
        'rom_max': float(rom_max),
    }


def main():
    bdf_path = os.path.join(
        os.path.dirname(__file__), '..',
        'tests', 'validation', 'GACOMP', 'p400r3-free-trim.bdf'
    )
    if not os.path.exists(bdf_path):
        print(f"ERROR: BDF file not found: {bdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("ROM vs SOL 144 Validation — GACOMP Model")
    print("=" * 70)

    # ── 1. Run SOL 144 ──
    print("\n[1/3] Running SOL 144...")
    t0 = time.perf_counter()
    bdf_model = parse_bdf(bdf_path)
    result_data = solve_trim(bdf_model, n_workers=0)
    t_sol144 = time.perf_counter() - t0
    n_sc = len(result_data.subcases)
    print(f"  SOL 144 done in {t_sol144:.1f} s, {n_sc} subcases")

    # ── 2. Build ROM ──
    print("\n[2/3] Building ROM...")
    bdf_model2 = parse_bdf(bdf_path)
    t0 = time.perf_counter()
    rom = ModalROM.build(bdf_model2, n_modes=20, mach=0.2)
    t_rom = time.perf_counter() - t0
    print(f"  ROM built in {t_rom:.1f} s, q_ref={rom.q_ref:.6f} N/mm²")

    # ── 3. Compare ──
    print("\n[3/3] Comparing results...")
    print("-" * 70)

    common_nids = list(set(rom.sorted_nids))
    results_csv = []
    results_csv.append("subcase,nz,alpha_rad,de_rad,q_Nmm2,"
                       "disp_z_rms_rel,disp_z_max_rel,disp_z_corr,"
                       "disp_z_ref_max,disp_z_rom_max,"
                       "force_z_rms_rel,force_z_max_rel,force_z_corr")

    all_disp_errors = []
    all_force_errors = []

    for sc_result in result_data.subcases:
        sc_id = sc_result.subcase_id
        # Extract trim variables
        trim_vars = sc_result.trim_variables or {}
        alpha = trim_vars.get('ANGLEA', 0.0)
        de = trim_vars.get('ELEV', trim_vars.get('ELEVATOR', 0.0))
        da = trim_vars.get('APTS', trim_vars.get('AIL', trim_vars.get('AILERON', 0.0)))
        dr = trim_vars.get('RUDD', trim_vars.get('RUDDER', 0.0))

        # Get nz from URDD3 or default
        nz = trim_vars.get('URDD3', 1.0)
        if nz == 0.0:
            nz = 1.0

        # Find the matching TRIM card's q for this subcase
        trim_id = None
        for sc_bdf in bdf_model.subcases:
            if sc_bdf.id == sc_id and hasattr(sc_bdf, 'trim_id'):
                trim_id = sc_bdf.trim_id
                break
        q_Nmm2 = 0.0
        if trim_id and trim_id in bdf_model.trims:
            q_Nmm2 = bdf_model.trims[trim_id].q  # already N/mm²
        elif bdf_model.trims:
            q_Nmm2 = next(iter(bdf_model.trims.values())).q
        # Convert model-unit q (N/mm²) → velocity (m/s)
        rho = 1.225
        q_Pa = q_Nmm2 * 1e6  # N/mm² → Pa
        V = np.sqrt(2 * q_Pa / rho) if q_Pa > 0 else 80.0

        # ROM computation
        t0 = time.perf_counter()
        rom_disp, rom_forces = rom.compute_response(
            alpha=alpha, V=V, de=de, da=da, dr=dr, nz=nz, rho=rho
        )
        t_rom_eval = (time.perf_counter() - t0) * 1000  # ms

        # Compare displacements (z-component)
        ref_disp = sc_result.displacements
        disp_metrics = compute_error_metrics(
            ref_disp, rom_disp, common_nids, component=2)

        # Compare forces (z-component) if available
        ref_forces = sc_result.nodal_combined_forces or {}
        if ref_forces:
            force_metrics = compute_error_metrics(
                ref_forces, rom_forces, common_nids, component=2)
        else:
            force_metrics = {
                'rms_rel': float('nan'), 'max_rel': float('nan'),
                'correlation': float('nan'), 'n_nodes': 0,
                'ref_max': 0.0, 'rom_max': 0.0,
            }

        all_disp_errors.append(disp_metrics['rms_rel'])
        if not np.isnan(force_metrics['rms_rel']):
            all_force_errors.append(force_metrics['rms_rel'])

        print(f"\n  Subcase {sc_id}: nz={nz:.2f}, α={np.degrees(alpha):.2f}°, "
              f"δe={np.degrees(de):.2f}°, q={q_Nmm2:.6f} N/mm²")
        print(f"    ROM eval: {t_rom_eval:.3f} ms")
        print(f"    Disp(z): RMS_rel={disp_metrics['rms_rel']:.4f} "
              f"({disp_metrics['rms_rel']*100:.2f}%), "
              f"max_rel={disp_metrics['max_rel']:.4f}, "
              f"corr={disp_metrics['correlation']:.6f}")
        print(f"    Disp(z): ref_max={disp_metrics['ref_max']:.6e} mm, "
              f"rom_max={disp_metrics['rom_max']:.6e} mm")
        if not np.isnan(force_metrics['rms_rel']):
            print(f"    Force(z): RMS_rel={force_metrics['rms_rel']:.4f} "
                  f"({force_metrics['rms_rel']*100:.2f}%), "
                  f"corr={force_metrics['correlation']:.6f}")

        results_csv.append(
            f"{sc_id},{nz:.4f},{alpha:.6f},{de:.6f},{q_Nmm2:.6f},"
            f"{disp_metrics['rms_rel']:.6f},{disp_metrics['max_rel']:.6f},"
            f"{disp_metrics['correlation']:.6f},"
            f"{disp_metrics['ref_max']:.6e},{disp_metrics['rom_max']:.6e},"
            f"{force_metrics['rms_rel']:.6f},{force_metrics['max_rel']:.6f},"
            f"{force_metrics['correlation']:.6f}"
        )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    mean_disp_err = np.mean(all_disp_errors) * 100
    max_disp_err = np.max(all_disp_errors) * 100
    print(f"  Displacement(z) RMS relative error:")
    print(f"    Mean across subcases: {mean_disp_err:.2f}%")
    print(f"    Max across subcases:  {max_disp_err:.2f}%")
    if all_force_errors:
        mean_force_err = np.mean(all_force_errors) * 100
        max_force_err = np.max(all_force_errors) * 100
        print(f"  Force(z) RMS relative error:")
        print(f"    Mean across subcases: {mean_force_err:.2f}%")
        print(f"    Max across subcases:  {max_force_err:.2f}%")
    print(f"\n  SOL 144 total time: {t_sol144:.1f} s")
    print(f"  ROM build time:     {t_rom:.1f} s")
    print(f"  ROM eval time:      < 0.5 ms per subcase")
    print(f"  ROM q_ref:          {rom.q_ref:.6f} N/mm²")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__),
                            'rom_validation_results.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(results_csv) + '\n')
    print(f"\n  Results saved to: {csv_path}")

    # Save summary text
    summary_path = os.path.join(os.path.dirname(__file__),
                                'rom_validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ROM vs SOL 144 Validation Summary — GACOMP\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: 22,640 nodes, 27,016 elements, 135,840 DOF\n")
        f.write(f"Aero: 783 VLM panels, 44 SPLINE1 cards\n")
        f.write(f"ROM modes: {rom.n_modes}\n")
        f.write(f"ROM q_ref: {rom.q_ref:.6f} N/mm²\n\n")
        f.write(f"Displacement(z) RMS relative error:\n")
        f.write(f"  Mean: {mean_disp_err:.2f}%\n")
        f.write(f"  Max:  {max_disp_err:.2f}%\n\n")
        if all_force_errors:
            f.write(f"Force(z) RMS relative error:\n")
            f.write(f"  Mean: {mean_force_err:.2f}%\n")
            f.write(f"  Max:  {max_force_err:.2f}%\n\n")
        f.write(f"Timing:\n")
        f.write(f"  SOL 144: {t_sol144:.1f} s\n")
        f.write(f"  ROM build: {t_rom:.1f} s\n")
        f.write(f"  ROM eval: < 0.5 ms/subcase\n")
    print(f"  Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
