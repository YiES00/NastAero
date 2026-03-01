#!/usr/bin/env python3
"""NastAero Scalability Benchmark.

Runs SOL 101 on progressively larger plate models and measures:
- BDF parse time
- Assembly time
- Solve time
- Total time
- Memory usage (peak RSS)

Usage:
    python benchmark_scalability.py
    python benchmark_scalability.py --max-dof 500000
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_large_model import generate_gravity_model


def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for Windows without psutil
        try:
            import ctypes
            import ctypes.wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.GetCurrentProcess()
            psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            return 0.0


def run_benchmark(nx, ny, temp_dir):
    """Run a single benchmark case and return timing results."""
    n_nodes = (nx + 1) * (ny + 1)
    n_elems = nx * ny
    n_dof = n_nodes * 6

    result = {
        'nx': nx, 'ny': ny,
        'n_nodes': n_nodes, 'n_elements': n_elems, 'n_dof': n_dof,
        'status': 'failed',
    }

    bdf_path = os.path.join(temp_dir, f"bench_{nx}x{ny}.bdf")

    # Generate model
    print(f"\n{'='*70}")
    print(f"Benchmark: {nx}x{ny} CQUAD4 = {n_elems:,} elements, {n_dof:,} DOFs ({n_dof/1e6:.2f}M)")
    print(f"{'='*70}")

    gc.collect()
    mem_start = get_memory_mb()

    try:
        # Step 1: Generate BDF file
        t0 = time.perf_counter()
        generate_gravity_model(nx, ny, output_path=bdf_path)
        t_gen = time.perf_counter() - t0
        result['t_generate'] = t_gen
        print(f"  [1/4] BDF generated: {t_gen:.2f} s")

        # Step 2: Parse BDF (includes cross-reference)
        from nastaero.bdf.parser import BDFParser
        parser = BDFParser()
        t0 = time.perf_counter()
        bdf_model = parser.parse(bdf_path)
        t_parse = time.perf_counter() - t0
        result['t_parse'] = t_parse
        result['t_cross_ref'] = 0.0  # included in parse
        print(f"  [2/4] BDF parsed: {t_parse:.2f} s ({n_nodes:,} nodes, {n_elems:,} elements)")

        # Step 3: Assemble
        from nastaero.fem.dof_manager import DOFManager
        from nastaero.fem.assembly import assemble_global_matrices
        dof_mgr = DOFManager(list(bdf_model.nodes.keys()))

        t0 = time.perf_counter()
        K, M = assemble_global_matrices(bdf_model, dof_mgr)
        t_assemble = time.perf_counter() - t0
        result['t_assemble'] = t_assemble
        result['nnz_K'] = K.nnz
        result['nnz_M'] = M.nnz
        print(f"  [3b/4] Assembly: {t_assemble:.2f} s (K: {K.nnz:,} nnz, M: {M.nnz:,} nnz)")

        mem_after_assembly = get_memory_mb()
        result['mem_assembly_mb'] = mem_after_assembly

        # Step 4: Solve directly (partition + spsolve) - avoids re-assembly
        from nastaero.fem.load_vector import assemble_load_vector
        from nastaero.fem.boundary import apply_spcs
        import scipy.sparse.linalg as spla

        subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
        subcase = subcases[0]
        effective = bdf_model.get_effective_subcase(subcase)

        t0 = time.perf_counter()
        F = assemble_load_vector(bdf_model, effective, dof_mgr)
        t_load = time.perf_counter() - t0
        result['t_load_vector'] = t_load
        print(f"  [4a/4] Load vector: {t_load:.2f} s")

        spc_list = bdf_model.spcs.get(effective.spc_id, [])
        constrained, enforced = dof_mgr.get_constrained_dofs(spc_list, bdf_model.nodes)

        t0 = time.perf_counter()
        K_ff, M_ff, F_f, f_dofs, s_dofs = apply_spcs(K, M, F, constrained, enforced)
        t_partition = time.perf_counter() - t0
        result['t_partition'] = t_partition
        result['n_free_dof'] = K_ff.shape[0]
        result['n_constrained_dof'] = len(s_dofs)
        print(f"  [4b/4] Partition: {t_partition:.2f} s ({K_ff.shape[0]:,} free, {len(s_dofs):,} constrained)")

        t0 = time.perf_counter()
        u_f = spla.spsolve(K_ff, F_f)
        t_solve = time.perf_counter() - t0
        result['t_solve'] = t_solve
        print(f"  [4c/4] Solve (spsolve): {t_solve:.2f} s")

        max_disp = float(np.max(np.abs(u_f)))
        result['max_displacement'] = max_disp
        print(f"        Max displacement: {max_disp:.6e} m")

        t_solve_total = t_load + t_partition + t_solve
        result['t_solve_total'] = t_assemble + t_solve_total

        mem_peak = get_memory_mb()
        result['mem_peak_mb'] = mem_peak
        result['mem_delta_mb'] = mem_peak - mem_start

        t_total = t_gen + t_parse + t_assemble + t_solve_total
        result['t_total'] = t_total
        result['status'] = 'success'

        print(f"\n  Summary: parse={t_parse:.1f}s, assembly={t_assemble:.1f}s, "
              f"load={t_load:.1f}s, solve={t_solve:.1f}s, total={t_total:.1f}s")
        print(f"  Memory: start={mem_start:.0f}MB, peak={mem_peak:.0f}MB, "
              f"delta={mem_peak-mem_start:.0f}MB")

    except MemoryError:
        result['status'] = 'out_of_memory'
        print(f"  ERROR: Out of memory!")
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"  ERROR: {e}")
        traceback.print_exc()
    finally:
        # Clean up temp file
        if os.path.exists(bdf_path):
            try:
                os.remove(bdf_path)
            except OSError:
                pass
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="NastAero Scalability Benchmark")
    parser.add_argument("--max-dof", type=int, default=1000000,
                        help="Maximum DOF count to test (default: 1,000,000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    # Benchmark sizes: progressively larger
    # nx × ny CQUAD4 → (nx+1)*(ny+1) nodes → (nx+1)*(ny+1)*6 DOFs
    test_sizes = [
        (10, 10),       # 726 DOF
        (20, 20),       # 2,646 DOF
        (50, 50),       # 15,606 DOF
        (100, 100),     # 61,206 DOF
        (150, 150),     # 136,806 DOF
        (200, 200),     # 242,406 DOF
        (250, 250),     # 378,006 DOF
        (300, 300),     # 543,606 DOF
        (350, 350),     # 739,206 DOF
        (400, 400),     # 964,806 DOF
        (410, 410),     # 1,013,766 DOF
    ]

    # Filter by max DOF
    test_sizes = [(nx, ny) for nx, ny in test_sizes
                  if (nx + 1) * (ny + 1) * 6 <= args.max_dof * 1.2]

    # Ensure temp directory exists
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Setup logging
    from nastaero.config import setup_logging
    setup_logging("WARNING")  # Suppress most output during benchmarks

    print("=" * 70)
    print("NastAero Scalability Benchmark")
    print("=" * 70)
    print(f"Max DOF target: {args.max_dof:,}")
    print(f"Test cases: {len(test_sizes)}")
    print()

    all_results = []
    for nx, ny in test_sizes:
        result = run_benchmark(nx, ny, temp_dir)
        all_results.append(result)

        # Stop if we're running out of memory or taking too long
        if result['status'] == 'out_of_memory':
            print("\n*** Stopping: Out of memory ***")
            break
        if result['status'] == 'success' and result.get('t_solve_total', 0) > 3600:
            print("\n*** Stopping: Solve time > 1 hour ***")
            break

    # Summary table
    print("\n\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Size':>10s} {'Elements':>10s} {'DOFs':>10s} {'Parse':>8s} "
          f"{'Assembly':>10s} {'Load':>8s} {'Solve':>10s} {'Total':>10s} {'Memory':>8s} {'Status':>10s}")
    print("-" * 106)
    for r in all_results:
        size_str = f"{r['nx']}x{r['ny']}"
        elem_str = f"{r['n_elements']:,}"
        dof_str = f"{r['n_dof']:,}"
        if r['status'] == 'success':
            parse_s = f"{r['t_parse']:.1f}s"
            asm_s = f"{r['t_assemble']:.1f}s"
            load_s = f"{r.get('t_load_vector', 0):.1f}s"
            solve_s = f"{r.get('t_solve', 0):.1f}s"
            total_s = f"{r['t_total']:.1f}s"
            mem_s = f"{r.get('mem_peak_mb', 0):.0f}MB"
        else:
            parse_s = asm_s = load_s = solve_s = total_s = mem_s = "---"
        print(f"{size_str:>10s} {elem_str:>10s} {dof_str:>10s} {parse_s:>8s} "
              f"{asm_s:>10s} {load_s:>8s} {solve_s:>10s} {total_s:>10s} {mem_s:>8s} {r['status']:>10s}")

    # Save results
    output_file = args.output or os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Clean up temp dir
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
