#!/usr/bin/env python3
"""Plot benchmark scalability results."""
import json
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plot generation")
    sys.exit(0)

# Load results
script_dir = os.path.dirname(__file__)
results_file = os.path.join(script_dir, 'benchmark_results.json')
if not os.path.exists(results_file):
    print(f"Results file not found: {results_file}")
    sys.exit(1)

with open(results_file) as f:
    results = json.load(f)

# Filter successful results
data = [r for r in results if r['status'] == 'success']
if not data:
    print("No successful results to plot")
    sys.exit(1)

n_dof = np.array([r['n_dof'] for r in data]) / 1000  # kDOF
n_elem = np.array([r['n_elements'] for r in data]) / 1000  # kElements
t_parse = np.array([r['t_parse'] for r in data])
t_assembly = np.array([r['t_assemble'] for r in data])
t_load = np.array([r.get('t_load_vector', 0) for r in data])
t_solve = np.array([r.get('t_solve', 0) for r in data])
t_total = np.array([r['t_total'] for r in data])

# Styling
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (14, 10),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NastAero Scalability Benchmark\n(CQUAD4 Flat Plate, SOL 101, SciPy spsolve)',
             fontsize=14, fontweight='bold')

# Plot 1: Total time vs DOF
ax = axes[0, 0]
ax.plot(n_dof, t_total, 'ko-', linewidth=2, markersize=6, label='Total')
ax.plot(n_dof, t_assembly, 'b^-', linewidth=1.5, markersize=5, label='Assembly')
ax.plot(n_dof, t_solve, 'rs-', linewidth=1.5, markersize=5, label='Solve')
ax.plot(n_dof, t_parse, 'gD-', linewidth=1.5, markersize=4, label='Parse')
ax.plot(n_dof, t_load, 'mv-', linewidth=1.5, markersize=4, label='Load Vector')
ax.set_xlabel('Degrees of Freedom (kDOF)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Execution Time vs Model Size')
ax.legend(loc='upper left')

# Plot 2: Stacked bar chart of time components
ax = axes[0, 1]
labels = [f"{r['nx']}x{r['ny']}" for r in data]
# Show only selected labels for readability
skip = max(1, len(labels) // 6)
x_pos = np.arange(len(labels))
bar_w = 0.6
bottoms = np.zeros(len(labels))
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
names = ['Assembly', 'Solve', 'Parse', 'Load Vector']
times = [t_assembly, t_solve, t_parse, t_load]
for i, (name, t, c) in enumerate(zip(names, times, colors)):
    ax.bar(x_pos, t, bar_w, bottom=bottoms, label=name, color=c, alpha=0.85)
    bottoms += t
ax.set_xticks(x_pos[::skip])
ax.set_xticklabels([labels[i] for i in range(0, len(labels), skip)], rotation=45)
ax.set_xlabel('Mesh Size')
ax.set_ylabel('Time (seconds)')
ax.set_title('Time Breakdown by Phase')
ax.legend(loc='upper left', fontsize=9)

# Plot 3: Assembly rate (elements/sec)
ax = axes[1, 0]
rate = n_elem * 1000 / t_assembly  # elements/sec
rate[t_assembly < 0.01] = 0  # avoid division issues
ax.plot(n_dof, rate, 'b^-', linewidth=2, markersize=6)
ax.set_xlabel('Degrees of Freedom (kDOF)')
ax.set_ylabel('Elements / second')
ax.set_title('Assembly Throughput')
ax.axhline(y=np.mean(rate[rate > 0]), color='r', linestyle='--', alpha=0.5,
           label=f'Mean: {np.mean(rate[rate > 0]):.0f} elem/s')
ax.legend()

# Plot 4: Log-log scaling
ax = axes[1, 1]
mask = t_total > 0.01
if mask.sum() > 2:
    ax.loglog(n_dof[mask], t_total[mask], 'ko-', linewidth=2, markersize=6, label='Total')
    ax.loglog(n_dof[mask], t_assembly[mask], 'b^-', linewidth=1.5, markersize=5, label='Assembly')
    ax.loglog(n_dof[mask], t_solve[mask], 'rs-', linewidth=1.5, markersize=5, label='Solve')

    # Fit power law to total: t = a * N^b
    log_n = np.log(n_dof[mask])
    log_t = np.log(t_total[mask])
    b, log_a = np.polyfit(log_n, log_t, 1)
    n_fit = np.linspace(n_dof[mask].min(), n_dof[mask].max(), 50)
    t_fit = np.exp(log_a) * n_fit ** b
    ax.loglog(n_fit, t_fit, 'k--', alpha=0.5, label=f'Total fit: O(N^{b:.2f})')

    # Fit assembly
    log_ta = np.log(t_assembly[mask])
    ba, log_aa = np.polyfit(log_n, log_ta, 1)
    ta_fit = np.exp(log_aa) * n_fit ** ba
    ax.loglog(n_fit, ta_fit, 'b--', alpha=0.5, label=f'Assembly fit: O(N^{ba:.2f})')

    ax.legend(fontsize=9)
ax.set_xlabel('Degrees of Freedom (kDOF)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Scaling Behavior (Log-Log)')

plt.tight_layout()
output_path = os.path.join(script_dir, 'benchmark_scalability.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also print a summary table
print("\n\nSCALABILITY SUMMARY")
print("=" * 80)
print(f"{'DOFs':>12s} {'Elements':>10s} {'Parse':>8s} {'Assembly':>10s} {'Solve':>8s} {'Total':>8s} {'elem/s':>8s}")
print("-" * 80)
for r in data:
    asm_rate = r['n_elements'] / r['t_assemble'] if r['t_assemble'] > 0 else 0
    print(f"{r['n_dof']:>12,d} {r['n_elements']:>10,d} {r['t_parse']:>8.2f} "
          f"{r['t_assemble']:>10.2f} {r.get('t_solve', 0):>8.2f} {r['t_total']:>8.1f} {asm_rate:>8.0f}")

print(f"\nMax model size: {data[-1]['n_dof']:,} DOFs ({data[-1]['n_dof']/1e6:.2f}M)")
print(f"Total time at max: {data[-1]['t_total']:.1f} seconds")
print(f"Max displacement converged to: {data[-1]['max_displacement']:.6e} m")
