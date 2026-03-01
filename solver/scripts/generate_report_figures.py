"""Generate all figures for the verification report."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nastaero.bdf.parser import BDFParser
from nastaero.solvers.sol101 import solve_static
from nastaero.solvers.sol103 import solve_modes
from nastaero.solvers.sol144 import solve_trim
from nastaero.aero.panel import generate_panel_mesh, generate_all_panels
from nastaero.aero.dlm import build_aic_matrix, circulation_to_delta_cp
from types import SimpleNamespace

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'validation')
VM_DIR = os.path.join(VALIDATION_DIR, 'nastran_vm')
AERO_VM_DIR = os.path.join(VALIDATION_DIR, 'aero_vm')
GOLAND_DIR = os.path.join(VALIDATION_DIR, 'goland_wing')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
})

print("Generating report figures...")

# ============================================================
# Fig 1: Cantilever Beam Deflection Curve
# ============================================================
print("  Fig 1: Cantilever beam deflection curve...")
model = BDFParser().parse(os.path.join(VALIDATION_DIR, 'cantilever_beam', 'cantilever.bdf'))
results = solve_static(model)
sc = results.subcases[0]

P, L, E, I = 100.0, 1.0, 7.0e10, 8.333e-10
x_nodes = [(nid-1)*0.1 for nid in range(1, 12)]
w_nastaero = [sc.displacements[nid][2] for nid in range(1, 12)]
x_fine = np.linspace(0, L, 100)
w_exact = P * x_fine**2 * (3*L - x_fine) / (6*E*I)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Beam mesh diagram
ax1.set_xlim(-0.15, 1.15)
ax1.set_ylim(-0.3, 0.3)
for i in range(10):
    x0 = i * 0.1
    x1 = (i+1) * 0.1
    ax1.plot([x0, x1], [0, 0], 'b-', lw=2.5)
    ax1.plot(x0, 0, 'ko', ms=5, zorder=5)
ax1.plot(1.0, 0, 'ko', ms=5, zorder=5)
# Fixed support
ax1.plot([-0.02, -0.02], [-0.15, 0.15], 'k-', lw=3)
for y in np.linspace(-0.12, 0.12, 5):
    ax1.plot([-0.02, -0.08], [y, y-0.04], 'k-', lw=1)
# Force arrow
ax1.annotate('', xy=(1.0, 0.12), xytext=(1.0, 0),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(1.02, 0.06, 'P=100N', color='red', fontsize=10)
ax1.text(0.5, -0.22, 'L = 1.0 m, 10 CBAR elements', ha='center', fontsize=10)
ax1.set_title('(a) Model: Cantilever Beam')
ax1.set_xlabel('x (m)')
ax1.set_aspect('equal')
ax1.axis('off')

# Deflection plot
ax2.plot(x_fine, w_exact*1000, 'b-', lw=2, label='Analytical (Euler-Bernoulli)')
ax2.plot(x_nodes, [w*1000 for w in w_nastaero], 'ro', ms=7, label='NastAero', zorder=5)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('Deflection w (mm)')
ax2.set_title('(b) Deflection Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig01_cantilever_beam.png'))
plt.close()

# ============================================================
# Fig 2: VM2 Cantilever Moment Deflection
# ============================================================
print("  Fig 2: VM2 cantilever moment...")
model = BDFParser().parse(os.path.join(VM_DIR, 'vm2_cantilever_moment.bdf'))
results = solve_static(model)
sc = results.subcases[0]
M_val, L, E, I = 10000.0, 10.0, 200e9, 1e-4

x_nodes = [(nid-1)*1.0 for nid in range(1, 12)]
w_na = [abs(sc.displacements[nid][2]) for nid in range(1, 12)]
x_fine = np.linspace(0, 10, 100)
w_exact = M_val * x_fine**2 / (2*E*I)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x_fine, w_exact*1e3, 'b-', lw=2, label='Analytical: $w = Mx^2/(2EI)$')
ax.plot(x_nodes, [w*1e3 for w in w_na], 'ro', ms=7, label='NastAero (10 CBAR)')
ax.set_xlabel('x (m)')
ax.set_ylabel('|Deflection| (mm)')
ax.set_title('VM2: Cantilever Under End Moment (M=10,000 N-m)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, 'fig02_vm2_moment.png'))
plt.close()

# ============================================================
# Fig 3: VM3 Plate Deflection Contour
# ============================================================
print("  Fig 3: VM3 plate deflection...")
model = BDFParser().parse(os.path.join(VM_DIR, 'vm3_plate_pressure.bdf'))
results = solve_static(model)
sc = results.subcases[0]

# Extract grid
nx, ny = 9, 9
x_grid = np.zeros((ny, nx))
y_grid = np.zeros((ny, nx))
w_grid = np.zeros((ny, nx))
for nid, node in model.nodes.items():
    i = (nid - 1) % nx
    j = (nid - 1) // nx
    x_grid[j, i] = node.xyz_global[0]
    y_grid[j, i] = node.xyz_global[1]
    w_grid[j, i] = abs(sc.displacements[nid][2]) * 1e4  # mm x 10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Mesh
for j in range(8):
    for i in range(8):
        corners = [
            (x_grid[j,i], y_grid[j,i]),
            (x_grid[j,i+1], y_grid[j,i+1]),
            (x_grid[j+1,i+1], y_grid[j+1,i+1]),
            (x_grid[j+1,i], y_grid[j+1,i]),
        ]
        poly = plt.Polygon(corners, fill=False, edgecolor='blue', lw=0.8)
        ax1.add_patch(poly)
ax1.plot(x_grid.ravel(), y_grid.ravel(), 'k.', ms=2)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_aspect('equal')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('(a) 8x8 CQUAD4 Mesh (SS all edges)')

# Contour
cf = ax2.contourf(x_grid, y_grid, w_grid, levels=20, cmap='RdYlBu_r')
plt.colorbar(cf, ax=ax2, label='|w| (x$10^{-4}$ m)')
ax2.set_aspect('equal')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title('(b) Deflection Under Uniform Pressure')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig03_vm3_plate.png'))
plt.close()

# ============================================================
# Fig 4: CL_alpha vs AR
# ============================================================
print("  Fig 4: CL_alpha vs aspect ratio...")
AR_values = [2, 3, 4, 5, 6, 8, 10, 15, 20]
cl_vlm = []
cl_ll = []
cl_2d = 2 * np.pi

for AR in AR_values:
    nspan = max(int(AR)*2, 8)
    caero = SimpleNamespace(
        nspan=nspan, nchord=4,
        p1=np.array([0.0, 0.0, 0.0]),
        p4=np.array([0.0, float(AR), 0.0]),
        chord1=1.0, chord4=1.0,
    )
    boxes = generate_panel_mesh(caero)
    D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
    D_inv = np.linalg.inv(D)
    w = -np.ones(len(boxes))
    gamma = D_inv @ w
    dcp = circulation_to_delta_cp(boxes, gamma)
    S_ref = float(AR)
    CL = sum(dcp[i] * boxes[i].area for i in range(len(boxes))) / S_ref
    cl_vlm.append(CL)
    cl_ll.append(2*np.pi*AR/(AR+2))

AR_fine = np.linspace(1, 25, 100)
cl_ll_fine = 2*np.pi*AR_fine/(AR_fine+2)
# Helmbold
cl_helm = 2*np.pi*AR_fine / np.sqrt(4 + AR_fine**2) / (1 + 2/np.sqrt(4 + AR_fine**2)) * AR_fine / AR_fine
cl_helm = 2*np.pi / (np.sqrt(1 + (2*np.pi/(np.pi*AR_fine))**2) + 2*np.pi/(np.pi*AR_fine))
# Simpler: Helmbold formula
cl_helm = 2*np.pi*AR_fine / (2 + np.sqrt(AR_fine**2 + 4))

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(AR_fine, cl_ll_fine, 'b-', lw=2, label='Lifting Line: $C_{L_\\alpha}=2\\pi AR/(AR+2)$')
ax.plot(AR_fine, cl_helm, 'g--', lw=1.5, label='Helmbold Formula')
ax.plot(AR_values, cl_vlm, 'rs', ms=9, label='NastAero VLM', zorder=5)
ax.axhline(y=cl_2d, color='gray', linestyle=':', lw=1, label='$2\\pi$ (2D limit)')
ax.set_xlabel('Aspect Ratio (AR)')
ax.set_ylabel('$C_{L_\\alpha}$ (per radian)')
ax.set_title('Lift Curve Slope vs. Aspect Ratio')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)
ax.set_ylim(0, 7)
plt.savefig(os.path.join(FIG_DIR, 'fig04_cl_alpha_ar.png'))
plt.close()

# ============================================================
# Fig 5: Goland Wing Model + Aero Mesh
# ============================================================
print("  Fig 5: Goland wing model...")
model = BDFParser().parse(os.path.join(GOLAND_DIR, 'goland_static.bdf'))
boxes = generate_all_panels(model)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plan view: structural + aero
for eid, elem in model.elements.items():
    if elem.type == "CBAR":
        n1 = model.nodes[elem.node_ids[0]]
        n2 = model.nodes[elem.node_ids[1]]
        ax1.plot([n1.xyz_global[1], n2.xyz_global[1]],
                 [n1.xyz_global[0], n2.xyz_global[0]], 'b-', lw=2.5)

# Structural nodes
for nid, node in model.nodes.items():
    ax1.plot(node.xyz_global[1], node.xyz_global[0], 'bo', ms=5, zorder=5)

# Aero panels
for box in boxes:
    c = box.corners
    poly = plt.Polygon(
        [(c[0,1], c[0,0]), (c[1,1], c[1,0]),
         (c[2,1], c[2,0]), (c[3,1], c[3,0])],
        fill=False, edgecolor='red', lw=0.7, linestyle='-')
    ax1.add_patch(poly)
    # Control point
    ax1.plot(box.control_point[1], box.control_point[0], 'r.', ms=2)

ax1.set_xlabel('Span y (m)')
ax1.set_ylabel('Chord x (m)')
ax1.set_title('(a) Goland Wing: Structure + Aero Mesh')
ax1.set_aspect('equal')
ax1.legend(['Structural beam (CBAR)', 'Aero panels (CAERO1)'],
           loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.invert_yaxis()

# Spanwise displacement (from results)
results = solve_trim(model)
sc = results.subcases[0]

y_span = []
uz_span = []
rx_span = []
for nid in sorted(sc.displacements.keys()):
    y_span.append(model.nodes[nid].xyz_global[1])
    uz_span.append(sc.displacements[nid][2] * 1e3)
    rx_span.append(np.degrees(sc.displacements[nid][3]))

ax2a = ax2
ax2b = ax2.twinx()

l1, = ax2a.plot(y_span, uz_span, 'b-o', ms=5, lw=2, label='$u_z$ (bending)')
ax2a.set_xlabel('Span y (m)')
ax2a.set_ylabel('$u_z$ (mm)', color='b')
ax2a.tick_params(axis='y', labelcolor='b')

l2, = ax2b.plot(y_span, rx_span, 'r-s', ms=5, lw=2, label='$\\theta_x$ (twist)')
ax2b.set_ylabel('Twist $\\theta_x$ (deg)', color='r')
ax2b.tick_params(axis='y', labelcolor='r')

ax2.set_title('(b) Spanwise Deformation (M=0.3, 1g trim)')
ax2.legend([l1, l2], ['Bending $u_z$', 'Twist $\\theta_x$'], loc='lower left')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig05_goland_wing.png'))
plt.close()

# ============================================================
# Fig 6: Goland Enhanced (AVM2) Pressure Distribution
# ============================================================
print("  Fig 6: Goland enhanced pressure...")
model = BDFParser().parse(os.path.join(AERO_VM_DIR, 'avm2_goland_enhanced.bdf'))
results = solve_trim(model)
sc = results.subcases[0]
boxes = generate_all_panels(model)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Pressure color map on panels
cp = sc.aero_pressures
cp_min, cp_max = np.min(cp), np.max(cp)

for i, box in enumerate(boxes):
    c = box.corners
    color_val = (cp[i] - cp_min) / (cp_max - cp_min + 1e-15)
    color = plt.cm.RdYlBu_r(color_val)
    poly = plt.Polygon(
        [(c[0,1], c[0,0]), (c[1,1], c[1,0]),
         (c[2,1], c[2,0]), (c[3,1], c[3,0])],
        facecolor=color, edgecolor='k', lw=0.3)
    ax1.add_patch(poly)

# Structural nodes
for nid, node in model.nodes.items():
    ax1.plot(node.xyz_global[1], node.xyz_global[0], 'ko', ms=3, zorder=5)

sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                            norm=plt.Normalize(vmin=cp_min, vmax=cp_max))
plt.colorbar(sm, ax=ax1, label='$\\Delta C_p$')
ax1.set_xlabel('Span y (m)')
ax1.set_ylabel('Chord x (m)')
ax1.set_title('(a) Pressure Distribution (M=0.5, q=3920 Pa)')
ax1.set_aspect('equal')
ax1.invert_yaxis()

# Spanwise deformation
y_span = []
uz_span = []
rx_span = []
for nid in sorted(sc.displacements.keys()):
    y_span.append(model.nodes[nid].xyz_global[1])
    uz_span.append(sc.displacements[nid][2] * 1e3)
    rx_span.append(np.degrees(sc.displacements[nid][3]))

ax2a = ax2
ax2b = ax2.twinx()
l1, = ax2a.plot(y_span, uz_span, 'b-o', ms=5, lw=2)
ax2a.set_xlabel('Span y (m)')
ax2a.set_ylabel('$u_z$ (mm)', color='b')
ax2a.tick_params(axis='y', labelcolor='b')

l2, = ax2b.plot(y_span, rx_span, 'r-s', ms=5, lw=2)
ax2b.set_ylabel('Twist $\\theta_x$ (deg)', color='r')
ax2b.tick_params(axis='y', labelcolor='r')

ax2.set_title(f'(b) Spanwise Deformation ($\\alpha$={np.degrees(sc.trim_variables["ANGLEA"]):.2f}$^\\circ$)')
ax2.legend([l1, l2], ['Bending $u_z$', 'Twist $\\theta_x$'], loc='lower left')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig06_goland_enhanced.png'))
plt.close()

# ============================================================
# Fig 7: VLM Aero Panel Mesh Example
# ============================================================
print("  Fig 7: VLM panel mesh...")
caero = SimpleNamespace(
    nspan=8, nchord=4,
    p1=np.array([0.0, 0.0, 0.0]),
    p4=np.array([0.0, 6.0, 0.0]),
    chord1=1.0, chord4=1.0,
)
boxes = generate_panel_mesh(caero)

fig, ax = plt.subplots(figsize=(10, 4))
for box in boxes:
    c = box.corners
    poly = plt.Polygon(
        [(c[0,1], c[0,0]), (c[1,1], c[1,0]),
         (c[2,1], c[2,0]), (c[3,1], c[3,0])],
        fill=False, edgecolor='blue', lw=1)
    ax.add_patch(poly)
    # 1/4 chord (doublet)
    ax.plot(box.doublet_point[1], box.doublet_point[0], 'r^', ms=4, zorder=5)
    # 3/4 chord (control)
    ax.plot(box.control_point[1], box.control_point[0], 'gs', ms=4, zorder=5)

# Horseshoe vortex for one box (illustrative)
b0 = boxes[12]  # middle box
c = b0.corners
a_pt = c[0] + 0.25*(c[1]-c[0])
b_pt = c[3] + 0.25*(c[2]-c[3])
# Bound vortex
ax.plot([a_pt[1], b_pt[1]], [a_pt[0], b_pt[0]], 'm-', lw=3, alpha=0.6)
# Trailing vortices
ax.annotate('', xy=(a_pt[1], a_pt[0]+1.5), xytext=(a_pt[1], a_pt[0]),
            arrowprops=dict(arrowstyle='->', color='m', lw=1.5, alpha=0.5))
ax.annotate('', xy=(b_pt[1], b_pt[0]+1.5), xytext=(b_pt[1], b_pt[0]),
            arrowprops=dict(arrowstyle='->', color='m', lw=1.5, alpha=0.5))

ax.plot([], [], 'r^', ms=6, label='Doublet point (1/4 chord)')
ax.plot([], [], 'gs', ms=6, label='Control point (3/4 chord)')
ax.plot([], [], 'm-', lw=3, alpha=0.6, label='Horseshoe vortex')

ax.set_xlabel('Span y (m)')
ax.set_ylabel('Chord x (m)')
ax.set_title('VLM Panel Discretization (8 span x 4 chord)')
ax.set_aspect('equal')
ax.legend(loc='upper right')
ax.invert_yaxis()
ax.grid(True, alpha=0.2)
plt.savefig(os.path.join(FIG_DIR, 'fig07_vlm_mesh.png'))
plt.close()

# ============================================================
# Fig 8: Error Summary Bar Chart
# ============================================================
print("  Fig 8: Error summary...")
test_names = [
    'VM1\nAxial Rod',
    'VM2\nCant. Moment',
    'VM3\nPlate Pressure',
    'VM4-f1\nBeam Mode 1',
    'VM5\n3-Bar Truss',
    'VM6\nFixed-Fixed',
    'VM9\nPropped Cant.',
    'VM10-f1\nPlate Mode 1',
]
errors = [
    0.0000,  # VM1
    0.0000,  # VM2
    1.85,    # VM3
    0.00,    # VM4
    0.0000,  # VM5
    0.0000,  # VM6
    0.66,    # VM9
    2.04,    # VM10
]

fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.bar(range(len(test_names)), errors, color=['green' if e < 1 else 'orange' if e < 5 else 'red' for e in errors],
              edgecolor='black', lw=0.8)
ax.set_xticks(range(len(test_names)))
ax.set_xticklabels(test_names, fontsize=9)
ax.set_ylabel('Relative Error (%)')
ax.set_title('Structural Verification: Error vs Analytical Solutions')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='1% threshold')
ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, max(errors)*1.5 + 0.5)

# Add value labels
for bar, val in zip(bars, errors):
    if val < 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                'exact', ha='center', va='bottom', fontsize=8, fontweight='bold', color='green')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig08_error_summary.png'))
plt.close()

# ============================================================
# Fig 9: Aero Error Summary
# ============================================================
print("  Fig 9: Aero error summary...")
aero_names = ['AR=4', 'AR=6', 'AR=8', 'AR=10', 'AR=15', 'AR=20']
aero_errors = [6.72, 6.07, 5.50, 5.04, 4.21, 3.66]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(aero_names)), aero_errors,
              color='steelblue', edgecolor='black', lw=0.8)
ax.set_xticks(range(len(aero_names)))
ax.set_xticklabels(aero_names)
ax.set_ylabel('Error vs Lifting Line Theory (%)')
ax.set_title('VLM $C_{L_\\alpha}$ Error vs. Aspect Ratio')
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='5% reference')
ax.legend()

for bar, val in zip(bars, aero_errors):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig09_aero_cl_error.png'))
plt.close()

# ============================================================
# Fig 10: Fixed-Fixed Beam Deflection
# ============================================================
print("  Fig 10: VM6 Fixed-Fixed beam...")
model = BDFParser().parse(os.path.join(VM_DIR, 'vm6_fixed_fixed_beam.bdf'))
results = solve_static(model)
sc = results.subcases[0]

P, L, E, I = 10000, 10, 200e9, 8.333e-6
x_nodes = [(nid-1)*1.0 for nid in range(1, 12)]
w_na = [abs(sc.displacements[nid][1])*1e3 for nid in range(1, 12)]

# Exact: w(x) for fixed-fixed with center load
x_fine = np.linspace(0, L, 200)
w_exact = np.zeros_like(x_fine)
for i, x in enumerate(x_fine):
    if x <= L/2:
        w_exact[i] = P*x**2*(3*L - 4*x) / (48*E*I)
    else:
        w_exact[i] = P*(L-x)**2*(3*L - 4*(L-x)) / (48*E*I)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x_fine, w_exact*1e3, 'b-', lw=2, label='Analytical')
ax.plot(x_nodes, w_na, 'ro', ms=7, label='NastAero', zorder=5)
ax.set_xlabel('x (m)')
ax.set_ylabel('|Deflection| (mm)')
ax.set_title('VM6: Fixed-Fixed Beam with Center Load (P=10,000 N)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, 'fig10_vm6_fixed_fixed.png'))
plt.close()

# ============================================================
# Fig 11: AVM3 rigid wing schematic
# ============================================================
print("  Fig 11: AVM3 rigid wing...")
model = BDFParser().parse(os.path.join(AERO_VM_DIR, 'avm3_rigid_wing_trim.bdf'))
boxes = generate_all_panels(model)

fig, ax = plt.subplots(figsize=(10, 4))
for box in boxes:
    c = box.corners
    poly = plt.Polygon(
        [(c[0,1], c[0,0]), (c[1,1], c[1,0]),
         (c[2,1], c[2,0]), (c[3,1], c[3,0])],
        fill=True, facecolor='lightblue', edgecolor='blue', lw=0.7, alpha=0.5)
    ax.add_patch(poly)

for nid, node in model.nodes.items():
    ax.plot(node.xyz_global[1], node.xyz_global[0], 'ko', ms=4, zorder=5)

# Beam
for eid, elem in model.elements.items():
    if elem.type == "CBAR":
        n1 = model.nodes[elem.node_ids[0]]
        n2 = model.nodes[elem.node_ids[1]]
        ax.plot([n1.xyz_global[1], n2.xyz_global[1]],
                [n1.xyz_global[0], n2.xyz_global[0]], 'k-', lw=3)

ax.set_xlabel('Span y (m)')
ax.set_ylabel('Chord x (m)')
ax.set_title('AVM3: Rigid Wing Model (AR=4, c=2m, b=8m)')
ax.set_aspect('equal')
ax.invert_yaxis()
ax.grid(True, alpha=0.2)

# Annotation
ax.text(4.0, -0.3, 'CBAR structural beam\n(E = 2×$10^{14}$ Pa)',
        ha='center', fontsize=9, style='italic')
ax.text(4.0, 2.3, '8×4 VLM panels (32 boxes)',
        ha='center', fontsize=9, color='blue')

plt.savefig(os.path.join(FIG_DIR, 'fig11_avm3_rigid.png'))
plt.close()

print(f"\nAll figures saved to {FIG_DIR}/")
print("Done!")
