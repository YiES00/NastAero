"""Collect all verification test results for the report."""
import os
import sys
import json
import numpy as np

# Add solver to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nastaero.bdf.parser import BDFParser
from nastaero.solvers.sol101 import solve_static
from nastaero.solvers.sol103 import solve_modes
from nastaero.solvers.sol144 import solve_trim
from nastaero.aero.panel import generate_panel_mesh, generate_all_panels
from nastaero.aero.dlm import build_aic_matrix, circulation_to_delta_cp

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'validation')
VM_DIR = os.path.join(VALIDATION_DIR, 'nastran_vm')
AERO_VM_DIR = os.path.join(VALIDATION_DIR, 'aero_vm')
GOLAND_DIR = os.path.join(VALIDATION_DIR, 'goland_wing')


def parse_and_solve_101(path):
    model = BDFParser().parse(path)
    return solve_static(model), model

def parse_and_solve_103(path):
    model = BDFParser().parse(path)
    return solve_modes(model), model

def parse_and_solve_144(path):
    model = BDFParser().parse(path)
    return solve_trim(model), model


results = {}

# =========== VM1: Axial Rod ===========
print("=" * 60)
print("VM1: Axial Rod Under Tension")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VM_DIR, 'vm1_rod_axial.bdf'))
sc = r.subcases[0]
P, L, A, E = 1000.0, 10.0, 1.0, 30.0e6
delta_analytical = P * L / (A * E)
tip_x = sc.displacements[2][0]
error = abs(tip_x - delta_analytical) / delta_analytical * 100
rx = sc.spc_forces[1][0]
print(f"  Analytical:  delta = {delta_analytical:.6e}")
print(f"  NastAero:    delta = {tip_x:.6e}")
print(f"  Error:       {error:.4f}%")
print(f"  Reaction Fx: {rx:.2f} N (expected -1000)")
results['VM1'] = {
    'analytical': delta_analytical,
    'nastaero': tip_x,
    'error_pct': error,
    'reaction': rx,
}

# =========== VM2: Cantilever Moment ===========
print("\n" + "=" * 60)
print("VM2: Cantilever Beam Under End Moment")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VM_DIR, 'vm2_cantilever_moment.bdf'))
sc = r.subcases[0]
M, L, E, I = 10000.0, 10.0, 200.0e9, 1.0e-4
delta_a = M * L**2 / (2 * E * I)
theta_a = M * L / (E * I)
tip_z = sc.displacements[11][2]
tip_ry = sc.displacements[11][4]
err_d = abs(abs(tip_z) - delta_a) / delta_a * 100
err_t = abs(abs(tip_ry) - theta_a) / theta_a * 100
print(f"  Tip Deflection:")
print(f"    Analytical: {delta_a:.6e} m")
print(f"    NastAero:   {abs(tip_z):.6e} m")
print(f"    Error:      {err_d:.4f}%")
print(f"  Tip Rotation:")
print(f"    Analytical: {theta_a:.6e} rad")
print(f"    NastAero:   {abs(tip_ry):.6e} rad")
print(f"    Error:      {err_t:.4f}%")

# Deflection curve
print(f"  Deflection curve (w(x) = Mx^2/(2EI)):")
deflection_data = []
for nid in range(1, 12):
    x = (nid - 1) * 1.0
    w_exact = M * x**2 / (2 * E * I)
    w_comp = abs(sc.displacements[nid][2])
    err = abs(w_comp - w_exact) / max(w_exact, 1e-15) * 100 if w_exact > 1e-12 else 0
    deflection_data.append({'x': x, 'analytical': w_exact, 'nastaero': w_comp, 'error': err})
    if nid in [1, 3, 6, 9, 11]:
        print(f"    x={x:.1f}m: analytical={w_exact:.6e}, NastAero={w_comp:.6e}, err={err:.3f}%")

results['VM2'] = {
    'tip_deflection': {'analytical': delta_a, 'nastaero': abs(tip_z), 'error_pct': err_d},
    'tip_rotation': {'analytical': theta_a, 'nastaero': abs(tip_ry), 'error_pct': err_t},
    'deflection_curve': deflection_data,
}

# =========== VM3: Plate Pressure ===========
print("\n" + "=" * 60)
print("VM3: Simply-Supported Plate Under Uniform Pressure")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VM_DIR, 'vm3_plate_pressure.bdf'))
sc = r.subcases[0]
E, nu, t, p, a = 200.0e9, 0.3, 0.01, 1000.0, 1.0
D = E * t**3 / (12 * (1 - nu**2))
alpha_coef = 0.00406
w_a = alpha_coef * p * a**4 / D
w_center = abs(sc.displacements[41][2])
err_p = abs(w_center - w_a) / w_a * 100
print(f"  Timoshenko D = {D:.2f} N-m")
print(f"  Center Deflection:")
print(f"    Analytical: {w_a:.6e} m")
print(f"    NastAero:   {w_center:.6e} m")
print(f"    Error:      {err_p:.2f}%")
results['VM3'] = {
    'analytical': w_a, 'nastaero': w_center, 'error_pct': err_p,
}

# =========== VM4: Beam Modes ===========
print("\n" + "=" * 60)
print("VM4: Cantilever Beam Natural Frequencies")
print("=" * 60)
r, m = parse_and_solve_103(os.path.join(VM_DIR, 'vm4_beam_modes_20elem.bdf'))
sc = r.subcases[0]
E, I, rho, A, L = 200.0e9, 8.333e-10, 7850.0, 1.0e-4, 10.0
C = np.sqrt(E * I / (rho * A))
betas = [1.8751, 4.6941, 7.8548]
analytical_freqs = [b**2 * C / (2 * np.pi * L**2) for b in betas]

# Extract unique frequencies
all_f = sc.frequencies
unique_f = []
for f in all_f:
    if not unique_f or abs(f - unique_f[-1]) / max(f, 1e-10) > 0.01:
        unique_f.append(f)

mode_data = []
print(f"  Mode  |  Analytical (Hz)  |  NastAero (Hz)  |  Error (%)")
print(f"  ------+-------------------+-----------------+-----------")
for i in range(min(3, len(unique_f))):
    err = abs(unique_f[i] - analytical_freqs[i]) / analytical_freqs[i] * 100
    print(f"  {i+1}     |  {analytical_freqs[i]:>15.4f}  |  {unique_f[i]:>13.4f}  |  {err:.2f}")
    mode_data.append({
        'mode': i+1, 'analytical': analytical_freqs[i],
        'nastaero': unique_f[i], 'error_pct': err
    })
results['VM4'] = {'modes': mode_data}

# =========== VM5: Three-Bar Truss ===========
print("\n" + "=" * 60)
print("VM5: Three-Bar Truss")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VM_DIR, 'vm5_3bar_truss.bdf'))
sc = r.subcases[0]
ux = sc.displacements[4][0]
uy = sc.displacements[4][1]
err_ux = abs(ux - 0.004) / 0.004 * 100
err_uy = abs(uy - (-0.004)) / 0.004 * 100
print(f"  Node 4 Displacement:")
print(f"    ux: analytical=0.004, NastAero={ux:.6e}, error={err_ux:.4f}%")
print(f"    uy: analytical=-0.004, NastAero={uy:.6e}, error={err_uy:.4f}%")
results['VM5'] = {
    'ux': {'analytical': 0.004, 'nastaero': ux, 'error_pct': err_ux},
    'uy': {'analytical': -0.004, 'nastaero': uy, 'error_pct': err_uy},
}

# =========== VM6: Fixed-Fixed Beam ===========
print("\n" + "=" * 60)
print("VM6: Fixed-Fixed Beam with Center Load")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VM_DIR, 'vm6_fixed_fixed_beam.bdf'))
sc = r.subcases[0]
P, L, E, I = 10000, 10, 200e9, 8.333e-6
delta_a = P * L**3 / (192 * E * I)
delta_c = abs(sc.displacements[6][1])
err = abs(delta_c - delta_a) / delta_a * 100
ry1 = abs(sc.spc_forces[1][1])
ry11 = abs(sc.spc_forces[11][1])
mz1 = abs(sc.spc_forces[1][5])
mz11 = abs(sc.spc_forces[11][5])
print(f"  Center Deflection:")
print(f"    Analytical: {delta_a:.6e} m")
print(f"    NastAero:   {delta_c:.6e} m")
print(f"    Error:      {err:.4f}%")
print(f"  Reactions: R1={ry1:.2f}N, R11={ry11:.2f}N (expected 5000 each)")
print(f"  Moments:  M1={mz1:.2f}Nm, M11={mz11:.2f}Nm (expected 12500 each)")
results['VM6'] = {
    'deflection': {'analytical': delta_a, 'nastaero': delta_c, 'error_pct': err},
    'reactions': {'R1': ry1, 'R11': ry11, 'expected': 5000.0},
    'moments': {'M1': mz1, 'M11': mz11, 'expected': 12500.0},
}

# =========== VM9: Propped Cantilever ===========
print("\n" + "=" * 60)
print("VM9: Propped Cantilever with Uniform Load")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VM_DIR, 'vm9_propped_cantilever.bdf'))
sc = r.subcases[0]
w, L, E, I = 1000, 10, 200e9, 8.333e-6
delta_a = w * L**4 / (185 * E * I)
max_defl = max(abs(sc.displacements[nid][1]) for nid in range(2, 11))
err = abs(max_defl - delta_a) / delta_a * 100
ry_fixed = sc.spc_forces[1][1]
ry_roller = sc.spc_forces[11][1]
print(f"  Max Deflection:")
print(f"    Analytical: {delta_a:.6e} m")
print(f"    NastAero:   {max_defl:.6e} m")
print(f"    Error:      {err:.2f}%")
print(f"  Reactions: Fixed={ry_fixed:.2f}N(exp 6250), Roller={ry_roller:.2f}N(exp 3750)")
results['VM9'] = {
    'deflection': {'analytical': delta_a, 'nastaero': max_defl, 'error_pct': err},
    'reactions': {'fixed': ry_fixed, 'roller': ry_roller},
}

# =========== VM10: Plate Modes 8x8 ===========
print("\n" + "=" * 60)
print("VM10: 8x8 Plate Modal Analysis")
print("=" * 60)
r, m = parse_and_solve_103(os.path.join(VM_DIR, 'vm10_plate_modes_8x8.bdf'))
sc = r.subcases[0]
E, nu, t, rho, a = 210e9, 0.3, 0.01, 7850, 1.0
D = E * t**3 / (12 * (1 - nu**2))
def plate_freq(m_mode, n_mode):
    return (np.pi / 2) * np.sqrt(D / (rho * t)) * (m_mode**2/a**2 + n_mode**2/a**2)

positive_f = [f for f in sc.frequencies if f > 1.0]
f11_a = plate_freq(1, 1)
f12_a = plate_freq(1, 2)
f22_a = plate_freq(2, 2)
print(f"  Mode |  (m,n) | Analytical (Hz) | NastAero (Hz) | Error (%)")
print(f"  -----+--------+-----------------+---------------+----------")
mode_entries = [
    (1, '(1,1)', f11_a, positive_f[0]),
    (2, '(1,2)', f12_a, positive_f[1]),
    (3, '(2,1)', f12_a, positive_f[2]),
]
if len(positive_f) >= 5:
    mode_entries.append((4, '(2,2)', f22_a, positive_f[4] if len(positive_f) > 4 else positive_f[3]))

plate_modes = []
for idx, label, fa, fc in mode_entries:
    err = abs(fc - fa) / fa * 100
    print(f"  {idx}    | {label}  | {fa:>15.2f} | {fc:>13.2f} | {err:.2f}")
    plate_modes.append({'mode': idx, 'mn': label, 'analytical': fa, 'nastaero': fc, 'error_pct': err})
results['VM10'] = {'modes': plate_modes}

# =========== Cantilever Beam (original) ===========
print("\n" + "=" * 60)
print("Original Cantilever Beam (L=1m, P=100N)")
print("=" * 60)
r, m = parse_and_solve_101(os.path.join(VALIDATION_DIR, 'cantilever_beam', 'cantilever.bdf'))
sc = r.subcases[0]
P, L, E, I = 100.0, 1.0, 7.0e10, 8.333e-10
delta_a = P * L**3 / (3 * E * I)
tip_z = sc.displacements[11][2]
err = abs(tip_z - delta_a) / delta_a * 100
print(f"  Tip Deflection:")
print(f"    Analytical: {delta_a:.6e} m")
print(f"    NastAero:   {tip_z:.6e} m")
print(f"    Error:      {err:.4f}%")

cant_curve = []
for nid in range(1, 12):
    x = (nid - 1) * 0.1
    w_exact = P * x**2 * (3*L - x) / (6 * E * I)
    w_comp = sc.displacements[nid][2]
    cant_curve.append({'x': x, 'analytical': w_exact, 'nastaero': w_comp})
results['cantilever'] = {
    'tip': {'analytical': delta_a, 'nastaero': tip_z, 'error_pct': err},
    'curve': cant_curve,
}

# =========== DLM CL_alpha ===========
print("\n" + "=" * 60)
print("DLM CL_alpha Validation")
print("=" * 60)
from types import SimpleNamespace
cl_data = []
for AR in [2, 4, 6, 8, 10, 15, 20]:
    nspan = max(int(AR) * 2, 8)
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
    delta_cp = circulation_to_delta_cp(boxes, gamma)
    S_ref = float(AR)
    CL = sum(delta_cp[i] * boxes[i].area for i in range(len(boxes))) / S_ref
    cl_ll = 2 * np.pi * AR / (AR + 2)
    err = abs(CL - cl_ll) / cl_ll * 100
    print(f"  AR={AR:>2d}: CL_alpha(VLM)={CL:.4f}, CL_alpha(LL)={cl_ll:.4f}, error={err:.2f}%")
    cl_data.append({'AR': AR, 'VLM': CL, 'lifting_line': cl_ll, 'error_pct': err})
results['CL_alpha'] = cl_data

# =========== Goland Wing Trim (original) ===========
print("\n" + "=" * 60)
print("Goland Wing Trim (M=0.3, q=1531.25 Pa)")
print("=" * 60)
r, m = parse_and_solve_144(os.path.join(GOLAND_DIR, 'goland_static.bdf'))
sc = r.subcases[0]
alpha = sc.trim_variables['ANGLEA']
alpha_deg = np.degrees(alpha)
total_fz = np.sum(sc.aero_forces[:, 2])
total_mass = 0
for eid, elem in m.elements.items():
    if elem.type == "CBAR":
        n1 = m.nodes[elem.node_ids[0]]
        n2 = m.nodes[elem.node_ids[1]]
        L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
        prop = elem.property_ref
        mat = prop.material_ref
        total_mass += mat.rho * prop.A * L
weight = total_mass * 9.81
print(f"  ANGLEA = {alpha:.6f} rad ({alpha_deg:.4f} deg)")
print(f"  Total Fz = {total_fz:.2f} N")
print(f"  Weight = {weight:.2f} N")
print(f"  Lift/Weight = {total_fz/weight:.6f}")

# Tip displacement and twist
tip_z = sc.displacements[11][2]
tip_rx = sc.displacements[11][3]
print(f"  Tip z-disp = {tip_z:.6e} m")
print(f"  Tip twist = {np.degrees(tip_rx):.6f} deg")

# Spanwise displacement
goland_span = []
for nid in sorted(sc.displacements.keys()):
    d = sc.displacements[nid]
    y = m.nodes[nid].xyz_global[1]
    goland_span.append({'nid': nid, 'y': y, 'uz': d[2], 'rx': d[3]})

results['goland_original'] = {
    'alpha_rad': alpha, 'alpha_deg': alpha_deg,
    'total_fz': total_fz, 'weight': weight,
    'tip_z': tip_z, 'tip_twist_deg': np.degrees(tip_rx),
    'mass_kg': total_mass,
    'spanwise': goland_span,
}

# =========== AVM1: Flat Plate CL ===========
print("\n" + "=" * 60)
print("AVM1: Flat Plate AR=6 CL_alpha")
print("=" * 60)
r, m = parse_and_solve_144(os.path.join(AERO_VM_DIR, 'avm1_flat_plate_cl.bdf'))
sc = r.subcases[0]
alpha = sc.trim_variables['ANGLEA']
total_fz = np.sum(sc.aero_forces[:, 2])
q, S = 6125.0, 6.0
CL = total_fz / (q * S)
cl_alpha = CL / alpha if abs(alpha) > 1e-10 else 0
print(f"  ANGLEA = {np.degrees(alpha):.4f} deg")
print(f"  CL = {CL:.6f}")
print(f"  CL_alpha = {cl_alpha:.4f} per rad")
results['AVM1'] = {
    'alpha_deg': np.degrees(alpha), 'CL': CL,
    'cl_alpha': cl_alpha,
}

# =========== AVM2: Goland Enhanced ===========
print("\n" + "=" * 60)
print("AVM2: Goland Wing Enhanced (M=0.5, q=3920)")
print("=" * 60)
r, m = parse_and_solve_144(os.path.join(AERO_VM_DIR, 'avm2_goland_enhanced.bdf'))
sc = r.subcases[0]
alpha = sc.trim_variables['ANGLEA']
total_fz = np.sum(sc.aero_forces[:, 2])
tip_z = sc.displacements[11][2]
tip_rx = sc.displacements[11][3]
total_mass = 0
for eid, elem in m.elements.items():
    if elem.type == "CBAR":
        n1 = m.nodes[elem.node_ids[0]]
        n2 = m.nodes[elem.node_ids[1]]
        L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
        total_mass += elem.property_ref.material_ref.rho * elem.property_ref.A * L
weight = total_mass * 9.81
print(f"  ANGLEA = {np.degrees(alpha):.4f} deg")
print(f"  Total Fz = {total_fz:.2f} N, Weight = {weight:.2f} N")
print(f"  Tip z = {tip_z:.6e} m, Tip twist = {np.degrees(tip_rx):.6f} deg")

goland2_span = []
for nid in sorted(sc.displacements.keys()):
    d = sc.displacements[nid]
    y = m.nodes[nid].xyz_global[1]
    goland2_span.append({'nid': nid, 'y': y, 'uz': d[2], 'rx': d[3]})

results['AVM2'] = {
    'alpha_deg': np.degrees(alpha), 'total_fz': total_fz,
    'weight': weight, 'tip_z': tip_z,
    'tip_twist_deg': np.degrees(tip_rx),
    'spanwise': goland2_span,
}

# =========== AVM3: Rigid Wing ===========
print("\n" + "=" * 60)
print("AVM3: Rigid Wing Trim (AR=4)")
print("=" * 60)
r, m = parse_and_solve_144(os.path.join(AERO_VM_DIR, 'avm3_rigid_wing_trim.bdf'))
sc = r.subcases[0]
alpha = sc.trim_variables['ANGLEA']
total_fz = np.sum(sc.aero_forces[:, 2])
W = 8.0 * 9.81
# Compute VLM CL_alpha
boxes = generate_all_panels(m)
D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
D_inv = np.linalg.inv(D)
w = -np.ones(len(boxes))
gamma = D_inv @ w
dcp = circulation_to_delta_cp(boxes, gamma)
S = 16.0
cl_alpha_vlm = sum(dcp[i] * boxes[i].area for i in range(len(boxes))) / S
alpha_pred = W / (6125 * S * cl_alpha_vlm)
err = abs(alpha - alpha_pred) / abs(alpha_pred) * 100
print(f"  ANGLEA = {np.degrees(alpha):.6f} deg")
print(f"  Total Fz = {total_fz:.2f} N (Weight = {W:.2f} N)")
print(f"  CL_alpha(VLM) = {cl_alpha_vlm:.4f}")
print(f"  alpha predicted = {np.degrees(alpha_pred):.6f} deg")
print(f"  Error = {err:.2f}%")
results['AVM3'] = {
    'alpha_deg': np.degrees(alpha),
    'alpha_predicted_deg': np.degrees(alpha_pred),
    'total_fz': total_fz, 'weight': W,
    'cl_alpha_vlm': cl_alpha_vlm,
    'error_pct': err,
}

# Save all results
output_path = os.path.join(os.path.dirname(__file__), 'verification_data.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
print(f"\n\nResults saved to {output_path}")
print(f"\nTotal test results collected successfully!")
