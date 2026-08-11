"""HA144A Forward Swept Wing - Stability Derivative Validation.

Runs NastAero SOL 144 on the FSW model and computes rigid stability
derivatives for comparison against MSC Nastran Aeroelastic Manual Table 7-1.

Reference: MSC Nastran Aeroelastic Analysis User's Guide, Example HA144A

Known limitation: the SYMXZ=1 image vortex implementation has a sign issue
that affects symmetric half-model results. Results are shown both with
sym_xz=1 (current code) and sym_xz=0 (no image, better for canard).
"""
import sys
import os
import numpy as np

sys.path.insert(0, '.')

from nastaero.bdf.parser import BDFParser
from nastaero.solvers.sol144 import (
    _build_shared_data, _trim_variable_normalwash, solve_trim,
    _compute_cg_x
)
from nastaero.aero.panel import generate_all_panels, get_box_index_map
from nastaero.aero.dlm import build_aic_matrix
from nastaero.config import logger
import logging

logger.setLevel(logging.WARNING)

# ============================================================
# 1. Parse the FSW BDF model
# ============================================================
BDF_PATH = os.path.join("tests", "validation", "FSW", "ha144a.bdf")
print(f"Parsing BDF: {BDF_PATH}")
parser = BDFParser()
model = parser.parse(BDF_PATH)
model.cross_reference()

print(f"  SOL = {model.sol}")
print(f"  Nodes: {len(model.nodes)}")
print(f"  Elements: {len(model.elements)}")
print(f"  CAERO1 panels: {list(model.caero_panels.keys())}")
print(f"  AEROS: refc={model.aeros.refc}, refb={model.aeros.refb}, "
      f"refs={model.aeros.refs}, symxz={model.aeros.symxz}")

# Verify wing sweep
wing = model.caero_panels[1100]
dx = wing.p1[0] - wing.p4[0]
dy = wing.p4[1] - wing.p1[1]
sweep_deg = np.degrees(np.arctan2(dx, dy))
print(f"  Wing LE sweep: {sweep_deg:.1f} deg forward")
print()

# ============================================================
# 2. Generate panels and build AIC
# ============================================================
boxes = generate_all_panels(model, use_nastran_eid=True)
box_idx = get_box_index_map(boxes)
n_boxes = len(boxes)

mach = 0.9
REFS = model.aeros.refs   # 200.0
REFC = model.aeros.refc   # 10.0
REFB = model.aeros.refb   # 40.0
x_ref = 15.0  # support point (grid 90), moment reference for derivatives

# CG for pitch rate reference
cg_x = _compute_cg_x(model)
print(f"  Aero boxes: {n_boxes}")
print(f"  CG_x = {cg_x:.4f} ft")
print(f"  X_ref (support point) = {x_ref}")
print()

# Build AIC with SYMXZ=1 (includes image vortices)
D_sym = build_aic_matrix(boxes, mach, reduced_freq=0.0, sym_xz=1, kernel='dlm')
D_inv_sym = np.linalg.inv(D_sym)

# Build AIC without image (sym_xz=0)
D_nosym = build_aic_matrix(boxes, mach, reduced_freq=0.0, sym_xz=0, kernel='dlm')
D_inv_nosym = np.linalg.inv(D_nosym)

# ============================================================
# 3. Compute RIGID stability derivatives
# ============================================================
# For SYMXZ=1 half-model: Cz = Fz_half / REFS (no factor of 2)
# This matches Nastran's convention where REFS accounts for symmetry.

print("=" * 72)
print("RIGID STABILITY DERIVATIVES (M=0.9)")
print(f"  REFS={REFS}, REFC={REFC}, REFB={REFB}, x_ref={x_ref}")
print("=" * 72)
print()

nastran_rigid = {
    "Cz_alpha": -5.071,
    "Cm_alpha": -2.871,
    "Cz_de":    -0.2461,
    "Cm_de":     0.5715,
    "Cz_q":    -12.074,
    "Cm_q":     -9.954,
    "Cz_0":    -0.008421,
    "Cm_0":    -0.006008,
}

def compute_derivs(D_inv, label_suffix=""):
    """Compute stability derivatives from AIC inverse."""
    derivs = {}
    for var_label, deriv_cz, deriv_cm in [
        ("ANGLEA", "Cz_alpha", "Cm_alpha"),
        ("ELEV",   "Cz_de",    "Cm_de"),
        ("PITCH",  "Cz_q",     "Cm_q"),
    ]:
        w = _trim_variable_normalwash(var_label, boxes, box_idx, model)
        gamma = D_inv @ w
        force_nd = np.array([2*gamma[j]*boxes[j].area/boxes[j].chord
                             for j in range(n_boxes)])
        Fz_half = np.sum(force_nd)
        My_half = np.sum(force_nd * np.array([boxes[j].control_point[0] - x_ref
                                               for j in range(n_boxes)]))
        derivs[deriv_cz] = -Fz_half / REFS
        derivs[deriv_cm] = -My_half / (REFS * REFC)
    derivs["Cz_0"] = 0.0
    derivs["Cm_0"] = 0.0
    return derivs

derivs_sym = compute_derivs(D_inv_sym)
derivs_nosym = compute_derivs(D_inv_nosym)

print(f"{'Derivative':<12s} {'Nastran':>10s} {'No Image':>10s} {'Err%':>8s} {'W/Image':>10s} {'Err%':>8s}")
print("-" * 62)
for key in ["Cz_alpha", "Cm_alpha", "Cz_de", "Cm_de", "Cz_q", "Cm_q", "Cz_0", "Cm_0"]:
    nast = nastran_rigid[key]
    nosym = derivs_nosym[key]
    wsym = derivs_sym[key]
    err_nosym = 100*(nosym - nast)/abs(nast) if abs(nast) > 1e-10 else 0.0
    err_wsym = 100*(wsym - nast)/abs(nast) if abs(nast) > 1e-10 else 0.0
    print(f"  {key:<12s} {nast:10.4f} {nosym:10.4f} {err_nosym:7.1f}% {wsym:10.4f} {err_wsym:7.1f}%")
print()

# ============================================================
# 4. Run SOL 144 trim
# ============================================================
print("=" * 72)
print("SOL 144 TRIM ANALYSIS (M=0.9, q=40 psf)")
print("=" * 72)

logger.setLevel(logging.INFO)
results = solve_trim(model)
logger.setLevel(logging.WARNING)

sc = results.subcases[0]
print()
print("Trim variables:")
for label, val in sc.trim_variables.items():
    if label in ("ANGLEA", "ELEV"):
        print(f"  {label:12s} = {val:.6f} rad = {np.degrees(val):.4f} deg")
    else:
        print(f"  {label:12s} = {val:.6f}")

total_fz = np.sum(sc.aero_forces[:, 2])
print(f"  Total Fz    = {total_fz:.2f} (half-model)")
print()

nastran_trim = {
    "ANGLEA": 0.169191,
    "ELEV":   0.492457,
}

print(f"{'Variable':<12s} {'Nastran':>12s} {'NastAero':>12s} {'Error %':>10s} {'Units':>8s}")
print("-" * 56)
for key in ["ANGLEA", "ELEV"]:
    nast_val = nastran_trim[key]
    nastaero_val = sc.trim_variables.get(key, 0.0)
    err_pct = 100*(nastaero_val - nast_val)/abs(nast_val) if abs(nast_val) > 1e-10 else 0.0
    print(f"  {key:<12s} {nast_val:12.6f} {nastaero_val:12.6f} {err_pct:10.2f}% {'rad':>8s}")
    print(f"  {'':12s} {np.degrees(nast_val):12.4f} {np.degrees(nastaero_val):12.4f} {'':>10s} {'deg':>8s}")
print()

# ============================================================
# 5. Summary
# ============================================================
print("=" * 72)
print("SUMMARY")
print("=" * 72)
print()
print("Rigid stability derivatives (no image, best accuracy):")
print(f"  Cz_alpha: NastAero={derivs_nosym['Cz_alpha']:.4f}, Nastran={nastran_rigid['Cz_alpha']:.4f}, "
      f"err={100*(derivs_nosym['Cz_alpha']-nastran_rigid['Cz_alpha'])/abs(nastran_rigid['Cz_alpha']):.1f}%")
print(f"  Cm_alpha: NastAero={derivs_nosym['Cm_alpha']:.4f}, Nastran={nastran_rigid['Cm_alpha']:.4f}, "
      f"err={100*(derivs_nosym['Cm_alpha']-nastran_rigid['Cm_alpha'])/abs(nastran_rigid['Cm_alpha']):.1f}%")
print(f"  Cz_de:    NastAero={derivs_nosym['Cz_de']:.4f}, Nastran={nastran_rigid['Cz_de']:.4f}, "
      f"err={100*(derivs_nosym['Cz_de']-nastran_rigid['Cz_de'])/abs(nastran_rigid['Cz_de']):.1f}%")
print(f"  Cm_de:    NastAero={derivs_nosym['Cm_de']:.4f}, Nastran={nastran_rigid['Cm_de']:.4f}, "
      f"err={100*(derivs_nosym['Cm_de']-nastran_rigid['Cm_de'])/abs(nastran_rigid['Cm_de']):.1f}%")
print(f"  Cz_q:     NastAero={derivs_nosym['Cz_q']:.4f}, Nastran={nastran_rigid['Cz_q']:.4f}, "
      f"err={100*(derivs_nosym['Cz_q']-nastran_rigid['Cz_q'])/abs(nastran_rigid['Cz_q']):.1f}%")
print(f"  Cm_q:     NastAero={derivs_nosym['Cm_q']:.4f}, Nastran={nastran_rigid['Cm_q']:.4f}, "
      f"err={100*(derivs_nosym['Cm_q']-nastran_rigid['Cm_q'])/abs(nastran_rigid['Cm_q']):.1f}%")
print()
print("Notes:")
print("  - SYMXZ=1 image vortex has a known sign issue affecting half-model results")
print("  - 'No Image' results omit the image contribution (conservative approximation)")
print("  - 'W/Image' uses the current image implementation")
print("  - Trim solution uses the coupled aeroelastic solver with SYMXZ=1")
print("  - Trim discrepancy is partly due to image vortex issue and partly due")
print("    to elastic deformation effects in the simplified structural model")
