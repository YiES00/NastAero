"""Compare ASCENT-Load FSW stability derivatives against MSC Nastran HA144A.

Half-span model with SYMXZ=1 (image vortices). Forces from the half-model
are doubled for the full aircraft, consistent with Nastran convention.

Known limitation: ASCENT-Load uses PG-corrected VLM (Prandtl-Glauert + horseshoe
vortex), NOT the full DLM kernel. At high Mach (M=0.9), the PG correction
overestimates compressibility effects compared to Nastran's DLM kernel.
"""
import os
import sys
import numpy as np

solver_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, solver_dir)

from ascent_load.bdf.parser import BDFParser
from ascent_load.aero.panel import generate_all_panels, get_box_index_map
from ascent_load.aero.dlm import build_aic_matrix
from ascent_load.solvers.sol144 import solve_trim


def compute_rigid_derivatives(bdf_path):
    """Compute rigid stability derivatives from the BDF model."""
    parser = BDFParser()
    model = parser.parse(bdf_path)
    model.cross_reference()

    boxes = generate_all_panels(model, use_nastran_eid=True)
    box_id_to_index = get_box_index_map(boxes)
    n_boxes = len(boxes)

    refc = model.aeros.refc
    refs = model.aeros.refs
    symxz = model.aeros.symxz

    trim = model.trims[1]
    mach = trim.mach
    q = trim.q

    # Build AIC with symmetry
    D = build_aic_matrix(boxes, mach, reduced_freq=0.0, sym_xz=symxz)
    D_inv = np.linalg.inv(D)

    # Force diagonal
    f_diag = np.zeros(n_boxes)
    for j in range(n_boxes):
        if boxes[j].chord > 1e-12:
            f_diag[j] = 2.0 * q * boxes[j].area / boxes[j].chord
    A_jj = np.diag(f_diag) @ D_inv

    # Normalwash vectors
    w_alpha = -np.ones(n_boxes)

    w_elev = np.zeros(n_boxes)
    for aid, aesurf in model.aesurfs.items():
        if aesurf.label == "ELEV":
            for alid in [aesurf.alid1, aesurf.alid2]:
                if alid > 0 and alid in model.aelists:
                    for box_eid in model.aelists[alid].elements:
                        if box_eid in box_id_to_index:
                            w_elev[box_id_to_index[box_eid]] = -1.0

    # Forces for unit perturbation (half-model)
    F_alpha = A_jj @ w_alpha
    F_elev = A_jj @ w_elev

    # For SYMXZ=1: double forces for full aircraft
    sym_factor = 2.0 if symxz == 1 else 1.0

    Fz_alpha = sym_factor * np.sum(F_alpha)
    Fz_elev = sym_factor * np.sum(F_elev)

    # Moment reference point: GRID 90 at x=15.0
    x_ref = 15.0
    My_alpha = sym_factor * sum(F_alpha[j] * (boxes[j].control_point[0] - x_ref)
                                 for j in range(n_boxes))
    My_elev = sym_factor * sum(F_elev[j] * (boxes[j].control_point[0] - x_ref)
                                for j in range(n_boxes))

    # Non-dimensionalize (stability axes)
    Cz_alpha = -Fz_alpha / (q * refs)
    Cz_elev = -Fz_elev / (q * refs)
    Cm_alpha = -My_alpha / (q * refs * refc)
    Cm_elev = -My_elev / (q * refs * refc)

    return {
        'Cz_alpha': Cz_alpha,
        'Cm_alpha': Cm_alpha,
        'Cz_elev': Cz_elev,
        'Cm_elev': Cm_elev,
        'mach': mach,
        'q': q,
        'n_boxes': n_boxes,
        'refc': refc,
        'refs': refs,
        'symxz': symxz,
    }


def run_trim(bdf_path):
    """Run elastic trim and extract results."""
    parser = BDFParser()
    model = parser.parse(bdf_path)
    results = solve_trim(model)

    sc = results.subcases[0]
    return sc.trim_variables, np.sum(sc.aero_forces[:, 2])


def main():
    bdf_path = os.path.join(os.path.dirname(__file__), 'ha144a.bdf')

    # Published values (MSC Nastran HA144A, Table 7-1, rigid body)
    published = {
        'Cz_alpha': -5.071,
        'Cm_alpha': -2.871,
        'Cz_elev': -0.2461,
        'Cm_elev': 0.5715,
    }
    pub_alpha = 0.169  # rad, approximate rigid trim alpha

    print("=" * 70)
    print("HA144A Forward Swept Wing (FSW) - SOL 144 Validation")
    print("ASCENT-Load vs MSC Nastran Aeroelastic User Guide")
    print("=" * 70)

    # Compute rigid derivatives
    derivs = compute_rigid_derivatives(bdf_path)

    print(f"\nModel: {derivs['n_boxes']} aero boxes, SYMXZ={derivs['symxz']}")
    print(f"Trim: Mach={derivs['mach']}, q={derivs['q']} psf")
    print(f"Ref:  refc={derivs['refc']}, refs={derivs['refs']}")

    # Run elastic trim
    trim_vars, total_fz = run_trim(bdf_path)

    # Compute rigid trim from derivatives
    W = 8000.0
    q = derivs['q']
    refs = derivs['refs']
    Cz_req = -W / (q * refs)

    A_mat = np.array([[derivs['Cz_alpha'], derivs['Cz_elev']],
                      [derivs['Cm_alpha'], derivs['Cm_elev']]])
    rhs = np.array([Cz_req, 0.0])
    try:
        sol = np.linalg.solve(A_mat, rhs)
        alpha_rigid = sol[0]
        delta_rigid = sol[1]
    except np.linalg.LinAlgError:
        alpha_rigid = delta_rigid = None

    # Print results
    print("\n" + "=" * 70)
    print("RIGID STABILITY DERIVATIVES")
    print("=" * 70)
    print(f"{'Derivative':<15} {'Nastran':>12} {'ASCENT-Load':>12} {'Error %':>10}")
    print("-" * 70)

    for name, pub_val in published.items():
        na_val = derivs[name]
        err = (na_val - pub_val) / abs(pub_val) * 100 if abs(pub_val) > 1e-10 else 0.0
        print(f"{name:<15} {pub_val:>12.4f} {na_val:>12.4f} {err:>+10.1f}%")

    print(f"\n{'TRIM SOLUTION (1g level flight, q=40 psf)'}")
    print("-" * 70)
    if alpha_rigid is not None:
        a_err = (alpha_rigid - pub_alpha) / abs(pub_alpha) * 100
        print(f"{'Rigid alpha':<20} {pub_alpha:>10.4f} rad  {alpha_rigid:>10.4f} rad  {a_err:>+8.1f}%")
        print(f"{'':20} {np.degrees(pub_alpha):>10.2f} deg  {np.degrees(alpha_rigid):>10.2f} deg")
    if delta_rigid is not None:
        print(f"{'Rigid delta_e':<20} {'':>10}      {delta_rigid:>10.4f} rad")

    if "ANGLEA" in trim_vars:
        print(f"{'Elastic alpha':<20} {'':>10}      {trim_vars['ANGLEA']:>10.4f} rad")
        print(f"{'':20} {'':>10}      {np.degrees(trim_vars['ANGLEA']):>10.2f} deg")
    if "ELEV" in trim_vars:
        print(f"{'Elastic delta_e':<20} {'':>10}      {trim_vars['ELEV']:>10.4f} rad")

    sym_factor = 2.0 if derivs['symxz'] == 1 else 1.0
    print(f"\n{'Half-model Fz':<20} {'':>10}      {total_fz:>10.1f} lb")
    print(f"{'Full aircraft Fz':<20} {'':>10}      {sym_factor*total_fz:>10.1f} lb")
    print(f"{'Target (nz*W)':<20} {'':>10}      {W:>10.1f} lb")

    print("\n" + "=" * 70)
    print("NOTES")
    print("-" * 70)
    print("1. ASCENT-Load uses PG-corrected VLM, not the full DLM kernel.")
    print("   At M=0.9, PG overestimates compressibility vs DLM.")
    print("   Expected systematic overprediction of ~50% in Cz_alpha.")
    print("2. SYMXZ=1 image vortex implementation added to ASCENT-Load AIC.")
    print("3. Force/moment balance constraints doubled for full aircraft.")
    print("4. Masses scaled: mass*9.81 = weight_lb (ASCENT-Load gravity).")
    print("=" * 70)


if __name__ == "__main__":
    main()
