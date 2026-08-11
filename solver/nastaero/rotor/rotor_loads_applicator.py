"""Convert rotor hub loads to structural FORCE/MOMENT cards.

Transforms RotorLoads (in rotor shaft frame) to forces and moments
at the hub structural node in the global coordinate system.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .bemt_solver import RotorLoads
from .rotor_config import RotorDef, RotationDir


def rotor_loads_to_nodal_forces(rotor: RotorDef,
                                 loads: RotorLoads,
                                 ) -> Dict[int, np.ndarray]:
    """Convert rotor loads to nodal force vector at hub node.

    Transforms thrust, torque, H-force from shaft frame to global
    coordinates and returns as a nodal force dictionary compatible
    with the existing loads analysis framework.

    Parameters
    ----------
    rotor : RotorDef
        Rotor definition with position, axis, and hub node ID.
    loads : RotorLoads
        BEMT-computed rotor loads in shaft frame.

    Returns
    -------
    dict
        {hub_node_id: ndarray(6)} where value = [Fx, Fy, Fz, Mx, My, Mz]
        in global coordinates.
    """
    # Build rotation matrix from shaft frame to global
    # Shaft frame: z_shaft = shaft_axis (thrust direction)
    # Need to construct full orthonormal basis
    z_shaft = rotor.shaft_axis / np.linalg.norm(rotor.shaft_axis)

    # Choose an arbitrary perpendicular vector for x_shaft
    if abs(z_shaft[0]) < 0.9:
        v = np.array([1., 0., 0.])
    else:
        v = np.array([0., 1., 0.])
    x_shaft = np.cross(z_shaft, v)
    x_shaft /= np.linalg.norm(x_shaft)
    y_shaft = np.cross(z_shaft, x_shaft)

    # Rotation matrix: columns are shaft frame axes in global coords
    R = np.column_stack([x_shaft, y_shaft, z_shaft])

    # Forces in shaft frame
    # Thrust along z_shaft, H-force along x_shaft
    F_shaft = np.array([loads.H_force, 0.0, loads.thrust])

    # Moments in shaft frame
    # Torque about z_shaft (reaction torque opposes rotation)
    torque_sign = 1.0 if rotor.rotation_dir == RotationDir.CW else -1.0
    M_shaft = np.array([loads.roll_moment,
                        loads.pitch_moment,
                        -torque_sign * loads.torque])

    # Transform to global
    F_global = R @ F_shaft
    M_global = R @ M_shaft

    # Combine into 6-DOF force vector
    force_vec = np.zeros(6)
    force_vec[:3] = F_global
    force_vec[3:] = M_global

    return {rotor.hub_node_id: force_vec}


def all_rotor_forces(rotors: List[RotorDef],
                     loads_map: Dict[int, RotorLoads],
                     ) -> Dict[int, np.ndarray]:
    """Combine forces from all rotors into nodal force dictionary.

    Parameters
    ----------
    rotors : list of RotorDef
        All rotor definitions.
    loads_map : dict
        {rotor_id: RotorLoads} for each active rotor.

    Returns
    -------
    dict
        {node_id: ndarray(6)} combined forces from all rotors.
    """
    combined: Dict[int, np.ndarray] = {}

    for rotor in rotors:
        if rotor.rotor_id not in loads_map:
            continue
        loads = loads_map[rotor.rotor_id]
        nodal = rotor_loads_to_nodal_forces(rotor, loads)

        for nid, fvec in nodal.items():
            if nid in combined:
                combined[nid] = combined[nid] + fvec
            else:
                combined[nid] = fvec.copy()

    return combined


def generate_force_moment_cards(nodal_forces: Dict[int, np.ndarray],
                                 load_set_id: int,
                                 ) -> List[str]:
    """Generate Nastran FORCE/MOMENT bulk data cards from nodal forces.

    Parameters
    ----------
    nodal_forces : dict
        {node_id: ndarray(6)} forces and moments at each node.
    load_set_id : int
        Load set ID for the FORCE/MOMENT cards.

    Returns
    -------
    list of str
        BDF card strings (8-character fixed format).
    """
    cards = []
    for nid, fvec in nodal_forces.items():
        fx, fy, fz = fvec[0], fvec[1], fvec[2]
        mx, my, mz = fvec[3], fvec[4], fvec[5]

        f_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        if f_mag > 1e-10:
            n1, n2, n3 = fx / f_mag, fy / f_mag, fz / f_mag
            cards.append(
                f"FORCE   {load_set_id:8d}{nid:8d}       0"
                f"{f_mag:8.1f}{n1:8.4f}{n2:8.4f}{n3:8.4f}")

        m_mag = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        if m_mag > 1e-10:
            n1, n2, n3 = mx / m_mag, my / m_mag, mz / m_mag
            cards.append(
                f"MOMENT  {load_set_id:8d}{nid:8d}       0"
                f"{m_mag:8.1f}{n1:8.4f}{n2:8.4f}{n3:8.4f}")

    return cards
