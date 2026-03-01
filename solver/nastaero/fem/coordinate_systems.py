"""Coordinate system transformations for FEA."""
from __future__ import annotations
import numpy as np

def build_beam_transform(node1_xyz: np.ndarray, node2_xyz: np.ndarray, v_vector: np.ndarray) -> np.ndarray:
    ex = node2_xyz - node1_xyz
    L = np.linalg.norm(ex)
    if L < 1e-12: raise ValueError("Zero-length beam element")
    ex = ex / L
    v = v_vector.copy()
    ey = np.cross(ex, v)
    ey_norm = np.linalg.norm(ey)
    if ey_norm < 1e-12:
        v = np.array([0., 0., 1.]) if abs(ex[2]) < 0.9 else np.array([1., 0., 0.])
        ey = np.cross(ex, v)
        ey_norm = np.linalg.norm(ey)
    ey = ey / ey_norm
    ez = np.cross(ex, ey)
    return np.array([ex, ey, ez])

def build_transform_12x12(Lambda: np.ndarray) -> np.ndarray:
    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*(i+1), 3*i:3*(i+1)] = Lambda
    return T
