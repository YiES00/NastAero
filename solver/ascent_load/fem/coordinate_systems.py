"""Coordinate system transformations for FEA."""
from __future__ import annotations
import numpy as np

def build_beam_transform(node1_xyz: np.ndarray, node2_xyz: np.ndarray, v_vector: np.ndarray) -> np.ndarray:
    ex = node2_xyz - node1_xyz
    L = np.linalg.norm(ex)
    if L < 1e-12: raise ValueError("Zero-length beam element")
    ex = ex / L
    # MSC 규약: plane 1은 (x_elem, v) 평면 — z = x×v, y = z×x 로 v가
    # local y(플레인 1) 방향에 놓여 I1이 plane-1 굽힘을 지배한다.
    # (이전 구현은 y = x×v 라서 v가 local -z가 되어 I1/I2 평면이 뒤바뀜.)
    v = v_vector.copy()
    ez = np.cross(ex, v)
    ez_norm = np.linalg.norm(ez)
    if ez_norm < 1e-12:
        v = np.array([0., 0., 1.]) if abs(ex[2]) < 0.9 else np.array([1., 0., 0.])
        ez = np.cross(ex, v)
        ez_norm = np.linalg.norm(ez)
    ez = ez / ez_norm
    ey = np.cross(ez, ex)
    return np.array([ex, ey, ez])

def build_transform_12x12(Lambda: np.ndarray) -> np.ndarray:
    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*(i+1), 3*i:3*(i+1)] = Lambda
    return T
