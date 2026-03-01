"""Coordinate system card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from ..field_parser import nastran_int, nastran_float

@dataclass
class CORD2R:
    cid: int = 0
    rid: int = 0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    xz_plane: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    transform: np.ndarray = field(default_factory=lambda: np.eye(3))

    @classmethod
    def from_fields(cls, fields: List[str]) -> CORD2R:
        coord = cls()
        coord.cid = nastran_int(fields[1])
        coord.rid = nastran_int(fields[2])
        coord.origin = np.array([nastran_float(fields[3]), nastran_float(fields[4]), nastran_float(fields[5])])
        coord.z_axis = np.array([nastran_float(fields[6]), nastran_float(fields[7]), nastran_float(fields[8])])
        if len(fields) > 11:
            coord.xz_plane = np.array([nastran_float(fields[9]), nastran_float(fields[10]), nastran_float(fields[11])])
        coord._build_transform()
        return coord

    def _build_transform(self) -> None:
        ez = self.z_axis - self.origin
        ez = ez / np.linalg.norm(ez)
        v = self.xz_plane - self.origin
        ex = v - np.dot(v, ez) * ez
        ex = ex / np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        self.transform = np.column_stack([ex, ey, ez])

    def to_global(self, xyz_local: np.ndarray) -> np.ndarray:
        return self.origin + self.transform @ xyz_local

    def to_local(self, xyz_global: np.ndarray) -> np.ndarray:
        return self.transform.T @ (xyz_global - self.origin)
