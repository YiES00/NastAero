"""Coordinate system card parsers.

Supports chained coordinate systems (rid != 0) where the definition
points (origin, z_axis, xz_plane) are given in a reference coordinate
system rather than BASIC.  Call ``resolve_all(coords)`` after parsing
to recursively resolve the chains.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Union, Any
import numpy as np
from ..field_parser import nastran_int, nastran_float


def _gram_schmidt(origin: np.ndarray, z_point: np.ndarray,
                  xz_point: np.ndarray) -> np.ndarray:
    """Gram-Schmidt to build [ex, ey, ez] from three BASIC-frame points."""
    ez = z_point - origin
    norm_ez = np.linalg.norm(ez)
    if norm_ez < 1e-30:
        return np.eye(3)
    ez = ez / norm_ez
    v = xz_point - origin
    ex = v - np.dot(v, ez) * ez
    norm_ex = np.linalg.norm(ex)
    if norm_ex < 1e-30:
        return np.eye(3)
    ex = ex / norm_ex
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez])


@dataclass
class CORD2R:
    """Rectangular coordinate system.

    The three definition points (origin, z_axis, xz_plane) are stored
    as read from the BDF -- they live in the *rid* coordinate system.
    After ``resolve()`` is called the ``origin_basic``, ``transform``
    fields contain BASIC-frame quantities suitable for ``to_global()``.
    """
    cid: int = 0
    rid: int = 0
    # Raw definition points (in rid frame)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    xz_plane: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    # Resolved BASIC-frame data
    origin_basic: np.ndarray = field(default_factory=lambda: np.zeros(3))
    transform: np.ndarray = field(default_factory=lambda: np.eye(3))
    _resolved: bool = False

    @classmethod
    def from_fields(cls, fields: List[str]) -> CORD2R:
        coord = cls()
        coord.cid = nastran_int(fields[1])
        coord.rid = nastran_int(fields[2])
        coord.origin = np.array([nastran_float(fields[3]),
                                  nastran_float(fields[4]),
                                  nastran_float(fields[5])])
        coord.z_axis = np.array([nastran_float(fields[6]),
                                  nastran_float(fields[7]),
                                  nastran_float(fields[8])])
        if len(fields) > 11:
            coord.xz_plane = np.array([nastran_float(fields[9]),
                                        nastran_float(fields[10]),
                                        nastran_float(fields[11])])
        # For rid==0 we can resolve immediately
        if coord.rid == 0:
            coord._resolve_basic()
        return coord

    def _resolve_basic(self) -> None:
        """Resolve when rid==0 (points already in BASIC)."""
        self.origin_basic = self.origin.copy()
        self.transform = _gram_schmidt(self.origin, self.z_axis, self.xz_plane)
        self._resolved = True

    # Legacy compat: _build_transform for tests that call it directly
    def _build_transform(self) -> None:
        self._resolve_basic()

    def resolve(self, coords: Dict[int, Any]) -> None:
        """Resolve definition points through the rid chain to BASIC."""
        if self._resolved:
            return
        # Ensure the reference CS is resolved first
        if self.rid != 0 and self.rid in coords:
            ref = coords[self.rid]
            if not ref._resolved:
                ref.resolve(coords)
        # Transform the three definition points to BASIC
        o_basic = _point_to_basic(self.origin, self.rid, coords)
        z_basic = _point_to_basic(self.z_axis, self.rid, coords)
        xz_basic = _point_to_basic(self.xz_plane, self.rid, coords)
        self.origin_basic = o_basic
        self.transform = _gram_schmidt(o_basic, z_basic, xz_basic)
        self._resolved = True

    def to_global(self, xyz_local: np.ndarray) -> np.ndarray:
        """Transform point from this CS's local Cartesian frame to BASIC."""
        return self.origin_basic + self.transform @ xyz_local

    def to_local(self, xyz_global: np.ndarray) -> np.ndarray:
        """Transform point from BASIC to this CS's local Cartesian frame."""
        return self.transform.T @ (xyz_global - self.origin_basic)


@dataclass
class CORD2C:
    """Cylindrical coordinate system.

    Points are defined as (R, theta_deg, Z) in the local system.
    Conversion: X = R*cos(theta), Y = R*sin(theta), Z = Z  (local Cartesian)
    Then apply the rotation + translation to get BASIC coordinates.

    Field format identical to CORD2R:
    CORD2C CID RID A1 A2 A3 B1 B2 B3
           C1  C2  C3
    """
    cid: int = 0
    rid: int = 0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    xz_plane: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    origin_basic: np.ndarray = field(default_factory=lambda: np.zeros(3))
    transform: np.ndarray = field(default_factory=lambda: np.eye(3))
    _resolved: bool = False

    @classmethod
    def from_fields(cls, fields: List[str]) -> CORD2C:
        coord = cls()
        coord.cid = nastran_int(fields[1])
        coord.rid = nastran_int(fields[2])
        coord.origin = np.array([nastran_float(fields[3]),
                                  nastran_float(fields[4]),
                                  nastran_float(fields[5])])
        coord.z_axis = np.array([nastran_float(fields[6]),
                                  nastran_float(fields[7]),
                                  nastran_float(fields[8])])
        if len(fields) > 11:
            coord.xz_plane = np.array([nastran_float(fields[9]),
                                        nastran_float(fields[10]),
                                        nastran_float(fields[11])])
        if coord.rid == 0:
            coord._resolve_basic()
        return coord

    def _resolve_basic(self) -> None:
        """Resolve when rid==0 (points already in BASIC)."""
        self.origin_basic = self.origin.copy()
        self.transform = _gram_schmidt(self.origin, self.z_axis, self.xz_plane)
        self._resolved = True

    # Legacy compat
    def _build_transform(self) -> None:
        self._resolve_basic()

    def resolve(self, coords: Dict[int, Any]) -> None:
        """Resolve definition points through the rid chain to BASIC."""
        if self._resolved:
            return
        if self.rid != 0 and self.rid in coords:
            ref = coords[self.rid]
            if not ref._resolved:
                ref.resolve(coords)
        o_basic = _point_to_basic(self.origin, self.rid, coords)
        z_basic = _point_to_basic(self.z_axis, self.rid, coords)
        xz_basic = _point_to_basic(self.xz_plane, self.rid, coords)
        self.origin_basic = o_basic
        self.transform = _gram_schmidt(o_basic, z_basic, xz_basic)
        self._resolved = True

    def to_global(self, rtz_local: np.ndarray) -> np.ndarray:
        """Convert cylindrical (R, theta_deg, Z) to BASIC Cartesian."""
        R = rtz_local[0]
        theta = np.radians(rtz_local[1])  # Nastran uses degrees
        Z = rtz_local[2]
        xyz_local = np.array([R * np.cos(theta), R * np.sin(theta), Z])
        return self.origin_basic + self.transform @ xyz_local

    def to_local(self, xyz_global: np.ndarray) -> np.ndarray:
        """Convert BASIC Cartesian to cylindrical (R, theta_deg, Z)."""
        xyz_local = self.transform.T @ (xyz_global - self.origin_basic)
        R = np.sqrt(xyz_local[0]**2 + xyz_local[1]**2)
        theta = np.degrees(np.arctan2(xyz_local[1], xyz_local[0]))
        Z = xyz_local[2]
        return np.array([R, theta, Z])


def _point_to_basic(pt: np.ndarray, rid: int, coords: Dict[int, Any]) -> np.ndarray:
    """Transform a point defined in coordinate system *rid* to BASIC.

    If *rid* is 0 or absent from *coords*, the point is already in BASIC.
    The referenced coordinate must already be resolved (``_resolved``).
    """
    if rid == 0 or rid not in coords:
        return pt.copy()
    ref = coords[rid]
    return ref.to_global(pt)


def resolve_all(coords: Dict[int, Any]) -> None:
    """Resolve all coordinate systems in *coords* through their rid chains.

    Must be called after all coordinate systems have been parsed and before
    any ``to_global()`` calls on coordinate systems with ``rid != 0``.
    """
    for cid, cs in coords.items():
        if not cs._resolved:
            cs.resolve(coords)
