"""Load card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from ..field_parser import nastran_int, nastran_float

@dataclass
class FORCE:
    sid: int = 0; node_id: int = 0; cid: int = 0; mag: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    @property
    def type(self) -> str: return "FORCE"
    @classmethod
    def from_fields(cls, fields: List[str]) -> FORCE:
        l = cls(); l.sid = nastran_int(fields[1]); l.node_id = nastran_int(fields[2])
        l.cid = nastran_int(fields[3]); l.mag = nastran_float(fields[4])
        l.direction = np.array([nastran_float(fields[5]), nastran_float(fields[6]),
            nastran_float(fields[7]) if len(fields) > 7 else 0.0])
        return l
    def get_force_vector(self) -> np.ndarray:
        n = np.linalg.norm(self.direction)
        return self.mag * self.direction / n if n > 0 else np.zeros(3)

@dataclass
class MOMENT:
    sid: int = 0; node_id: int = 0; cid: int = 0; mag: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    @property
    def type(self) -> str: return "MOMENT"
    @classmethod
    def from_fields(cls, fields: List[str]) -> MOMENT:
        l = cls(); l.sid = nastran_int(fields[1]); l.node_id = nastran_int(fields[2])
        l.cid = nastran_int(fields[3]); l.mag = nastran_float(fields[4])
        l.direction = np.array([nastran_float(fields[5]), nastran_float(fields[6]),
            nastran_float(fields[7]) if len(fields) > 7 else 0.0])
        return l
    def get_moment_vector(self) -> np.ndarray:
        n = np.linalg.norm(self.direction)
        return self.mag * self.direction / n if n > 0 else np.zeros(3)

@dataclass
class GRAV:
    sid: int = 0; cid: int = 0; scale: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    @property
    def type(self) -> str: return "GRAV"
    @classmethod
    def from_fields(cls, fields: List[str]) -> GRAV:
        l = cls(); l.sid = nastran_int(fields[1]); l.cid = nastran_int(fields[2])
        l.scale = nastran_float(fields[3])
        l.direction = np.array([nastran_float(fields[4]), nastran_float(fields[5]),
            nastran_float(fields[6]) if len(fields) > 6 else 0.0])
        return l
    def get_acceleration_vector(self) -> np.ndarray:
        n = np.linalg.norm(self.direction)
        return self.scale * self.direction / n if n > 0 else np.zeros(3)

@dataclass
class LoadCombination:
    sid: int = 0; scale: float = 1.0
    scale_factors: List[float] = field(default_factory=list)
    load_ids: List[int] = field(default_factory=list)
    @property
    def type(self) -> str: return "LOAD"
    @classmethod
    def from_fields(cls, fields: List[str]) -> LoadCombination:
        lc = cls(); lc.sid = nastran_int(fields[1]); lc.scale = nastran_float(fields[2])
        i = 3
        while i + 1 < len(fields):
            sf_str = fields[i].strip(); lid_str = fields[i + 1].strip()
            if not sf_str and not lid_str: break
            if sf_str and lid_str:
                lc.scale_factors.append(nastran_float(fields[i]))
                lc.load_ids.append(nastran_int(fields[i + 1]))
            i += 2
        return lc
