"""Mass element card parsers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from ..field_parser import nastran_int, nastran_float

@dataclass
class CONM2:
    eid: int = 0; node_id: int = 0; cid: int = -1; mass: float = 0.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    I11: float = 0.0; I21: float = 0.0; I22: float = 0.0
    I31: float = 0.0; I32: float = 0.0; I33: float = 0.0
    @property
    def type(self) -> str: return "CONM2"
    @property
    def node_ids(self) -> List[int]: return [self.node_id]
    @classmethod
    def from_fields(cls, fields: List[str]) -> CONM2:
        m = cls(); m.eid = nastran_int(fields[1]); m.node_id = nastran_int(fields[2])
        # QRG: 공란 = 0 (기본좌표계 오프셋). 명시적 -1은 X1~X3가
        # 오프셋이 아니라 기본좌표계 기준 질량 CG의 절대좌표라는 뜻이라
        # 공란과 구분해서 보존한다.
        m.cid = nastran_int(fields[3]) if fields[3].strip() else 0
        m.mass = nastran_float(fields[4])
        m.offset = np.array([nastran_float(fields[5]) if len(fields)>5 else 0.,
            nastran_float(fields[6]) if len(fields)>6 else 0.,
            nastran_float(fields[7]) if len(fields)>7 else 0.])
        if len(fields) > 9:
            m.I11 = nastran_float(fields[9]) if len(fields)>9 else 0.
            m.I21 = nastran_float(fields[10]) if len(fields)>10 else 0.
            m.I22 = nastran_float(fields[11]) if len(fields)>11 else 0.
            m.I31 = nastran_float(fields[12]) if len(fields)>12 else 0.
            m.I32 = nastran_float(fields[13]) if len(fields)>13 else 0.
            m.I33 = nastran_float(fields[14]) if len(fields)>14 else 0.
        return m
