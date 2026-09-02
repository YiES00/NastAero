"""Base element interface."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
import numpy as np


def drill_scale() -> float:
    """드릴링 정칙화 스케일 훅 (연구용, 기본 1.0).

    ASCENT_DRILL_SCALE 환경변수로 alpha_drill = E*t*A*1e-6 항을
    일괄 스케일한다. 발산 스펙트럼의 정칙화 의존성 연구(T1-E1)
    전용이며 생산 해석에서는 건드리지 않는다.
    """
    try:
        return float(os.environ.get("ASCENT_DRILL_SCALE", "1.0"))
    except ValueError:
        return 1.0


class BaseElement(ABC):
    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray: ...
    @abstractmethod
    def mass_matrix(self) -> np.ndarray: ...
    @abstractmethod
    def dof_count(self) -> int: ...
