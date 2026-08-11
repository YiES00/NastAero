"""Base element interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BaseElement(ABC):
    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray: ...
    @abstractmethod
    def mass_matrix(self) -> np.ndarray: ...
    @abstractmethod
    def dof_count(self) -> int: ...
