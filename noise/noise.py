from random import Random
from math import sin, cos, pi
from abc import ABC, abstractmethod
from typing import Tuple


class Noise(ABC):
    
    _seed: int | None

    def __init__(self, seed: int | None = None):
        self._seed = seed

    def _random_value(self, x: float, y: float) -> float:
        return Random(self._seed + hash(x) + hash(y)).random()
    
    def _get_xy(self, t: float) -> Tuple[float, float]:
        angle = t * 2.0 * pi
        return cos(angle), sin(angle)
    
    @abstractmethod
    def _get_value(self, x: float, y: float) -> float:
        raise NotImplementedError("@abstractmethod _get_value")

    def get_value(self, t: float) -> float:
        x, y = self._get_xy(t)
        return self._get_value(x, y)
