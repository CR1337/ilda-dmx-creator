from dataclasses import dataclass
from typing import Tuple


@dataclass
class Projector:

    number: int
    angle: float
    kpps: Tuple[int, float]
