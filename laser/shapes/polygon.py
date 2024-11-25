from laser.shapes.polyline import Polyline
from laser.color import ColorGradient
from typing import List
import numpy as np


class Polygon(Polyline):

    def __init__(
        self, 
        points: List[np.ndarray], 
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__(points, True, color_gradient, point_density)
