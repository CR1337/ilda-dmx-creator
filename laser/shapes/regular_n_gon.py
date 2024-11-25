from laser.shapes.polygon import Polygon
from laser.color import ColorGradient
import numpy as np


class RegularNGon(Polygon):

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        n: int,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        points = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append(np.array([x, y]))
        super().__init__(points, color_gradient, point_density)
