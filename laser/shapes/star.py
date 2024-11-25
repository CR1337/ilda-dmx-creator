from laser.shapes.polygon import Polygon
from laser.color import ColorGradient
import numpy as np


class Star(Polygon):

    def __init__(
        self,
        center: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        n: int,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        points = []
        for i in range(2 * n):
            radius = inner_radius if i % 2 == 0 else outer_radius
            angle = 2 * np.pi * i / (2 * n)
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append(np.array([x, y]))
        super().__init__(points, color_gradient, point_density)
        