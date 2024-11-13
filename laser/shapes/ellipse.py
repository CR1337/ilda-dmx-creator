import numpy as np
from math import pi, sqrt
from typing import List, Tuple

from laser.color import ColorGradient, Color
from laser.displacement import Displacement
from laser.shapes.shape import Shape


class Ellipse(Shape):

    _center: np.ndarray
    _radii: np.ndarray

    def __init__(
        self,
        center: np.ndarray,
        radii: np.ndarray,
        color_gradient: ColorGradient,
        displacement: Displacement | None = None,
        point_density: float | None = None
    ):
        super().__init__(color_gradient, displacement, point_density)
        self._center = center
        self._radii = radii

    def get_centroid(self) -> np.ndarray:
        return self._center
    
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        cirumference = pi * (3.0 * (self._radii[0] + self._radii[1]) - sqrt((3 * self._radii[0] + self._radii[1]) * (self._radii[0] + 3.0 * self._radii[1])))

        spacing = 1.0 / (self._point_density * self.ILDA_RESOLUTION)
        n_points = int(round(cirumference / spacing))
        
        points = []
        colors = []
        ts = []
        accumulated_length = 0.0
        segment_length = 0.0
        previous_point = None

        for i in range(n_points):
            angle = (2.0 * pi * i) / n_points
            x = self._center[0] + self._radii[0] * np.cos(angle)
            y = self._center[1] + self._radii[1] * np.sin(angle)

            if previous_point is not None:
                dx = x - previous_point[0]
                dy = y - previous_point[1]
                segment_length = sqrt(dx * dx + dy * dy)
                accumulated_length += segment_length

            t = accumulated_length / cirumference
            previous_point = np.array([x, y])

            points.append(np.array([x, y]))
            colors.append(self._color_gradient.get_color(t))
            ts.append(t)

        return points, colors, ts
    
    def is_point_inside(self, p: np.ndarray) -> bool:
        return (
            (p[0] - self._center[0]) ** 2 / self._radii[0] ** 2 
            + (p[1] - self._center[1]) ** 2 / self._radii[1] ** 2 
            <= 1
        )
    
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        # check if endpoints are inside the ellipse
        if self.is_point_inside(p0) or self.is_point_inside(p1):
            return True
        
        # check if the line intersects the ellipse
        A = (p1[0] - p0[0]) ** 2 / self._radii[0] ** 2 + (p1[1] - p0[1]) ** 2 / self._radii[1] ** 2
        B = 2 * p0[0] * (p1[0] - p0[0]) / self._radii[0] ** 2 + 2 * p0[1] * (p1[1] - p0[1]) / self._radii[1] ** 2
        C = p0[0] ** 2 / self._radii[0] ** 2 + p0[1] ** 2 / self._radii[1] ** 2 - 1

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return False
        
        t1 = (-B + sqrt(discriminant)) / (2 * A)
        t2 = (-B - sqrt(discriminant)) / (2 * A)

        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
        
        return False
    
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_point_inside(p0) or not self.is_point_inside(p1)