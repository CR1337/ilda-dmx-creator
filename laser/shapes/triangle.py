import numpy as np

from laser.color import ColorGradient
from laser.shapes.polyline import Polyline  


class Triangle(Polyline):

    def __init__(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__([point1, point2, point3], True, color_gradient, point_density)

    def is_point_inside(self, p):
        denominator = (self._points[1][1] - self._points[2][1]) * (self._points[0][0] - self._points[2][0]) + (self._points[2][0] - self._points[1][0]) * (self._points[0][1] - self._points[2][1])
        a = ((self._points[1][1] - self._points[2][1]) * (p[0] - self._points[2][0]) + (self._points[2][0] - self._points[1][0]) * (p[1] - self._points[2][1])) / denominator
        b = ((self._points[2][1] - self._points[0][1]) * (p[0] - self._points[2][0]) + (self._points[0][0] - self._points[2][0]) * (p[1] - self._points[2][1])) / denominator
        c = 1 - a - b
        return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

    def is_line_inside(self, p0, p1):
        if self.is_point_inside(p0) or self.is_point_inside(p1):
            return True
        for q0, q1 in zip(self._points, self._points[1:] + [self._points[0]]):
            if self._do_lines_intersect(p0, p1, q0, q1):
                return True
        return False

    def is_line_outside(self, p0, p1):
        return not self.is_point_inside(p0) and not self.is_point_inside(p1)
        