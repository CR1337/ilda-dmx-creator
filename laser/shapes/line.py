import numpy as np

from laser.color import ColorGradient
from laser.displacement import Displacement
from laser.shapes.polyline import Polyline  


class Line(Polyline):

    def __init__(
        self, 
        start: np.ndarray, 
        end: np.ndarray, 
        color_gradient: ColorGradient,
        displacement: Displacement | None = None,
        point_density: float | None = None
    ):
        super().__init__([start, end], False, color_gradient, displacement, point_density)

    def get_centroid(self) -> np.ndarray:
        return 0.5 * (self._points[0] + self._points[1])
    
    def is_point_inside(self, p: np.ndarray) -> bool:
        return (
            (p[1] - self._points[0][1]) * (self._points[1][0] - self._points[0][0]) == (self._points[1][1] - self._points[0][1]) * (p[0] - self._points[0][0])
            and (
                min(self._points[0][0], self._points[1][0]) <= p[0] <= max(self._points[0][0], self._points[1][0])
                and min(self._points[0][1], self._points[1][1]) <= p[1] <= max(self._points[0][1], self._points[1][1])
            )
        )
    
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return self._do_lines_intersect(p0, p1, self._points[0], self._points[1])

    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_line_inside(p0, p1)
    