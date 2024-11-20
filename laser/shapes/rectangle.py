import numpy as np

from laser.color import ColorGradient
from laser.shapes.polyline import Polyline  


class Rectangle(Polyline):

    def __init__(
        self,
        top_left: np.ndarray,
        bottom_right: np.ndarray,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        top_right = np.array([bottom_right[0], top_left[1]])
        bottom_left = np.array([top_left[0], bottom_right[1]])
        points = [top_left, bottom_left, bottom_right, top_right]
        super().__init__(points, True, color_gradient, point_density)

    def is_point_inside(self, p: np.ndarray) -> bool:
        return (
            self._points[0][0] <= p[0] <= self._points[2][0]
            and self._points[0][1] <= p[1] <= self._points[2][1]
        )

    def is_line_inside(self, p0, p1) -> bool:
        if self.is_point_inside(p0) or self.is_point_inside(p1):
            return True
        for q0, q1 in zip(self._points, self._points[1:] + [self._points[0]]):
            if self._do_lines_intersect(p0, p1, q0, q1):
                return True
        return False
    
    def is_line_outside(self, p0, p1) -> bool:
        return self.is_point_inside(p0) and self.is_point_inside(p1)
    
    def signed_distance(self, p: np.ndarray) -> float:
        ...  # TODO
    
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        ...  # TODO
    