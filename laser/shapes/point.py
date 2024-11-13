import numpy as np

from laser.color import ColorGradient
from laser.displacement import Displacement
from laser.shapes.polyline import Polyline  



class Point(Polyline):

    def __init__(
        self, 
        point: np.ndarray, 
        color_gradient: ColorGradient,
        displacement: Displacement | None = None,
        point_density: float | None = None
    ):
        super().__init__([point, point], False, color_gradient, displacement, point_density)

    def get_centroid(self) -> np.ndarray:
        return self._points[0]
    
    def is_point_inside(self, p: np.ndarray) -> bool:
        return np.array_equal(self._points[0], p)
    
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return (
            (self._points[0][1] - p0[1]) * (p1[0] - p0[0]) == (p1[1] - p0[1]) * (self._points[0][0] - p0[0])
            and (
                min(p0[0], p1[0]) <= self._points[0][0] <= max(p0[0], p1[0])
                and min(p0[1], p1[1]) <= self._points[0][1] <= max(p0[1], p1[1])
            )
        )
    
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_line_inside(p0, p1)