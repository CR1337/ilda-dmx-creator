import numpy as np

from laser.color import ColorGradient
from laser.shapes.polyline import Polyline  
from util import ensure_np_array


class Line(Polyline):

    def __init__(
        self, 
        start: np.ndarray, 
        end: np.ndarray, 
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__([start, end], False, color_gradient, point_density)
    
    @ensure_np_array
    def signed_distance(self, p: np.ndarray) -> float:
        return np.linalg.norm(p - self.nearest_point(p))
    
    @ensure_np_array
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        p_t = self._inv_transform(p)
        p1 = self._points[0]
        p2 = self._points[1]
        t = (p_t - p1).dot(p2 - p1) / np.linalg.norm(p2 - p1) ** 2
        t = np.clip(t, 0.0, 1.0)
        q = p1 + t * (p2 - p1)
        return q

    def point_by_s(self, s: float, t: float) -> np.ndarray:
        point = self._points[0] + s * (self._points[1] - self._points[0])
        point = self._displace(self._transform(point), t, s)
        return point
    
    def tangent(self, s: float) -> np.ndarray:
        tangent_vector =  self.end - self.start
        tangent_vector = self._transform(tangent_vector)
        return tangent_vector / np.linalg.norm(tangent_vector)

    @property
    def center(self) -> np.ndarray:
        return (self._points[0] + self._points[1]) / 2
    
    @property
    def start(self) -> np.ndarray:
        return self._points[0]
    
    @property
    def end(self) -> np.ndarray:
        return self._points[1]
    