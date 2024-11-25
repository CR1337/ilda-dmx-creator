import numpy as np

from laser.color import ColorGradient, Color
from laser.shapes.shape import Shape 
from util import ensure_np_array

from typing import List, Tuple


class Point(Shape):

    _point: np.ndarray

    def __init__(
        self, 
        point: np.ndarray, 
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        self._point = point
        self._tangent_noise = None
        super().__init__(color_gradient, point_density)
    
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        return (
            [self._point, self._point], 
            [self._color_gradient.get_color(0), self._color_gradient.get_color(0)], 
            [0.0, 0.0]
        )
    
    @ensure_np_array
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        p0_t = self._inv_transform(p0)
        p1_t = self._inv_transform(p1)
        return (
            (self._point[1] - p0_t[1]) * (p1_t[0] - p0_t[0]) == (p1_t[1] - p0_t[1]) * (self._point[0] - p0_t[0])
            and (
                min(p0_t[0], p1_t[0]) <= self._point[0] <= max(p0_t[0], p1_t[0])
                and min(p0_t[1], p1_t[1]) <= self._point[1] <= max(p0_t[1], p1_t[1])
            )
        )
    
    @ensure_np_array
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_line_inside(p0, p1)
    
    @ensure_np_array
    def signed_distance(self, p: np.ndarray, t: float) -> float:
        p_t = self._inv_transform(p)
        return np.linalg.norm(p_t - self._point)
    
    @ensure_np_array
    def nearest_point(self, p: np.ndarray, t: float) -> np.ndarray:
        return self._transform(self._point)

    def point_by_s(self, s: float, t: float) -> np.ndarray:
        return self._displace(self._transform(self._point), t, s)
    
    def tangent(self, s: float) -> np.ndarray:
        np.array([0.0, 0.0])

    def copy(self) -> Shape:
        point = Point(
            self._point.copy(), 
            self._color_gradient.copy(),
            self._point_density
        )
        point._transformations = [t.copy() for t in self._transformations]
        point._inverse_transformations = [t.copy() for t in self._inverse_transformations]
        point._displacements = self._displacements
        return point 
