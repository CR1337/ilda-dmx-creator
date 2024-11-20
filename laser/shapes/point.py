import numpy as np

from laser.color import ColorGradient, Color
from noise import Noise1D, Noise2D, Noise3D
from laser.shapes.shape import Shape 

from typing import List, Tuple




class Point(Shape):

    _point: np.ndarray
    _tangent_noise: Noise1D | None

    def __init__(
        self, 
        point: np.ndarray, 
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        self._point = point
        self._tangent_noise = None
        super().__init__(color_gradient, point_density)

    def get_centroid(self) -> np.ndarray:
        return self._point
    
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        return (
            [self._point, self._point], 
            [self._color_gradient.get_color(0), self._color_gradient.get_color(0)], 
            [0.0, 0.0]
        )
    
    def is_point_inside(self, p: np.ndarray) -> bool:
        return np.array_equal(self._point, p)
    
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return (
            (self._point[1] - p0[1]) * (p1[0] - p0[0]) == (p1[1] - p0[1]) * (self._point[0] - p0[0])
            and (
                min(p0[0], p1[0]) <= self._point[0] <= max(p0[0], p1[0])
                and min(p0[1], p1[1]) <= self._point[1] <= max(p0[1], p1[1])
            )
        )
    
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_line_inside(p0, p1)
    
    def _apply_displacements(self, point_index: int, timestamp: float, points: np.ndarray) -> np.ndarray:
        if self._tangent_noise is None:
            return points[point_index]
        tangent_angle = self._tangent_noise.get_value(np.array([]), timestamp)
        tangent = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])
        normal = np.array([-tangent[1], tangent[0]])
        point = points[point_index]
        displacement = np.array([0.0, 0.0])

        for noise, swizzle in self._displacements:
            if isinstance(noise, Noise2D):
                displacement += noise.get_value(point, None, swizzle=swizzle) * normal
            elif isinstance(noise, Noise3D):
                displacement += noise.get_value(point, timestamp, swizzle=swizzle) * normal

        return points[point_index] + displacement
    
    def displace(self, noise: Noise2D | Noise3D, tangent_frequency: float = 1.0, tangent_amplitude: float = 1.0, *, swizzle: str | None = None) -> Shape:
        self._tangent_noise = Noise1D.circle(np.array([tangent_frequency]), tangent_amplitude)
        return super().displace(noise, swizzle=swizzle)