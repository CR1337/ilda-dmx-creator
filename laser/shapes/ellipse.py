import numpy as np
from math import pi, sqrt
from typing import List, Tuple
from scipy.optimize import minimize
from util import np_cache, ensure_np_array

from laser.color import ColorGradient, Color
from laser.shapes.shape import Shape
from laser.shapes.line import Line


class Ellipse(Shape):

    _center: np.ndarray
    _radii: np.ndarray

    def __init__(
        self,
        center: np.ndarray,
        radii: np.ndarray,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__(color_gradient, point_density)
        self._center = center
        self._radii = radii

    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        cirumference = pi * (3.0 * (self._radii[0] + self._radii[1]) - sqrt((3 * self._radii[0] + self._radii[1]) * (self._radii[0] + 3.0 * self._radii[1])))

        spacing = 1.0 / (self._point_density * self.ILDX_RESOLUTION)
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

        points.append(points[0])
        colors.append(colors[0])
        ts.append(1)

        return points, colors, ts
    
    @ensure_np_array
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        # check if endpoints are inside the ellipse
        if self.is_point_inside(p0) or self.is_point_inside(p1):
            return True
        
        p0_t = self._inv_transform(p0)
        p1_t = self._inv_transform(p1)
        
        # check if the line intersects the ellipse
        A = (p1_t[0] - p0_t[0]) ** 2 / self._radii[0] ** 2 + (p1_t[1] - p0_t[1]) ** 2 / self._radii[1] ** 2
        B = 2 * p0_t[0] * (p1_t[0] - p0_t[0]) / self._radii[0] ** 2 + 2 * p0_t[1] * (p1_t[1] - p0_t[1]) / self._radii[1] ** 2
        C = p0_t[0] ** 2 / self._radii[0] ** 2 + p0_t[1] ** 2 / self._radii[1] ** 2 - 1

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return False
        
        t1 = (-B + sqrt(discriminant)) / (2 * A)
        t2 = (-B - sqrt(discriminant)) / (2 * A)

        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
        
        return False
    
    @ensure_np_array
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_point_inside(p0) or not self.is_point_inside(p1)
    
    @ensure_np_array
    def _normalized_point_and_radius(self, p: np.ndarray) -> Tuple[np.ndarray, float]:
        p_norm = p - self._center / self._radii
        r = np.linalg.norm(p_norm)
        return p_norm, r

    @ensure_np_array
    def signed_distance(self, p: np.ndarray) -> float:
        p_t = self._inv_transform(p)
        p_r = p_t - self._center
        q = self.nearest_point(p)
        return np.sign((p_r[0] / self._radii[0]) ** 2 + (p_r[1] / self._radii[1]) ** 2 - 1) * np.linalg.norm(p_r - q)

    @np_cache
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        p_t = self._inv_transform(p)

        def objective(theta: float) -> float:
            x = self._center[0] + self._radii[0] * np.cos(theta)
            y = self._center[1] + self._radii[1] * np.sin(theta)
            return (x - p_t[0]) ** 2 + (y - p_t[1]) ** 2
    
        theta0 = np.arctan2(p_t[1] - self._center[1], p_t[0] - self._center[0])
        result = minimize(objective, theta0, bounds=[(0, 2 * np.pi)])
        theta_opt = result.x[0]
        x = self._center[0] + self._radii[0] * np.cos(theta_opt)
        y = self._center[1] + self._radii[1] * np.sin(theta_opt)
        return np.array([x, y])
        
    def point_by_s(self, s: float, t: float) -> np.ndarray:
        if s < 0.0 or s > 1.0:
            raise ValueError("t must be in the range [0, 1]")
        
        angle = 2.0 * pi * s
        x = self._center[0] + self._radii[0] * np.cos(angle)
        y = self._center[1] + self._radii[1] * np.sin(angle)
        
        point = np.array([x, y])
        point = self._displace(self._transform(point), t, s)
        return point

    def tangent(self, s: float) -> np.ndarray:
        tangent_vector = np.array([
            -self._radii[0] * np.sin(2.0 * pi * s),
            self._radii[1] * np.cos(2.0 * pi * s)
        ])
        tangent_vector = self._transform(tangent_vector)
        tangent_vector /= np.linalg.norm(tangent_vector)
        return tangent_vector

    def copy(self) -> Shape:
        ellipse = Ellipse(
            self._center.copy(),
            self._radii.copy(),
            self._color_gradient.copy(),
            self._point_density
        )
        ellipse._transformations = [t.copy() for t in self._transformations]
        ellipse._inverse_transformations = [t.copy() for t in self._inverse_transformations]
        ellipse._displacements = self._displacements
        return ellipse

    def segments(self, n: int, t: float = 0.0) -> List[Line]:
        S = np.linspace(0, 1, n + 1)
        segments = [
            Line(
                self.point_by_s(S[i], t), 
                self.point_by_s(S[i + 1], t), 
                self._color_gradient, 
                self._point_density
            )
            for i in range(n)
        ]
        return segments
    
    @property
    def center(self) -> np.ndarray:
        return self._center
    
    @property
    def radii(self) -> np.ndarray:
        return self._radii
    