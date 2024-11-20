from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Generator, Tuple

from laser.color import ColorGradient, Color
from laser.render_line import RenderLine

from noise import Noise2D, Noise3D


class Shape(ABC):

    ILDA_RESOLUTION: int = 2 ** 16

    _point_density: float | None
    _color_gradient: ColorGradient

    _transformation_matrices: List[np.ndarray]
    _displacements: List[Tuple[Noise2D | Noise3D, str | None]]

    def __init__(
        self, 
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        self._color_gradient = color_gradient
        self._point_density = point_density

        self._transformation_matrices = []
        self._displacements = []

    @abstractmethod
    def get_centroid(self) -> np.ndarray:
        raise NotImplementedError("@abstractmethod get_centroid")
    
    @abstractmethod
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        raise NotImplementedError("@abstractmethod _compute_points")

    @abstractmethod
    def is_point_inside(self, p: np.ndarray) -> bool:
        raise NotImplementedError("@abstractmethod _is_inside")

    @abstractmethod
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError("@abstractmethod _is_inside")

    @abstractmethod
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError("@abstractmethod _is_outside")
    
    def _get_tangent(self, point_index: int, points: np.ndarray) -> np.ndarray:
        if len(points) == 1:  # Single point
            raise ValueError("Cannot get tangent for single point")
        elif len(points) == 2:  # Two points
            if np.array_equal(points[0], points[1]):
                raise ValueError("Cannot get tangent for two identical points")
            tangent_vector = points[1] - points[0]
        else:  # Any other number of points
            if point_index == 0:  # First point
                tangent_vector = points[1] - points[0]
            elif point_index == len(points) - 1:  # Last point
                tangent_vector = points[-1] - points[-2]
            else:  # Any other point
                # compute chordal tangent:
                (x1, y1), (x2, y2), (x3, y3) = points[point_index - 1:point_index + 2]

                v12 = np.array([x2 - x1, y2 - y1])
                v23 = np.array([x3 - x2, y3 - y2])

                w12 = np.linalg.norm(v12)
                w23 = np.linalg.norm(v23)
                
                tangent_vector = (w12 * v12 + w23 * v23) / (w12 + w23)

        norm = np.linalg.norm(tangent_vector)
        if norm > 0:
            tangent_vector = tangent_vector / norm
        return tangent_vector
            
    
    def get_render_lines(self, timestamp: float) -> Generator[RenderLine, None, None]:
        points, colors, ts = self._compute_points()
        points = [
            self._apply_transformartions(point)
            for point in points
        ]
        points = [
            self._apply_displacements(i, timestamp, points)
            for i in range(len(points))
        ]

        filtered_points = []
        filtered_colors = []
        for p, c in zip(points, colors):
            if p[0] <= -1 or p[0] >= 1 or p[1] <= -1 or p[1] >= 1:
                continue
            filtered_points.append(p)
            filtered_colors.append(c)

        for p0, p1, color in zip(filtered_points[:-1], filtered_points[1:], filtered_colors):
            yield RenderLine(p0, p1, color)
        
    
    def _apply_transformartions(self, p: np.ndarray) -> np.ndarray:
        affine_coordinates = np.array([p[0], p[1], 1])
        for matrix in self._transformation_matrices:
            affine_coordinates = np.dot(matrix, affine_coordinates)
        return np.array([affine_coordinates[0], affine_coordinates[1]]) / affine_coordinates[2]
    
    def _apply_displacements(self, point_index: int, timestamp: float, points: np.ndarray) -> np.ndarray:
        tangent = self._get_tangent(point_index, points)
        normal = np.array([-tangent[1], tangent[0]])
        point = points[point_index]
        displacement = np.array([0.0, 0.0])

        for noise, swizzle in self._displacements:
            if isinstance(noise, Noise2D):
                displacement += noise.get_value(point, None, swizzle=swizzle) * normal
            elif isinstance(noise, Noise3D):
                displacement += noise.get_value(point, timestamp, swizzle=swizzle) * normal

        return points[point_index] + displacement
            
    
    def translate(self, translation: np.ndarray) -> Shape:
        matrix = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])
        self._transformation_matrices.append(matrix)
        return self

    def rotate(self, angle: float, center: np.ndarray | None = None) -> Shape:
        if center is None:
            center = np.array([0.0, 0.0])
        matrix = np.array([
            [np.cos(angle), -np.sin(angle), center[0]],
            [np.sin(angle), np.cos(angle), center[1]],
            [0, 0, 1]
        ])
        self._transformation_matrices.append(matrix)
        return self

    def scale(self, scale: np.ndarray, center: np.ndarray | None = None) -> Shape:
        if center is None:
            center = np.array([0.0, 0.0])
        matrix = np.array([
            [scale[0], 0, center[0]],
            [0, scale[1], center[1]],
            [0, 0, 1]
        ])
        self._transformation_matrices.append(matrix)
        return self
    
    def displace(self, noise: Noise2D | Noise3D, *, swizzle: str | None = None) -> Shape:
        self._displacements.append((noise, swizzle))
        return self

    @property
    def point_density(self) -> float | None:
        return self._point_density
    
    @point_density.setter
    def point_density(self, value: float | None):
        self._point_density = value
        