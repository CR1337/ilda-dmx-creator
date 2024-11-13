import numpy as np
from abc import ABC, abstractmethod
from typing import List, Generator, Tuple

from laser.color import ColorGradient, Color
from laser.displacement import Displacement
from laser.render_line import RenderLine


class Shape(ABC):

    ILDA_RESOLUTION: int = 2 ** 16

    _point_density: float | None
    _color_gradient: ColorGradient
    _displacement: Displacement | None

    _transformation_matrices: List[np.ndarray]

    def __init__(
        self, 
        color_gradient: ColorGradient,
        displacement: Displacement | None = None,
        point_density: float | None = None
    ):
        self._color_gradient = color_gradient
        self._displacement = displacement
        self._point_density = point_density

        self._transformation_matrices = []

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
    
    def get_render_lines(self) -> Generator[RenderLine, None, None]:
        points, colors, ts = self._compute_points()
        points = [
            self._apply_transformartions(point)
            for point in points
        ]
        if self._displacement:
            points = [
                point + self._displacement.get_displacement(t, prev_p, point, next_p)
                for prev_p, point, next_p, t in zip([None] + points[:-1], points, points[1:] + [None], ts)
            ]
        for p0, p1, color in zip(points[:-1], points[1:], colors):
            yield RenderLine(p0, p1, color)
        
    
    def _apply_transformartions(self, p: np.ndarray) -> np.ndarray:
        affine_coordinates = np.array([p[0], p[1], 1])
        for matrix in self._transformation_matrices:
            affine_coordinates = np.dot(matrix, affine_coordinates)
        return np.array([affine_coordinates[0], affine_coordinates[1]]) / affine_coordinates[2]
    
    def translate(self, translation: np.ndarray):
        matrix = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ])
        self._transformation_matrices.append(matrix)

    def rotate(self, angle: float, center: np.ndarray | None = None):
        if center is None:
            center = np.array([0.0, 0.0])
        matrix = np.array([
            [np.cos(angle), -np.sin(angle), center[0]],
            [np.sin(angle), np.cos(angle), center[1]],
            [0, 0, 1]
        ])
        self._transformation_matrices.append(matrix)

    def scale(self, scale: np.ndarray, center: np.ndarray | None = None):
        if center is None:
            center = np.ndarray([0.0, 0.0])
        matrix = np.array([
            [scale[0], 0, center[0]],
            [0, scale[1], center[1]],
            [0, 0, 1]
        ])
        self._transformation_matrices.append(matrix)

    @property
    def point_density(self) -> float | None:
        return self._point_density
    
    @point_density.setter
    def point_density(self, value: float | None):
        self._point_density = value
        