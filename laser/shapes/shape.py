from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Generator, Tuple, Callable
from skimage import measure
from functools import cache
import warnings

from laser.color import ColorGradient, Color
from laser.render_line import RenderLine
from util import np_hash, np_cache, ensure_np_array


Displacement = Callable[['Shape', np.ndarray, float, float], np.ndarray]


class Shape(ABC):

    ILDX_RESOLUTION: int = 2 ** 16
    NEEDED_COMBINATION_DENSITY: int = 300
    DEFAULT_POINT_DENSITY: float = 0.0005

    _point_density: float | None
    _color_gradient: ColorGradient

    _transformations: List[np.ndarray]
    _inverse_transformations: List[np.ndarray]
    _displacements: List[Displacement]

    def __init__(
        self, 
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        self._color_gradient = color_gradient
        self._point_density = point_density

        self._transformations = []
        self._inverse_transformations = []
        self._displacements = []

    def __eq__(self, other: Shape) -> bool:
        if self._point_density != other._point_density:
            return False
        if self._color_gradient != other._color_gradient:
            return False
        if len(self._transformations) != len(other._transformations):
            return False
        if len(self._displacements) != len(other._displacements):
            return False
        for transformation, other_transformation in zip(self._transformations, other._transformations):
            if not np.allclose(transformation, other_transformation):
                return False
        for displacement, other_displacement in zip(self._displacements, other._displacements):
            if displacement != other_displacement:
                return False
        return True
    
    def __hash__(self) -> int:
        return hash((
            self._point_density,
            self._color_gradient,
            *(np_hash(transformation) for transformation in self._transformations),
            tuple(self._displacements)
        ))

    @abstractmethod
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        raise NotImplementedError("@abstractmethod _compute_points")

    @np_cache
    def _transform(self, p: np.ndarray) -> np.ndarray:
        affine_coordinates = np.array([p[0], p[1], 1])
        for transformation in self._transformations:
            affine_coordinates = np.dot(transformation, affine_coordinates)
        return np.array([affine_coordinates[0], affine_coordinates[1]]) / affine_coordinates[2]

    @np_cache
    def _inv_transform(self, p: np.ndarray) -> np.ndarray:
        affine_coordinates = np.array([p[0], p[1], 1])
        for transformation in reversed(self._inverse_transformations):
            affine_coordinates = np.dot(transformation, affine_coordinates)
        return np.array([affine_coordinates[0], affine_coordinates[1]]) / affine_coordinates[2]

    @np_cache
    def _displace(self, p: np.ndarray, s: float, t: float) -> np.ndarray:
        p_disp = p.copy()
        for displacement in self._displacements:
            p_disp = displacement(self, p_disp, s, t)
        return p_disp

    @ensure_np_array
    def is_point_inside(self, p: np.ndarray) -> bool:
        return self.signed_distance(p) <= 0

    def get_render_lines(self, t: float) -> Generator[RenderLine, None, None]:
        points, colors, ts = self._compute_points()
        points = [
            self._displace(
                self._transform(point), t, t
            )
            for point, t in zip(points, ts)
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

    @ensure_np_array
    @abstractmethod
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError("@abstractmethod _is_inside")

    @ensure_np_array
    @abstractmethod
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError("@abstractmethod _is_outside")

    @ensure_np_array
    @abstractmethod
    def signed_distance(self, p: np.ndarray) -> float:
        raise NotImplementedError("@abstractmethod signed_distance")
    
    @ensure_np_array
    @abstractmethod
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError("@abstractmethod nearest_point")
    
    @ensure_np_array
    @abstractmethod
    def point_by_s(self, s: float, t: float) -> np.ndarray:
        raise NotImplementedError("@abstractmethod get_point")

    @ensure_np_array
    @abstractmethod
    def tangent(self, s: float) -> np.ndarray:
        raise NotImplementedError("@abstractmethod tangent")

    @ensure_np_array
    @abstractmethod
    def copy(self) -> Shape:
        raise NotImplementedError("@abstractmethod copy")
    
    def normal(self, s: float, t: float) -> np.ndarray:
        tangent = self.tangent(s, t)
        return np.array([-tangent[1], tangent[0]])
    
    def reset_transformations(self) -> Shape:
        self._transformations = []
        self._inverse_transformations = []
        return self

    def reset_displacements(self) -> Shape:
        self._displacements = []
        return self
    
    @ensure_np_array
    def translate(self, translation: np.ndarray) -> Shape:
        return self.transform(np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ]))

    def rotate(self, angle: float) -> Shape:
        return self.transform(np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ]))

    @ensure_np_array
    def scale(self, scale: np.ndarray) -> Shape:
        return self.transform(np.array([
            [scale[0], 0, 0],
            [0, scale[1], 0],
            [0, 0, 1]
        ]))

    @ensure_np_array
    def shear(self, shear: np.ndarray) -> Shape:
        return self.transform(np.array([
            [1, shear[0], 0],
            [shear[1], 1, 0],
            [0, 0, 1]
        ]))

    @ensure_np_array
    def skew(self, skew: np.ndarray) -> Shape:
        return self.transform(np.array([
            [1, np.tan(skew[0]), 0],
            [np.tan(skew[1]), 1, 0],
            [0, 0, 1]
        ]))
    
    @ensure_np_array
    def reflect(self, axis: np.ndarray) -> Shape:
        return self.transform(np.array([
            [1 - 2 * axis[0] ** 2, -2 * axis[0] * axis[1], 0],
            [-2 * axis[0] * axis[1], 1 - 2 * axis[1] ** 2, 0],
            [0, 0, 1]
        ]))
    
    def project(self, angle: float) -> Shape:
        return self.transform(np.array([
            [np.cos(angle) ** 2, np.sin(angle) * np.cos(angle), 0],
            [np.sin(angle) * np.cos(angle), np.sin(angle) ** 2, 0],
            [0, 0, 1]
        ]))
    
    def identity(self) -> Shape:
        return self.transform(np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]))

    @ensure_np_array
    def transform(self, matrix: np.ndarray) -> Shape:
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            inv_matrix = np.linalg.pinv(matrix)  # TODO: Check if this is correct
            warnings.warn(
                "You used a non-invertable transformation matrix. The pseudo-inverse was used instead.",
                RuntimeWarning
            )
        self._transformations.append(matrix)
        self._inverse_transformations.append(inv_matrix)
        return self

    def displace(self, func: Displacement) -> Shape:
        self._displacements.append(func)
        return self

    @property
    def point_density(self) -> float | None:
        return self._point_density
    
    @point_density.setter
    def point_density(self, value: float | None):
        self._point_density = value
        
    def union(self, other: Shape, color_gradient: ColorGradient) -> List[Shape]:
        def sdf(p: np.ndarray) -> float:
            return np.minimum(self.signed_distance(p), other.signed_distance(p))
        return self._combine_shapes(other, sdf, color_gradient)

    def smooth_union(self, other: Shape, k: float, color_gradient: ColorGradient) -> List[Shape]:
        def sdf(p: np.ndarray) -> float:
            return -np.log(np.exp(-k * self.signed_distance(p)) + np.exp(-k * other.signed_distance(p))) / k
        return self._combine_shapes(other, sdf, color_gradient)

    def intersection(self, other: Shape, color_gradient: ColorGradient) -> List[Shape]:
        def sdf(p: np.ndarray) -> float:
            return np.maximum(self.signed_distance(p), other.signed_distance(p))
        return self._combine_shapes(other, sdf, color_gradient)

    def difference(self, other: Shape, color_gradient: ColorGradient) -> List[Shape]:
        def sdf(p: np.ndarray) -> float:
            return np.maximum(self.signed_distance(p), -other.signed_distance(p))
        return self._combine_shapes(other, sdf, color_gradient)
   
    def lerp(self, other: Shape, r: float, color_gradient: ColorGradient) -> List[Shape]:
        def sdf(p: np.ndarray) -> float:
            return (1 - r) * self.signed_distance(p) + r * other.signed_distance(p)
        return self._combine_shapes(other, sdf, color_gradient)
    
    def custom_sdf_operation(self, other: Shape, custom_sdf: Callable[[np.ndarray, np.ndarray], float], color_gradient: ColorGradient) -> List[Shape]:
        def sdf(p: np.ndarray) -> float:
            return custom_sdf(self.signed_distance(p), other.signed_distance(p) if other else None)
        return self._combine_shapes(other, sdf, color_gradient)

    @cache
    def _combine_shapes(self, other: Shape, sdf: Callable[[np.ndarray], float], color_gradient: ColorGradient) -> List[Shape]:
        from laser.shapes.polyline import Polyline
        from laser.shapes.point import Point

        if other:
            if self._point_density is None and other._point_density is not None:
                point_density = other._point_density
            elif self._point_density is not None and other._point_density is None:
                point_density = other._point_density
            elif self._point_density is None and other._point_density is None:
                point_density = self.DEFAULT_POINT_DENSITY
            else:
                point_density = max(self._point_density, other._point_density)
        else:
            if self._point_density is None:
                point_density = self.DEFAULT_POINT_DENSITY
            else:
                point_density = self._point_density

        X, Y = np.meshgrid(np.linspace(-1, 1, self.NEEDED_COMBINATION_DENSITY), np.linspace(-1, 1, self.NEEDED_COMBINATION_DENSITY))
        Z = np.array([sdf(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        contours = measure.find_contours(Z, 0)

        polylines = []
        for contour in contours:
            points = contour[:, ::-1] / self.NEEDED_COMBINATION_DENSITY * 2 - 1
            sampled_points = points[::max(int(self.NEEDED_COMBINATION_DENSITY / (point_density * self.ILDX_RESOLUTION)), 1)]
            if len(points) > 1:
                is_closed = np.allclose(points[0], points[-1])
                polylines.append((Polyline(sampled_points, is_closed, color_gradient, point_density), len(points)))
            elif len(points) == 1:
                polylines.append((Point(points[0], color_gradient), len(points)))

        polylines.sort(key=lambda x: x[1], reverse=True)

        return [polyline for polyline, _ in polylines]
