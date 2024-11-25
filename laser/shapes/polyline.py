from __future__ import annotations
import numpy as np
from typing import List, Tuple, Callable
from math import sqrt
from itertools import chain
from util import np_cache, ensure_np_array
from functools import cache

from laser.color import ColorGradient, Color
from laser.shapes.shape import Shape


class Polyline(Shape):

    DEFAULT_PARAMETRIC_STEP_SIZE: float = 0.01

    _points: List[np.ndarray]
    _closed: bool

    _total_length: float

    @classmethod
    def from_sdf(cls, sdf: Callable[[np.ndarray], float], color_gradient: ColorGradient, point_density: float | None = None) -> List[Polyline]:
        dummy_shape = cls(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])], False, 
            color_gradient, point_density
        )
        return dummy_shape._combine_shapes(None, sdf, color_gradient)
    
    @classmethod
    def from_parametric_equation(
        cls, 
        f: Callable[[float], np.ndarray], 
        closed: bool,
        color_gradient: ColorGradient,
        point_amount: int | None = None, 
        step_size: float | None = None,
        point_density: float | None = None
    ):
        if step_size is None and point_amount is None:
            step_size = cls.DEFAULT_PARAMETRIC_STEP_SIZE
        elif step_size is None:
            step_size = 1.0 / point_amount
        
        points = [f(s) for s in np.arange(0.0, 1.0 + step_size, step_size)]
        return cls(points, closed, color_gradient, point_density)

    def __init__(
        self, 
        points: List[np.ndarray], 
        closed: bool,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__(color_gradient, point_density)
        self._points = points
        self._closed = closed

        self._compute_total_length()

    def _compute_total_length(self):
        self._total_length = 0.0
        for i in range(1, len(self._points)):
            (x1, y1) = self._points[i - 1]
            (x2, y2) = self._points[i]
            self._total_length += sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if self._closed:
            (x1, y1) = self._points[-1]
            (x2, y2) = self._points[0]
            self._total_length += sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        spacing = 1.0 / (self._point_density * self.ILDX_RESOLUTION)

        points = [self._points[0]]
        colors = [self._color_gradient.get_color(0.0)]
        ts = [0.0]
        accumulated_length = 0.0
        
        for i in range(1, len(self._points)):
            (x1, y1) = self._points[i - 1]
            (x2, y2) = self._points[i]

            segment_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if segment_length > spacing:
                n_points = int(round(segment_length / spacing))
                dx = (x2 - x1) / (n_points + 1)
                dy = (y2 - y1) / (n_points + 1)

                for j in range(1, n_points + 1):
                    new_x = x1 + j * dx
                    new_y = y1 + j * dy

                    accumulated_length += sqrt(dx ** 2 + dy ** 2) 
                    t = accumulated_length / self._total_length

                    points.append(np.array([new_x, new_y]))
                    colors.append(self._color_gradient.get_color(t))
                    ts.append(t)
            else:
                accumulated_length += segment_length
                
            t = accumulated_length / self._total_length

            points.append(np.array([x2, y2]))
            colors.append(self._color_gradient.get_color(t))
            ts.append(t)

        if self._closed:
            (x1, y1) = self._points[-1]
            (x2, y2) = self._points[0]

            segment_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if segment_length > spacing:
                n_points = int(round(segment_length / spacing))
                dx = (x2 - x1) / (n_points + 1)
                dy = (y2 - y1) / (n_points + 1)

                for j in range(1, n_points + 1):
                    new_x = x1 + j * dx
                    new_y = y1 + j * dy

                    accumulated_length += sqrt(dx ** 2 + dy ** 2) * j
                    t = accumulated_length / self._total_length

                    points.append(np.array([new_x, new_y]))
                    colors.append(self._color_gradient.get_color(t))
                    ts.append(t)

            accumulated_length += segment_length
            if self._total_length == 0.0:
                t = 0.0
            else:
                t = accumulated_length / self._total_length

            points.append(np.array([x2, y2]))
            colors.append(self._color_gradient.get_color(t))
            ts.append(t)

        return points, colors, ts

    def _orientation(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> int:
        """
        Return the orientation of the triplet (p0, p1, p2):
        -1 -> Clockwise
        0 -> Collinear
        1 -> Counterclockwise
        """
        val = (p1[1] - p0[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p1[0] - p0[0])
        if val > 0:
            return 1  # Counterclockwise
        elif val < 0:
            return -1  # Clockwise
        else:
            return 0  # Collinear

    def _do_lines_intersect(self, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> bool:
        """
        Return True if the line segments p0p1 and q0q1 intersect.
        """
        def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
            """
            Check if point q lies on the segment pr (assumes q is collinear with pr).
            """
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        # Calculate orientation values
        o1 = self._orientation(p0, p1, q0)
        o2 = self._orientation(p0, p1, q1)
        o3 = self._orientation(q0, q1, p0)
        o4 = self._orientation(q0, q1, p1)

        # General case: the segments intersect if the orientations are different
        if o1 != o2 and o3 != o4:
            return True

        # Collinear cases: check if one segment's endpoint lies on the other segment
        if o1 == 0 and _on_segment(p0, q0, p1):
            return True
        if o2 == 0 and _on_segment(p0, q1, p1):
            return True
        if o3 == 0 and _on_segment(q0, p0, q1):
            return True
        if o4 == 0 and _on_segment(q0, p1, q1):
            return True

        # No intersection
        return False
    
    @ensure_np_array
    def is_point_inside(self, p: np.ndarray) -> bool:
        if not self._closed:
            return super().is_point_inside(p)
        
        winding_number = 0
        n = len(self._points)

        for i in range(n):
            p0 = self._points[i]
            p1 = self._points[(i + 1) % n]
            if p0[1] <= p[1]:
                if p1[1] > p[1] and self._orientation(p0, p1, p) == 1:
                    winding_number += 1
            else:
                if p1[1] <= p[1] and self._orientation(p0, p1, p) == -1:
                    winding_number -= 1

        return winding_number != 0
        
    @np_cache
    @ensure_np_array
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        line_segments = (
            (self._points[i - 1], self._points[i]) 
            for i in range(1, len(self._points))
        )

        if self._closed:
            line_segments = chain(line_segments, [(self._points[-1], self._points[0])])
            if self.is_point_inside(p0) or self.is_point_inside(p1):
                return True
            for (q0, q1) in line_segments:
                if self._do_lines_intersect(p0, p1, q0, q1):
                    return True
            return False
        
        else:
            for (q0, q1) in line_segments:
                if self._do_lines_intersect(p0, p1, q0, q1):
                    return True
            return False
    
    @ensure_np_array
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        return not self.is_line_inside(p0, p1)
    
    @np_cache
    @ensure_np_array
    def _nearest_point_and_distance(self, p: np.ndarray) -> Tuple[np.ndarray, float]:
        p = self._inv_transform(p)
        line_segments = (
            (self._points[i - 1], self._points[i]) 
            for i in range(1, len(self._points))
        )
        best_p = None
        best_distance = float("inf")
        for (p_0, p_1) in chain(line_segments, [(self._points[-1], self._points[0])] if self._closed else []):
            t = (p - p_0).dot(p_1 - p_0) / np.linalg.norm(p_1 - p_0) ** 2
            t = np.clip(t, 0.0, 1.0)
            q = p_0 + t * (p_1 - p_0)
            if t == 0.0:
                q = p_0
            elif t == 1.0:
                q = p_1       
            distance = np.linalg.norm(p - q)
            if distance < best_distance:
                best_distance = distance
                best_p = q
        return self._transform(best_p), best_distance
    
    @ensure_np_array
    def signed_distance(self, p: np.ndarray) -> float:
        distance = self._nearest_point_and_distance(p)[1]
        if self._closed and self.is_point_inside(p):
            return -distance
        return distance
        
    @ensure_np_array
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        return self._nearest_point_and_distance(p)[0]

    @cache
    def point_by_s(self, s: float, t: float) -> np.ndarray:       
        if s == 1.0:
            p = self._points[-1]
        else:
            accumulated_length = 0.0
            for i in range(1, len(self._points)):
                (x1, y1) = self._points[i - 1]
                (x2, y2) = self._points[i]

                segment_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                accumulated_length += segment_length

                if s * self._total_length <= accumulated_length:
                    p = np.array([x1 + (x2 - x1) * s, y1 + (y2 - y1) * s])
                
            if self._closed:
                (x1, y1) = self._points[-1]
                (x2, y2) = self._points[0]

                segment_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                accumulated_length += segment_length

                if s * self._total_length <= accumulated_length:
                    p = np.array([x1 + (x2 - x1) * s, y1 + (y2 - y1) * s])

        return self._displace(self._transform(p), t, s)
                
    def _find_line_segment_index(self, s: float) -> Tuple[int, float]:
        accumulated_length = 0.0
        for i in range(1, len(self._points)):
            (x1, y1) = self._points[i - 1]
            (x2, y2) = self._points[i]

            segment_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            accumulated_length += segment_length

            if s * self._total_length <= accumulated_length:
                return i - 1

        if self._closed:
            (x1, y1) = self._points[-1]
            (x2, y2) = self._points[0]

            segment_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            accumulated_length += segment_length

            if s * self._total_length <= accumulated_length:
                return len(self._points) - 1

        return -1
    
    def _two_point_tangent(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        return (p1 - p0) 
    
    def _three_point_tangent(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        v01 = p1 - p0
        v12 = p2 - p1
        w01 = np.linalg.norm(v01)
        w12 = np.linalg.norm(v12)
        return (w01 * v01 + w12 * v12) / (w01 + w12)

    @cache
    def tangent(self, s: float) -> np.ndarray:
        i = self._find_line_segment_index(s)
        if s == 0.0:
            if i == 0 and not self._closed:
                v = self._two_point_tangent(self._points[0], self._points[1])
            else:
                v = self._three_point_tangent(self._points[-1], self._points[0], self._points[1])
        elif s == 1.0:
            if i == len(self._points) - 1 and not self._closed:
                v = self._two_point_tangent(self._points[-2], self._points[-1])
            else:
                v = self._three_point_tangent(self._points[-2], self._points[-1], self._points[0])
        else:
            if self._closed:
                v = self._three_point_tangent(
                    self._points[i - 1], self._points[i], self._points[(i + 1) % len(self._points)]
                )
            else:
                if i == 0:
                    v = self._two_point_tangent(self._points[0], self._points[1])
                elif i == len(self._points) - 1:
                    v = self._two_point_tangent(self._points[-2], self._points[-1])
                else:
                    v = self._three_point_tangent(
                        self._points[i - 1], self._points[i], self._points[i + 1]
                    )
                
        tagent_vector = self._transform(v)
        return tagent_vector / np.linalg.norm(tagent_vector)

    def copy(self) -> Shape:
        polyline = Polyline(
            self._points.copy(), 
            self._closed, 
            self._color_gradient.copy(), 
            self._point_density
        )
        polyline._transformations = [t.copy() for t in self._transformations]
        polyline._inverse_transformations = [t.copy() for t in self._inverse_transformations]
        polyline._displacements = self._displacements
        return polyline
    
