import numpy as np
from typing import List, Tuple
from math import sqrt

from laser.color import ColorGradient, Color
from laser.displacement import Displacement
from laser.shapes.shape import Shape


class Polyline(Shape):

    _points: List[np.ndarray]
    _closed: bool

    _total_length: float

    def __init__(
        self, 
        points: List[np.ndarray], 
        closed: bool,
        color_gradient: ColorGradient,
        displacement: Displacement | None = None,
        point_density: float | None = None
    ):
        super().__init__(color_gradient, displacement, point_density)
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

    def get_centroid(self) -> np.ndarray:
        area = 0.5 * sum(
            self._points[i][0] * self._points[i + 1][1] 
            - self._points[i + 1][0] * self._points[i][1]
            for i in range(len(self._points) - 1)
        )
        x = 1.0 / (6.0 * area) * sum(
            (self._points[i][0] + self._points[i + 1][0]) 
            * (self._points[i][0] * self._points[i + 1][1] - self._points[i + 1][0] * self._points[i][1])
            for i in range(len(self._points) - 1)
        )
        y = 1.0 / (6.0 * area) * sum(
            (self._points[i][1] + self._points[i + 1][1]) 
            * (self._points[i][0] * self._points[i + 1][1] - self._points[i + 1][0] * self._points[i][1])
            for i in range(len(self._points) - 1)
        )
        return np.array([x, y])
    
    def _compute_points(self) -> Tuple[List[np.ndarray], List[Color], List[float]]:
        spacing = 1.0 / (self._point_density * self.ILDA_RESOLUTION)

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

                    accumulated_length += sqrt(dx ** 2 + dy ** 2) * j
                    t = accumulated_length / self._total_length

                    points.append(np.array([new_x, new_y]))
                    colors.append(self._color_gradient.get_color(t))
                    ts.append(t)

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
            t = accumulated_length / self._total_length

            points.append(np.array([x2, y2]))
            colors.append(self._color_gradient.get_color(t))
            ts.append(t)

        return points, colors, ts
    
    def _orientation(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        return (p1[1] - p0[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p1[0] - p0[0])
    
    def _do_lines_intersect(self, p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> bool:
        o1 = self._orientation(p0, q0, p1)
        o2 = self._orientation(p0, q0, q1)
        o3 = self._orientation(p1, q1, p0)
        o4 = self._orientation(p1, q1, q0)
        return o1 != o2 and o3 != o4
    
    def is_point_inside(self, p: np.ndarray) -> bool:
        raise NotImplementedError()
    
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError()
    
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError()
    