import numpy as np
from typing import List, Tuple
from math import sqrt

from laser.color import ColorGradient, Color
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

    def get_centroid(self) -> np.ndarray:
        x = sum([p[0] for p in self._points]) / len(self._points)
        y = sum([p[1] for p in self._points]) / len(self._points)
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

    
    def is_point_inside(self, p: np.ndarray) -> bool:
        raise NotImplementedError()
    
    def is_line_inside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError()
    
    def is_line_outside(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        raise NotImplementedError()
    
    def signed_distance(self, p: np.ndarray) -> float:
        ...  # TODO
    
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        ...  # TODO
        