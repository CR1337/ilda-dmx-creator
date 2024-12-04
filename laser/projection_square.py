from __future__ import annotations
import numpy as np
from typing import List
from laser.shapes import Shape, Polygon
from laser.color import ColorGradient, Color


# https://chatgpt.com/share/67477209-38c4-8010-ad9a-48b0c0d716e9


class ProjectionSquare:

    _is_default: bool = False

    # Plane
    _o: np.ndarray  # Origin in world coordinates
    _n: np.ndarray  # Normal vector in world coordinates
    _a: np.ndarray  # Rotation angle around the normal vector in radians

    _u: np.ndarray  # Base vector u in world coordinates
    _v: np.ndarray  # Base vector v in world coordinates

    _corners = List[np.ndarray]  # Corners of the plane in world coordinates

    # Pyramid
    _b: np.ndarray  # Base vector b in world coordinates
    _h: float  # Height of the pyramid
    _l: float  # Base side length of the pyramid
    _t: np.ndarray  # Tip of the pyramid in world coordinates

    _pyramid_corners = List[np.ndarray]  # Corners of the pyramid in world coordinates
    _face_normals = List[np.ndarray]  # Normals of the pyramid faces in world coordinates

    # Projection
    _rotation_matrix: np.ndarray
    _inverse_rotation_matrix: np.ndarray
    _to_world_projection_matrix: np.ndarray
    _to_plane_projection_matrix: np.ndarray
    _plane_to_plane_projection_matrix: np.ndarray
    _t_numerator: float

    # Drawable Polygon
    _drawable_polygon: Polygon
    _drawable_polygon_on_default: Polygon

    @classmethod
    def default(cls, angle: float) -> ProjectionSquare:
        z = np.cos(np.radians(angle) / 2)
        projection_square = cls(
            np.array([0, 0, z]),
            np.array([0, 0, -1]),
            0
        )
        projection_square._is_default = True
        return projection_square
    
    @property
    def is_default(self) -> bool:
        return self._is_default
    
    @property
    def drawable_polygon(self) -> Polygon:
        return self._drawable_polygon
    
    @property
    def drawable_polygon_on_default(self) -> Polygon:
        return self._drawable_polygon_on_default

    def _construct_plane(self):
        o = self._origin
        n = self._normal
        a = self._angle

        N = np.ndarray([
            [    0, -n[2],  n[1]],
            [ n[2],     0, -n[0]],
            [-n[1],  n[0],     0]
        ])
        R = np.identity(3) + np.sin(a) * N + (1 - np.cos(a)) * N @ N

        if (
            np.array_equal(n, np.array([1, 0, 0])) 
            or np.array_equal(n, np.array([-1, 0, 0]))
        ):
            w = np.array([0, 1, 0])
        else:
            w = np.array([1, 0, 0])

        u = np.cross(w, n)
        u /= np.linalg.norm(u)
        u = R @ u

        v = np.cross(n, u)
        v /= np.linalg.norm(v)
        v = R @ v
        
        self._to_world_projection_matrix = np.array([
            [u[0], v[0], o[0]],
            [u[1], v[1], o[1]],
            [u[2], v[2], o[2]],
            [   0,    0,    1]
        ])

        self._to_plane_projection_matrix = np.array([
            [u[0], u[1], u[2], -(u * o)],
            [v[0], v[1], v[2], -(v * o)],
            [   0,    0,    0,        1]
        ])

        self._plane_to_plane_projection_matrix = np.array([
            [n[0] * n[0], n[0] * n[1], n[0] * n[2]],
            [n[1] * n[0], n[1] * n[1], n[1] * n[2]],
            [n[2] * n[0], n[2] * n[1], n[2] * n[2]]
        ])

        self._t_numerator = np.dot(o, n)
    
        self._u = u
        self._v = v
        self._rotation_matrix = R
        self._inverse_rotation_matrix = np.linalg.inv(R)

        self._corners = [
            o + u + v,
            o - u + v,
            o - u - v,
            o + u - v
        ]

    def _construct_pyramid(self):
        a = self._angle
        b = np.array([0, 0, 2 * np.maximum(c[2] for c in self._corners)])
        h = np.linalg.norm(b)
        l = 2 * h * np.tan(a / 2)
        t = np.array([0, 0, 0])

        self._b = b
        self._h = h
        self._l = l
        self._t = t

        v_b = np.array([1, 0, 0])
        u_b = np.array([0, 1, 0])

        self._pyramid_corners = [
            b + l / 2 * u_b + l / 2 * v_b,
            b - l / 2 * u_b + l / 2 * v_b,
            b - l / 2 * u_b - l / 2 * v_b,
            b + l / 2 * u_b - l / 2 * v_b
        ]

        self._face_normals = [
            np.cross(self._pyramid_corners[1] - self._pyramid_corners[0], t - self._pyramid_corners[0]),
            np.cross(self._pyramid_corners[2] - self._pyramid_corners[1], t - self._pyramid_corners[1]),
            np.cross(self._pyramid_corners[3] - self._pyramid_corners[2], t - self._pyramid_corners[2]),
            np.cross(self._pyramid_corners[0] - self._pyramid_corners[3], t - self._pyramid_corners[3])
        ]

    def _construct_drawable_polygon(self):
        o = self._o
        n = self._n
        u = self._u
        v = self._v
        t = self._t
        subject_corners = []
        lies_on_square = []
        for q1, q2 in zip(self._pyramid_corners, self._pyramid_corners[1:] + [self._pyramid_corners[0]]):
            t = (o - q1) @ n / (q2 - q1) @ n
            if 0 <= t <= 1:
                p = q1 + t * (q2 - q1)
                subject_corners.append(self._to_plane_projection_matrix @ p)
    
                # check if point lies on square
                u_local = (p - o) * u
                v_local = (p - o) * v
                lies_on_square.append(u_local <= 1 and v_local <= 1)

        assert len(subject_corners) == 4

        square_corners = [
            self._to_plane_projection_matrix @ c
            for c in self._corners
        ]

        # Sutherland-Hodgman Algorithm for Convex Clipping:

        def is_inside(p: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> bool:
            return (e1[0] - e0[0]) * (p[1] - e0[1]) - (e1[1] - e0[1]) * (p[0] - e0[0]) >= 0
        
        def compute_intersection(p0: np.ndarray, p1: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> np.ndarray:
            denom = (p0[0] - p1[0]) * (e0[1] - e1[1]) - (p0[1] - p1[1]) * (e0[0] - e1[0])
            if denom == 0:
                return None  # Lines are parallel, no intersection
    
            t = ((p0[0] - e0[0]) * (e0[1] - e1[1]) - (p0[1] - e0[1]) * (e0[0] - e1[0])) / denom
            return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        
        clipped_corners = subject_corners[:]

        for i in range(len(square_corners)):
            e0 = subject_corners[i]
            e1 = subject_corners[(i + 1) % 4]

            input_polygon = clipped_corners[:]
            clipped_corners = []

            for j in range(len(input_polygon)):
                p1 = input_polygon[j]
                p0 = input_polygon[j - 1]

                if is_inside(p1, e0, e1):
                    if not is_inside(p0, e0, e1):
                        intersection = compute_intersection(p0, p1, e0, e1)
                        if intersection:
                            clipped_corners.append(intersection)
                    clipped_corners.append(p1)

                elif is_inside(p0, e0, e1):
                    intersection = compute_intersection(p0, p1, e0, e1)
                    if intersection:
                            clipped_corners.append(intersection)

        color_gradient = ColorGradient(Color(0, 1, 0))
        self._drawable_polygon = Polygon(clipped_corners, color_gradient)
        self._drawable_polygon_on_default = self.project_to_default_plane(self._drawable_polygon)

    def __init__(self, origin: np.ndarray, normal: np.ndarray, angle: float):
        self._o = origin
        self._n = normal / np.linalg.norm(normal)
        self._a = np.radians(angle)

        self._construct_plane()
        self._construct_pyramid()
        self._construct_drawable_polygon()

    def project_to_default_plane(self, p: np.ndarray | Shape) -> np.ndarray:
        if self._is_default:
            return p
        
        n = self._n

        t_denominator = np.dot(p, n)
        t = self._t_numerator / t_denominator

        M_1 = self._to_world_projection_matrix
        M_2 = self._plane_to_plane_projection_matrix * t
        M_3 = self._to_plane_projection_matrix

        M = M_3 @ M_2 @ M_1

        if isinstance(p, Shape):
            return p.copy().transform(M)
        else:
            return M @ p
