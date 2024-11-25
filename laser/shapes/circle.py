import numpy as np

from laser.color import ColorGradient
from laser.shapes.ellipse import Ellipse
from util import ensure_np_array


class Circle(Ellipse):

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__(center, np.array([radius, radius]), color_gradient, point_density)
        
    @ensure_np_array
    def signed_distance(self, p: np.ndarray) -> float:
        p_t = self._inv_transform(p)
        v = p_t - self._center
        d_c = np.linalg.norm(v)
        return d_c - self._radii[0]

    @ensure_np_array
    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        p_t = self._inv_transform(p)
        v = p_t - self._center
        d_c = np.linalg.norm(v)
        if d_c == 0:
            q = self._center + self._radii[0] * np.array([1.0, 0.0])
        else:
            q = self._center + self._radii[0] * v / d_c
        return q