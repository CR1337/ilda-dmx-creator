import numpy as np

from laser.color import ColorGradient
from laser.shapes.polygon import Polygon  


class Rectangle(Polygon):

    def __init__(
        self,
        top_left: np.ndarray,
        bottom_right: np.ndarray,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        top_left = np.array(top_left)
        bottom_right = np.array(bottom_right)
        top_right = np.array([bottom_right[0], top_left[1]])
        bottom_left = np.array([top_left[0], bottom_right[1]])
        points = [top_left, bottom_left, bottom_right, top_right]
        super().__init__(points, color_gradient, point_density)

    @property
    def center(self) -> np.ndarray:
        return (self._points[0] + self._points[2]) / 2

    @property
    def width(self) -> float:
        return self._points[2][0] - self._points[0][0]
    
    @property
    def height(self) -> float:
        return self._points[3][1] - self._points[1][1]