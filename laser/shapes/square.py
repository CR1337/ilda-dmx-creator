import numpy as np

from laser.color import ColorGradient
from laser.shapes.rectangle import Rectangle  


class Square(Rectangle):

    def __init__(
        self,
        top_left: np.ndarray,
        side_length: float,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        bottom_right = top_left + np.array([side_length, side_length])
        super().__init__(top_left, bottom_right, color_gradient, point_density)
        