import numpy as np

from laser.color import ColorGradient
from laser.displacement import Displacement
from laser.shapes.rectangle import Rectangle  


class Square(Rectangle):

    def __init__(
        self,
        top_left: np.ndarray,
        side_length: float,
        color_gradient: ColorGradient,
        displacement: Displacement | None = None,
        point_density: float | None = None
    ):
        bottom_right = top_left + np.array([side_length, side_length])
        super().__init__(top_left, bottom_right, color_gradient, displacement, point_density)
        