import numpy as np

from laser.color import ColorGradient
from laser.shapes.ellipse import Ellipse


class Circle(Ellipse):

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        color_gradient: ColorGradient,
        point_density: float | None = None
    ):
        super().__init__(center, np.array([radius, radius]), color_gradient, point_density)
        