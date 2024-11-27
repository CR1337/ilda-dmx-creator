import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Ellipse, Shape
from noise import Noise3D

import numpy as np


class Displace:

    def __init__(self, noise: Noise3D):
        self._noise = noise

    def __call__(self, shape: Shape, p: np.ndarray, s: float, t: float) -> np.ndarray:
        return p + self._noise.get_value([p[0], p[1], t])


def factory_function(frame: Frame):
    color_gradient = ColorGradient(Color(1, 0, 0), Color(1, 0, 0))
    color_gradient.add_color(frame.progress, Color(0, 0, 1))

    ellipse = Ellipse(
        np.array([0.0, 0.0]), 
        np.array([0.5, 0.75]), 
        color_gradient
    )
    frame += ellipse.displace(Displace(Noise3D.cylindrical(amplitude = 0.2)))


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        durations=3.0,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/noise.ildx",
        point_density=0.001
    )
    factory.run()
