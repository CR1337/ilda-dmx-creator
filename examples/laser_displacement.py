import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Shape, Circle

import numpy as np


DURATION: float = 3.0


class Displace:

    def __init__(self, other_circle: Circle):
        self.other_circle = other_circle

    def __call__(self, shape: Shape, p: np.ndarray, t: float, time: float) -> np.ndarray:
        q = self.other_circle.nearest_point(p)
        d = self.other_circle.signed_distance(p)
        return p - (0.05 / d ** 2) * (q - p)


def factory_function(frame: Frame):
    progress = frame.t / DURATION

    color_gradient = ColorGradient(Color(1, 0, 0))
    color_gradient.add_color(0.5, Color(0, 0, 1))

    frame += (circle := Circle(np.array([-0.5, 0.0]), 0.4, color_gradient))

    frame += Circle(np.array([0.4, 2 * progress - 1]), 0.4, color_gradient).displace(
        Displace(circle)
    )


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        durations=DURATION,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/displacement.ildx",
        point_density=0.0005,
        show_excluision_zones=False,
        flip_x=False,
        flip_y=False
    )
    factory.run()
