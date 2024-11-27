import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Circle

import numpy as np


def factory_function(frame: Frame):
    color_gradient = ColorGradient(Color(1, 0, 0))
    color_gradient.add_color(0.5, Color(0, 0, 1))

    circle1 = Circle(np.array([-0.5, 0.0]), 0.4, color_gradient)
    circle2 = Circle(np.array([0.4, 2 * frame.progress - 1]), 0.8, color_gradient)
    shapes = circle1.smooth_union(circle2, 16, color_gradient)
    if shapes:
        frame += shapes[0]


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        durations=3.0,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/sdf.ildx",
        point_density=0.0005
    )
    factory.run()
