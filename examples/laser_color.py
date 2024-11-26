import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Ellipse

import numpy as np


def factory_function(frame: Frame):
    color_gradient = ColorGradient(Color(1, 0, 0), Color(1, 0, 0))
    color_gradient.add_color(frame.progress, Color(0, 0, 1))

    ellipse = Ellipse(
        np.array([0.0, 0.0]), 
        np.array([0.5, 0.75]), 
        color_gradient
    )
    frame += ellipse


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        durations=3.0,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/color.ildx",
        point_density=0.001
    )
    factory.run()
