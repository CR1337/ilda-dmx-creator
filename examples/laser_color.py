import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Ellipse

import numpy as np


DURATION: float = 3.0


def factory_function(frame: Frame):
    progress = frame.t / DURATION

    color_gradient = ColorGradient(Color(1, 0, 0), Color(1, 0, 0))
    color_gradient.add_color(progress, Color(0, 0, 1))

    ellipse = Ellipse(
        np.array([0.0, 0.0]), 
        np.array([0.5, 0.75]), 
        color_gradient
    )
    frame += ellipse


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        duration=DURATION,
        start_t=0,
        factory_function=factory_function,
        ildx_filename="examples/output/color.ildx",
        point_density=0.001,
        show_excluision_zones=False,
        flip_x=False,
        flip_y=False
    )
    factory.run()
