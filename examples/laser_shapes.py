import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Circle, Ellipse, Line, Point, Polygon, Polyline, Rectangle, RegularNGon, Square, Star

import numpy as np


def factory_function(frame: Frame):
    color_gradient = ColorGradient(Color(1, 0, 0))
    color_gradient.add_color(0.5, Color(0, 0, 1))

    # square    rectangle   Polygon
    # Ellipse   Circle      RegularNGon
    # Star      Line        Polyline
    
    frame += Square(np.array([-0.5, 0.5]), 0.2, color_gradient)
    frame += Rectangle(np.array([0.0, 0.5]), np.array([0.2, 0.6]), color_gradient)
    frame += Polygon([np.array([0.5, 0.5]), np.array([0.6, 0.6]), np.array([0.7, 0.5])], color_gradient)

    frame += Ellipse(np.array([-0.5, 0.0]), np.array([0.2, 0.1]), color_gradient)
    frame += Circle(np.array([0.0, 0.0]), 0.2, color_gradient)
    frame += RegularNGon(np.array([0.5, 0.0]), 0.2, 5, color_gradient)

    frame += Star(np.array([-0.5, -0.5]), 0.1, 0.2, 5, color_gradient)
    frame += Line(np.array([0.0, -0.5]), np.array([0.2, -0.7]), color_gradient)
    frame += Polyline([np.array([0.5, -0.5]), np.array([0.6, -0.6]), np.array([0.7, -0.5])], False, color_gradient)

    frame += Point(np.array([0.0, 0.0]), color_gradient)


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        durations=3.0,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/shapes.ildx",
        point_density=0.001
    )
    factory.run()
