import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ilda_factory import IldaFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Ellipse, Ellipse, Line, Point, Polyline, Rectangle, Square, Triangle

import numpy as np


DURATION: float = 3.0


def factory_function(frame: Frame):
    shape_classes = [Ellipse, Ellipse, Line, Point, Polyline, Rectangle, Square, Triangle]
    mid_points = [
        np.array([
            np.cos(2 * np.pi * i / len(shape_classes)) / 2, 
            np.sin(2 * np.pi * i / len(shape_classes)) / 2
        ])
        for i in range(len(shape_classes))
    ]

    color_gradient = ColorGradient(Color(1, 0, 0), Color(0, 0, 1))
    color_gradient.add_color(0.5, Color(0, 1, 0))

    # place shapes (with an inner circle of radius 0.1) in a circle of radius 0.5 
    shapes = []
    shapes.append(Ellipse(mid_points[0], 0.1, color_gradient))
    shapes.append(Ellipse(mid_points[1], np.array([0.1, 0.2]), color_gradient))
    shapes.append(Line(mid_points[2] - np.array([0.1, 0.1]), mid_points[2] + np.array([0.1, 0.1]), color_gradient))
    shapes.append(Point(mid_points[3], color_gradient))
    shapes.append(Polyline(
        [
            mid_points[4] - np.array([0.1, 0.1]), 
            mid_points[4] + np.array([0.1, 0.1]),
            mid_points[4] + np.array([0.1, -0.1]),
            mid_points[4] - np.array([0.1, -0.1]),
            mid_points[4] - np.array([0.1, 0.1])
        ],
        True,
        color_gradient
    ))
    shapes.append(Rectangle(mid_points[5] - np.array([0.1, 0.1]), mid_points[5] + np.array([0.1, 0.2]), color_gradient))
    shapes.append(Square(mid_points[6] - np.array([0.1, 0.1]), 0.2, color_gradient))
    shapes.append(Triangle(
        mid_points[7] + np.array([0.0, 0.1]), 
        mid_points[7] + np.array([0.1, -0.1]), 
        mid_points[7] + np.array([-0.1, -0.1]), 
        color_gradient
    ))

    progress = frame.timestamp / DURATION
    for i, shape in enumerate(shapes):
        shape.rotate(2 * np.pi * progress, np.array([0, 0]))
        frame += shape

    if progress >= 1:
        frame.set_last()


if __name__ == "__main__":
    factory = IldaFactory(
        fps=30,
        start_timestamp=0,
        factory_function=factory_function,
        ilda_filename="examples/output/shapes.ildx",
        point_density=0.001,
        show_excluision_zones=False,
        flip_x=False,
        flip_y=False
    )
    factory.run()
