import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Ellipse, Circle, Line, Rectangle, Square, Star

import numpy as np


DURATION: float = 3.0


def create_exclusion_zones():
    shape_classes = [Ellipse, Ellipse, Line, Rectangle, Square, Star]
    mid_points = [
        np.array([
            np.cos(2 * np.pi * i / len(shape_classes)) / 2, 
            np.sin(2 * np.pi * i / len(shape_classes)) / 2
        ])
        for i in range(len(shape_classes))
    ]
    # place shapes (with an inner circle of radius 0.1) in a circle of radius 0.5 
    zones = []
    zones.append(Circle(mid_points[0], 0.1, ColorGradient(Color(1, 0, 0))))
    zones.append(Ellipse(mid_points[1], np.array([0.1, 0.2]), ColorGradient(Color(1, 0, 0))))
    zones.append(Line(mid_points[2] - np.array([0.1, 0.1]), mid_points[2] + np.array([0.1, 0.1]), ColorGradient(Color(1, 0, 0))))
    zones.append(Rectangle(mid_points[3] - np.array([0.1, 0.1]), mid_points[3] + np.array([0.1, 0.2]), ColorGradient(Color(1, 0, 0))))
    zones.append(Square(mid_points[4] - np.array([0.1, 0.1]), 0.2, ColorGradient(Color(1, 0, 0))))
    zones.append(Star(
        mid_points[5], 0.1, 0.2, 5,
        ColorGradient(Color(1, 0, 0))
    ))
    return zones


def factory_function(frame: Frame):
    progress = frame.t / DURATION
    circle = Circle(
        np.array([0.0, 0.0]), 
        0.5, 
        ColorGradient(Color(0, 1, 0), Color(0, 1, 1))
    ).rotate(2 * np.pi * progress)
    frame += circle


if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        duration=DURATION,
        start_t=0,
        factory_function=factory_function,
        ildx_filename="examples/output/exclusion_zones.ildx",
        point_density=0.0002,
        show_excluision_zones=True,
        flip_x=False,
        flip_y=False
    )
    for zone in create_exclusion_zones():
        factory.add_exclusion_zone(zone)
    factory.run()
