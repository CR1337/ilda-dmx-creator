import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ilda_factory import IldaFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Circle, Point
from noise import Noise2D, Noise3D

import numpy as np


DURATION: float = 3.0

offsets = [
    np.random.rand(2) * 0.1,
    np.random.rand(2) * 0.1,
    np.random.rand(2) * 0.1,
    np.random.rand(2) * 0.1,
    np.random.rand(2) * 0.1
]


def factory_function(frame: Frame):
    progress = frame.timestamp / DURATION
    
    ul_circle = Circle(np.array([-0.5, 0.5]), 0.1, ColorGradient(Color(1, 0, 0), Color(0, 0, 1))).displace(Noise2D.plane())
    ur_circle = Circle(np.array([ 0.5, 0.5]), 0.1, ColorGradient(Color(1, 0, 0), Color(0, 0, 1))).displace(Noise3D.toroidal(frequency=np.array([3.0, 5.0, 3.0]), amplitude = 0.1))

    points = [
        Point(np.array([0.0,  -0.5]) + offsets[0], ColorGradient(Color(0, 0, 1))).displace(Noise3D.cylindrical()),
        Point(np.array([0.0,  -0.6]) + offsets[1], ColorGradient(Color(0, 1, 0))).displace(Noise3D.cylindrical()),
        Point(np.array([0.0,  -0.7]) + offsets[2], ColorGradient(Color(0, 1, 1))).displace(Noise3D.cylindrical()),
        Point(np.array([-0.5, -0.8]) + offsets[3], ColorGradient(Color(1, 0, 0))).displace(Noise3D.cylindrical()),
        Point(np.array([ 0.5, -0.9]) + offsets[4], ColorGradient(Color(1, 0, 1))).displace(Noise3D.cylindrical()),
    ]
   

    for point in points:
        frame += point

    frame += ul_circle
    frame += ur_circle

    if progress >= 1:
        frame.set_last()


if __name__ == "__main__":
    factory = IldaFactory(
        fps=30,
        start_timestamp=0,
        factory_function=factory_function,
        ilda_filename="examples/output/displacement.ildx",
        point_density=0.001,
        show_excluision_zones=False,
        flip_x=False,
        flip_y=True
    )
    factory.run()
