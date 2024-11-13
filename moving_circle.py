from laser.frame_factory import IldaFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Circle, Square

from math import sin, cos, pi
import numpy as np


DURATION: float = 3.0


def factory_function(frame: Frame):
    progress = frame.timestamp / DURATION
    angle = progress * 2.0 * pi
    position = np.array([cos(angle), sin(angle)]) * 0.5
    circle = Circle(position, 0.5, ColorGradient(Color(0, 1, 0), Color(0, 1, 1)))
    circle.rotate(10 * angle)
    frame.add_shape(circle)
    if progress >= 1:
        frame.set_last()


if __name__ == "__main__":
    factory = IldaFactory(
        fps=30,
        start_timestamp=0,
        factory_function=factory_function,
        ilda_filename="moving_circle.ildx",
        point_density=0.01,
        show_excluision_zones=True,
        flip_x=False,
        flip_y=False
    )
    exclusion_square = Square([-1, -1], 1, ColorGradient(Color(1, 0, 0)))
    factory.add_exclusion_zone(exclusion_square)
    factory.run()
