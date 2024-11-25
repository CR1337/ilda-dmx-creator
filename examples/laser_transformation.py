import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Star

import numpy as np


DURATION: float = 3.0


def factory_function(frame: Frame):
    progress = frame.t / DURATION

    color_gradient = ColorGradient(Color(1, 0, 0))
    color_gradient.add_color(0.5, Color(0, 0, 1))
    
    frame += (star00 := Star(np.array([0.0, 0.0]), 0.1, 0.2, 5, color_gradient))
    frame += (star10 := star00.copy())
    frame += (star20 := star00.copy())
    frame += (star01 := star00.copy())
    frame += (star11 := star00.copy())
    frame += (star21 := star00.copy())
    frame += (star02 := star00.copy())
    frame += (star12 := star00.copy())
    frame += (star22 := star00.copy())

    star00.identity()
    star10.translate(np.array([0.1, 0.1]) * (progress - DURATION / 2))
    star20.rotate(progress * 2 * np.pi)
    star01.scale(np.array([1.1, 1.1]) * (1 - progress))
    star11.shear(np.array([0.1, 1.0]) * progress)
    star21.skew(np.array([0.1, 1.0]) * progress)
    star02.reflect(np.array([1, 0]) if int(10 * progress) % 2 == 0 else np.array([0, 1]))
    star12.project(progress * 2 * np.pi)
    star22.transform(np.array([
        [0.8 + 0.2 * progress, -0.3 + 0.3 * progress, 0],
        [0.6 - 0.6 * progress, 0.9 + 0.1 * progress, 0],
        [0, 0, 1]
    ]))

    star00.translate(np.array([-0.5,  0.5]))
    star10.translate(np.array([ 0.0,  0.5]))
    star20.translate(np.array([ 0.5,  0.5]))
    star01.translate(np.array([-0.5,  0.0]))
    star11.translate(np.array([ 0.0,  0.0]))
    star21.translate(np.array([ 0.5,  0.0]))
    star02.translate(np.array([-0.5, -0.5]))
    star12.translate(np.array([ 0.0, -0.5]))
    star22.translate(np.array([ 0.5, -0.5]))
    

if __name__ == "__main__":
    factory = IldxFactory(
        fps=30,
        duration=DURATION,
        start_t=0,
        factory_function=factory_function,
        ildx_filename="examples/output/transformation.ild",
        point_density=0.0005,
        show_excluision_zones=False,
        flip_x=False,
        flip_y=False,
        legacy_mode=True
    )
    factory.run()
