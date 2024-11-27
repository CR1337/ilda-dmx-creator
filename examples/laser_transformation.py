import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laser.ildx_factory import IldxFactory, Frame
from laser.color import Color, ColorGradient
from laser.shapes import Star

import numpy as np


def factory_function(frame: Frame):
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
    star10.translate(np.array([0.1, 0.1]) * (frame.progress - frame.duration / 2))
    star20.rotate(frame.progress * 2 * np.pi)
    star01.scale(np.array([1.1, 1.1]) * (1 - frame.progress))
    star11.shear(np.array([0.1, 1.0]) * frame.progress)
    star21.skew(np.array([0.1, 1.0]) * frame.progress)
    star02.reflect(np.array([1, 0]) if int(10 * frame.progress) % 2 == 0 else np.array([0, 1]))
    star12.project(frame.progress * 2 * np.pi)
    star22.transform(np.array([
        [0.8 + 0.2 * frame.progress, -0.3 + 0.3 * frame.progress, 0],
        [0.6 - 0.6 * frame.progress, 0.9 + 0.1 * frame.progress, 0],
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
        durations=3.0,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/transformation.ildx",
        point_density=0.0005
    )
    factory.run()
