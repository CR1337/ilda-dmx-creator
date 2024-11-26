import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory import Factory, IldxFrame, DmxFrame
import json
import numpy as np
from laser.shapes import Circle
from laser.color import Color, ColorGradient
from dmx.fixture import Fixture


with open("dmx/fixtures/lixada_rgbw_leds.json", 'r') as f:
    lamp = Fixture.from_dict(json.load(f), 1)


def factory_function(ildx_frame: IldxFrame, dmx_frame: DmxFrame):
    dmx_frame += lamp.dimmer << 1
    dmx_frame += lamp.red << dmx_frame.progress

    ildx_frame += Circle(
        np.array([0.0, 0.0]), 
        0.5, 
        ColorGradient(Color(1, 0, 0), Color(0, 1, 0))
    ).rotate(2 * np.pi * ildx_frame.progress)


if __name__ == "__main__":
    factory = Factory(
        fps=30,
        durations=3.0,
        start_ts=0,
        factory_functions=factory_function,
        ildx_filename="examples/output/combined.ildx",
        dmx_filename="examples/output/combined.json",
        point_density=0.001,
        flip_y=True,
        save_dmx_as_binary=False
    )
    factory.run()
