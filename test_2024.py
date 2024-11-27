import json
import numpy as np

from factory import Factory, IldxFrame, DmxFrame
from noise import Noise3D

from laser.color import Color, ColorGradient
from laser.shapes import Star, Shape

from dmx.fixture import Fixture


with open("dmx/fixtures/lixada_rgbw_leds.json", 'r') as f:
    lamp1 = Fixture.from_dict(json.load(f), 1)
    f.seek(0)
    lamp2 = Fixture.from_dict(json.load(f), 1 + len(lamp1))


class StarNoise:

    def __init__(self):
        self.noise = Noise3D.toroidal(amplitude = 0.1)

    def __call__(self, shape: Shape, p: np.ndarray, s: float, t: float) -> np.ndarray:
        return p + self.noise.get_value([p[0], p[1], t])


def factory_function(ildx_frame: IldxFrame, dmx_frame: DmxFrame):
    color_gradient = ColorGradient(Color(1, 0, 0), Color(0, 0, 1))  # FIXME
    color_gradient.add_color(0.5, Color(0, 1, 0))
    star = Star([0.0, 0.0], 0.3, 0.6, 5, color_gradient).rotate(2 * np.pi * ildx_frame.t / 10)
    if ildx_frame.progress > 0.5:
        star.displace(StarNoise())
    ildx_frame += star

    dmx_frame += lamp1.dimmer << 1
    dmx_frame += lamp1.red.default.pulse(dmx_frame.t)

    dmx_frame += lamp2.dimmer << 1
    dmx_frame += lamp2.green.default.pulse(dmx_frame.t)


if __name__ == "__main__":
    factory = Factory(
        fps=30,
        duration=20,
        start_t=0,
        factory_function=factory_function,
        ildx_filename="test_2024/laser.ildx",
        dmx_filename="test_2024/dmx.bin",
        point_density=0.001
    )
    factory.run()