import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from dmx.dmx_factory import DmxFactory, Frame
from dmx.fixture import Fixture

with open("dmx/fixtures/lixada_rgbw_leds.json", "r") as f:
    lamp1 = Fixture.from_dict(json.load(f), 1, "lamp1")
    f.seek(0)
    lamp2 = Fixture.from_dict(json.load(f), 1 + len(lamp1), "lamp2")


DURATION: float = 3.0


def init(frame: Frame):
    frame += lamp1.dimmer << 1
    frame += lamp2.dimmer << 1


def factory_function(frame: Frame):
    progress = frame.timestamp / 3.0

    if progress == 0.0:
        init(frame)

    frame += lamp1.red.default.pulse(
        frame.timestamp,
        1.0, 1.5,
        0.0, 0.0
    )
    frame += lamp2.red.default.pulse(
        frame.timestamp,
        0.5, 0.15,
        0.0, 0.0
    )

    if progress >= 1:
        frame.set_last()


if __name__ == "__main__":
    factory = DmxFactory(
        fps=30,
        start_timestamp=0,
        factory_function=factory_function,
        dmx_filename="examples/output/dmx_pulse_square.json",
        universe=0,
        save_as_binary=False
    )
    factory.run()
