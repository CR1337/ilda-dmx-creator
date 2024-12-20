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
    frame += lamp1.red << 1


def factory_function(frame: Frame):
    if frame.progress == 0.0:
        init(frame)

    if frame.progress < 0.5:
        frame += lamp1.strobe.speed.lerp(
            frame.t,
            0.0, frame.duration / 2,
            0.0, 1.0
        )
    else:
        frame += lamp1.strobe.speed.lerp(
            frame.t,
            frame.duration / 2, frame.duration,
            1.0, 0.0
        )

    frame += lamp2.red.default.lerp(
        frame.t,
        0.0, frame.duration,
        0.0, 1.0
    )


if __name__ == "__main__":
    factory = DmxFactory(
        fps=30,
        start_ts=0,
        durations=3.0,
        factory_functions=factory_function,
        dmx_filename="examples/output/dmx_lerp.json",
        universe=0,
        save_as_binary=False
    )
    factory.run()
