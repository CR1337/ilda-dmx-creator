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
    frame += lamp2.green << 1


def factory_function(frame: Frame):
    progress = frame.t / 3.0

    if progress == 0.0:
        init(frame)

    if int(progress * 10) % 2 == 0:
        frame += lamp1.strobe.off()
        frame += lamp2.strobe.max()
    else:
        frame += lamp1.strobe.max()
        frame += lamp2.strobe.off()

    if progress >= 1:
        frame.set_last()


if __name__ == "__main__":
    factory = DmxFactory(
        fps=30,
        start_t=0,
        factory_function=factory_function,
        dmx_filename="examples/output/dmx_on_off.json",
        universe=0,
        save_as_binary=False
    )
    factory.run()
