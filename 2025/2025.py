import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory import Factory, IldxFrame, DmxFrame
from noise import Noise1D, Noise3D
from laser.color import Color, ColorGradient
import laser.shapes as ls

from dmx.fixture import Fixture
from dmx.channel import Channel

import json
import random
import numpy as np
from typing import List, Tuple
from itertools import cycle, product


class ShowFactory:

    factory: Factory
    start_ts: List[float]
    durations: List[float]

    # region settings

    FPS: int = 24
    TIMINGS: List[str] = [
        "00:00",
        "00:01",
        "04:00",
        "06:15",
        "07:30",
        "08:50",
        "10:20",
        "11:20",
        "11:24",
        "12:25",
        "13:43",
        "14:30"
    ]
    SAVE_DMX_AS_BINARY: bool = True
    POINT_DENSITY: float = 0.0003
    ILDX_COMPANY_NAME: str = "CR"
    ILDX_FRAME_NAMES: List[str] = [
        "Setup",
        "PreIntro",
        "Intro",
        "Wormhole",
        "NewSitu",
        "Attack",
        "Battle",
        "Freeze",
        "FinPush",
        "Aftrmath",
        "Epilogue"
    ]
    ILDX_FILENAME: str = "2025/2025.ildx"
    if SAVE_DMX_AS_BINARY:
        DMX_FILENAME: str = "2025/2025dmx.bin"
    else:
        DMX_FILENAME: str = "2025/2025dmx.json"

    # endregion

    # region: DMX fixtures
    lamp_l: Fixture
    lamp_r: Fixture
    fog_fury_l: Fixture
    fog_fury_r: Fixture
    relays: Fixture
    txt_laser: Fixture

    firewheel_motor: Channel

    LAMP_CHANNEL_COUNT: int = 8
    FOG_FURY_CHANNEL_COUNT: int = 7
    RELAY_CHANNEL_COUNT: int = 4
    TXT_LASER_CHANNEL_COUNT: int = 20

    LAMP_L_START_CHANNEL: int = 1
    LAMP_R_START_CHANNEL: int = 9  # 1 + 8
    FOG_FURY_L_START_CHANNEL: int = 17  # 9 + 8
    FOG_FURY_R_START_CHANNEL: int = 24  # 17 + 7
    RELAYS_START_CHANNEL: int = 31  # 24 + 7
    TXT_LASER_START_CHANNEL: int = 35  # 31 + 4

    def _load_fixtures(self):
        with open("dmx/fixtures/lixada_rgbw_leds.json", 'r') as f:
            self.lamp_l = Fixture.from_dict(json.load(f), self.LAMP_L_START_CHANNEL)
            f.seek(0)
            self.lamp_r = Fixture.from_dict(json.load(f), self.LAMP_R_START_CHANNEL)
        with open("dmx/fixtures/fog_fury_jett_7ch.json", 'r') as f:
            self.fog_fury_l = Fixture.from_dict(json.load(f), self.FOG_FURY_L_START_CHANNEL)
            f.seek(0)
            self.fog_fury_r = Fixture.from_dict(json.load(f), self.FOG_FURY_R_START_CHANNEL)
        with open("dmx/fixtures/dmx_relays.json", 'r') as f:
            self.relays = Fixture.from_dict(json.load(f), self.RELAYS_START_CHANNEL)
        with open("dmx/fixtures/showtec_galactic_txt.json", 'r') as f:
            self.txt_laser = Fixture.from_dict(json.load(f), self.TXT_LASER_START_CHANNEL)

        self.firewheel_motor = self.relays['relay_1']

    # endregion

    # region exclusion zones

    viewer_box: ls.Rectangle
    left_house_box: ls.Rectangle
    upper_house_box: ls.Rectangle
    house_triangle: ls.Polygon
    exclusion_zones: List[ls.Shape]

    def _create_exclusion_zones(self):
        color_gradient = ColorGradient(Color(1, 0, 0))
        self.viewer_box = ls.Rectangle(
            [-0.368347167968750, 0.327392578125000], 
            [ 0.390106201171875, 0.084564208984375], 
            color_gradient
        )
        self.left_house_box = ls.Rectangle(
            [-1,                 1],
            [-0.65618896484375, -1],
            color_gradient
        )
        self.upper_house_box = ls.Rectangle(
            [-1, 1],
            [ 1, 0.91204833984375],
            color_gradient
        )
        self.house_triangle = ls.Polygon(
            [
                [-1,                1], 
                [-0.65618896484375, 0.32739257812500], 
                [-0.36834716796875, 0.91204833984375]
            ],
            color_gradient
        )
        self.exclusion_zones = [
            self.viewer_box,
            self.left_house_box,
            self.upper_house_box,
            self.house_triangle
        ]

    # endregion

    E: float = 2.1 * (1 / FPS)

    def __init__(self):
        self._load_fixtures()
        self._create_exclusion_zones()
        self.start_ts, self.durations = self._build_timings()
        self.factory = Factory(
            fps=self.FPS,
            durations=self.durations,
            start_ts=self.start_ts,
            factory_functions=[
                self.setup_factory_function,
                self.pre_intro_factory_function,
                self.intro_factory_function,
                self.wormhole_factory_function,
                self.new_situation_factory_function,
                self.attack_shields_factory_function,
                self.battle_factory_function,
                self.freeze_factory_function,
                self.final_push_factory_function,
                self.aftermath_factory_function,
                self.epilogue_factory_function
            ],
            ildx_filename=self.ILDX_FILENAME,
            dmx_filename=self.DMX_FILENAME,
            save_dmx_as_binary=self.SAVE_DMX_AS_BINARY,
            point_density=self.POINT_DENSITY,
            ildx_company_name=self.ILDX_COMPANY_NAME,
            ildx_frame_names=self.ILDX_FRAME_NAMES,
            show_exclusion_zones=False  # TODO
        )
        for exclusion_zone in self.exclusion_zones:
            self.factory._ildx_factory.add_exclusion_zone(exclusion_zone)

    def _eqal_time(self, t1: float, t2: float) -> bool:
        return t1 - self.E <= t2 <= t1 + self.E
    
    def _time_between(self, t: float, t1: float, t2: float) -> bool:
        return t1 <= t <= t2

    def _string_to_seconds(self, t: str) -> float:
        m, s = map(int, t.split(':'))
        return m * 60 + s

    def _build_timings(self) -> Tuple[List[float], List[float]]:
        start_ts = []
        durations = []
        for t_start, t_end in zip(self.TIMINGS[:-1], self.TIMINGS[1:]):
            start_t = self._string_to_seconds(t_start)
            end_t = self._string_to_seconds(t_end)
            start_ts.append(start_t)
            durations.append(end_t - start_t)
        return start_ts, durations

    def setup_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        00:00 - 00:01
        """
        dmx_frame += self.relays['relay_1'] << 0
        dmx_frame += self.relays['relay_2'] << 0
        dmx_frame += self.relays['relay_3'] << 0
        dmx_frame += self.relays['relay_4'] << 0

        dmx_frame += self.lamp_l['red'] << 0
        dmx_frame += self.lamp_l['green'] << 0
        dmx_frame += self.lamp_l['blue'] << 0
        dmx_frame += self.lamp_l['white'] << 0
        dmx_frame += self.lamp_l['strobe']['off']()
        dmx_frame += self.lamp_l['dimmer'] << 1

        dmx_frame += self.lamp_r['red'] << 0
        dmx_frame += self.lamp_r['green'] << 0
        dmx_frame += self.lamp_r['blue'] << 0
        dmx_frame += self.lamp_r['white'] << 0
        dmx_frame += self.lamp_r['strobe']['off']()
        dmx_frame += self.lamp_r['dimmer'] << 1

        dmx_frame += self.fog_fury_l['fog']['off']()
        dmx_frame += self.fog_fury_l['red'] << 0
        dmx_frame += self.fog_fury_l['green'] << 0
        dmx_frame += self.fog_fury_l['blue'] << 0
        dmx_frame += self.fog_fury_l['amber'] << 0
        dmx_frame += self.fog_fury_l['strobing']['off']()
        dmx_frame += self.fog_fury_l['dimmer'] << 1

        dmx_frame += self.fog_fury_r['fog']['off']()
        dmx_frame += self.fog_fury_r['red'] << 0
        dmx_frame += self.fog_fury_r['green'] << 0
        dmx_frame += self.fog_fury_r['blue'] << 0
        dmx_frame += self.fog_fury_r['amber'] << 0
        dmx_frame += self.fog_fury_r['strobing']['off']()
        dmx_frame += self.fog_fury_r['dimmer'] << 1

        dmx_frame += self.txt_laser['shutter']['off']()

    def pre_intro_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        00:01 - 04:00
        """
        if ildx_frame.index == 0:
            self.lamp_l['white'] << 1
            self.lamp_r['white'] << 1

        t = 5 * 60 - 10
        if self._eqal_time(ildx_frame.t, t):
            dmx_frame += self.txt_laser['shutter']['off']()
            dmx_frame += self.txt_laser['color']['cyan']()
            dmx_frame += self.txt_laser['patterns1']['form7']()

        for i in range(9, -1, -1):
            t = 5 * 60 - i
            if self._eqal_time(ildx_frame.t, t):
                dmx_frame += self.txt_laser['patterns2'][f'form{i+1}']()

        t = 5 * 60 + 1
        if self._eqal_time(ildx_frame.t, t):
            dmx_frame += self.txt_laser['shutter']['on']()

        if ildx_frame.t >= 4 * 60 + 50:
            self.lamp_l['white']['default'].smooth(
                t=ildx_frame.t,
                start_t=4 * 60 + 50,
                end_t=5 * 60,
                start_value=1,
                end_value=0
            )
            self.lamp_r['white']['default'].smooth(
                t=ildx_frame.t,
                start_t=4 * 60 + 50,
                end_t=5 * 60,
                start_value=1,
                end_value=0
            )

    def intro_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        04:00 - 06:15
        """
        beauty_start = 4 * 60
        beauty_end = 4 * 60 + 23
        beauty_duration = beauty_end - beauty_start

        impulse_start = 4 * 60 + 23
        impulse_end = 5 * 60 + 9
        impulse_duration = impulse_end - impulse_start

        warp_start = 5 * 60 + 9
        warp_end = 6 * 60 + 15
        warp_duration = warp_end - warp_start

        impulse_fog_duration = 3
        warp_fog_duration = 4

        if beauty_start <= ildx_frame.t < beauty_end:
            beauty_progress = (ildx_frame.t - beauty_start) / beauty_duration
            amplitude = 0.8 * beauty_progress + 0.2
            dmx_frame += self.lamp_l['white']['default'].pulse(
                t=ildx_frame.t,
                amplitude=amplitude,
                frequency=4,
                phase=0
            )
            dmx_frame += self.lamp_r['white']['default'].pulse(
                t=ildx_frame.t,
                amplitude=amplitude,
                frequency=4,
                phase=2
            )
        if ildx_frame.t >= 5 * 60 + 23:
            dmx_frame += self.lamp_l['white'] << 0
            dmx_frame += self.lamp_r['white'] << 0

        if impulse_start <= ildx_frame.t < impulse_end:
            impulse_progress = (ildx_frame.t - impulse_start) / impulse_duration
            dmx_frame += self.lamp_l['red'] << 1
            dmx_frame += self.lamp_r['red'] << 1
            dmx_frame += self.lamp_l['strobe']['speed'] << impulse_progress
            dmx_frame += self.lamp_r['strobe']['speed'] << impulse_progress

            if ildx_frame.t < impulse_end - impulse_fog_duration:
                dmx_frame += self.fog_fury_l['red'] << 1
                dmx_frame += self.fog_fury_r['red'] << 1
                dmx_frame += self.fog_fury_l['strobing']['random'] << 1
                dmx_frame += self.fog_fury_r['strobing']['random'] << 1
                dmx_frame += self.fog_fury_l['fog']['on']()
                dmx_frame += self.fog_fury_r['fog']['on']()

        if warp_start <= ildx_frame.t < warp_end:
            warp_progress = (ildx_frame.t - warp_start) / warp_duration
            dmx_frame += self.lamp_l['strobe']['off']()
            dmx_frame += self.lamp_r['strobe']['off']()
            dmx_frame += self.lamp_l['red'] << 0
            dmx_frame += self.lamp_r['red'] << 0
            dmx_frame += self.fog_fury_l['fog']['off']()
            dmx_frame += self.fog_fury_r['fog']['off']()

            dmx_frame += self.lamp_l['blue'] << 1
            dmx_frame += self.lamp_r['blue'] << 1
            dmx_frame += self.lamp_l['white'] << warp_progress
            dmx_frame += self.lamp_r['white'] << warp_progress

            if ildx_frame.t < warp_end - warp_fog_duration:
                dmx_frame += self.fog_fury_l['blue'] << 1
                dmx_frame += self.fog_fury_r['blue'] << 1
                dmx_frame += self.fog_fury_l['strobing']['pulse'] << 0.1
                dmx_frame += self.fog_fury_r['strobing']['pulse'] << 0.1
                dmx_frame += self.fog_fury_l['fog']['on']()
                dmx_frame += self.fog_fury_r['fog']['on']()
            
        if ildx_frame.index == ildx_frame.total_frames - 1:
            dmx_frame += self.lamp_l['blue'] << 0
            dmx_frame += self.lamp_r['blue'] << 0
            dmx_frame += self.lamp_l['white'] << 0
            dmx_frame += self.lamp_r['white'] << 0
            dmx_frame += self.fog_fury_l['blue'] << 0
            dmx_frame += self.fog_fury_r['blue'] << 0
            dmx_frame += self.fog_fury_l['strobing']['off']()
            dmx_frame += self.fog_fury_r['strobing']['off']()
            dmx_frame += self.fog_fury_l['fog']['off']()
            dmx_frame += self.fog_fury_r['fog']['off']()

    class WormholeDisplacement:

        _warp_noise: Noise3D

        def __init__(self, rel_t: float):
            self._warp_noise = Noise3D.toroidal(
                amplitude=(rel_t / 15) * 0.04
            )

        def __call__(self, shape: ls.Shape, p: np.ndarray, s: float, t: float) -> np.ndarray:
            return p + self._warp_noise([p[0], p[1], t])

    def wormhole_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        6:15 - 07:30
        """
        fade_duration = 5

        rotation_start = 15
        wobble_start = 30
        displacement_start = 45
        fade_out_start_1 = 55
        fade_out_start_2 = 65

        n_colors = 14

        if ildx_frame.rel_t <= fade_duration:
            purple = Color.black().interpolate_hsv(Color.purple(), ildx_frame.rel_t / fade_duration)
            yellow = Color.black().interpolate_hsv(Color.yellow(), ildx_frame.rel_t / fade_duration)
        elif ildx_frame.rel_t > fade_out_start_1:
            purple = Color.purple().interpolate_hsv(Color.white(), (ildx_frame.rel_t - fade_out_start_1) / fade_duration)
            yellow = Color.yellow().interpolate_hsv(Color.white(), (ildx_frame.rel_t - fade_out_start_1) / fade_duration)
        elif ildx_frame.rel_t > fade_out_start_2:
            purple = Color.white().interpolate_hsv(Color.black(), (ildx_frame.rel_t - fade_out_start_2) / fade_duration)
            yellow = purple.copy()
        else:
            purple = Color.purple()
            yellow = Color.yellow()

        if random.random() < ildx_frame.progress / 2:
            if random.random() < 0.1:
                purple = Color.lime()
                yellow = Color.lime()
            elif random.random() < 0.55:
                purple = Color.white()
            else:
                yellow = Color.white()

        colors = []
        for i, color in enumerate(cycle((purple, yellow))):
            if i == n_colors:
                break
            colors.append((i / n_colors, color))

        color_gradient = ColorGradient(colors[0][1], colors[-1][1])
        for s, color in colors[1:-1]:
            if ildx_frame.rel_t >= rotation_start:
                s += ildx_frame.progress * (ildx_frame.rel_t - rotation_start) * 0.15
                while s > 1:
                    s -= 1
            color_gradient.add_color(s, color)
        
        wormhole = ls.Ellipse(
            self.viewer_box.center, 
            [
                1.15 * self.viewer_box.width / np.sqrt(2),
                1.15 * self.viewer_box.height / np.sqrt(2)
            ],
            color_gradient
        )

        if ildx_frame.rel_t >= wobble_start:
            wobble_noise = Noise1D(
                amplitude=0.02
            )
            wormhole.translate([
                wobble_noise.get_value(ildx_frame.rel_t),
                wobble_noise.get_value(ildx_frame.rel_t + 1337)
            ])

        if ildx_frame.rel_t >= displacement_start:
            wormhole.displace(self.WormholeDisplacement(ildx_frame.rel_t - displacement_start))

        ildx_frame += wormhole

        if ildx_frame.index == 0:
            self.lamp_l['dimmer'] << 1
            self.lamp_r['dimmer'] << 1
            self.firewheel_motor << 1

        if ildx_frame.index == self.FPS:
            self.firewheel_motor << 0

        if ildx_frame.rel_t <= fade_out_start_1:
            self.lamp_l['red']['default'].pulse(
                ildx_frame.t
            )
            self.lamp_r['red']['default'].pulse(
                ildx_frame.t
            )
            self.lamp_l['green']['default'].pulse(
                ildx_frame.t,
                frequency=0.5
            )
            self.lamp_r['green']['default'].pulse(
                ildx_frame.t,
                frequency=0.5,
                phase=0.5
            )
            self.lamp_l['blue']['default'].pulse(
                ildx_frame.t,
                frequency=0.5,
                phase=0.5
            )
            self.lamp_r['blue']['default'].pulse(
                ildx_frame.t,
                frequency=0.5
            )
        else:
            self.lamp_l['red'] << 0
            self.lamp_r['red'] << 0
            self.lamp_l['green'] << 0
            self.lamp_r['green'] << 0
            self.lamp_l['blue'] << 0
            self.lamp_r['blue'] << 0
            self.lamp_l['white'] << 1
            self.lamp_r['white'] << 1
            self.lamp_l['strobe']['speed'] << ildx_frame.rel_t / fade_duration
            self.lamp_r['strobe']['speed'] << ildx_frame.rel_t / fade_duration

        if dmx_frame.index == dmx_frame.total_frames:
            self.firewheel_motor << 1
            self.firewheel_motor << 0
            self.lamp_l['white'] << 0
            self.lamp_r['white'] << 0

    def new_situation_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        07:30 - 08:50
        """
        pass

    def attack_shields_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        08:50 - 10:20
        """
        explosion_times = [
            self._string_to_seconds(t) for t in [
                "9:30",
                "9:33",
                "9:36",
                
                "9:38",
                "9:40",
                "9:42",

                "9:43",
                "9:44",
                "9:45",
            ]
        ]
        explosion_positions = [
            [-0.8798828124999997,  -0.6477468390213816],
            [ 0.12808227539062497, -0.7698171515213816],
            [ 0.5894968133223687,  -0.7299901058799343],

            [ 0.1225425318667763,  -0.6837334883840461],
            [-0.8833473607113483,  -0.35269325657894746],
            [ 0.10670712119654602, -0.55047607421875],

            [ 0.11356393914473681, -0.4182241339432566],
            [ 0.6720902292351977,  -0.29776482833059215],
            [-0.8872600354646378,  -0.17821141293174353]
        ]

    def battle_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        10:20 - 11:20
        """
        lamps = [self.lamp_l, self.lamp_r]
        colors = ['red', 'green']

        if dmx_frame.index != dmx_frame.total_frames - 1:
            amplitude_noises = [Noise1D(0.1, 0.1, seed=random.ranint(0, 100)) for _ in range(len(product(lamps, colors)))]
            frequency_noises = [Noise1D(0.1, 0.1, seed=random.ranint(0, 100)) for _ in range(len(product(lamps, colors)))]
            
            for i, lamp, color in enumerate(product(lamps, colors)):
                dmx_frame += lamp[color]['default'].pulse(
                    t=dmx_frame.t,
                    amplitude=amplitude_noises[i].get_value(dmx_frame.rel_t),
                    frequency=frequency_noises[i].get_value(dmx_frame.rel_t),
                    shape=0,
                    duty=0.1
                )
        else:
            for lamp, color in product(lamps, colors):
                dmx_frame += lamp[color] << 0
      
    def freeze_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        11:20 - 11:24
        """
        if dmx_frame.index != dmx_frame.total_frames - 1:
            self.lamp_l['white'] << dmx_frame.progress
            self.lamp_r['white'] << dmx_frame.progress
            self.lamp_l['blue'] << dmx_frame.progress * 0.3
            self.lamp_r['blue'] << dmx_frame.progress * 0.3
            if dmx_frame.rel_t > dmx_frame.start_t - 3:
                self.fog_fury_l['red'] << 0.7
                self.fog_fury_l['green'] << 0.7
                self.fog_fury_l['blue'] << 1
                self.fog_fury_r['red'] << 0.7
                self.fog_fury_r['green'] << 0.7
                self.fog_fury_r['blue'] << 1
                self.fog_fury_l['fog']['on']()
                self.fog_fury_r['fog']['on']()
        else:
            self.lamp_l['white'] << 0
            self.lamp_r['white'] << 0
            self.lamp_l['blue'] << 0
            self.lamp_r['blue'] << 0

            self.fog_fury_l['red'] << 0
            self.fog_fury_l['green'] << 0
            self.fog_fury_l['blue'] << 0
            self.fog_fury_r['red'] << 0
            self.fog_fury_r['green'] << 0
            self.fog_fury_r['blue'] << 0
            self.fog_fury_l['fog']['off']()
            self.fog_fury_r['fog']['off']()

    def final_push_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        11:24 - 12:25
        """
        if dmx_frame.t >= 11 * 60 + 38:
            self.lamp_l['green'] << 1
            self.lamp_r['green'] << 1
            self.lamp_l['strobe']['speed'] << 0.5
            self.lamp_r['strobe']['speed'] << 0.5
        elif dmx_frame.t >= 11 * 60 + 40:
            rest_progress = (dmx_frame.t - 11 * 60 + 40) / 45
            self.lamp_l['green'] << 1 - rest_progress
            self.lamp_r['green'] << 1 - rest_progress
            self.lamp_l['white'] << rest_progress
            self.lamp_r['white'] << rest_progress
            self.lamp_l['strobe']['off']()
            self.lamp_r['strobe']['off']()
        elif dmx_frame.index == dmx_frame.total_frames - 1:
            self.lamp_l['green'] << 0
            self.lamp_r['green'] << 0
            self.lamp_l['white'] << 1
            self.lamp_r['white'] << 1

    def aftermath_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        12:25 - 13:43
        """
        if dmx_frame.index == 0:
            self.firewheel_motor << 1
        elif dmx_frame.index == dmx_frame.fps:
            self.firewheel_motor << 0
        elif dmx_frame.t >= 9 * 60 + 10:
            self.firewheel_motor << 1
        elif dmx_frame.t >= 9 * 60 + 11:
            self.firewheel_motor << 0

    def epilogue_factory_function(self, ildx_frame: IldxFrame, dmx_frame: DmxFrame):
        """
        13:43 - 14:30
        """
        self.lamp_l['white'] << 1
        self.lamp_r['white'] << 1
        self.lamp_l['red'] << 1
        self.lamp_r['red'] << 1
        self.lamp_l['green'] << 1
        self.lamp_r['green'] << 1
        self.lamp_l['blue'] << 1
        self.lamp_r['blue'] << 1

    def run(self):
        self.factory.run()


if __name__ == "__main__":
    show_factory = ShowFactory()
    show_factory.run()
