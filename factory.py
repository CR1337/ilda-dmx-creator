from laser.ildx_factory import IldxFactory
from dmx.dmx_factory import DmxFactory
from dmx.frame import Frame as DmxFrame
from laser.frame import Frame as IldxFrame
from laser.shapes import Shape
from laser.color import Color
from typing import Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from math import ceil
from tqdm import tqdm


class FillFrame:

    _factory_function: Callable[[IldxFrame, DmxFrame], None]
    _exclusion_zones: List[Tuple[Shape, Color]]
    _shoe_exclusion_zones: bool

    def __init__(self, factory_function: Callable[[IldxFrame, DmxFrame], None], exclusion_zones: List[Tuple[Shape, Color]] = [], show_exclusion_zones: bool = False):
        self._factory_function = factory_function
        self._exclusion_zones = exclusion_zones
        self._show_exclusion_zones = show_exclusion_zones

    def __call__(self, frames: Tuple[IldxFrame, DmxFrame]) -> Tuple[IldxFrame, DmxFrame]:
        frame, dmx_frame = frames
        self._factory_function(frame, dmx_frame)
        if self._show_exclusion_zones:
            for exclusion_shape, _ in self._exclusion_zones:
                frame.add_shape(exclusion_shape, is_exclusion_shape=True)
        return frame, dmx_frame


class Factory:

    _factory_functions: List[Callable[[IldxFrame, DmxFrame], None]]

    _fps: float
    _start_ts: List[float]
    _durations: List[float]
    _point_density: float

    _ildx_factory: IldxFactory
    _dmx_factory: DmxFactory

    def _empty_ildx_factory_function(frame: IldxFrame):
        pass

    def _empty_dmx_factory_function(frame: DmxFrame):
        pass

    def __init__(
        self,
        fps: float,
        durations: List[float],
        start_ts: List[float],
        factory_functions: List[Callable[[IldxFrame, DmxFrame], None]],
        ildx_filename: str,
        dmx_filename: str,
        point_density: float,
        show_exclusion_zones: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        ildx_frame_name: str = "",
        ildx_company_name: str = "",
        ildx_projector_number: int = 0,
        dmx_universe: int = 0,
        save_dmx_as_binary: bool = True
    ):
        self._factory_functions = factory_functions if isinstance(factory_functions, list) else [factory_functions]
        self._start_ts = start_ts if isinstance(start_ts, list) else [start_ts]
        self._fps = fps
        self._durations = durations if isinstance(durations, list) else [durations]
        self._point_density = point_density
        self._ildx_factory = IldxFactory(
            fps,
            durations,
            start_ts,
            [self._empty_ildx_factory_function] * len(self._factory_functions),
            ildx_filename,
            point_density,
            show_exclusion_zones,
            flip_x,
            flip_y,
            ildx_frame_name,
            ildx_company_name,
            ildx_projector_number
        )
        self._dmx_factory = DmxFactory(
            fps,
            durations,
            start_ts,
            [self._empty_dmx_factory_function] * len(self._factory_functions),
            dmx_filename,
            dmx_universe,
            save_dmx_as_binary
        )
    
    def _compute_frames(self) -> Tuple[List[List[IldxFrame]], List[List[DmxFrame]]]:
        print("Computing frames...")
        ildx_animations, dmx_animations = [], []
        for start_t, duration, factory_function in zip(self._start_ts, self._durations, self._factory_functions):
            frame_count = ceil(self._fps * duration)
            empty_frames = (
                (IldxFrame(start_t, start_t + (i / self._fps), self._fps, duration, self._point_density), DmxFrame(start_t, start_t + (i / self._fps), self._fps, duration))
                for i in range(frame_count)
            )
            with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
                exclusion_zones = self._ildx_factory._exclusion_zones
                show_exclusion_zones = self._ildx_factory._show_exclusion_zones
                frames = list(tqdm(
                    executor.map(FillFrame(factory_function, exclusion_zones, show_exclusion_zones), empty_frames),
                    total=frame_count,
                    desc=f"Animation {len(ildx_animations) + 1}/{len(self._durations)}"
                ))
            ildx_frames, dmx_frames = zip(*frames)
            ildx_animations.append(ildx_frames)
            dmx_animations.append(dmx_frames)
        return ildx_animations, dmx_animations

    def run(self):
        ildx_animations, dmx_animations = self._compute_frames()

        render_lines = self._ildx_factory._compute_render_lines(ildx_animations)
        self._ildx_factory._write_file(render_lines)

        channels = self._dmx_factory._compute_channels(dmx_animations)
        self._dmx_factory._write_file(channels)

        print("Done!")
