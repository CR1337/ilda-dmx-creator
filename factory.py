from laser.ildx_factory import IldxFactory
from dmx.dmx_factory import DmxFactory
from dmx.frame import Frame as DmxFrame
from laser.frame import Frame as IldxFrame
from typing import Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from math import ceil


class Factory:

    _factory_function: Callable[[IldxFrame, DmxFrame], None]

    _start_t: float
    _fps: float
    _duration: float
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
        duration: float,
        start_t: float,
        factory_function: Callable[[IldxFrame, DmxFrame], None],
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
        self._factory_function = factory_function
        self._start_t = start_t
        self._fps = fps
        self._duration = duration
        self._point_density = point_density
        self._ildx_factory = IldxFactory(
            fps,
            duration,
            start_t,
            self._empty_ildx_factory_function,
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
            duration,
            start_t,
            self._empty_dmx_factory_function,
            dmx_filename,
            dmx_universe,
            save_dmx_as_binary
        )

    def _fill_frame(self, frame: IldxFrame, dmx_frame: DmxFrame) -> Tuple[IldxFrame, DmxFrame]:
        self._factory_function(frame, dmx_frame)
        if self._ildx_factory._show_exclusion_zones:
            for exclusion_shape, _ in self._ildx_factory._exclusion_zones:
                frame.add_shape(exclusion_shape, is_exclusion_shape=True)
        return frame, dmx_frame
    
    def _compute_frames(self) -> Tuple[List[IldxFrame], List[DmxFrame]]:
        print("Computing frames...")
        empty_frames = (
            (IldxFrame(self._start_t + (i / self._fps), self._fps, self._duration, self._point_density), DmxFrame(self._start_t + (i / self._fps), self._fps, self._duration))
            for i in range(ceil(self._fps * self._duration))
        )
        with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
            frames = list(executor.map(lambda x: self._fill_frame(*x), empty_frames))
        ildx_frames, dmx_frames = zip(*frames)
        return ildx_frames, dmx_frames

    def run(self):
        ildx_frames, dmx_frames = self._compute_frames()

        render_lines = self._ildx_factory._compute_render_lines(ildx_frames)
        self._ildx_factory._write_file(render_lines)

        channels = self._dmx_factory._compute_channels(dmx_frames)
        self._dmx_factory._write_file(channels)

        print("Done!")
