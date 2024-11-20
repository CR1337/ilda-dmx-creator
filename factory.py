from laser.ilda_factory import IldaFactory
from dmx.dmx_factory import DmxFactory
from dmx.frame import Frame as DmxFrame
from laser.frame import Frame as IldxFrame
from typing import Callable, List, Tuple


class Factory:

    _factory_function: Callable[[IldxFrame, DmxFrame], None]

    _start_timestamp: float
    _fps: float

    _ildx_factory: IldaFactory
    _dmx_factory: DmxFactory

    def __init__(
        self,
        fps: float,
        start_timestamp: float,
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
        self._start_timestamp = start_timestamp
        self._fps = fps
        self._ildx_factory = IldaFactory(
            fps,
            start_timestamp,
            lambda _: None,
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
            start_timestamp,
            lambda _: None,
            dmx_filename,
            dmx_universe,
            save_dmx_as_binary
        )

    def _compute_frames(self) -> Tuple[List[IldxFrame], List[DmxFrame]]:
        print("Computing frames...")
        ilda_frames = []
        dmx_frames = []
        ilda_done, dmx_done = False, False
        timestamp = self._start_timestamp
        while not (ilda_done and dmx_done):
            ilda_frame = IldxFrame(timestamp, self._fps, self._ildx_factory._point_density)
            dmx_frame = DmxFrame(timestamp, self._fps)
            self._factory_function(ilda_frame, dmx_frame)
            if ilda_frame.is_last:
                ilda_done = True
            if dmx_frame.is_last:
                dmx_done = True
            timestamp += 1.0 / self._ildx_factory._fps
            if not ilda_done:
                ilda_frames.append(ilda_frame)
            if not dmx_done:
                dmx_frames.append(dmx_frame)
        return ilda_frames, dmx_frames

    def run(self):
        ilda_frames, dmx_frames = self._compute_frames()

        render_lines = self._ildx_factory._compute_render_lines(ilda_frames)
        self._ildx_factory._write_file(render_lines)

        channels = self._dmx_factory._compute_channels(dmx_frames)
        self._dmx_factory._write_file(channels)

        print("Done!")
