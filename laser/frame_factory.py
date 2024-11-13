from laser.shapes.shape import Shape
from laser.frame import Frame
from laser.color import Color
from laser.render_line import RenderLine
from laser.ildx import ILDX_MAGIC, IldxHeader, Ilda2dTrueColorRecord, adjust_start_timestamp, ILDA_STATUS_CODE_BLANKING_MASK, ILDA_STATUS_CODE_LAST_POINT_MASK
from typing import Callable, List, Tuple

from tqdm import tqdm


class IldaFactory:
    
    _fps: float
    _start_timestamp: float
    _factory_function: Callable[[Frame], None]
    _ilda_filename: str
    _point_density: float
    _show_exclusion_zones: bool
    _flip_x: bool
    _flip_y: bool

    _exclusion_zones: List[Tuple[Shape, bool]]

    def __init__(
        self, 
        fps: float, 
        start_timestamp: float, 
        factory_function: Callable[[Frame], None],
        ilda_filename: str,
        point_density: float,
        show_excluision_zones: bool = False,
        flip_x: bool = False,
        flip_y: bool = False
    ):
        self._fps = fps
        self._start_timestamp = start_timestamp
        self._factory_function = factory_function
        self._ilda_filename = ilda_filename
        self._point_density = point_density
        self._show_exclusion_zones = show_excluision_zones
        self._flip_x = flip_x
        self._flip_y = flip_y

        self._exclusion_zones = []

    def add_exclusion_zone(self, shape: Shape, inside: bool = True):
        self._exclusion_zones.append((shape, inside))

    def _compute_frames(self) ->List[Frame]:
        print("Computing frames...")
        frames = []
        timestamp = self._start_timestamp
        while True:
            frame = Frame(timestamp, self._fps, self._point_density)
            self._factory_function(frame)
            if self._show_exclusion_zones:
                for exclusion_shape, _ in self._exclusion_zones:
                    frame.add_shape(exclusion_shape, is_exclusion_shape=True)
            frames.append(frame)
            if frame.is_last:
                break
            timestamp += 1.0 / self._fps
        return frames
    
    def _compute_render_lines(self, frames: List[Frame]) -> List[List[RenderLine]]:
        print("Computing lines...")
        render_lines = []
        for frame in tqdm(frames, total=len(frames)):
            render_lines.append([])

            for (shape, is_exclusion_shape), (next_shape, _) in zip(frame.shapes, frame.shapes[1:] + [(None, None)]):

                for render_line in shape.get_render_lines():
                    if not is_exclusion_shape:

                        for exclusion_shape, is_inside in self._exclusion_zones:
                            if (
                                is_inside and exclusion_shape.is_line_inside(render_line.start, render_line.end)
                                or not is_inside and exclusion_shape.is_line_outside(render_line.start, render_line.end)
                            ):
                                render_line.blank()

                    render_lines[-1].append(render_line)
                
                if next_shape:
                    render_lines[-1].append(
                        RenderLine(
                            render_lines[-1][-1].end,
                            next(next_shape.get_render_lines()).start,
                            Color.black(),
                            blanked=True
                        )
                    )

        if self._flip_x:
            for frame in render_lines:
                for render_line in frame:
                    render_line.flip_x()

        if self._flip_y:
            for frame in render_lines:
                for render_line in frame:
                    render_line.flip_y()

        return render_lines
    
    def _write_file(self, render_lines: List[List[RenderLine]]):
        print("Writing file...")
        target = bytearray()
        for frame_idx, frame in tqdm(enumerate(render_lines), total=len(render_lines)):
            header = IldxHeader(
                ildxMagic=ILDX_MAGIC,
                startTimestamp=adjust_start_timestamp(self._start_timestamp),
                formatCode=5,  # TODO
                frameName=bytes("", encoding="ascii"),  # TODO
                companyName=bytes("", encoding="ascii"),  # TODO
                numberOfRecords=len(frame),
                frameOrPaletteNumber=frame_idx,
                totalFrames=len(render_lines),
                projectorNumber=0,  # TODO
                framePerSecondOrFrameAmout=self._fps if frame_idx == 0 else 1
            )
            target.extend(bytearray(header))
            for line_idx, render_line in enumerate(frame):
                # TODO: maybe we need another point at the start?
                status_code = 0
                if render_line.blanked:
                    status_code |= ILDA_STATUS_CODE_BLANKING_MASK
                if line_idx == len(frame) - 1:
                    status_code |= ILDA_STATUS_CODE_LAST_POINT_MASK
                record = Ilda2dTrueColorRecord(
                    x=int((render_line.end[0] * 0.5 + 0.5) * Shape.ILDA_RESOLUTION),
                    y=int((render_line.end[1] * 0.5 + 0.5) * Shape.ILDA_RESOLUTION),
                    statusCode=status_code,
                    r=int(255 * render_line.color.r),
                    g=int(255 * render_line.color.g),
                    b=int(255 * render_line.color.b)
                )
                target.extend(bytearray(record))
        with open(self._ilda_filename, 'wb') as file:
            file.write(target)
    
    def run(self):
        frames = self._compute_frames()
        render_lines = self._compute_render_lines(frames)
        self._write_file(render_lines)
        print("Done!")
