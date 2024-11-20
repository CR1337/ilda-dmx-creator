from laser.shapes.shape import Shape
from laser.frame import Frame
from laser.color import Color
from laser.render_line import RenderLine
from laser.ildx import ILDX_MAGIC, IldxHeader, Ilda2dTrueColorRecord, adjust_start_timestamp, ILDA_STATUS_CODE_BLANKING_MASK, ILDA_STATUS_CODE_LAST_POINT_MASK
from typing import Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from tqdm import tqdm


class IldaFactory:

    ILDA_NAME_LENGTH: int = 8
    FORMAT_CODE_2D_TRUE_COLOR: int = 5
    
    _fps: float
    _start_timestamp: float
    _factory_function: Callable[[Frame], None]
    _ilda_filename: str
    _point_density: float
    _show_exclusion_zones: bool
    _flip_x: bool
    _flip_y: bool
    _frame_name: str
    _company_name: str
    _projector_number: int

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
        flip_y: bool = False,
        frame_name: str = "",
        company_name: str = "",
        projector_number: int = 0
    ):
        self._fps = fps
        self._start_timestamp = start_timestamp
        self._factory_function = factory_function
        self._ilda_filename = ilda_filename
        self._point_density = point_density
        self._show_exclusion_zones = show_excluision_zones
        self._flip_x = flip_x
        self._flip_y = flip_y
        self._frame_name = self._format_ilda_name(frame_name)
        self._company_name = self._format_ilda_name(company_name)
        self._projector_number = projector_number

        self._exclusion_zones = []

    def add_exclusion_zone(self, shape: Shape, inside: bool = True):
        self._exclusion_zones.append((shape, inside))

    def _format_ilda_name(self, name: str) -> str:
        if len(name) > self.ILDA_NAME_LENGTH:
            return name[:self.ILDA_NAME_LENGTH]
        elif len(name) < self.ILDA_NAME_LENGTH:
            return name + " " * (self.ILDA_NAME_LENGTH - len(name))
        else:
            return name

    def _compute_frames(self) -> List[Frame]:
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
    
    def _compute_render_lines_for_frame(self, frame: Frame) -> List[RenderLine]:
        render_lines = []
        for (shape, is_exclusion_shape), (next_shape, _) in zip(frame.shapes, frame.shapes[1:] + [(None, None)]):

            for render_line in shape.get_render_lines(frame.timestamp):
                if not is_exclusion_shape:

                    for exclusion_shape, is_inside in self._exclusion_zones:
                        if (
                            is_inside and exclusion_shape.is_line_inside(render_line.start, render_line.end)
                            or not is_inside and exclusion_shape.is_line_outside(render_line.start, render_line.end)
                        ):
                            render_line.blank()

                render_lines.append(render_line)
            
            if next_shape:
                try:
                    render_lines.append(
                        RenderLine(
                            render_lines[-1].end,
                            next(next_shape.get_render_lines(frame.timestamp)).start,
                            Color.black(),
                            blanked=True
                        )
                    )
                except (StopIteration, IndexError):
                    pass
                
        render_lines.insert(0, render_lines[0].copy())
        return render_lines
    
    def _compute_render_lines(self, frames: List[Frame]) -> List[List[RenderLine]]:
        print("Computing ILDX lines...")
        render_lines = []
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            for frame in tqdm(executor.map(self._compute_render_lines_for_frame, frames), total=len(frames)):
                render_lines.append(frame)

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
        print("Writing ILDX file...")
        target = bytearray()
        for frame_idx, frame in tqdm(enumerate(render_lines), total=len(render_lines)):
            header = IldxHeader(
                ildxMagic=ILDX_MAGIC,
                startTimestamp=adjust_start_timestamp(self._start_timestamp),
                formatCode=self.FORMAT_CODE_2D_TRUE_COLOR,
                companyName=bytes(self._company_name, encoding="ascii"),
                frameName=bytes(self._frame_name, encoding="ascii"),
                numberOfRecords=len(frame),
                frameOrPaletteNumber=frame_idx,
                totalFrames=len(render_lines),
                projectorNumber=self._projector_number,
                framesPerSecondOrFrameAmount=self._fps if frame_idx == 0 else 1
            )
            target.extend(bytearray(header))
            for line_idx, render_line in enumerate(frame):
                status_code = 0
                if render_line.blanked:
                    status_code |= ILDA_STATUS_CODE_BLANKING_MASK
                if line_idx == len(frame) - 1:
                    status_code |= ILDA_STATUS_CODE_LAST_POINT_MASK
                record = Ilda2dTrueColorRecord(
                    x=int(render_line.end[0] * Shape.ILDA_RESOLUTION * 0.5),
                    y=int(render_line.end[1] * Shape.ILDA_RESOLUTION * 0.5),
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
