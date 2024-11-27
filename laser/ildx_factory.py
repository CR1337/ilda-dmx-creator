from laser.shapes.shape import Shape
from laser.frame import Frame
from laser.color import Color
from laser.render_line import RenderLine
from laser.ildx import ILDA_MAGIC, ILDX_MAGIC, IldxHeader, Ilda2dTrueColorRecord, adjust_start_time, zero_start_time, ILDX_STATUS_CODE_BLANKING_MASK, ILDX_STATUS_CODE_LAST_POINT_MASK
from typing import Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from math import ceil

from tqdm import tqdm


class FillFrame:

    _factory_function: Callable[[Frame], None]
    _exclusion_zones: List[Tuple[Shape, bool]]
    _show_exclusion_zones: bool

    def __init__(self, factory_function: Callable[[Frame], None], exclusion_zones: List[Tuple[Shape, bool]], show_exclusion_zones: bool):
        self._factory_function = factory_function
        self._exclusion_zones = exclusion_zones
        self._show_exclusion_zones = show_exclusion_zones
        
    def __call__(self, frame: Frame) -> Frame:
        self._factory_function(frame)
        if self._show_exclusion_zones:
            for exclusion_shape, _ in self._exclusion_zones:
                frame.add_shape(exclusion_shape, is_exclusion_shape=True)
        return frame


class IldxFactory:

    ILDX_NAME_LENGTH: int = 8
    FORMAT_CODE_2D_TRUE_COLOR: int = 5
    FRAME_BATCH_SIZE_FACTOR: int = 2
    
    _fps: float
    _durations: List[float]
    _start_ts: List[float]
    _factory_functions: List[Callable[[Frame], None]]
    _ildx_filename: str
    _point_density: float
    _show_exclusion_zones: bool
    _flip_x: bool
    _flip_y: bool
    _frame_names: List[str]
    _company_name: str
    _projector_number: int
    _legacy_mode: bool

    _exclusion_zones: List[Tuple[Shape, bool]]

    def __init__(
        self, 
        fps: float, 
        start_ts: List[float], 
        durations: List[float],
        factory_functions: List[Callable[[Frame], None]],
        ildx_filename: str,
        point_density: float,
        show_excluision_zones: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        frame_names: List[str] = [],
        company_name: str = "",
        projector_number: int = 0,
        legacy_mode: bool = False
    ):
        self._fps = fps
        self._durations = durations if isinstance(durations, list) else [durations]
        self._start_ts = start_ts if isinstance(start_ts, list) else [start_ts]
        self._factory_functions = factory_functions if isinstance(factory_functions, list) else [factory_functions]
        self._ildx_filename = ildx_filename
        self._point_density = point_density
        self._show_exclusion_zones = show_excluision_zones
        self._flip_x = flip_x
        self._flip_y = flip_y
        if isinstance(frame_names, str):
            self._frame_names = [self._format_ildx_name(frame_names)]
        else:
            self._frame_names = [self._format_ildx_name(name) for name in frame_names]
        self._company_name = self._format_ildx_name(company_name)
        self._projector_number = projector_number
        self._legacy_mode = legacy_mode

        while len(self._frame_names) < len(self._durations):
            self._frame_names.append(self._format_ildx_name(""))

        self._exclusion_zones = []

    def add_exclusion_zone(self, shape: Shape, inside: bool = True):
        self._exclusion_zones.append((shape, inside))

    def _format_ildx_name(self, name: str) -> str:
        if len(name) > self.ILDX_NAME_LENGTH:
            return name[:self.ILDX_NAME_LENGTH]
        elif len(name) < self.ILDX_NAME_LENGTH:
            return name + " " * (self.ILDX_NAME_LENGTH - len(name))
        else:
            return name

    def _compute_frames(self) -> List[List[Frame]]:
        print("Computing ILDX animations...")
        animations = []
        for start_t, duration, factory_function in zip(self._start_ts, self._durations, self._factory_functions):
            empty_frames = (
                Frame(start_t, start_t + (i / self._fps), self._fps, duration, self._point_density) 
                for i in range(ceil(self._fps * duration))
            )
            with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
                frames = list(tqdm(
                    executor.map(FillFrame(factory_function, self._exclusion_zones, self._show_exclusion_zones), empty_frames), 
                    total=ceil(self._fps * duration), 
                    desc=f"Animation {len(animations) + 1}/{len(self._durations)}"
                ))
            animations.append(frames)
        return animations
    
    def _compute_render_lines_for_frame(self, frame: Frame) -> List[RenderLine]:
        render_lines = []
        for (shape, is_exclusion_shape), (next_shape, _) in zip(frame.shapes, frame.shapes[1:] + [(None, None)]):

            for render_line in shape.get_render_lines(frame.t):
                if not is_exclusion_shape:

                    for exclusion_shape, is_inside in self._exclusion_zones:
                        if (
                            is_inside and exclusion_shape.is_line_inside(render_line.p0, render_line.p1)
                            or not is_inside and exclusion_shape.is_line_outside(render_line.p0, render_line.p1)
                        ):
                            render_line.blank()

                render_lines.append(render_line)
            
            if next_shape:
                try:
                    render_lines.append(
                        RenderLine(
                            render_lines[-1].p1,
                            next(next_shape.get_render_lines(frame.t)).p0,
                            Color.black(),
                            blanked=True
                        )
                    )
                except (StopIteration, IndexError):
                    pass
                
        if render_lines:
            render_lines.insert(0, render_lines[0].copy())
        return render_lines
    
    def _compute_render_lines(self, animations: List[List[Frame]]) -> List[List[List[RenderLine]]]:
        print("Computing ILDX lines...")
        all_render_lines = []
        for animation in animations:
            render_lines = []
            with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
                for frame in tqdm(
                    executor.map(self._compute_render_lines_for_frame, animation), 
                    total=len(animation),
                    desc=f"Animation {len(all_render_lines) + 1}/{len(animations)}"
                ):
                    render_lines.append(frame)
            # for frame in frames:
            #     render_lines.append(self._compute_render_lines_for_frame(frame))

            if self._flip_x:
                for frame in render_lines:
                    for render_line in frame:
                        render_line.flip_x()

            if self._flip_y:
                for frame in render_lines:
                    for render_line in frame:
                        render_line.flip_y()

            all_render_lines.append(render_lines)
        return all_render_lines
    
    def _write_file(self, render_lines: List[List[List[RenderLine]]]):
        print("Writing ILDX file...")
        target = bytearray()
        for animation_idx, animation in enumerate(render_lines):
            for frame_idx, frame in tqdm(
                enumerate(animation), 
                total=len(animation),
                desc=f"Animation {len(target) + 1}/{len(render_lines)}"
            ):
                header = IldxHeader(
                    ildxMagic=ILDA_MAGIC,
                    starttime=(
                        zero_start_time() if self._legacy_mode
                        else adjust_start_time(self._start_ts[animation_idx])
                    ),
                    formatCode=self.FORMAT_CODE_2D_TRUE_COLOR,
                    companyName=bytes(self._company_name, encoding="ascii"),
                    frameName=bytes(self._frame_names[animation_idx], encoding="ascii"),
                    numberOfRecords=len(frame),
                    frameOrPaletteNumber=frame_idx,
                    totalFrames=len(animation),
                    projectorNumber=self._projector_number,
                    framesPerSecondOrFrameAmount=(
                        0 if self._legacy_mode
                        else self._fps if frame_idx == 0 else 1
                    )
                )
                target.extend(bytearray(header))
                for line_idx, render_line in enumerate(frame):
                    status_code = 0
                    if render_line.blanked:
                        status_code |= ILDX_STATUS_CODE_BLANKING_MASK
                    if line_idx == len(frame) - 1:
                        status_code |= ILDX_STATUS_CODE_LAST_POINT_MASK
                    record = Ilda2dTrueColorRecord(
                        x=int(render_line.p1[0] * Shape.ILDX_RESOLUTION * 0.5),
                        y=int(render_line.p1[1] * Shape.ILDX_RESOLUTION * 0.5),
                        statusCode=status_code,
                        r=int(255 * render_line.color.r),
                        g=int(255 * render_line.color.g),
                        b=int(255 * render_line.color.b)
                    )
                    target.extend(bytearray(record))

        last_header = IldxHeader(
            ildxMagic=ILDA_MAGIC,
            starttime=adjust_start_time(0),
            formatCode=self.FORMAT_CODE_2D_TRUE_COLOR,
            companyName=bytes(self._format_ildx_name(""), encoding='ascii'),
            frameName=bytes(self._format_ildx_name(""), encoding='ascii'),
            numberOfRecords=0,
            frameOrPaletteNumber=0,
            totalFrames=0,
            projectorNumber=0,
            framesPerSecondOrFrameAmount=0
        )
        target.extend(bytearray(last_header))
        with open(self._ildx_filename, 'wb') as file:
            file.write(target)
    
    def run(self):
        animations = self._compute_frames()
        render_lines = self._compute_render_lines(animations)
        self._write_file(render_lines)
        print("Done!")
