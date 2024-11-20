import numpy as np
from laser.color import Color


class RenderLine:
    
    _start: np.ndarray
    _end: np.ndarray
    _color: Color
    _blanked: bool

    def __init__(self, start: np.ndarray, end: np.ndarray, color: Color, blanked: bool = False):
        self._start = start.copy()
        self._end = end.copy()
        self._color = color
        self._blanked = blanked

    def copy(self):
        return RenderLine(self._start, self._end, self._color, self._blanked)

    def blank(self):
        self._blanked = True

    def flip_x(self):
        self._start[0] = -self._start[0]
        self._end[0] = -self._end[0]

    def flip_y(self):
        self._start[1] = -self._start[1]
        self._end[1] = -self._end[1]

    @property
    def start(self) -> np.ndarray:
        return self._start
    
    @property
    def end(self) -> np.ndarray:
        return self._end
    
    @property
    def color(self) -> Color:
        return self._color
    
    @property
    def blanked(self) -> bool:
        return self._blanked
    