import numpy as np
from laser.color import Color
from util import ensure_np_array


class RenderLine:
    
    _p0: np.ndarray
    _p1: np.ndarray
    _color: Color
    _blanked: bool

    @ensure_np_array
    def __init__(self, p0: np.ndarray, p1: np.ndarray, color: Color, blanked: bool = False):
        self._p0 = p0.copy()
        self._p1 = p1.copy()
        self._color = color
        self._blanked = blanked

    def copy(self):
        return RenderLine(self._p0, self._p1, self._color, self._blanked)

    def blank(self):
        self._blanked = True

    def flip_x(self):
        self._p0[0] = -self._p0[0]
        self._p1[0] = -self._p1[0]

    def flip_y(self):
        self._p0[1] = -self._p0[1]
        self._p1[1] = -self._p1[1]

    @property
    def p0(self) -> np.ndarray:
        return self._p0
    
    @property
    def p1(self) -> np.ndarray:
        return self._p1
    
    @property
    def color(self) -> Color:
        return self._color
    
    @property
    def blanked(self) -> bool:
        return self._blanked
    