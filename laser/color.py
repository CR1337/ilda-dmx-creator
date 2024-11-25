from __future__ import annotations
from typing import Tuple, List


class Color:
    
    r: int
    g: int
    b: int

    h: int
    s: int
    v: int

    @classmethod
    def black(cls) -> Color:
        return Color(0, 0, 0)

    def __init__(self, component_0: float, component_1: float, component_2: float, is_rgb: bool = True):
        if is_rgb:
            self.r = component_0
            self.g = component_1
            self.b = component_2
            self._compute_hsv()
        else:
            self.h = component_0
            self.s = component_1
            self.v = component_2
            self._compute_rgb() 

    def __eq__(self, other: Color) -> bool:
        return self.r == other.r and self.g == other.g and self.b == other.b
    
    def __hash__(self) -> int:
        return hash((self.r, self.g, self.b))

    def _compute_rgb(self):
        c = self.v * self.s
        x = c * (1 - abs((self.h * 6) % 2 - 1))
        m = self.v - c

        if 0 <= self.h < 1/6:
            self.r = c
            self.g = x
            self.b = 0
        elif 1/6 <= self.h < 2/6:
            self.r = x
            self.g = c
            self.b = 0
        elif 2/6 <= self.h < 3/6:
            self.r = 0
            self.g = c
            self.b = x
        elif 3/6 <= self.h < 4/6:
            self.r = 0
            self.g = x
            self.b = c
        elif 4/6 <= self.h < 5/6:
            self.r = x
            self.g = 0
            self.b = c
        else:
            self.r = c
            self.g = 0
            self.b = x

        self.r += m
        self.g += m
        self.b += m

    def _compute_hsv(self):
        c_max = max(self.r, self.g, self.b)
        c_min = min(self.r, self.g, self.b)
        delta = c_max - c_min

        if delta == 0:
            self.h = 0
        elif c_max == self.r:
            self.h = 60 * (((self.g - self.b) / delta) % 6)
        elif c_max == self.g:
            self.h = 60 * (((self.b - self.r) / delta) + 2)
        else:
            self.h = 60 * (((self.r - self.g) / delta) + 4)

        self.h = self.h / 360.0

        if c_max == 0:
            self.s = 0
        else:
            self.s = delta / c_max

        self.v = c_max

    def interpolate_rgb(self, other: Color, ratio: float):
        return Color(
            self.r + (other.r - self.r) * ratio,
            self.g + (other.g - self.g) * ratio,
            self.b + (other.b - self.b) * ratio
        )
    
    def interpolate_hsv(self, other: Color, ratio: float):
        return Color(
            self.h + (other.h - self.h) * ratio,
            self.s + (other.s - self.s) * ratio,
            self.v + (other.v - self.v) * ratio,
            is_rgb=False
        )
    
    def copy(self) -> Color:
        return Color(self.r, self.g, self.b)


class ColorGradient:
    
    _colors: List[Tuple[float, Color]]
    _interpolation_mode: str

    def __init__(self, start_color: Color, end_color: Color | None = None, interpolation_mode: str = 'hsv'):
        if end_color is None:
            end_color = start_color
        self._colors = [(0.0, start_color), (1.0, end_color)]
        self._interpolation_mode = interpolation_mode

    def __eq__(self, other: ColorGradient) -> bool:
        if len(self._colors) != len(other._colors):
            return False
        for i in range(len(self._colors)):
            if self._colors[i] != other._colors[i]:
                return False
        if self._interpolation_mode != other._interpolation_mode:
            return False
        return True
    
    def __hash__(self) -> int:
        return hash(tuple(self._colors) + (self._interpolation_mode,))

    def add_color(self, s: float, color: Color):
        self._colors.append((s, color))
        self._colors.sort(key=lambda x: x[0])

    def get_color(self, s: float) -> Color:
        for i in range(len(self._colors) - 1):
            if self._colors[i][0] < s <= self._colors[i + 1][0]:
                ratio = (s - self._colors[i][0]) / (self._colors[i + 1][0] - self._colors[i][0])
                if self._interpolation_mode == 'rgb':
                    return self._colors[i][1].interpolate_rgb(self._colors[i + 1][1], ratio)
                else:
                    return self._colors[i][1].interpolate_hsv(self._colors[i + 1][1], ratio)    
        color = self._colors[0][1]
        return color
    
    def copy(self) -> ColorGradient:
        color_gradient = ColorGradient(self._colors[0][1], interpolation_mode=self._interpolation_mode)
        for position, color in self._colors[1:]:
            color_gradient.add_color(position, color.copy())
        return color_gradient
