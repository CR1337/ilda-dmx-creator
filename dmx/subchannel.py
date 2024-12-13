from abc import ABC
import numpy as np
from typing import Callable, Tuple


class Subchannel(ABC):

    _name: str
    _min_value: int
    _max_value: int
    _callback: Callable[[int], Tuple[int, int]]

    def __init__(self, name: str, min_value: int, max_value: int, callback: Callable[[int], Tuple[int, int]]):
        self._name = name
        self._min_value = min_value
        self._max_value = max_value
        self._callback = callback

    def set_value(self, value: float) -> Tuple[int, int]:
        value = max(0, min(1, value))
        value = int(self._min_value + value * (self._max_value - self._min_value))
        return self._callback(value)
    
    def pulse_once(self, t: float, start_t: float, end_t: float, value: float = 1.0) -> Tuple[int, int]:
        if t < start_t or t > end_t:
            return self.set_value(0)
        return self.set_value(value)
    
    def __lshift__(self, value: float) -> Tuple[int, int]:
        return self.set_value(value)

    @property
    def name(self) -> str:
        return self._name


class ContinousSubchannel(Subchannel):
    
    def pulse(
        self, 
        t: float,
        amplitude: float | Callable[[float], float] = 0.5,
        frequency: float | Callable[[float], float] = 1.0,
        phase: float | Callable[[float], float] = 0.0,
        shape: float | Callable[[float], float] = 0.5,
        duty: float | Callable[[float], float] = 0.5,
        vertical_shift: float | Callable[[float], float] = 1.0
    ) -> Tuple[int, int]:
        A = amplitude(t) if callable(amplitude) else amplitude
        f = frequency(t) if callable(frequency) else frequency
        phi = phase(t) if callable(phase) else phase
        s = (shape(t) if callable(shape) else shape) * 2
        d = duty(t) if callable(duty) else duty
        v = vertical_shift(t) if callable(vertical_shift) else vertical_shift
        
        E = min(s, 1)
        alpha = max(s, 1) - 1

        t_m = np.modf((t - phi) / f)[0] * f
        t_1 = (f * t_m) / (2 * d)
        t_2 = 1 / ((2 / f) * (1 - d)) * t_m + 1 - 1 / (2 * (1 - d))
        T = t_1 if t_m <= d / f else t_2
 
        y_sine = np.sin(2 * np.pi * T)
        y_ext_sine = np.sign(y_sine) * np.abs(y_sine) ** E
        y_tri = (2 / np.pi) * np.arcsin(y_sine)
        y_lerp = (1 - alpha) * y_ext_sine + alpha * y_tri
        y = A * (y_lerp + v)

        return self.set_value(y)

    def lerp(
        self, 
        t: float, 
        start_t: float, 
        end_t: float, 
        start_value: float, 
        end_value: float
    ) -> Tuple[int, int]:
        if t < start_t:
            y = start_value
        elif t > end_t:
            y = end_value
        else:
            progress = (t - start_t) / (end_t - start_t)
            y = progress * end_value + (1 - progress) * start_value
        return self.set_value(y)
    
    def smooth(
        self,
        t: float,
        start_t: float,
        end_t: float,
        start_value: float,
        end_value: float,
    ) -> Tuple[int, int]:
        if t < start_t:
            y = start_value
        elif t > end_t:
            y = end_value
        else:
            progress = (t - start_t) / (end_t - start_t)
            y = (end_value - start_value) * (progress ** 2) + start_value
        return self.set_value(y)

    def zero(self) -> Tuple[int, int]:
        return self.set_value(0)
    
    def max(self) -> Tuple[int, int]:
        return self.set_value(1)
    

class CategorySubchannel(Subchannel):

    def activate(self) -> Tuple[int, int]:
        return self.set_value(0.5)
    
    def __call__(self) -> Tuple[int, int]:
        return self.activate()
    