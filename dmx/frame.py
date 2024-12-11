from typing import List, Tuple


class Frame:

    _start_t: float
    _t: float
    _fps: float
    _duration: float
    _channel_values: List[Tuple[int, int]]

    def __init__(self, start_t: float, t: float, fps: float, duration: float):
        self._start_t = start_t
        self._t = t
        self._fps = fps
        self._duration = duration
        self._channel_values = []

    @property
    def start_t(self) -> float:
        return self._start_t
    
    @property
    def t(self) -> float:
        return self._t
    
    @property
    def rel_t(self) -> float:
        return self._t - self._start_t
    
    @property
    def duration(self) -> float:
        return self._duration
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def channel_values(self) -> List[Tuple[int, int]]:
        return self._channel_values

    @property
    def progress(self) -> float:
        return (self._t - self._start_t) / self._duration
    
    @property
    def index(self) -> int:
        return int((self._t - self._start_t) * self._fps)
    
    @property
    def total_frames(self) -> int:
        return int(self._duration * self._fps)
    
    def add_value(self, channel_value: Tuple[int, int]):
        self._channel_values.append(channel_value)

    def __iadd__(self, channel_value: Tuple[int, int]):
        self.add_value(channel_value)
        return self
