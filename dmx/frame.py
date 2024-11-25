from typing import List, Tuple


class Frame:

    _t: float
    _fps: float
    _duration: float
    _channel_values: List[Tuple[int, int]]

    def __init__(self, t: float, fps: float, duration: float):
        self._t = t
        self._fps = fps
        self._duration = duration
        self._channel_values = []
    
    @property
    def t(self) -> float:
        return self._t
    
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
        return self._t / self._duration
    
    def add_value(self, channel_value: Tuple[int, int]):
        self._channel_values.append(channel_value)

    def __iadd__(self, channel_value: Tuple[int, int]):
        self.add_value(channel_value)
        return self
