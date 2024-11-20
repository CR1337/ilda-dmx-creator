from typing import List, Tuple


class Frame:

    _timestamp: float
    _fps: float
    _channel_values: List[Tuple[int, int]]
    _is_last: bool

    def __init__(self, timestamp: float, fps: float):
        self._timestamp = timestamp
        self._fps = fps
        self._is_last = False
        self._channel_values = []

    def set_last(self):
        self._is_last = True

    @property
    def is_last(self) -> bool:
        return self._is_last
    
    @property
    def timestamp(self) -> float:
        return self._timestamp
    
    @property
    def channel_values(self) -> List[Tuple[int, int]]:
        return self._channel_values
    
    def add_value(self, channel_value: Tuple[int, int]):
        self._channel_values.append(channel_value)

    def __iadd__(self, channel_value: Tuple[int, int]):
        self.add_value(channel_value)
        return self
