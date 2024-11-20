from dmx.subchannel import Subchannel, ContinousSubchannel, CategorySubchannel
from typing import Dict, Tuple


class Channel:
    
    _index: int
    _name: str
    _subchannels: Dict[str, Subchannel]

    _subchannel_added: bool = False

    def __init__(self, index: int, name: str):
        self._index = index
        self._name = name
        self._subchannels = {
            'default': ContinousSubchannel('default', 0, 255, self._value_changed)
        }

    def _value_changed(self, value: int):
        return (self._index, value)

    def add_continous_subchannel(self, name: str, min_value: int, max_value: int) -> ContinousSubchannel:
        if not self._subchannel_added:
            self._subchannel_added = True
            self._subchannels = {}
        self._subchannels[name] = ContinousSubchannel(name, min_value, max_value, self._value_changed)
        return self._subchannels[name]
    
    def add_category_subchannel(self, name: str, min_value: int, max_value: int) -> CategorySubchannel:
        if not self._subchannel_added:
            self._subchannel_added = True
            self._subchannels = {}
        self._subchannels[name] = CategorySubchannel(name, min_value, max_value, self._value_changed)
        return self._subchannels[name]

    def __getattr__(self, name: str) -> Subchannel:
        return self._subchannels[name]
    
    def __getitem__(self, name: str) -> Subchannel:
        return self._subchannels[name]
    
    def set_value(self, value: float) -> Tuple[int, int]:
        return (self._index, int(value * 0xff))
    
    def __lshift__(self, value: float) -> Tuple[int, int]:
        return self.set_value(value)
    
    def zero(self) -> Tuple[int, int]:
        return (self._index, 0)
    
    def max(self) -> Tuple[int, int]:
        return (self._index, 0xff)
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def value(self) -> int:
        return self._value
    