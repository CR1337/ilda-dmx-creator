from laser.shapes.shape import Shape
from typing import List, Tuple


class Frame:
    
    _timestamp: float
    _fps: float
    _point_density: float
    _shapes: List[Tuple[Shape, bool]]
    _is_last: bool

    def __init__(self, timestamp: float, fps: float, point_density: float):
        self._timestamp = timestamp
        self._fps = fps
        self._point_density = point_density
        self._shapes = []
        self._is_last = False

    def set_last(self):
        self._is_last = True

    @property
    def is_last(self) -> bool:
        return self._is_last
    
    @property
    def timestamp(self) -> float:
        return self._timestamp
    
    @property
    def shapes(self) -> List[Tuple[Shape, bool]]:
        return self._shapes
    
    def add_shape(self, shape: Shape, is_exclusion_shape: bool = False):
        if shape.point_density is None:
            shape.point_density = self._point_density
        self._shapes.append((shape, is_exclusion_shape))

    def __iadd__(self, shape: Shape):
        self.add_shape(shape)
        return self
