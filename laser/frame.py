from laser.shapes.shape import Shape
from typing import List, Tuple


class Frame:
    
    _t: float
    _fps: float
    _duration: float
    _point_density: float
    _shapes: List[Tuple[Shape, bool]]

    def __init__(self, t: float, fps: float, duration: float, point_density: float):
        self._t = t
        self._fps = fps
        self._duration = duration
        self._point_density = point_density
        self._shapes = []
    
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
    def point_density(self) -> float:
        return self._point_density
    
    @property
    def shapes(self) -> List[Tuple[Shape, bool]]:
        return self._shapes

    @property
    def progress(self) -> float:
        return self._t / self._duration
    
    def add_shape(self, shape: Shape, is_exclusion_shape: bool = False):
        if shape is None:
            return
        if shape.point_density is None:
            shape.point_density = self._point_density
        self._shapes.append((shape.copy(), is_exclusion_shape))

    def __iadd__(self, shape: Shape):
        self.add_shape(shape)
        return self
