from abc import ABC, abstractmethod
import opensimplex
from random import randint
import numpy as np
from typing import Dict


class Noise(ABC):

    _frequency: np.ndarray
    _amplitude: float
    _n_wrapped_dimensions: int

    SWIZZLE_TRANSLATION: Dict[str, int] = {
        "x": 0,
        "y": 1,
        "z": 2,
        "w": 3,
        "t": -1
    }
    
    def __init__(
        self, 
        frequency: np.ndarray, 
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0, 
        seed: int | None = None
    ):
        if frequency.shape == (1,):
            frequency = np.repeat(frequency[0], self.N_DIMENSIONS)
        elif frequency.shape != (self.N_DIMENSIONS,):
            raise ValueError(f"frequency must have shape ({self.N_DIMENSIONS},)")
        self._frequency = frequency
        
        self._amplitude = amplitude

        if n_wrapped_dimensions < 0 or n_wrapped_dimensions > self.MAX_N_WRAPPED_DIMENSIONS:
            raise ValueError(f"n_wrapped_dimensions must be between 0 and {self.MAX_N_WRAPPED_DIMENSIONS}")
        self._n_wrapped_dimensions = n_wrapped_dimensions

        if seed is None:
            seed = randint(0, 2 ** 64 - 1)
        opensimplex.seed(seed)

    @abstractmethod
    def get_value(self, position: np.ndarray, time: float | None, *, swizzle: str | None = None) -> float:
        raise NotImplementedError("@abstractmethod get_value")
    
    def __call__(self, position: np.ndarray, time: float | None = None, *, swizzle: str | None = None) -> float:
        return self.get_value(position, time, swizzle=swizzle)
    
    def _swizzle_coordinates(self, position: np.ndarray, time: float | None, swizzle: str | None) -> np.ndarray:
        if time is None:
            if position.shape != (self.N_DIMENSIONS,):
                raise ValueError(f"position must have shape ({self.N_DIMENSIONS},)")
            default_swizzle = "xyzw"[:self.N_DIMENSIONS]
        else:
            if position.shape != (self.N_DIMENSIONS - 1,):
                raise ValueError(f"position must have shape ({self.N_DIMENSIONS - 1},)")
            default_swizzle = "xyzw"[:self.N_DIMENSIONS - 1] + "t"
            position = np.append(position, time)

        swizzle = swizzle or default_swizzle
        if len(swizzle) != self.N_DIMENSIONS:
            raise ValueError(f"Invalid swizzle length {len(swizzle)}")
        for c in swizzle:
            if c not in default_swizzle:
                raise ValueError(f"Invalid swizzle character '{c}'")

        new_position = np.array([
            position[self.SWIZZLE_TRANSLATION[swizzle[i]]] 
            for i in range(self.N_DIMENSIONS)
        ])
        
        return new_position
     

class Noise1D(Noise):

    N_DIMENSIONS: int = 1
    MAX_N_WRAPPED_DIMENSIONS: int = 1

    @classmethod
    def line(cls, frequency: np.ndarray = np.array([1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 0, seed)
    
    @classmethod
    def circle(cls, frequency: np.ndarray = np.array([1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 1, seed)
    
    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, seed)

    def get_value(self, position: np.ndarray, time: float | None, *, swizzle: str | None = None) -> float:
        pos = self._swizzle_coordinates(position, time, swizzle)
        if self._n_wrapped_dimensions == 0:  # line
            x = pos[0] * self._frequency[0]
            y = 0
        elif self._n_wrapped_dimensions == 1:  # circle
            x = np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)
            y = np.sin(pos[0] * self._frequency[0] * np.pi * 2.0)

        v = opensimplex.noise2(x, y)
        return self._amplitude * v

class Noise2D(Noise):

    N_DIMENSIONS: int = 2
    MAX_N_WRAPPED_DIMENSIONS: int = 2

    @classmethod
    def plane(cls, frequency: np.ndarray = np.array([1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 0, seed)
    
    @classmethod
    def cylinder(cls, frequency: np.ndarray = np.array([1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 1, seed)
    
    @classmethod
    def torus(cls, frequency: np.ndarray = np.array([1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 2, seed)

    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0, 1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, seed)

    def get_value(self, position: np.ndarray, time: float | None, *, swizzle: str | None = None) -> float:
        pos = self._swizzle_coordinates(position, time, swizzle)
        if self._n_wrapped_dimensions == 0:  # plane
            x = pos[0] * self._frequency[0]
            y = pos[1] * self._frequency[1]
            v = opensimplex.noise2(x, y)
        elif self._n_wrapped_dimensions == 1:  # cylinder
            x = np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)
            y = np.sin(pos[0] * self._frequency[0] * np.pi * 2.0)
            z = pos[1] * self._frequency[1]
            v = opensimplex.noise3(x, y, z)
        elif self._n_wrapped_dimensions == 2:  # torus
            x = (1 + np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)) * np.cos(pos[1] * self._frequency[1] * np.pi * 2.0)
            y = (1 + np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)) * np.sin(pos[1] * self._frequency[1] * np.pi * 2.0)
            z = np.sin(pos[0] * self._frequency[0] * np.pi * 2.0)
            v = opensimplex.noise3(x, y, z)

        return self._amplitude * v


class Noise3D(Noise):

    N_DIMENSIONS: int = 3
    MAX_N_WRAPPED_DIMENSIONS: int = 2

    @classmethod
    def space(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 0, seed)
    
    @classmethod
    def cylindrical(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 1, seed)
    
    @classmethod
    def toroidal(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 2, seed)

    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0, 1.0, 1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, seed)

    def get_value(self, position: np.ndarray, time: float | None, *, swizzle: str | None = None) -> float:
        pos = self._swizzle_coordinates(position, time, swizzle)
        if self._n_wrapped_dimensions == 0:  # space
            x = pos[0] * self._frequency[0]
            y = pos[1] * self._frequency[1]
            z = pos[2] * self._frequency[2]
            v = opensimplex.noise3(x, y, z)
        elif self._n_wrapped_dimensions == 1:  # cylindrical
            x = np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)
            y = np.sin(pos[0] * self._frequency[0] * np.pi * 2.0)
            z = position[1] * self._frequency[1]
            w = pos[2] * self._frequency[2]
            v = opensimplex.noise4(x, y, z, w)
        elif self._n_wrapped_dimensions == 2:  # toroidal
            x = (1 + np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)) * np.cos(pos[1] * self._frequency[1] * np.pi * 2.0)
            y = (1 + np.cos(pos[0] * self._frequency[0] * np.pi * 2.0)) * np.sin(pos[1] * self._frequency[1] * np.pi * 2.0)
            z = np.sin(pos[0] * self._frequency[0] * np.pi * 2.0)
            w = pos[2] * self._frequency[2]
            v = opensimplex.noise4(x, y, z, w)

        return self._amplitude * v
    

class Noise4D(Noise):

    N_DIMENSIONS: int = 4
    MAX_N_WRAPPED_DIMENSIONS: int = 0

    @classmethod
    def hyperspace(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0]), amplitude: float = 1.0, seed: int | None = None):
        return cls(frequency, amplitude, 0, seed)
    
    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, seed)
    
    def get_value(self, position: np.ndarray, time: float | None, *, swizzle: str | None = None) -> float:
        pos = self._swizzle_coordinates(position, time, swizzle)
        x = pos[0] * self._frequency[0]
        y = pos[1] * self._frequency[1]
        z = pos[2] * self._frequency[2]
        w = pos[3] * self._frequency[3]
        v = opensimplex.noise4(x, y, z, w)
        return self._amplitude * v
    