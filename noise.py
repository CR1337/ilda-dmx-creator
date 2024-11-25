from abc import ABC, abstractmethod
import opensimplex
from random import randint
import numpy as np
from util import ensure_np_array


class Noise(ABC):

    _frequency: np.ndarray
    _amplitude: float
    _n_wrapped_dimensions: int
    _radii: np.ndarray
    
    @ensure_np_array
    def __init__(
        self, 
        frequency: np.ndarray, 
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0, 
        radii: np.ndarray = np.array([]),
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

        if radii.shape[0] != n_wrapped_dimensions:
            raise ValueError(f"radii must have shape ({n_wrapped_dimensions},)")
        self._radii = radii

        if seed is None:
            seed = randint(0, 2 ** 64 - 1)
        opensimplex.seed(seed)

    @ensure_np_array
    @abstractmethod
    def get_value(self, p: np.ndarray) -> float:
        raise NotImplementedError("@abstractmethod get_value")
    
    @ensure_np_array
    def __call__(self, p: np.ndarray) -> float:
        return self.get_value(p)
         

class Noise1D(Noise):

    N_DIMENSIONS: int = 1
    MAX_N_WRAPPED_DIMENSIONS: int = 1

    @classmethod
    @ensure_np_array
    def line(cls, frequency: np.ndarray = np.array([1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([]), seed: int | None = None):
        return cls(frequency, amplitude, 0, radii, seed)
    
    @classmethod
    @ensure_np_array
    def circle(cls, frequency: np.ndarray = np.array([1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([1.0]), seed: int | None = None):
        return cls(frequency, amplitude, 1, radii, seed)
    
    @ensure_np_array
    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        radii: np.ndarray = np.array([]),
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, radii, seed)

    @ensure_np_array
    def get_value(self, p: np.ndarray) -> float:
        if self._n_wrapped_dimensions == 0:  # line
            x = p[0] * self._frequency[0]
            y = 0
        elif self._n_wrapped_dimensions == 1:  # circle
            x = np.cos(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[0]
            y = np.sin(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[0]

        v = opensimplex.noise2(x, y)
        return self._amplitude * v

class Noise2D(Noise):

    N_DIMENSIONS: int = 2
    MAX_N_WRAPPED_DIMENSIONS: int = 2

    @ensure_np_array
    @classmethod
    def plane(cls, frequency: np.ndarray = np.array([1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([]), seed: int | None = None):
        return cls(frequency, amplitude, 0, radii, seed)
    
    @classmethod
    @ensure_np_array
    def cylinder(cls, frequency: np.ndarray = np.array([1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([1.0]), seed: int | None = None):
        return cls(frequency, amplitude, 1, radii, seed)
    
    @classmethod
    @ensure_np_array
    def torus(cls, frequency: np.ndarray = np.array([1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([1.0, 1.0]), seed: int | None = None):
        return cls(frequency, amplitude, 2, radii, seed)

    @ensure_np_array
    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0, 1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        radii: np.ndarray = np.array([]),
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, radii, seed)

    @ensure_np_array
    def get_value(self, p: np.ndarray) -> float:
        if self._n_wrapped_dimensions == 0:  # plane
            x = p[0] * self._frequency[0]
            y = p[1] * self._frequency[1]
            v = opensimplex.noise2(x, y)
        elif self._n_wrapped_dimensions == 1:  # cylinder
            x = np.cos(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[0]
            y = np.sin(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[0]
            z = p[1] * self._frequency[1]
            v = opensimplex.noise3(x, y, z)
        elif self._n_wrapped_dimensions == 2:  # torus
            x = (1 + np.cos(p[0] * self._frequency[0] * np.pi * 2.0)) * np.cos(p[1] * self._frequency[1] * np.pi * 2.0) * self._radii[0]
            y = (1 + np.cos(p[0] * self._frequency[0] * np.pi * 2.0)) * np.sin(p[1] * self._frequency[1] * np.pi * 2.0) * self._radii[0]
            z = np.sin(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[1]
            v = opensimplex.noise3(x, y, z)

        return self._amplitude * v


class Noise3D(Noise):

    N_DIMENSIONS: int = 3
    MAX_N_WRAPPED_DIMENSIONS: int = 2

    @classmethod
    @ensure_np_array
    def space(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([]), seed: int | None = None):
        return cls(frequency, amplitude, 0, radii, seed)
    
    @classmethod
    @ensure_np_array
    def cylindrical(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([1.0]), seed: int | None = None):
        return cls(frequency, amplitude, 1, radii, seed)
    
    @classmethod
    @ensure_np_array
    def toroidal(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([1.0, 1.0]), seed: int | None = None):
        return cls(frequency, amplitude, 2, radii, seed)

    @ensure_np_array
    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0, 1.0, 1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        radii: np.ndarray = np.array([]),
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, radii, seed)

    @ensure_np_array
    def get_value(self, p: np.ndarray) -> float:
        if self._n_wrapped_dimensions == 0:  # space
            x = p[0] * self._frequency[0]
            y = p[1] * self._frequency[1]
            z = p[2] * self._frequency[2]
            v = opensimplex.noise3(x, y, z)
        elif self._n_wrapped_dimensions == 1:  # cylindrical
            x = np.cos(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[0]
            y = np.sin(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[0]
            z = p[1] * self._frequency[1]
            w = p[2] * self._frequency[2]
            v = opensimplex.noise4(x, y, z, w)
        elif self._n_wrapped_dimensions == 2:  # toroidal
            x = (1 + np.cos(p[0] * self._frequency[0] * np.pi * 2.0)) * np.cos(p[1] * self._frequency[1] * np.pi * 2.0) * self._radii[0]
            y = (1 + np.cos(p[0] * self._frequency[0] * np.pi * 2.0)) * np.sin(p[1] * self._frequency[1] * np.pi * 2.0) * self._radii[0]
            z = np.sin(p[0] * self._frequency[0] * np.pi * 2.0) * self._radii[1]
            w = p[2] * self._frequency[2]
            v = opensimplex.noise4(x, y, z, w)

        return self._amplitude * v
    

class Noise4D(Noise):

    N_DIMENSIONS: int = 4
    MAX_N_WRAPPED_DIMENSIONS: int = 0

    @classmethod
    @ensure_np_array
    def hyperspace(cls, frequency: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0]), amplitude: float = 1.0, radii: np.ndarray = np.array([]), seed: int | None = None):
        return cls(frequency, amplitude, 0, radii, seed)
    
    @ensure_np_array
    def __init__(
        self,
        frequency: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0]),
        amplitude: float = 1.0,
        n_wrapped_dimensions: int = 0,
        radii: np.ndarray = np.array([]),
        seed: int | None = None
    ):
        super().__init__(frequency, amplitude, n_wrapped_dimensions, radii, seed)
    
    @ensure_np_array
    def get_value(self, p: np.ndarray) -> float:
        x = p[0] * self._frequency[0]
        y = p[1] * self._frequency[1]
        z = p[2] * self._frequency[2]
        w = p[3] * self._frequency[3]
        v = opensimplex.noise4(x, y, z, w)
        return self._amplitude * v
    