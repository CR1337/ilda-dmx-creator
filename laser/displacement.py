import numpy as np
from noise.noise import Noise
from noise.simplex_noise import SimplexNoise
from noise.white_noise import WhiteNoise


class Displacement:

    _amplitude: float
    _noise: Noise
    
    def __init__(self, noise_type: str, amplitude: float, seed: int | None = None):
        self._amplitude = amplitude
        if noise_type == 'simplex':
            self._noise = SimplexNoise(seed)
        elif noise_type == 'white':
            self._noise = WhiteNoise(seed)

    def _chrodal_tangent(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        d_prev = p1 - p0
        d_next = p2 - p1

        weight_prev = np.linalg.norm(d_next)
        weight_next = np.linalg.norm(d_prev)

        tangent = (weight_prev * d_prev + weight_next * d_next) / (weight_prev + weight_next)

        return tangent / np.linalg.norm(tangent)

    def get_displacement(self, t: float, p0: np.ndarray, p1: np.array, p2: np.ndarray) -> np.ndarray:
        chrodal_tangent = self._chrodal_tangent(p0, p1, p2)
        normal = np.array([-chrodal_tangent[1], chrodal_tangent[0]])
        displacement = normal * self._get_noise(t) * self._amplitude
        return displacement
    