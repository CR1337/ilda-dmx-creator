from noise.noise import Noise
import opensimplex



class SimplexNoise(Noise):

    def _get_value(self, x: float, y: float) -> float:
        opensimplex.seed(self._seed)
        return opensimplex.noise2d(x, y)
    