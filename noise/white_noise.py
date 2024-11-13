from noise.noise import Noise


class WhiteNoise(Noise):

    def _get_value(self, x: float, y: float) -> float:
        return self._random_value(x, y)
    