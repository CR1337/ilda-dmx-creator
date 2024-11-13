import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def distance(p0: np.ndarray, p1: np.ndarray) -> float:
    return np.linalg.norm(p0 - p1)


def fract(value: float) -> float:
    return value - np.floor(value)


def mix(x: float, y: float, a: float) -> float:
    return x * (1.0 - a) + y * a


def modf(value: float) -> tuple:
    return np.modf(value)


def reflect(i: np.ndarray, n: np.ndarray) -> np.ndarray:
    return i - 2.0 * np.dot(n, i) * n


def refract(i: np.ndarray, n: np.ndarray, eta: float) -> np.ndarray:
    cosi = -np.dot(i, n)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    return np.zeros(3) if k < 0.0 else eta * i + (eta * cosi - np.sqrt(k)) * n


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def step(edge: float, x: float) -> float:
    return 1.0 if x >= edge else 0.0
