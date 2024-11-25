import numpy as np
from util import ensure_np_array


def clamp(value: float | np.ndarray, min_value: float | np.ndarray, max_value: float | np.ndarray) -> float | np.ndarray:
    if isinstance(value, list):
        value = np.array(value)
    if isinstance(min_value, list):
        min_value = np.array(min_value)
    if isinstance(max_value, list):
        max_value = np.array(max_value)
    return np.maximum(min_value, np.minimum(value, max_value))


@ensure_np_array
def distance(p0: np.ndarray, p1: np.ndarray) -> float:
    return np.linalg.norm(p0 - p1)


def fract(value: float | np.ndarray) -> float | np.ndarray:
    if isinstance(value, list):
        value = np.array(value)
    return value - np.floor(value)


def mix(a: float | np.ndarray, b: float | np.ndarray, r: float) -> float | np.ndarray:
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return a * (1.0 - r) + b * r


@ensure_np_array
def reflect(i: np.ndarray, n: np.ndarray) -> np.ndarray:
    return i - 2.0 * np.dot(n, i) * n


@ensure_np_array
def refract(i: np.ndarray, n: np.ndarray, eta: float) -> np.ndarray:
    cosi = -np.dot(i, n)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    return np.zeros(3) if k < 0.0 else eta * i + (eta * cosi - np.sqrt(k)) * n


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    s = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def step(edge: float, x: float) -> float:
    return 1.0 if x >= edge else 0.0
