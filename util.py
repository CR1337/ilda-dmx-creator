import numpy as np
from functools import wraps
from typing import get_type_hints


def np_hash(a: np.ndarray) -> int:
    return hash(a.tobytes())


def np_as_key(a: np.ndarray) -> tuple:
    return tuple(a.flatten())


def np_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = tuple((np_as_key(a) if isinstance(a, np.ndarray) else a for a in args))
        if kwargs:
            key += tuple([np_as_key(a) if isinstance(a, np.ndarray) else a for a in kwargs.values()])
        if key in wrapper.cache:
            return wrapper.cache[key]
        result = func(*args, **kwargs)
        wrapper.cache[key] = result
        return result

    wrapper.cache = {}
    return wrapper


def ensure_np_array(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        type_hints = get_type_hints(func)
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        new_args = [
            np.array(arg) if arg_name in type_hints and type_hints[arg_name] is np.ndarray and isinstance(arg, list)
            else arg
            for arg, arg_name in zip(args, arg_names)
        ]
        new_kwargs = {
            k: (np.array(v) if k in type_hints and type_hints[k] is np.ndarray and isinstance(v, list) else v)
            for k, v in kwargs.items()
        }
        return func(*new_args, **new_kwargs)
    return wrapper
