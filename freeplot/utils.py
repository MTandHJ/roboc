
from typing import Dict, List
import json
import matplotlib.pyplot as plt

def load(filename: str) -> Dict:
    with open(filename, encoding="utf-8") as j:
        data = json.load(j)
    return data


def axis(func):
    def wrapper(*args, **kwargs):
        axis = kwargs[axis]
        results = func(*args, **kwargs)
        newresults = dict()
        for name, value in results.item():
            newresults[axis + results] = value
        return value
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def reset(set):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            results[index] = kwargs[index]
            return set(*results)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

def style_env(style: List[str]):
    def decorator(func):
        def wrapper(*arg, **kwargs):
            with plt.style.context(style, after_reset=False):
                results = func(*arg, **kwargs)
            return results
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

