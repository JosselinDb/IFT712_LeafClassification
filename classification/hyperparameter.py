from typing import Any, List

import numpy as np


class HyperParameter:
    """
    Represent an hyper parameter for a method

    Parameters
        name: str
            name of the argument of the hyper parameter
        list_values: List
            list of all the values taken by the hyperparameter we want to try
        bounds: (float, float)
            iff list_values is None
            min and max value of the values of the hyperparameter
        step: float
            iff list_values is None
            step for the range of values taken by the hyperparameter

    Attributes
        name: str
            name of the argument of the hyper paramater
        list_values: np.ndarray
            list of all the values taken by the hyperparameter we want to try
    """
    def __init__(self, name: str, *, list_values: List[Any]=None, bounds: tuple[float, float]=None, step: float=None) -> None:
        self.name = name
        if list_values is not None:
            self.list_values = np.array(list_values)
        else:
            if bounds is None or step is None:
                raise Exception("list_values or (bounds and step) must not be None")

            start, stop = bounds
            self.list_values = np.arange(start, stop, step)
