from typing import Any, List
from numpy import ndarray

from classification.hyperparameters.hyperparameter import HyperParameter


class ClassifierHyperParameters:
    """
    Contains the hyper parameters and their ranges for a classification's reserach

    Parameters
        parameters: dict[str, List | ((float, float), float)]
            hyperparameters and their range to store
            format {name: list_of_value} or {name: ((start, stop), step)}

    Arguments
        parameters: List[Hyperparameter]
            list of the hyperparameters
        grid: dict[str, np.ndarray]
            grid format for sklearn

    Methods
        add_hyperparameter(name, list_values, bounds, step):
            add a hyperparameter to `parameters`
    """
    def __init__(self, parameters: dict[str, Any]={}) -> None:
        self.parameters = []

        for name, value in parameters.items():
            if isinstance(value, list):
                self.add_hyperparameter(name, list_values=value)
            else:
                self.add_hyperparameter(name, bounds=value[0], step=value[1])

    @property
    def grid(self) -> dict[str, ndarray]:
        """
        Create the grid format for sklearn from our hyperparameters

        Returns: dict[str, ndarray]
            The grid of the ranges of the hyperparameters
        """
        return {
            hp.name: hp.list_values
            for hp in self.parameters
        }

    def add_hyperparameter(
            self, name: str, *,
            list_values: List[Any]=None, bounds: tuple[float, float]=None, step: float=None
    ) -> None:
        """
        Add a hyperparameter to `parameters`

        Parameters
            see `HyperParameter`
        """
        self.parameters.append(
            HyperParameter(name, list_values=list_values, bounds=bounds, step=step)
        )
