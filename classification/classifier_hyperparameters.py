from typing import Any, List, Union
from sklearn.base import ClassifierMixin

from classification.hyperparameter import HyperParameter



class ClassifierHyperParameters():
    """
    TODO à compléter
    """
    def __init__(self, parameters: dict[str, Union(List[Any], tuple[tuple[float, float], float])]=dict()) -> None:
        self.parameters = []
        for name, value in parameters.items():
            if type(value) == list:
                self.add_hyperparameter(name, list_values=value)
            else:
                self.add_hyperparameter(name, bounds=value[0], step=value[1])

    def add_hyperparameter(
            self, name: str, *,
            list_values: List[Any]=None, bounds: tuple[float, float]=None, step: float=None
    ) -> None:
        """
        TODO à compléter
        """
        self.parameters.append(
            HyperParameter(name, list_values=list_values, bounds=bounds, step=step)
        )

    def grid(self):
        """
        TODO à compléter
        """
        return {
            hp.name: hp.list_values
            for hp in self.parameters
        }