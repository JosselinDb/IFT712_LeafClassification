from typing import Any
from abc import ABCMeta

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV

from classification.hyperparameters.classifier_hyperparameters import ClassifierHyperParameters


class Classifier(metaclass=ABCMeta):
    """
    Abstract class
    Able to perform a classification with sklearn

    Parameters
        name: str
            name of the method
        method: ClassifierMixin
            sklearn class of our classifier
        params: dict
            keyword arguments of method
        hyperparameters: ClassifierHyperParameters
            all the hyperparameters we want for our hyperparameter search

    Attributes
        name: str
            name of the method
        method: ClassifierMixin
            sklearn class of our classifier
        params: dict
            default keyword arguments of method
        hyperparameters: dict
            keyword hyperparameters of our classifier
        model: ClassifierMixin
            fitted instance of our method
        best_model: ClassifierMixin
            fitted instance of the method after a hp search
        best_params: dict
            keyworks arguments of the method after a hp search

    Methods
        train(x_train, y_train):
            train our model according to our method on the set in arguments
        predict(X): np.ndarray
            compute the predictions f our model on new data
        score(X, y): float
            compute the performance of our model on a set
        hyperparameter_search(x_valid, y_valid, verbose): float
            perform a research on hyperparameters
    """
    def __init__(
        self, name: str,
        method: ClassifierMixin,
        params: dict[str, Any]=None,
        hyperparameters: ClassifierHyperParameters=None
    ) -> None:
        self.name = name

        self.method = method
        self.params = {} if params is None else params

        self.hyperparameters = hyperparameters

        self.model = None
        self.best_model = None
        self.best_params = None

    def train(self, x_train: DataFrame, y_train: DataFrame) -> None:
        """
        Train our model according to our method on the set in arguments.

        Arguments
            x_train: pd.DataFrame
                training set to train our model
            y_train: pd.DataFrame
                targets of the training set
        """
        model = self.method(**self.params)
        model.fit(x_train, y_train)

        self.model = model

    def predict(self, X: DataFrame) -> ndarray:
        """
        Compute the predictions f our model on new data

        Arguments
            X: pd.DataFrame
                data to predict their targets

        Returns: np.ndarray
            the predictions of the model for the entry X
        """
        return self.model.predict(X)

    def score(self, X: DataFrame, y: DataFrame) -> float:
        """
        Compute the performance of our model on a set

        Arguments
            X: pd.DataFrame
                data to test into our model
            y: pd.DataFrame
                true targets of the data X

        Returns: float
            The score of the model on the data (X, y)
        """
        return self.model.score(X, y)

    def hyperparameter_search(self, x_valid: DataFrame, y_valid: DataFrame, *, verbose: bool=0) -> float:
        """
        Perform a research on the hyperparameters

        Arguments
            x_valid: pd.DataFrame
                data where to perform the search
            y_valid: pd.DataFrame
                target of the data where to perform the search
            verbose: bool
                print information during the search

        Returns
            The score of the model after the search
        """
        search = GridSearchCV(
            self.model,
            self.hyperparameters.grid,
            verbose=verbose
        )
        search.fit(x_valid, y_valid)

        best_model = search.best_estimator_
        best_score = search.best_score_

        self.best_model = best_model
        self.best_params = search.best_params_

        return best_score

    def __str__(self) -> str:
        """
        The name of the method and the value of the parameters
        """
        s = f"### {self.name}\n"

        for arg, value in self.model.get_params().items():
            s += " "
            if arg in self.params:
                s += "*"
            s += f"{arg}: {value}\n"

        return s
