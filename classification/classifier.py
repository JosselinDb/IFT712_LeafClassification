from typing import List, Any
from abc import ABCMeta, abstractmethod

# from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

import numpy as np
from classification.classifier_hyperparameters import ClassifierHyperParameters


class Classifier(metaclass=ABCMeta):
    """ TODO à compléter
    Abstract class
    Represent a classifier with sklearn

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
    """
    def __init__(
        self, name,
        method: ClassifierMixin, params: dict[str, Any]=dict(),
        hyperparameters: ClassifierHyperParameters=None
    ) -> None:
        self.name = name

        self.method = method
        self.params = params

        self.hyperparameters = hyperparameters
        
        self.model = None
        self.best_model = None
        self.best_params = None

    def train(self, x_train, y_train) -> None:
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

    def predict(self, X) -> np.ndarray:
        """
        Compute the predictions f our model on new data

        Arguments
            X: pd.DataFrame
                data to predict their targets
        """
        return self.model.predict(X)

    def score(self, X, y) -> float:
        """
        FIXME see sklearn.model_selection.cross_val_score or sklearn.metrics.accuracy_score
        Compute the performance of our model on a set

        Argument
            X: pd.DataFrame
                data to test into our model
            y: pd.DataFrame
                true targets of the data X
        """
        return self.model.score(X, y)
        # return accuracy_score(y, self.predict(X))

    def hyperparameter_search(self, *, verbose=0) -> float:
        """
        TODO à compléter
        """
        search = GridSearchCV(
            self.model,
            self.hyperparameters.grid(),
            verbose=verbose
        )

        best_model = search.best_estimator_
        best_score = search.best_score_

        self.best_model = best_model
        self.best_params = search.best_params_

        return best_score

    def __str__(self) -> str:
        s = f"### {self.name}\n"

        for arg, value in self.model.get_params().items():
            s += " "
            if arg in self.params:
                s += "*"
            s += f"{arg}: {value}\n"

        return s