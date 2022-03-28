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
        method_kwargs: dict
            keyword arguments of method

    Attributes
        name: str
            name of the method
        method: ClassifierMixin
            sklearn class of our classifier
        hyperparameters: dict
            keyword hyperparameters of our classifier
        model: ClassifierMixin
            fitted instance of our method

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
        method: ClassifierMixin, method_kwargs: dict[str, Any]=dict(),
        hyperparameters: ClassifierHyperParameters=None
    ) -> None:
        self.name = name

        self.method = method
        self.method_kwargs = method_kwargs

        self.hyperparameters = hyperparameters
        
        self.model = None

    def train(self, x_train, y_train) -> None:
        """
        Train our model according to our method on the set in arguments.

        Arguments
            x_train: pd.DataFrame
                training set to train our model
            y_train: pd.DataFrame
                targets of the training set
        """
        model = self.method(**self.method_kwargs)
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
        FIXME see sklearn.model_selection.cross_val_score or accuracy_score
        Compute the performance of our model on a set

        Argument
            X: pd.DataFrame
                data to test into our model
            y: pd.DataFrame
                true targets of the data X
        """
        return self.model.score(X, y)
        # return accuracy_score(y, self.predict(X))

    def hyperparameter_search(self) -> None:
        """
        TODO à compléter
        """
        search = GridSearchCV(
            self.model,
            self.hyperparameters.grid()
        )

    @abstractmethod
    def show_parameters(self):
        raise NotImplementedError
    
    def __str__(self) -> str:
        s = f"### {self.name}\n"

        for arg, value in self.model.get_params().items():
            s += " "
            if arg in self.method_kwargs:
                s += "*"
            s += f"{arg}: {value}\n"

        return s