from typing import List, Any
from abc import ABCMeta, abstractmethod

# from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin

import numpy as np


class Classifier(metaclass=ABCMeta):
    """ TODO à compléter
    Abstract class
    Represent a classifier with sklearn

    Parameters
        method: ClassifierMixin
            sklearn class of our classifier
        method_kwargs: dict
            keyword arguments of method

    Attributes
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
    def __init__(self, method: ClassifierMixin, method_kwargs: dict[str, Any]=dict()) -> None:
        self.method = method
        self.hyperparameters = method_kwargs
        
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
        model = self.method(**self.hyperparameters)
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
        Compute the performance of our model on a set

        Arguments
            X: pd.DataFrame
                data to test into our model
            y: pd.DataFrame
                true targets of the data X
        """
        return self.model.score(X, y)
        # return accuracy_score(y, self.predict(X))