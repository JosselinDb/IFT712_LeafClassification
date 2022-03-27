import numpy as np
from classification.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNN(Classifier):
    """
    Classifier with K-Nearest-Neighbors method

    Parameters
        k: int > 0
            number of neighbors to consider
        **kwargs:
            keyword arguments for the sklearn classifier
    """
    def __init__(self, k: int=10, **kwargs) -> None:
        kwargs['n_neighbors'] = k

        super().__init__(KNeighborsClassifier, [], kwargs)