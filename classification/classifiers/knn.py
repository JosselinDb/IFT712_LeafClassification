from classification.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier


class KNN(Classifier):
    """
    Classifier with K-Nearest-Neighbors method

    Parameters
        k: int > 0
            number of neighbors to consider
        **hyperparameters:
            keyword arguments for the sklearn classifier
            see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    def __init__(self, k: int=5, **hyperparameters) -> None:
        hyperparameters['n_neighbors'] = k

        super().__init__("K-Nearest-Neighbors", KNeighborsClassifier, hyperparameters)