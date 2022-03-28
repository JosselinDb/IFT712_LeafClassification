from sklearn.ensemble import RandomForestClassifier

from classification.classifier import Classifier


class RandomForest(Classifier):
    """
    Classifier with Random Forest method

    Parameters
        **hyperparameters:
            keyword arguments for the sklearn classifier
            see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self, **hyperparameters) -> None:
        super().__init__("Random Forest", RandomForestClassifier, hyperparameters)
