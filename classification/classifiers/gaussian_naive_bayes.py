from sklearn.naive_bayes import GaussianNB as GNB

from classification.classifier import Classifier


class GaussianNB(Classifier):
    """
    Classifier with Gaussian Naive Bayes method

    Parameters
        hyperparameters:
            keyword arguments for the sklearn classifier
            https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

    Attributes
        see `Classifier`

    Methods
        see `Classifier`
    """
    def __init__(self, **hyperparameters) -> None:
        super().__init__("Gaussian Naive Bayes", GNB, hyperparameters)
