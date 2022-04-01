from sklearn.gaussian_process import GaussianProcessClassifier

from classification.classifier import Classifier


class GaussianProcess(Classifier):
    """
    Classifier with Gaussian Naive Bayes method

    Parameters
        hyperparameters:
            keyword arguments for the sklearn classifier
            https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html

    Attributes
        see `Classifier`

    Methods
        see `Classifier`
    """
    def __init__(self, **hyperparameters) -> None:
        super().__init__("Gaussian Process", GaussianProcessClassifier, hyperparameters)
