from sklearn.svm import SVC

from classification.classifier import Classifier


class SVM(Classifier):
    """
    Classifier with Support Vector Machine method

    Parameters
        hyperparameters:
            keyword arguments for the sklearn classifier
            see: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
    Attributes
        see `Classifier`

    Methods
        see `Classifier`
    """
    def __init__(self, **hyperparameters) -> None:
        super().__init__("Support Vector Machine", SVC, hyperparameters)
