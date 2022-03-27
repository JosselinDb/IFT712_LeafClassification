from classification.classifier import Classifier
from sklearn.neural_network import MLPClassifier



class MultiLayerPerceptron(Classifier):
    """
    Classifier with MultiLayer Perceptron method

    Parameters
        **hyperparameters:
            keyword arguments for the sklearn classifier
            see: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """
    def __init__(self, **hyperparameters) -> None:
        super().__init__("MultiLayer Perceptron", MLPClassifier, hyperparameters)