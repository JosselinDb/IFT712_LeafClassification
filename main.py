# ---
# DUBOIS Josselin (dubj0701)
# GUENARD Antoine (guea0702)
# JOURDIN Yann (jouy0701)
# --- 

import matplotlib.pyplot as plt
import numpy as np

from classification.classifiers.knn import KNN
from classification.classifiers.multi_layer import MultiLayerPerceptron
from classification.classifiers.random_forest import RandomForest
from classification.classifiers.svm import SVM
from data_management import DataManager



if __name__ == "__main__":
    dm = DataManager()
    dm.make_validation_set()
    
    