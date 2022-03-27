from typing import Generator
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class DataManager:
    """ TODO à compléter
    Loads our dataset and provides methods to split our data 
    for training purposes

    Attributes
    ----------
    train_data: pd.DataFrame
        supervised dataset of our leaves
    test_data: pd.Dataframe
        unsupervised dataset of our leaves
    x_train: pd.DataFrame
        training set
    y_train: pd.DataFrame
        targets of the training set
    x_test: pd.DataFrame
        testing set
    x_valid: pd.DataFrame
        validation set
    y_valid: pd.DataFrame
        targets of the validation set

    Methods
    -------
    make_validation_set(valid_prop):
        split our training data into a training set and a validation set
    kfolds(k): Generator((np.ndarray, np.ndarray))
        split our training data into K folds
    """
    train_file = "data/train.csv"
    test_file = "data/test.csv"

    def __init__(self) -> None:
        train_data = pd.read_csv(DataManager.train_file)
        test_data = pd.read_csv(DataManager.test_file)

        self.train_data = train_data.drop(columns=["id"])
        self.test_data = test_data.drop(columns=["id"])

        self.init_sets()

    def init_sets(self) -> None:
        """
        Make the first training set and testing set from the raw data
        with no validation set
        """
        self.x_train = DataManager.get_x(self.train_data)
        self.y_train = DataManager.get_y(self.train_data)

        self.x_test = self.test_data

        self.x_valid = None
        self.y_valid = None

    def make_validation_set(self, valid_prop: float = 0.125) -> None:
        """
        Split our training data into a training set and a validation set

        Parameters
            valid_prop (between 0 and 1)
                Proportion of rows in the validation set
        """
        if self.x_valid is not None:
            self.init_sets()

        nb_data = len(self.x_train)
        nb_valid = int(nb_data * valid_prop)

        rng = np.random.default_rng()
        valid_idxs = rng.choice(nb_data, nb_valid, replace=False)
        train_idxs = np.delete(np.arange(nb_data), valid_idxs)

        self.x_valid = self.x_train.loc[valid_idxs]
        self.y_valid = self.y_train.loc[valid_idxs]

        self.x_train = self.x_train.loc[train_idxs]
        self.y_train = self.y_train.loc[train_idxs]


    def kfolds(self, k: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split our training data into K folds

        Parameters
            k (> 0)
                Number of folds

        Returns
            A generator containing the K-folds from our training set
        Yields
            train_idxs, test_idxs: (np.ndarray, np.ndarray)
            the indexes for the training set and the testing set for each iteration
        """
        kf = KFold(n_splits=k)
        return kf.split(self.x_train)

    @staticmethod
    def get_x(set: pd.DataFrame) -> pd.DataFrame:
        """
        Get all the features from a complete dataset

        Parameters
            set: pd.DataFrame
                The complete set for our features

        Returns
            the DataFrame of the features
        """
        return set.loc[:, set.columns != 'species']

    @staticmethod
    def get_y(set: pd.DataFrame) -> pd.DataFrame:
        """
        Get the targets from a complete dataset

        Parameters
            set: pd.DataFrame
                The complete set for our targets

        Returns
            the DataFrame of the targets
        """
        return set['species']