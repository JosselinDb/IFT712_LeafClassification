from typing import Generator
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class DataManager:
    """
    Load our dataset
    TODO à compléter


    Attributes
    ---------- TODO à compléter
            
    train_data: pd.DataFrame
        supervised dataset of our leaves
    test_data: pd.Dataframe
        unsupervised dataset of our leaves

    Methods
    ------- TODO à compléter
    make_validation_set(valid_prop): tuple[pd.DataFrame, pd.DataFrame]
        split our train data into a train set and a validation set

    """
    train_file = "data/train.csv"
    test_file = "data/test.csv"

    def __init__(self) -> None:
        train_data = pd.read_csv(DataManager.train_file)
        test_data = pd.read_csv(DataManager.test_file)

        self.train_data = train_data.drop(columns=["id"])
        self.test_data = test_data.drop(columns=["id"])

    def make_validation_set(self, valid_prop: float = 0.125) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split our train data into a train set and a validation set

        Parameters
            valid_prop (between 0 and 1)
                Proportion of rows in the validation set

        Returns
            The training set and the validation set from our training set
        """
        nb_data = len(self.train_data)
        nb_valid = int(nb_data * valid_prop)

        rng = np.random.default_rng()
        valid_idxs = rng.choice(nb_data, nb_valid, replace=False)
        train_idxs = np.delete(np.arange(nb_data), valid_idxs)

        return (
            self.train_data.loc[train_idxs],
            self.train_data.loc[valid_idxs]
        )

    def make_kfolds(self, k: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Split our train data into a train set and a validation set

        Parameters
            k (> 0)
                Number of folds

        Returns
            A generator containing the K-folds from our training set
            Yields (train_idxs, test_idxs)
        """
        kf = KFold(n_splits=k)
        return kf.split(self.train_data)
        