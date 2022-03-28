import pandas as pd
from sklearn.model_selection import train_test_split


class DataManager:
    """
    Loads our dataset and provides methods to split our data
    for training purposes

    Attributes
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
        init_sets
            create the first training set and testing set from the raw data with no validation set
        make_validation_set(valid_prop)
            split our training data into a training set and a validation set
    """
    train_file = "data/train.csv"
    test_file = "data/test.csv"
    target_column = "species"
    useless_columns = ["id"]

    def __init__(self) -> None:
        train_data = pd.read_csv(DataManager.train_file)
        test_data = pd.read_csv(DataManager.test_file)

        self.train_data = train_data.drop(columns=DataManager.useless_columns)
        self.test_data = test_data.drop(columns=DataManager.useless_columns)

        self.x_train, self.y_train = None, None
        self.x_test = None,
        self.x_valid, self.y_valid = None, None

        self.init_sets()

    def init_sets(self) -> None:
        """
        Create the first training set and testing set from the raw data
        with no validation set
        """

        self.x_train = self.train_data.loc[:, self.train_data.columns != DataManager.target_column]
        self.y_train = self.train_data[DataManager.target_column]

        self.x_test = self.test_data

        self.x_valid = None
        self.y_valid = None

    def make_validation_set(self, valid_prop: float = 0.2) -> None:
        """
        Split our training data into a training set and a validation set

        Parameters
            valid_prop (between 0 and 1)
                Proportion of rows in the validation set
        """

        if self.x_valid is not None:
            self.init_sets()

        self.x_train, self.x_valid = train_test_split(self.x_train, valid_prop)

        self.y_valid = self.y_train.iloc[self.x_valid.index]
        self.y_train = self.y_train.iloc[self.x_train.index]
