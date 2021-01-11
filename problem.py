import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_absolute_error, r2_score

from rampwf.workflows import Estimator
from rampwf.score_types import BaseScoreType
from rampwf.prediction_types import make_regression
from rampwf.score_types import RMSE

DATA_HOME = ""
DATA_PATH = "data/"
PREDICT_COLUMN = "LOAD_FACTOR"

# --------------------------------------
# 1) Objects implementing the score type
# --------------------------------------


class MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="mae", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        """
        Calculate the absolute error for each instance (each combination of
        airline and date) and return the average over instances.

        Parameters
        ----------
        y_true : np.array representing the real values of the load factors 

        y_pred : np.array representing the predicted values of the load factors 

        Returns
        ------
        score : float representing the MAE of the 2 predictions
        """
        return mean_absolute_error(y_true, y_pred)


class RSquared(BaseScoreType):
    is_lower_the_better = False
    minimum = float("-inf")
    maximum = 1.0

    def __init__(self, name="r_squared", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        """
        Calculate the r squared coefficient between y_true and y_pred. A 
        constant model that always predicts the expected value of y, 
        disregarding the input features, would get a R² score of 0.0.

        Parameters
        ----------
        y_true : np.array representing the real values of the load factors 

        y_pred : np.array representing the predicted values of the load factors 

        Returns
        ------
        score : float representing the r² coefficient of the 2 predictions
        """
        return r2_score(y_true, y_pred)


# --------------------------------------
# 2) I/O functions
# --------------------------------------


def _read_data(path, dir_name):
    """RAMP function to read and get data for the challenge"""
    DATA_HOME = path
    X = pd.read_csv(
        os.path.join(DATA_HOME, DATA_PATH, dir_name, "features.csv"),
        index_col=0
    )
    X["DATE"] = pd.to_datetime(X["DATE"])
    X.sort_values(["DATE", "UNIQUE_CARRIER_NAME"], ascending=[True, True])
    y = X[PREDICT_COLUMN]
    del X[PREDICT_COLUMN]
    return X, y


def get_train_data(path="."):
    """RAMP wrapper function to get train data"""
    return _read_data(path, "train")


def get_test_data(path="."):
    """RAMP wrapper function to get test data"""
    return _read_data(path, "test")


# --------------------------------------
# 4) Ramp problem definition
# --------------------------------------
problem_title = "US Airlines Performance Forecast"
Predictions = make_regression()
workflow = Estimator()
score_types = [RMSE(precision=3), RSquared(precision=3)]


def _get_airline_cv(X, n_folds=3, test_size_in_months=12):
    """
    Splits the X dataset in a similar fashion as for the TimeSeriesSplit from
    sklearn, but with a custom airline/data groups separation, and usable for
    both RAMP and local data. Note that the function is not scalables for other 
    challenges since it wasn"t its purpose hence the naming for the parameters.

    Parameters
    ----------
    X : pandas DataFrame sorted by date and carrier name (n_rows, n_cols)
        Training data, where n_samples is the number of samples and 
        n_features is the number of features. Note that X is sorted by date.

    n_folds : integer representing how many folds to perform on X_used

    test_size_in_months : The number of datapoints to constitute a test sample 
    (for the current problem it"s 12 months)

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """
    max_date = max(X["DATE"]) + pd.DateOffset(months=1)
    X_index = X.index
    indices = []
    while n_folds > 0:
        test_split_start = max_date - \
            pd.DateOffset(months=test_size_in_months)
        test_indices = X_index[(X["DATE"] >= test_split_start) & (
            X["DATE"] < max_date)]
        train_indices = X_index[X["DATE"] < test_split_start]
        indices.append((train_indices, test_indices))
        max_date = test_split_start
        n_folds -= 1
    yield from reversed(indices)


def get_cv(X, y):
    """RAMP wrapper function for cross validation indices"""
    return _get_airline_cv(X)
