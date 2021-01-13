import pandas as pd
import numpy as np
import os
from dataclasses import dataclass

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from rampwf.workflows import Estimator
from rampwf.score_types import BaseScoreType
from rampwf.prediction_types import make_regression
from rampwf.score_types import RMSE

DATA_HOME = ""
DATA_PATH = "data/"
PREDICT_COLUMN = "LOAD_FACTOR"
NUM_AIRLINES = 20
NUM_MONTHS_TO_PREDICT = 12

# --------------------------------------
# 1) Objects implementing the score type
# --------------------------------------


class AirlineDataIndexer:
    def __init__(self, features, tweets):
        self.features = features
        self.tweets = tweets

    def __getitem__(self, key):
        """
        Allows the AirlineData class to be subscriptable. For a certain key,
        which here will mainly be list of x_train indices for cross validation,
        we get the corresponding date in the features dataframe, and since it
        has unique dates, and the tweets dataframe doesn't, we select all the 
        tweets corresponding to that date.
        """
        features_row_df = self.features.iloc[key]
        indexed_date = features_row_df["DATE"]
        tweets_for_date_df = self.tweets[self.tweets["DATE"].isin(
            indexed_date)]
        return AirlineData(
            features_row_df,
            tweets_for_date_df,
            AirlineDataIndexer(features_row_df, tweets_for_date_df)
        )


@dataclass
class AirlineData:
    """
    Wrapper class around the airlines traditional features (stats + weather) 
    and the tweets additional dataframe
    """

    features: pd.DataFrame
    tweets: pd.DataFrame
    # hack to allow custom indexing on the airline data object
    iloc: AirlineDataIndexer

    @classmethod
    def load_from_file(cls, path_features, path_tweets):
        features_df = pd.read_csv(path_features, index_col=0)
        features_df["DATE"] = pd.to_datetime(features_df["DATE"])
        features_df.sort_values(["DATE", "UNIQUE_CARRIER_NAME"], inplace=True)

        tweets_df = pd.read_csv(path_tweets, index_col=0)
        tweets_df["DATE"] = pd.to_datetime(tweets_df["DATE"])
        tweets_df.sort_values(["DATE", "UNIQUE_CARRIER_NAME"], inplace=True)

        return cls(
            features_df, tweets_df, AirlineDataIndexer(features_df, tweets_df)
        )

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
        num_preds = len(y_pred)
        num_expected_preds = NUM_MONTHS_TO_PREDICT*NUM_AIRLINES
        if num_preds < num_expected_preds:
            raise ValueError(
                f"Insufficient number of predictions, expected " +
                f"{num_expected_preds}, instead got {num_preds}.\n Make sure to" +
                f"have one preiction for each airline and month of the year."
            )
        y_true = y_true[-num_expected_preds:]
        y_pred = y_pred[-num_expected_preds:]
        return mean_absolute_error(y_true, y_pred)


class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="rmse", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        """
        Calculate the RMSE between the forecasted y_pred and y_true

        Parameters
        ----------
        y_true : np.array representing the real values of the load factors 

        y_pred : np.array representing the predicted values of the load factors 

        Returns
        ------
        score : float representing the RMSE with the predictions
        """
        num_preds = len(y_pred)
        num_expected_preds = NUM_MONTHS_TO_PREDICT*NUM_AIRLINES
        if num_preds < num_expected_preds:
            raise ValueError(
                f"Insufficient number of predictions, expected " +
                f"{num_expected_preds}, instead got {num_preds}.\n Make sure to" +
                f"have one preiction for each airline and month of the year."
            )
        y_true = y_true[-num_expected_preds:]
        y_pred = y_pred[-num_expected_preds:]
        return np.sqrt(mean_squared_error(y_true, y_pred))


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
        num_preds = len(y_pred)
        num_expected_preds = NUM_MONTHS_TO_PREDICT*NUM_AIRLINES
        if num_preds < num_expected_preds:
            raise ValueError(
                f"Insufficient number of predictions, expected " +
                f"{num_expected_preds}, instead got {num_preds}.\n Make sure to" +
                f"have one preiction for each airline and month of the year."
            )
        y_true = y_true[-num_expected_preds:]
        y_pred = y_pred[-num_expected_preds:]
        return r2_score(y_true, y_pred)


# --------------------------------------
# 2) I/O functions
# --------------------------------------


def _read_data(path, dir_name):
    """RAMP function to read and get data for the challenge"""
    DATA_HOME = path
    path_features = os.path.join(DATA_HOME, DATA_PATH, dir_name, "local.csv")
    path_tweets = os.path.join(
        DATA_HOME, DATA_PATH, dir_name, "tweets_local.csv"
    )
    X = AirlineData.load_from_file(path_features, path_tweets)
    airline_features = X.features
    y = airline_features[PREDICT_COLUMN].values
    del airline_features[PREDICT_COLUMN]
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


def _get_airline_cv(X, n_folds=1, test_size_in_months=12):
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
    X_features = X.features
    max_date = max(X_features["DATE"]) + pd.DateOffset(months=1)
    indices = []
    while n_folds > 0:
        test_split_start = max_date - \
            pd.DateOffset(months=test_size_in_months)
        test_indices = np.where((X_features["DATE"] >= test_split_start) & (
            X_features["DATE"] < max_date))[0]
        train_indices = np.where(X_features["DATE"] < test_split_start)[0]
        indices.append((train_indices, test_indices))
        max_date = test_split_start
        n_folds -= 1
    yield from reversed(indices)


def get_cv(X, y):
    """RAMP wrapper function for cross validation indices"""
    return _get_airline_cv(X)
