import pandas as pd
import numpy as np
import os

from sklearn.base import is_classifier
from sklearn.utils import _safe_indexing
from sklearn.model_selection import TimeSeriesSplit

from rampwf.utils.importing import import_module_from_source  # a revoir
from rampwf.workflows import SKLearnPipeline, Estimator
from rampwf.score_types import BaseScoreType
from rampwf.prediction_types.base import BasePrediction
import rampwf as rw

# https://paris-saclay-cds.github.io/ramp-workflow/problem.html
DATA_HOME = ""
DATA_PATH = "data/"
batch_size = 2  # a changer
#WINDOWS_SIZE = 12

##############################################
# 3 - Score class
##############################################


class AirlineRMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name="RMSE (airline forecast)", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        """
        Calculate the squared error for each instance (each combination of
        airline and date) and return the squared average over instances.
        The containers y_true and y_pred are dataframes with columns:
            - y_true: ['DATE'    , 'UNIQUE_CARRIER_NAME', 'LOAD_FACTOR']
                       2018-01-01   United Airlines Inc.      0.8457
                       ...
            - y_pred: Same columns and shape as y_true
        Arguments:
            y_true {pd.DataFrame} -- true load factors
            y_pred {pd.DataFrame} -- predicted load factors
        Returns:
            float -- RMSE-score, between 0.0 and inf
        """
        # y_true = pd.DataFrame(
        #     y_true, columns=['DATE', 'UNIQUE_CARRIER_NAME', 'LOAD_FACTOR']
        # )
        # y_pred = pd.DataFrame(
        #     y_pred, columns=['DATE', 'UNIQUE_CARRIER_NAME', 'LOAD_FACTOR_PRED']
        # )
        # merged_prediction = y_pred.merge(
        #     y_true, how='left', on=['DATE', 'UNIQUE_CARRIER_NAME']
        # )
        return np.sqrt(
            np.mean(
                np.square(y_true - y_pred)
            )
        )
        # return np.sqrt(
        #     np.mean(
        #         (
        #             merged_prediction['LOAD_FACTOR'] -
        #             merged_prediction['LOAD_FACTOR_PRED']
        #         ).pow(2).values
        #     )
        # )


# class MAE(BaseScoreType):
#     is_lower_the_better = True
#     minimum = 0.0
#     maximum = float('inf')

#     def __init__(self, name='rmse', precision=2):
#         self.name = name
#         self.precision = precision

#     def __call__(self, y_true, y_pred):
#         error = []
#         y_true = pd.DataFrame(
#             y_true, columns=['DATE', 'UNIQUE_CARRIER_NAME', 'LOAD_FACTOR'])
#         y_pred = pd.DataFrame(
#             y_pred, columns=['DATE', 'UNIQUE_CARRIER_NAME', 'LOAD_FACTOR'])
#         carriers = np.unique(y_true['UNIQUE_CARRIER_NAME'])
#         m_df = y_pred.merge(y_true, how='left', on=[
#                             'DATE', 'UNIQUE_CARRIER_NAME'])
#         for carrier in carriers:
#             if carrier != 'US Airways Inc.':
#                 y_true_c = m_df[m_df['UNIQUE_CARRIER_NAME'] ==
#                                 carrier]['LOAD_FACTOR_x']  # y_true.get(carrier)
#                 y_pred_c = m_df[m_df['UNIQUE_CARRIER_NAME']
#                                 == carrier]['LOAD_FACTOR_y']  # a changer
#                 error.extend(np.abs(y_true_c - y_pred_c) / len(y_true_c))
#         return np.mean(error)
##############################################
# 3 - Workflow class
##############################################


# class EstimatorAirlines(SKLearnPipeline):

#     def __init__(self):
#         super().__init__()

#     def train_submission(self, module_path, X, y, train_idx=None):
#         """Train the estimator of a given submission.
#         Parameters
#         ----------
#         module_path : str
#             The path to the submission where `filename` is located.
#         X : {array-like, sparse matrix, dataframe} of shape \
#                 (n_samples, n_features)
#             The data matrix.
#         y : array-like of shape (n_samples,)
#             The target vector.
#         train_idx : array-like of shape (n_training_samples,), default=None
#             The training indices. By default, the full dataset will be used
#             to train the model. If an array is provided, `X` and `y` will be
#             subsampled using these indices.
#         Returns
#         -------
#         estimator : estimator object
#             The scikit-learn fitted on (`X`, `y`).
#         """
#         train_idx = slice(None, None, None) if train_idx is None else train_idx
#         submission_module = import_module_from_source(
#             os.path.join(module_path, self.filename),
#             os.path.splitext(self.filename)[0],  # keep the module name only
#             sanitize=True
#         )
#         estimator = submission_module.get_estimator()
#         X_train = _safe_indexing(X, train_idx)
#         y_train = _safe_indexing(y, train_idx)
#         ####
#         y_train = pd.DataFrame(
#             y_train, columns=['DATE', 'UNIQUE_CARRIER_NAME', 'LOAD_FACTOR'])
#         estimators = {}
#         carriers = np.unique(X_train['UNIQUE_CARRIER_NAME'])
#         for carrier in carriers:
#             if carrier != 'US Airways Inc.':
#                 X = X_train[X_train['UNIQUE_CARRIER_NAME'] == carrier]
#                 X.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
#                 Y = y_train[y_train['UNIQUE_CARRIER_NAME'] == carrier]
#                 estimators[carrier] = estimator.fit(X, Y)
#         ####
#         return estimators

#     def test_submission(self, estimator_fitted, X):
#         """Predict using a fitted estimator.
#         Parameters
#         ----------
#         estimator_fitted : Estimator object
#             A fitted scikit-learn estimator.
#         X : {array-like, sparse matrix, dataframe} of shape \
#                 (n_samples, n_features)
#             The test data set.
#         Returns
#         -------
#         pred : ndarray of shape (n_samples, n_classes) or (n_samples)
#         """
#         ######
#         carriers = np.unique(X['UNIQUE_CARRIER_NAME'])
#         predictions = pd.DataFrame(columns=['UNIQUE_CARRIER_NAME', 'PRED'])
#         for carrier in carriers:
#             if carrier != 'US Airways Inc.':
#                 X_used = X[X['UNIQUE_CARRIER_NAME'] == carrier]
#                 y_pred_sub = pd.DataFrame(X[X['UNIQUE_CARRIER_NAME'] == carrier]['UNIQUE_CARRIER_NAME'][batch_size:], columns=[
#                                           'UNIQUE_CARRIER_NAME', 'PRED'])
#                 X_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
#                 predictions_tmp = estimator_fitted.get(carrier).predict(X_used)
#                 y_pred_sub['PRED'] = predictions_tmp
#                 predictions = predictions.append(y_pred_sub)
#         predictions.reset_index(level=0, inplace=True)
#         if predictions is None:
#             raise ValueError('NaNs found in the predictions.')
#         #########
#         return np.array(predictions)

##############################################
# 3 - Prediction type class
##############################################


class _Predictions(BasePrediction):
    def __init__(self, n_columns=1, y_pred=None, y_true=None, n_samples=None):
        """Essentially the same as in a regression task, but the prediction is a list not a float."""
        self.n_columns = n_columns
        if y_pred is not None:
            self.y_pred = y_pred
        elif y_true is not None:
            self.y_pred = y_true
        elif n_samples is not None:
            shape = (n_samples)
            self.y_pred = np.empty(shape, dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                "Missing init argument: y_pred, y_true, or n_samples"
            )
        # deja fait en base ##https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/prediction_types/base.py

    # def set_valid_in_train(self, predictions, test_is):
    #     """Set a cross-validation slice."""
    #     print(self.y_pred)
    #     print(predictions)
    #     self.y_pred[test_is] = predictions.y_pred


##############################################
# 4 - definir functions
##############################################

def make_load_factor_prediction():
    return _Predictions


def _read_data(path, dir_name):
    DATA_HOME = path
    X = pd.read_csv(os.path.join(DATA_HOME, DATA_PATH,
                                 dir_name, 'features.csv'), index_col=0)
    X['DATE'] = pd.to_datetime(X['DATE'])
    X.sort_values(['DATE', 'UNIQUE_CARRIER_NAME'], ascending=[True, True])
    y = X['LOAD_FACTOR']
    del X['LOAD_FACTOR']
    return X, np.array(y)


# --------------------------------------
# 4) Ramp problem definition
# --------------------------------------
problem_title = "US Airlines Performance Forecast"
Predictions = rampwf.prediction_types.regression.make_regression()
workflow = Estimator()
score_types = [AirlineRMSE()]


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")


def get_cv(X, y):
    tscv = TimeSeriesSplit(n_splits=4)
    return tscv.split(X, y)
