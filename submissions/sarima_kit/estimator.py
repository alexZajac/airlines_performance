import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

import statsmodels.api as sm


class SARIMAEstimator(BaseEstimator):
    def __init__(self, trend_orders=(0, 1, 0), seasonal_orders=(1, 0, 1, 12)):
        self.airline_models = {}
        self.trend_orders = trend_orders
        self.seasonal_orders = seasonal_orders

    def fit(self, X, y):
        X_features = X.features
        self.end_X_train = max(X_features["DATE"])
        carriers = np.unique(X_features["UNIQUE_CARRIER_NAME"])

        for carrier in carriers:
            X_used = X_features[X_features["UNIQUE_CARRIER_NAME"]
                                == carrier].copy()
            X_used.drop(columns="UNIQUE_CARRIER_NAME", inplace=True)
            X_used.set_index("DATE", inplace=True)
            y_used = pd.Series(y[np.where(
                X_features["UNIQUE_CARRIER_NAME"] == carrier)[0]])
            y_used.index = X_used.index
            model = sm.tsa.statespace.SARIMAX(
                y_used,
                order=self.trend_orders,
                seasonal_order=self.seasonal_orders,
                enforce_stationarity=False,
                enforce_invertibility=False,
                freq="MS"
            )
            results = model.fit(disp=False)
            self.airline_models[carrier] = results
        return self

    def predict(self, X):
        X_features = X.features
        # get all unique carriers
        carriers = np.unique(X_features["UNIQUE_CARRIER_NAME"])
        # init a dataframe of predictions to align with dates and carriers
        predictions = pd.DataFrame(columns=["UNIQUE_CARRIER_NAME", "PRED"])
        # the prediction dates boundaries
        start_predict_date = min(X_features["DATE"])
        end_predict_date = max(X_features["DATE"])
        print(start_predict_date, end_predict_date)
        # prediction with the model for each carrier
        for carrier in carriers:
            # get forecast for number of months
            airline_model = self.airline_models[carrier]
            # out samples forecast for last year and in sample prediction for the rest
            load_factor_forecast = airline_model.predict(
                start=start_predict_date, end=end_predict_date
            )

            # subsetting dataframe for current airline
            y_pred_df = X_features[
                X_features["UNIQUE_CARRIER_NAME"] == carrier
            ][["UNIQUE_CARRIER_NAME", "DATE"]].copy()
            y_pred_df.set_index("DATE", inplace=True)
            y_pred_df["PRED"] = load_factor_forecast
            predictions = predictions.append(y_pred_df)

        return predictions["PRED"].values


def get_estimator():
    # just create the pipeline
    pipeline = Pipeline([
        ("SARIMAEstimator", SARIMAEstimator()),
    ])
    return pipeline
