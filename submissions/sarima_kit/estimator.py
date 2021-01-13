import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

import statsmodels.api as sm


class SARIMAEstimator(BaseEstimator):
    def __init__(self, trend_orders=(0, 0, 0), seasonal_orders=(1, 0, 1, 12)):
        self.airline_models = {}
        self.trend_orders = trend_orders
        self.seasonal_orders = seasonal_orders
        self.end_X_train = None

    def fit(self, X, y):
        X_features = X.features
        self.end_X_train = max(X_features["DATE"])
        carriers = np.unique(X_features['UNIQUE_CARRIER_NAME'])

        for carrier in carriers:
            X_used = X_features[X_features['UNIQUE_CARRIER_NAME']
                                == carrier].copy()
            X_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
            X_used.set_index("DATE", inplace=True)
            y_used = pd.Series(y[np.where(
                X_features['UNIQUE_CARRIER_NAME'] == carrier)[0]])
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
        carriers = np.unique(X_features['UNIQUE_CARRIER_NAME'])
        # init a dataframe of predictions to align with dates and carriers
        predictions = pd.DataFrame(columns=['UNIQUE_CARRIER_NAME', 'PRED'])
        # the last prediction date
        end_predict_date = max(X_features["DATE"])
        print(self.end_X_train)
        print(end_predict_date)
        month_to_forecast = (
            end_predict_date.to_period('M') - self.end_X_train.to_period('M')
        ).n
        #  used to have dummy prediction before the months to predict
        y_dummy = pd.Series(
            0,
            index=np.unique(
                X_features[X_features["DATE"] <= self.end_X_train]["DATE"]
            )
        )
        # prediction with the model for each carrier
        for carrier in carriers:
            # get forecast for number of months
            airline_model = self.airline_models[carrier]
            pred_airline = airline_model.get_forecast(steps=month_to_forecast)

            load_factor_forecast = pred_airline.predicted_mean
            # subsetting dataframe for current airline
            y_pred_df = X_features[
                X_features['UNIQUE_CARRIER_NAME'] == carrier
            ][["UNIQUE_CARRIER_NAME", "DATE"]].copy()
            y_pred_df.set_index("DATE", inplace=True)

            # dummy prediction for previous years and needed predictions
            y_pred = pd.concat([y_dummy, load_factor_forecast])
            y_pred_df['PRED'] = y_pred
            predictions = predictions.append(y_pred_df)
        return predictions['PRED'].values


def get_estimator():
    # just create the pipeline
    pipeline = Pipeline([
        ('SARIMAEstimator', SARIMAEstimator()),
    ])
    return pipeline
