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

    def fit(self, X, y):
        X_features = X.features
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
        carriers = np.unique(X_features['UNIQUE_CARRIER_NAME'])
        predictions = pd.DataFrame(columns=['UNIQUE_CARRIER_NAME', 'PRED'])
        for carrier in carriers:
            airline_model = self.airline_models[carrier]
            pred_airline = airline_model.get_forecast(steps=12)
            y_pred = pred_airline.predicted_mean
            y_pred_df = pd.DataFrame(
                X_features[
                    X_features['UNIQUE_CARRIER_NAME'] == carrier
                ][['DATE', 'UNIQUE_CARRIER_NAME']]
            )
            y_pred_df['PRED'] = y_pred
            predictions = predictions.append(y_pred_df)
        predictions.reset_index(level=0, inplace=True)
        for carrier in carriers:
            print(predictions[
                predictions['UNIQUE_CARRIER_NAME'] == carrier
            ][['DATE', 'PRED']])
        # print(predictions)
        return predictions['PRED'].to_numpy()


def get_estimator():
    # just create the pipeline
    pipeline = Pipeline([
        ('SARIMAEstimator', SARIMAEstimator()),
    ])
    return pipeline
