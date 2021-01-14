import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin

pd.options.mode.chained_assignment = None
num_features = 127


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """Wrapper of a standard scaler for out AirlineData data structure"""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # out features are contained in this attribute
        X_features_mod = X.features.copy()
        scaler = StandardScaler()
        X_features_mod.set_index(['DATE', 'UNIQUE_CARRIER_NAME'], inplace=True)
        idx = X_features_mod.index
        # scling for mean=0 and std=1
        X_features_mod = scaler.fit_transform(X_features_mod)
        X_df = pd.DataFrame(X_features_mod, index=idx)
        X_df.reset_index(level=['DATE', 'UNIQUE_CARRIER_NAME'], inplace=True)
        X.features = X_df
        return X


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """Wrapper of a standard scaler for out AirlineData data structure"""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_features_mod = X.features
        enc = LabelEncoder()
        for column_name in X_features_mod.columns:
            # transform all non numeric columns except the carrier name
            if X_features_mod[column_name].dtype == object and column_name != 'UNIQUE_CARRIER_NAME':
                X_features_mod[column_name] = enc.fit_transform(
                    X_features_mod[column_name]
                )
        X.features = X_features_mod
        return X


class LSTMRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, win_length=2, batch_size=2, n_jobs=1):
        self.n_jobs = n_jobs
        self.win_length = win_length
        self.batch_size = batch_size
        self.model = {}

    def fit(self, X, y):
        X_features = X.features
        carriers = np.unique(X_features['UNIQUE_CARRIER_NAME'])

        for carrier in carriers:
            # subsetting for the current carrier
            X_used = X_features[X_features['UNIQUE_CARRIER_NAME']
                                == carrier].copy()
            X_used.drop(columns=['DATE', 'UNIQUE_CARRIER_NAME'], inplace=True)
            # finding the
            y_used = y[np.where(
                X_features['UNIQUE_CARRIER_NAME'] == carrier)[0]]
            train_generator = TimeseriesGenerator(
                X_used, y_used, length=self.win_length, sampling_rate=1, batch_size=self.batch_size)
            self.model[carrier] = create_model(self.win_length)
            self.model[carrier].fit(
                train_generator, epochs=1, shuffle=False, verbose=0
            )
        return self

    def predict(self, X):
        X_features = X.features
        carriers = np.unique(X_features['UNIQUE_CARRIER_NAME'])
        predictions = pd.DataFrame(columns=['UNIQUE_CARRIER_NAME', 'PRED'])
        for carrier in carriers:
            X_used = X_features[X_features['UNIQUE_CARRIER_NAME']
                                == carrier].copy()
            y_pred_sub = pd.DataFrame(X_features[X_features['UNIQUE_CARRIER_NAME'] == carrier]['UNIQUE_CARRIER_NAME'], columns=[
                                      'UNIQUE_CARRIER_NAME', 'PRED'])
            y_pred_sub['PRED'] = 0
            X_used.drop(columns=['DATE', 'UNIQUE_CARRIER_NAME'], inplace=True)
            y = np.zeros(X_used.shape[0])
            test_generator = TimeseriesGenerator(
                X_used, y, length=self.win_length, sampling_rate=1, batch_size=self.batch_size)
            predictions_tmp = self.model.get(carrier).predict(test_generator)
            y_pred_sub['PRED'][self.win_length:] = predictions_tmp.reshape(-1)
            predictions = predictions.append(y_pred_sub)
        return predictions['PRED'].to_numpy()


def create_model(win_length):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   input_shape=(win_length, num_features)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(num_features, return_sequences=False))
    model.add(Dense(1))

    model.compile(
        loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error']
    )
    return model


def get_estimator():
    label_encoder = CustomLabelEncoder()
    scaler = CustomStandardScaler()
    model = LSTMRegressor()

    # just create the pipeline
    pipeline = Pipeline([
        ('label_encoder', label_encoder),
        ('standard_scaler', scaler),
        ('model', model)
        # inverse scaler
    ], verbose=False)
    return pipeline
