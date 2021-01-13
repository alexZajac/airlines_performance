import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin, RegressorMixin

pd.options.mode.chained_assignment = None
num_features = 127
win_length = 2  # a changer

# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
      # but ==> ne pas scalerizer la column date
        # print("##################################################")
        #print("Debut scalar")
        # print("##################################################")
        #print(max(X['DATE']), "sc")
        scaler = StandardScaler()
        X.set_index(['DATE', 'UNIQUE_CARRIER_NAME'], inplace=True)
        idx = X.index
        X = scaler.fit_transform(X)
        X_df = pd.DataFrame(X, index=idx)
        X_df.reset_index(level='UNIQUE_CARRIER_NAME', inplace=True)
        return X_df


class Label_Encoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # print("##################################################")
        #print("Debut LE")
        # print("##################################################")
        #print(max(X['DATE']), "le")
        le = LabelEncoder()
        for column_name in X.columns:
            if X[column_name].dtype == object and column_name != 'UNIQUE_CARRIER_NAME':
                X[column_name] = le.fit_transform(X[column_name])
            else:
                pass
        return X


class DLClassifier(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, model, win_length=2, batch_size=1, n_jobs=1):
        self.n_jobs = n_jobs
        self.estimator = model
        self.win_length = win_length
        self.batch_size = batch_size
        self.model = {}
        self.batch = None
        #self._estimator_type =  "regressor"

    def fit(self, X, y):
        # print("##################################################")
        #print("Debut FIT")
        # print("##################################################")

        # print('---------')
        date_max = max(X.index.values)  # 'DATE'])#[:,1])
        #print(date_max, "1")
        #print(min(X.index.values), "1")
        # print("##################################################")
        date_min = date_max.astype(
            'M8[M]') - np.timedelta64(self.win_length, 'M')

        self.batch = X[(X.index > date_min) & (X.index <= date_max)]

        #y_train = pd.DataFrame(y, columns=['DATE','UNIQUE_CARRIER_NAME', 'LOAD_FACTOR'])
        carriers = np.unique(X['UNIQUE_CARRIER_NAME'])
        for carrier in carriers:
            X_used = X[X['UNIQUE_CARRIER_NAME'] == carrier].copy()
            #print(np.where(X['UNIQUE_CARRIER_NAME'] == carrier)[0])
            X_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
            # print(X_used)
            y_used = y[np.where(X['UNIQUE_CARRIER_NAME'] == carrier)[0]]
            #y_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
            train_generator = TimeseriesGenerator(
                X_used, y_used, length=self.win_length, sampling_rate=1, batch_size=self.batch_size)
            # tf.keras.models.clone_model(self.estimator)
            self.model[carrier] = create_model()
            print(y.shape)
            self.model[carrier].fit(
                train_generator, epochs=1, shuffle=False, verbose=0)
            # ou https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model
        return self

    def predict(self, X):
        print("##################################################")
        print("Debut predict")
        print("##################################################")
        print(self.batch)
        print("##################################################")
        # Sans batch#X_used = ###y = X_used[:, -1]
        X = pd.concat([self.batch, X])
        carriers = np.unique(X['UNIQUE_CARRIER_NAME'])
        predictions = pd.DataFrame(columns=['UNIQUE_CARRIER_NAME', 'PRED'])
        for carrier in carriers:
            X_used = X[X['UNIQUE_CARRIER_NAME'] == carrier].copy()
            y_pred_sub = pd.DataFrame(X[X['UNIQUE_CARRIER_NAME'] == carrier]['UNIQUE_CARRIER_NAME']
                                      [self.win_length:], columns=['UNIQUE_CARRIER_NAME', 'PRED'])
            X_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
            y = np.zeros(X_used.shape[0])
            print(carrier)
            print('X_used shape ==>', X_used.shape)
            print('y shape ==>', y.shape)
            print(X_used)
            print('-----------------------------------------')
            test_generator = TimeseriesGenerator(
                X_used, y, length=self.win_length, sampling_rate=1, batch_size=self.batch_size)
            # print(self.model)
            predictions_tmp = self.model.get(carrier).predict(test_generator)
            y_pred_sub['PRED'] = predictions_tmp
            predictions = predictions.append(y_pred_sub)
        predictions.reset_index(level=0, inplace=True)
        return predictions['PRED'].to_numpy()


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, return_sequences=True,
                                   input_shape=(win_length, num_features)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(num_features, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=tf.losses.MeanSquaredError(
    ), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanSquaredError()])
    return model


def get_estimator():
    labelEncoding_transforme = Label_Encoding()
    # preprocessor = make_column_transformer(('drop', ['UNIQUE_CARRIER_NAME' ]),remainder='passthrough')# ou aussi 'LOAD_FACTOR_SHIFTED'
    scaler = CustomScaler()  # StandardScaler()

    clf = create_model()
    custer = DLClassifier(clf)

    # just create the pipeline
    pipeline = Pipeline([
        ('LE_transformer', labelEncoding_transforme),
        #('Drop_transformer', preprocessor),
        ('scaling', scaler),
        ('clf', custer)
        # inverse scaler
    ], verbose=False)
    return pipeline
