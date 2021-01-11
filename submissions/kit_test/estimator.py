import pandas as pd
import numpy as np

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin, RegressorMixin

pd.options.mode.chained_assignment = None 
num_features = 128
win_length = 2# a changer

#https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
      ##but ==> ne pas scalerizer la column date
      scaler = StandardScaler()
      X.set_index('DATE', inplace=True)
      idx = X.index
      X = scaler.fit_transform(X)
      X_df = pd.DataFrame( X,index=idx)
      return X_df

class Label_Encoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        le = LabelEncoder()
        for column_name in X.columns:
            if X[column_name].dtype == object:
                X[column_name] = le.fit_transform(X[column_name])
            else:
                pass
        return X

class DLClassifier(BaseEstimator, ClassifierMixin,TransformerMixin):
    def __init__(self, model,win_length = 2,batch_size = 2, n_jobs=1):
        self.estimator = model
        self.win_length = win_length
        self.batch_size = batch_size
        self.model = {}
        self.batch = None

    def fit(self, X, y):
        #print(X)
        #print('---------')
        date_max = max(X.index.values)#'DATE'])#[:,1])
        date_min = date_max.astype('M8[M]')  - np.timedelta64(self.win_length,'M') 
        self.batch  = X[(X.index > date_min) & (X.index <= date_max)]
        
        y_train = pd.DataFrame(y, columns=['DATE','UNIQUE_CARRIER_NAME', 'LOAD_FACTOR'])
        carriers = np.unique( X['UNIQUE_CARRIER_NAME'])
        for carrier in carriers:
                X_used = X[X['UNIQUE_CARRIER_NAME'] == carrier]
                X_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
                y_used = y_train[y_train['UNIQUE_CARRIER_NAME'] == carrier]
                y_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
                train_generator = TimeseriesGenerator(X_used, y_used, length = self.win_length, sampling_rate=1, batch_size = self.batch_size)
                self.model[carrier] = self.estimator.fit(train_generator, epochs = 10 , shuffle=False, verbose = 0)
        
        return self
        
    def predict(self, X):
        #Sans batch#X_used = ###y = X_used[:, -1]
        X = pd.concat([self.batch ,X])## problem la il predi 22 elem??
        carriers = np.unique(X['UNIQUE_CARRIER_NAME'])
        predictions = pd.DataFrame(columns= ['UNIQUE_CARRIER_NAME','PRED'])
        for carrier in carriers:
            X_used = X[X['UNIQUE_CARRIER_NAME'] == carrier]
            y_pred_sub=pd.DataFrame( X[X['UNIQUE_CARRIER_NAME'] == carrier]['UNIQUE_CARRIER_NAME'][self.win_length:],columns= ['UNIQUE_CARRIER_NAME','PRED'])
            X_used.drop(columns='UNIQUE_CARRIER_NAME', inplace=True)
            y = np.zeros(X_used.shape[0])
            test_generator = TimeseriesGenerator(X_used,y,length=self.win_length, sampling_rate=1, batch_size=self.batch_size)
            predictions_tmp = self.model.get(carrier).predict(test_generator)
            y_pred_sub['PRED'] = predictions_tmp
            predictions = predictions.append(y_pred_sub)
        predictions.reset_index(level=0, inplace=True)
        return predictions['PRED'].to_numpy()


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(win_length, num_features)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LSTM(num_features, return_sequences=False)) 
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    return model


def get_estimator():
    labelEncoding_transforme = Label_Encoding()
    #preprocessor = make_column_transformer(('drop', ['UNIQUE_CARRIER_NAME' ]),remainder='passthrough')# ou aussi 'LOAD_FACTOR_SHIFTED'
    scaler = CustomScaler() #StandardScaler()

    clf = create_model ()#KerasRegressor(build_fn=create_model,epochs= 100, verbose=0)
    custer  = DLClassifier(clf)
    
    # just create the pipeline
    pipeline = Pipeline([
        ('LE_transformer', labelEncoding_transforme),
        #('Drop_transformer', preprocessor),
        ('scaling',scaler),
        ('clf',custer)
        ##inverse scaler
    ], verbose = False)
    return pipeline
