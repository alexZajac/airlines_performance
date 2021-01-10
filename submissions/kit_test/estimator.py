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
win_length = 2
batch_size = 2
num_features = 104

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
      #print(X)
      X = scaler.fit_transform(X)
      X_df = pd.DataFrame( X,index=idx)
      #print(X_df)
      #X_df.set_index('DATE', inplace=True)
      #X.reset_index(level=0, inplace=True)
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
    def __init__(self, model, n_jobs=1):
        self.model = model
    def fit(self, X, y):
        #print(X)
        #print('---------')
        date_max = max(X.index.values)#'DATE'])#[:,1])
        date_min = date_max.astype('M8[M]')  - np.timedelta64(12,'M') 
        self.batch  = X[(X.index > date_min) & (X.index <= date_max)]
        
        y = np.asarray(y.iloc[ :,-1].values)
        train_generator = TimeseriesGenerator(X, y, length=win_length, sampling_rate=1, batch_size=batch_size)
        return self.model.fit(train_generator, epochs = 50 , shuffle=False, verbose = 0)
    def predict(self, X):
        X_used = pd.concat([self.batch ,X])## problem la il predi 22 elem??
        #X_used = ##
        #y = X_used[:, -1]
        y =X_used[X_used.columns[-1]]
        
        ## pas trop imp :: comment on predit avec le Y avec nous ???
        test_generator = TimeseriesGenerator(X_used,y,length=win_length, sampling_rate=1, batch_size=batch_size)
        return self.model.predict(test_generator)


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
