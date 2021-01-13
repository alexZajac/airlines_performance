import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class DummyEstimator(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_features, X_tweets = X.features, X.tweets
        print(min(X_features["DATE"]), "1")
        print(max(X_features["DATE"]), "2")
        print(min(X_tweets["DATE"]), "3")
        print(max(X_tweets["DATE"]), "4")
        print("vrgf")
        return self

    def predict(self, X):
        X_features, X_tweets = X.features, X.tweets
        print(min(X_features["DATE"]), "5")
        print(max(X_features["DATE"]), "6")
        print(min(X_tweets["DATE"]), "7")
        print(max(X_tweets["DATE"]), "8")
        y = np.array([0] * len(X_features))
        return y


def get_estimator():
    # just create the pipeline
    pipeline = Pipeline([
        ('DummyEstimator', DummyEstimator()),
    ])
    return pipeline
