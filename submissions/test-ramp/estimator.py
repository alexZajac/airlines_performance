import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class DummyEstimator(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        print(min(X["DATE"]), "1")
        print(max(X["DATE"]), "2")
        print("vrgf")
        return self

    def predict(self, X):
        print(min(X["DATE"]), "5")
        print(max(X["DATE"]), "6")
        y = np.array([0] * len(X))
        return y


def get_estimator():
    # just create the pipeline
    pipeline = Pipeline([
        ('DummyEstimator', DummyEstimator()),
    ])
    return pipeline
