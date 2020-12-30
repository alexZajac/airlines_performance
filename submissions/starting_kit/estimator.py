import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class Detector(BaseEstimator):
    """Dummy detector"""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))


def get_estimator():
    detector = Detector()
    # make pipeline
    pipeline = Pipeline(steps=[('detector', detector)])
    return pipeline
