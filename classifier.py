from sklearn.base import BaseEstimator

import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import theano
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from pyearth import Earth


class Classifier(BaseEstimator):

    def __init__(self):

        self.clf = Pipeline([
            ('est', (ExtraTreesClassifier(max_depth=25, n_estimators=120))),
        ])

    def fit(self, X, y):
        X = X.astype(theano.config.floatX)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict_proba(X)

class LogScaler(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(1 + X)

"""
class MultiClassEarth(object):


    def fit(self, X, y):
        self.model = OneVsRestClassifier(Earth(max_degree=5, max_terms=10, penalty=4,
                                         thresh=0.01,
                                         minspan=100, endspan=100, check_every=100))
        self.model.fit(X, y)
        return

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.model.classes_)))
        for i, es in  enumerate(self.model.estimators_):
            probas[:, i] = es.predict(X)
        probas /= probas.sum(axis=1)[:, np.newaxis]
        return probas
"""
