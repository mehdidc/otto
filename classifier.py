from sklearn.base import BaseEstimator

import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet, channel_out
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import theano


class Classifier(BaseEstimator):

    def __init__(self):
        self.clf = Pipeline([
            ('log', LogScaler()),
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[1000, 1000],
                                          learning_rate=1.,
                                          optimization_method='adadelta',
                                          dropout_probs=[0.7, 0],
                                          activations=['relu', channel_out(10)],
                                          max_nb_epochs=30,
                                          batch_size=256,
                                          )),
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
