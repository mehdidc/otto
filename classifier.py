from sklearn.base import BaseEstimator

import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import theano


class Classifier(BaseEstimator):

    def __init__(self):
        self.clf = Pipeline([
            #('log', LogScaler()),
            ('scaler', StandardScaler()),
            ('nnet', SimpleNeuralNet(max_nb_epochs=150,
                                     nb_hidden_list=[500],
                                     activations=[('maxout', {"nb_components":3})],
                                     dropout_probs=[0.5],
                                     learning_rate=0.1,
                                     optimization_method='rmsprop',
                                     L1_factor=0.0001,
                                     batch_size=256)),
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
