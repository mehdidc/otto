from sklearn.base import BaseEstimator

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
os.environ["OMP_NUM_THREADS"] = "1"

import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, Adam, adadelta
from nolearn.lasagne import NeuralNet


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = None
        self.label_encoder = None
        self.scaler = None
        self.net = None

    def fit(self, X, y):
        layers0 = [('input', InputLayer),
                   ('dense0', DenseLayer),
                   ('dropout', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('output', DenseLayer)]
        X = X.astype(theano.config.floatX)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y).astype(np.int32)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        num_classes = len(self.label_encoder.classes_)
        num_features = X.shape[1]
        self.net = NeuralNet(layers=layers0,
                             input_shape=(None, num_features),
                             dense0_num_units=200,
                             dropout_p=0.5,
                             dense1_num_units=200,
                             output_num_units=num_classes,
                             output_nonlinearity=softmax,

                             update=Adam,
                             update_learning_rate=0.001,


                             eval_size=0.2,
                             verbose=1,
                             max_epochs=40)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        return self.label_encoder.inverse_transform(self.net.predict(self.scaler.transform(X)))

    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        return self.net.predict_proba(self.scaler.transform(X))
