import numpy as np
import pandas as pd


from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn import random_projection

from sklearn.mixture import GMM

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        """
        cols = X_dict[0].keys()
        X = np.array([[instance[col] for col in cols] for instance in X_dict])
        #min_dim = johnson_lindenstrauss_min_dim(n_samples=X.shape[0], eps=0.4)
        #print("min_dim : %d" % (min_dim,))
        min_dim = 200
        self.rp = random_projection.GaussianRandomProjection(n_components=min_dim)
        X_rp = self.rp.fit_transform(X)
        self.gmms = []
        self.classes = []
        self.props = []

        for cl in set(y):
            gmm = GMM(n_components=10).fit(X_rp[y==cl])
            self.gmms.append(gmm)
            self.classes.append(cl)
            self.props.append( np.sum(y==cl) / float(y.shape[0])  )
        print(self.props)
        """
        return self

    def transform(self, X_dict, test=False):
        cols = X_dict[0].keys()
        X = np.array([[instance[col] for col in cols] for instance in X_dict])
        return X

        """
        if test == False:
            X_list = []
            y_list = []
            for cl, gmm, prop in zip(self.classes, self.gmms, self.props):
                nb = int(100000 * prop)
                samples = gmm.sample(nb, random_state=1024).tolist()
                X_list.append(samples)
                y_list.append([cl] * nb)
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            return X, y
        else:
            cols = X_dict[0].keys()
            X = np.array([[instance[col] for col in cols] for instance in X_dict])
            return self.rp.transform(X), None
        """
