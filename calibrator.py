from sklearn.base import BaseEstimator


class Calibrator(BaseEstimator):
    def __init__(self):
        self.lambda_ = 0.0070100000000000006

    def fit(self, X_array, y_pred_array):
        return self

    def predict_proba(self, probas):
        return (probas + self.lambda_ / probas.shape[1]) / (1. + self.lambda_)
