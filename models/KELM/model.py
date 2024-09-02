import pandas as pd
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class KELM:
    def __init__(self, kernel='rbf', gamma=1.0, C=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.beta = None
        self.X_train = None

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** self.gamma
        elif self.kernel == 'rbf':
            if X1.ndim == 1:
                X1 = X1.reshape(1, -1)
            if X2.ndim == 1:
                X2 = X2.reshape(1, -1)
            K = np.exp(-self.gamma * np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
            return K
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")

    def fit(self, X, y):
        self.X_train = X
        H = self._kernel_function(X, X)
        N = X.shape[0]
        try:
            self.beta = np.linalg.inv(H + np.eye(N) / self.C).dot(y)
            return 0
        except:
            return -1

    def predict(self, X):
        K = self._kernel_function(X, self.X_train)
        pred = K.dot(self.beta)
        res = np.where(pred > 0, 1, -1)
        return res.astype(int)
    def predict_proba(self, X):
        K = self._kernel_function(X, self.X_train)
        pred = K.dot(self.beta)
        prob_positive = sigmoid(pred)
        return np.vstack((1 - prob_positive, prob_positive)).T