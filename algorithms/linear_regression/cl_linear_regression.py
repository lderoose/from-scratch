#!/usr/bin/python3.8


import numpy as np



class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init
        self.coef_ = np.zeros(n_features)
        if self.fit_intercept:
            self.intercept_ = 0

        # fitting
        for _ in range(self.num_iterations):

            y_pred = np.dot(X, self.coef_)
            
            if self.fit_intercept:
                y_pred += self.intercept_

            # compute gradient of coef_ and intercpt
            dw = (1/n_samples) * np.dot(X.values.T, (y_pred - y.values.ravel()))
            if self.fit_intercept:
                db = (1/n_samples) * np.sum(y_pred - y.values.ravel())

            # maj coef_
            self.coef_ -= self.learning_rate * dw
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * db

    def predict(self, X):
        p = np.dot(X, self.coef_)
        if self.fit_intercept:
            p += self.intercept_
        return p
    
    def score(self, X, y):
        y_bar = np.mean(y["label"])
        y_pred = self.predict(X)
        u = np.sum((y["label"] - y_pred)** 2)
        v = np.sum((y["label"] - y_bar)** 2)
        r2 = 1 - (u/v)
        return r2
