"""
Linear Regression

Ordinary Least Square (OLS)
"""

import numpy as np


class LinearRegression():

    """OLS model

    """

    def fit(self, X, y):
        """Fit the linear model

        Parameters:
        -----------
        X : a numpy array of form [num_samples, num_features]

        y : numpy array of form [num_samples, 1]

        """

        # copy X to avoid copy
        Xn = np.ndarray.copy(X)
        yn = np.ndarray.copy(y)

        Xm = Xn.mean(axis=0)
        Xs = Xn.std(axis=0)
        Xs[Xs == 0] = 1
        self.Xm = Xm
        self.Xs = Xs

        ym = yn.mean(axis=0)
        self.ym = ym
        yn = y - ym

        Xn = Xn - Xm
        Xn = Xn / Xs

        Xn = np.hstack((np.ones(X.shape[0])[np.newaxis].T, Xn))

        # Use Matrix Solution method from numpy
        try:
            self.coef_, self.residuals_, self.rank_, self.S_ = np.linalg.lstsq(
                Xn, yn)

        except Exception:
            print('Error occurred in Matrix operation.')
            raise

    def predict(self, X):
        """Predict regression values for X

        """
        Xn = np.ndarray.copy(X)

        Xn = Xn - self.Xm
        Xn = Xn / self.Xs

        Xn = np.hstack((np.ones(X.shape[0])[np.newaxis].T, Xn))

        return np.dot(Xn, self.coef_) + self.ym
