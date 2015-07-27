"""
Linear Regression

Ordinary Least Square (OLS)
"""

import numpy as np


class GradientDescentRegression():

    """Simple OLS implementation using Gradient Descent
    """

    def __init__(self, num_iters=2000, alpha=0.1):
        self.num_iters = num_iters
        self.alpha = alpha

    def _compute_cost(self, X, y, w):
        """Compute the value of cost function, J.
        Here J is total Least Square Error
        """
        m = X.shape[0]
        J = (1 / (2 * m)) * np.sum((np.dot(X, w) - y) ** 2)
        return J

    def _gradient_descent(self, X, y, w, num_iters, alpha):
        """Performs Graddient Descent.
        The threshold is set by num_iters, instead of some value in this implementation
        """
        m = X.shape[0]
        # Keep a history of Costs (for visualisation)
        J_all = np.zeros((num_iters, 1))

        # perform gradient descent
        for i in range(num_iters):
            #             print('GD: w: {0}'.format(w.shape))
            J_all[i] = self._compute_cost(X, y, w)

#             err0 = X.dot(w)
#             err = err0 - y[:,np.newaxis]
# print('X: {0}; y: {1}; w: {2}; err0: {3}; err: {4}'.format(X.shape, y.shape, w.shape, err0.shape, err.shape))
#             err2 = np.dot(X.T, err)
#             w = w - ((alpha / m) * err2)
            w = w - ((alpha / m) * np.dot(X.T, (X.dot(w) - y[:, np.newaxis])))

        return w, J_all

    def fit(self, X, y):
        """Fit the model
        """
        Xn = np.ndarray.copy(X)

        # initialise w params for lineaqr model, from w0 to w_num_features
        w = np.zeros((Xn.shape[1] + 1, 1))

        # normalise the X
        self.X_mean = np.mean(Xn, axis=0)
        self.X_std = np.std(Xn, axis=0)
        Xn -= self.X_mean
        Xn /= self.X_std

        # add ones fro intercept term
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        self.w, self.J_all = self._gradient_descent(
            Xn, y, w, self.num_iters, self.alpha)

        return self.w, self.J_all

    def predict(self, X):
        """Predict values for given X
        """
        Xn = np.ndarray.copy(X)
        Xn -= self.X_mean
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        return Xn.dot(self.w)
