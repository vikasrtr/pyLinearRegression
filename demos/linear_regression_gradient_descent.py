"""
Demo of linear regression (OLS)

"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

import sys
sys.path.append('../models')


def main():
    # include the OLS class
    from LinearRegressionGradientDescent import GradientDescentRegression

    data = pd.read_csv('data/machine.data.txt', header=None)

    # lets keep 9 attributes
    y = data[9]
    X = data.drop([0, 1, 9], axis=1)

    X = X.values
    y = y.values

    regr = GradientDescentRegression()

    w, J_all = regr.fit(X, y)
    outs = regr.predict(X)

    reg = linear_model.LinearRegression(normalize=True)
    reg.fit(X, y)
    skout = reg.predict(X)

    plt.scatter(y, outs, color='g', alpha=0.8, label='pyLinear')
    plt.scatter(y, skout, color='r', alpha=0.4, label='sklearn')
    plt.plot([0, 1200], [0, 1200], 'k-', lw=2)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
