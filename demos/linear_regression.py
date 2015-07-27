"""
Demo of linear regression (OLS)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

import sys
sys.path.append('../models')

# include the OLS class
from LinearRegression import LinearRegression

data = pd.read_csv('data/machine.data.txt', header=None)

# lets keep 9 attributes
y = data[9]
X = data.drop([0, 1, 9], axis=1)

X = X.values
y = y.values

reg = linear_model.LinearRegression(normalize=True)
reg.fit(X, y)
skout = reg.predict(X)
plt.scatter(y, skout, color='r', alpha=0.4, label='scikit-learn')
plt.plot([-200, 0], [0, 0], 'k--', lw=1)
plt.plot([0, 0], [-200, 0], 'k--', lw=1)


regr = LinearRegression()
regr.fit(X, y)
outs = regr.predict(X)
plt.scatter(y, outs, color='g', alpha=0.5, label='pyLinear')

plt.legend()
plt.show()
