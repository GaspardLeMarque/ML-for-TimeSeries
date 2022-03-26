import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.metrics import mean_squared_error

"""
Compare results of self-written linear regression with sklearn's one
"""

class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        n, k = X.shape
        X_train = X

# Dummy equals 1         
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y

        return self
        
    def predict(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w

        return y_pred
    
    def get_weights(self):
        return self.w

# Generate data to test a model
from sklearn.model_selection import train_test_split    

# Create simple linear function
def linear_expression(x):
    return 5 * x + 6

# Generate target values with random noise 
objects_num = 50
X = np.linspace(-5, 5, objects_num)
y = linear_expression(X) + np.random.randn(objects_num) * 5

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

# Draw a plot
plt.figure(figsize=(10, 7))
plt.plot(X, linear_expression(X), label='Real', c='g')
plt.scatter(X_train, y_train, label='Train', c='b')
plt.scatter(X_test, y_test, label='Test', c='orange')

plt.title("Generated data")
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# Fit the model and make prediction
regressor = MyLinearRegression()

# Adding one dimension with newaxis method because we have a vector of features instead of a matrix
regressor.fit(X_train[:, np.newaxis], y_train)

predictions = regressor.predict(X_test[:, np.newaxis])
w = regressor.get_weights()

# The weights not crucually deviate from equation's coefficients
print(w)

# Draw three plots for a separate comparison on both sets - train and test
plt.figure(figsize=(20, 7))

ax = None

for i, types in enumerate([['train', 'test'], ['train'], ['test']]):
    ax = plt.subplot(1, 3, i + 1, sharey=ax)
    if 'train' in types:
        plt.scatter(X_train, y_train, label='train', c='b')
    if 'test' in types:
        plt.scatter(X_test, y_test, label='test', c='orange')

    plt.plot(X, linear_expression(X), label='real', c='g')
    plt.plot(X, regressor.predict(X[:, np.newaxis]), label='predicted', c='r')

    plt.ylabel('target')
    plt.xlabel('feature')
    plt.title(" ".join(types))
    plt.grid(alpha=0.2)
    plt.legend()

plt.show()

# Compare to sklearn results
sk_reg = LinearRegression().fit(X_train[:, np.newaxis], y_train)

plt.figure(figsize=(10, 7))
plt.plot(X, linear_expression(X), label='real', c='g')

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X, regressor.predict(X[:, np.newaxis]), label='my', c='r', linestyle=':')
plt.plot(X, sk_reg.predict(X[:, np.newaxis]), label='sklearn', c='cyan', linestyle=':')

plt.title("Different Prediction")
plt.ylabel('target')
plt.xlabel('feature')
plt.grid(alpha=0.2)
plt.legend()

# Visually the dependencies are really close to each other
plt.show()

# Compare results with MSE
train_predictions = regressor.predict(X_train[:, np.newaxis])
test_predictions = regressor.predict(X_test[:, np.newaxis])

train_predictions_skl = sk_reg.predict(X_train[:, np.newaxis])
test_predictions_skl = sk_reg.predict(X_test[:, np.newaxis])

print('Train MSE: ', mean_squared_error(y_train, train_predictions), mean_squared_error(y_train, train_predictions_skl))
print('Test MSE: ', mean_squared_error(y_test, test_predictions), mean_squared_error(y_test, test_predictions_skl))
