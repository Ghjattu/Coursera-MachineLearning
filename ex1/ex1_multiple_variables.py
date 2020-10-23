import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

costs_list = []


def h(X, theta):
    return np.matmul(X, theta)


def compute_cost(hx, y, m):
    return np.sum((hx - y) ** 2) / (2 * m)


def gradient_descent(iterations, X, theta, y, m, learning_rate):
    for itr in range(iterations):
        hx = h(X, theta).ravel()
        cost = compute_cost(hx, y, m)
        costs_list.append(cost)
        delta = np.array([np.sum(np.multiply(hx - y, X[:, i])) / m for i in range(3)]).reshape(3, 1)
        theta -= learning_rate * delta


def plot_convergence(iterations):
    plt.xlabel('iteration number')
    plt.ylabel('cost function')
    plt.scatter(np.array([i for i in range(iterations)]), np.array(costs_list))
    plt.show()


def feature_normalization(X):
    means, stds = [], []
    for i in range(2):
        means.append(np.mean(X[:, i]))
        stds.append(np.std(X[:, i]))
        X[:, i] = ((X[:, i]) - means[i]) / stds[i]
    return means, stds

def normal_equation(X, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)


df = pd.read_csv('data/ex1data2.txt', header=None).to_numpy()
m = df.shape[0]
X = np.array(df[:, :2], dtype=np.float_)
means, stds = feature_normalization(X)
X = np.insert(X, 0, values=1.0, axis=1)
y = df[:, 2]
theta = np.zeros((3, 1))
iterations = 1500
learning_rate = 0.01

pred = np.array([[1., 1650., 3.]])
for i in range(1, 3):
    pred[0][i] = (pred[0][i] - means[i - 1]) / stds[i - 1]

gradient_descent(iterations, X, theta, y, m, learning_rate)
# plot_convergence(iterations)
print(np.sum(h(pred, theta)))
print(np.sum(h(pred, normal_equation(X, y).reshape(3, 1))))
