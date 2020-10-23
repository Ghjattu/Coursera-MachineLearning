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
        delta = np.array([np.sum(np.multiply(hx - y, X[:, i])) / m for i in range(2)]).reshape(2, 1)
        theta -= learning_rate * delta


def visualize_cost_function(X, y, m):
    theta0 = np.arange(-15, 10, 0.5)
    theta1 = np.arange(-2, 4, 0.1)
    jx, jy, jz = [], [], []
    for i in theta0:
        for j in theta1:
            jx.append(i)
            jy.append(j)
            hx = h(X, np.array([[i], [j]])).ravel()
            jz.append(compute_cost(hx, y, m))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(jx), np.array(jy), np.array(jz))
    plt.show()


def plot_convergence(iterations):
    plt.xlabel('iteration number')
    plt.ylabel('cost function')
    plt.scatter(np.array([i for i in range(iterations)]), np.array(costs_list))
    plt.show()


def plot_line(df, X, theta):
    plt.xlabel('Population of City in 10.000s')
    plt.ylabel('Profit in $10.000s')
    plt.plot(df[:, 0], df[:, 1], 'rx', label='Training data')
    plt.plot(df[:, 0], h(X, theta).ravel(), label='Linear regression')
    plt.show()


df = pd.read_csv('data/ex1data1.txt', header=None).to_numpy()
m = df.shape[0]
X = np.expand_dims(df[:, 0], axis=1)
X = np.insert(X, 0, values=1, axis=1)
y = df[:, 1]
theta = np.zeros((2, 1))
iterations = 1500
learning_rate = 0.01

gradient_descent(iterations, X, theta, y, m, learning_rate)
plot_line(df, X, theta)
