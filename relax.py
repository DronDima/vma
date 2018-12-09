import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
stack = []
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = n, nrows = 1)

def sum(A, fromVar, toVar, x, i):
    result = 0
    for j in range(fromVar, toVar):
        result += A[i,j] * x[j] / A[i, i]
    return result

def relax(A, b):
    w = 0.9
    x = np.zeros(n)
    X = np.zeros(n)
    iterations = 0
    for i in range(n):
        X[i] = (1 - w) * x[i] - w * sum(A, 0, i, X, i) - w * sum(A, i + 1, n, x, i) + w * b[i] / A[i, i]
    while(np.linalg.norm(X - x, np.inf) > eps):
        iterations += 1
        x = np.copy(X)
        for i in range(n):
            X[i] = (1 - w) * x[i] - w * sum(A, 0, i, X, i) - w * sum(A, i + 1, n, x, i) + w * b[i] / A[i, i]
    print("Количество итераций: ", iterations)
    print("Невязка: ", np.dot(A, X) - b)
    print("Решение: ", X)

b = np.copy(b)
b = np.dot(A.transpose(), b.reshape(5,1))
A = np.dot(A.transpose(), A)
print(A)
print(b)
relax(np.copy(A), np.copy(b).ravel())