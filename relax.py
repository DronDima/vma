import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
stack = []
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = 5, nrows = 1)

def sum(A, fromVar, toVar, x, i):
    result = 0
    print("i = ", i)
    for j in range(fromVar, toVar):
        print(j)
        result += A[i,j] * x[j] / A[i, i]
    return result

def relax(A, b):
    w = 0.9
    x = np.zeros(n)
    X = np.zeros(n)
    for i in range(n):
        X[i] = (1 - w) * x[i] - w * sum(A, 0, i, X, i) - w * sum(A, i + 1, n, x, i) + w * b[i] / A[i, i]
    while(np.linalg.norm(X - x, np.inf) > eps):
        x = np.copy(X)
        for i in range(n):
            X[i] = (1 - w) * x[i] - w * sum(A, 0, i, X, i) - w * sum(A, i + 1, n, x, i) + w * b[i] / A[i, i]
    print(X)

print("Расширенная матрица:")
print(np.insert(np.copy(A), n, b, axis=1))
relax(np.copy(A), np.copy(b).ravel())