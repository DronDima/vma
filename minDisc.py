import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
stack = []
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = 5, nrows = 1)

def iteration(A, b, x):
    r = np.dot(A, x) - b
    Ar = np.dot(A, r)
    t = np.dot(Ar, r) / np.dot(Ar, Ar)
    X = x - t * r
    return (X)

def minDisc(A, b):
    x = np.zeros(n)
    X = iteration(A, b, x)
    while (np.linalg.norm(X - x, np.inf) > eps):
        x = X
        X = iteration(A, b, x)
    print(X)

print("Расширенная матрица:")
print(np.insert(np.copy(A), n, b, axis=1))
minDisc(np.copy(A), np.copy(b).ravel())