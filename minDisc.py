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
    print(A)
    x = np.zeros(n)
    X = iteration(A, b, x)
    iterations = 0
    while (np.linalg.norm(X - x, np.inf) > eps):
        iterations += 1
        x = X
        X = iteration(A, b, x)
    print("Количество итераций: ", iterations)
    print("Невязка: ", np.dot(A, X) - b)
    print("Решение: ", X)

b = np.copy(b)
b = np.dot(A.transpose(), b.reshape(5,1))
A = np.dot(A.transpose(), A)
minDisc(np.copy(A), np.copy(b).ravel())