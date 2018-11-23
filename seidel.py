import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = 5, nrows = 1)

def seidel(A, b):
    E = np.identity(n)
    B = np.zeros((n, n))
    g = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        B[i,:] = -A[i,:]/A[i,i]
        B[i,i] = 0
        g[i] = b[i]/A[i,i]
    print("Расширенная матрица:")
    print(np.insert(np.copy(B), n, g, axis=1))
    H = np.tril(B)
    F = np.triu(B)
    X = np.dot(np.dot(np.linalg.inv(E-H), F), x) + np.dot(np.linalg.inv(E-H), g) # (E-H)^-1*F*x + (E-H)^-1 * g
    iterations = 0
    while (np.linalg.norm(X - x, np.inf) > eps):
        iterations += 1
        x = X
        X = np.dot(np.dot(np.linalg.inv(E - H), F), x) + np.dot(np.linalg.inv(E - H), g)
    print("Количество итераций: ", iterations)
    print("Невязка: ", np.dot(A, X) - b)
    print("Решение: ", X)

seidel(np.copy(A), np.copy(b).ravel())