import numpy as np
import pandas as pd
import numpy.linalg as alg

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
stack = []
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = 5, nrows = 1)

def jacobi(A, b):
    B = np.zeros((n,n))
    g = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        B[i,:] = -A[i,:]/A[i,i]
        B[i,i] = 0
        g[i] = b[i]/A[i,i]
    print("Расширенная матрица:")
    print(np.insert(np.copy(B), n, g, axis=1))
    X = np.dot(B,x) + g
    iterations = 0
    while(np.linalg.norm(X-x, np.inf) > eps):
        iterations += 1
        x = X
        X = np.dot(B,x) + g
    print("Количество итераций: ", iterations)
    print("Решение: ", X)
    print("Невязка: ", np.dot(A, X) - b)
    print(((np.log10(eps) + np.log10(1 - alg.norm(B, np.inf)) - np.log10(alg.norm(g, np.inf)))/np.log10(alg.norm(B, np.inf)))-1)

jacobi(np.copy(A), np.copy(b).ravel())
