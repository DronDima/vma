import numpy as np
import pandas as pd
from sympy.solvers import solve
import sympy as sy
from math import sqrt
n = 5
np.set_printoptions(precision = 4)
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)
def lenth(x):
    sum = 0
    for el in x:
        sum += el**2
    return sqrt(sum)
def krilov(A):
    M = np.eye(n)
    c = np.zeros(n)
    c[0] = 1
    M[:, n-1] = c
    for k in range(n-1,0,-1):
        M[:, k-1] = np.dot(A, M[:, k])
    b = np.dot(A, M[:, 0])
    Q = np.linalg.solve(M, b)
    print("Решение системы: ", Q)
    x = sy.Symbol("x")
    eigvals = np.array(solve(x ** 5 - Q[0] * x ** 4 - Q[1] * x ** 3 - Q[2] * x ** 2 - Q[3] * x - Q[4]))
    eigvect = np.zeros((n,n))
    for i in range(n):
        d = np.zeros(n)
        d[0] = 1
        for j in range(1,n):
            d[j] = eigvals[i]*d[j-1] - Q[j-1]
        for j in range(n):
            eigvect[:, i] += M[:, j]*d[j]
        eigvect[:, i] /= lenth(eigvect[:, i])
    return eigvals, eigvect

A = np.dot(np.copy(A).transpose(), np.copy(A))
eigvals, eigvect = krilov(A)
print("Собственные значения: ", eigvals.astype(np.double, 4))
print("Собственные векторы:")
print(eigvect)
print("Невязка:")
for i in range(n):
    disc = np.dot(A, eigvect[:, i]) - np.dot(eigvals[i], eigvect[:, i])
    print(disc.astype(np.double, 4))