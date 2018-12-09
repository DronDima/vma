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
    yz = np.zeros(n)
    yz[0] = 1
    M[:, n-1] = yz
    for k in range(n-1,0,-1):
        M[:, k-1] = np.dot(A, M[:, k])
    b = np.dot(A, M[:, 0])
    D = np.linalg.solve(M, b)
    x = sy.Symbol("x")
    eigvals = solve(x ** 5 - D[0] * x ** 4 - D[1] * x ** 3 - D[2] * x ** 2 - D[3] * x - D[4])
    eigvect = np.zeros((n,n))
    for i in range(n):
        q = np.zeros(n)
        q[0] = 1
        for j in range(1,n):
            q[j] = eigvals[i]*q[j-1] - D[j-1]
        for j in range(n):
            eigvect[:, i] += M[:, j]*q[j]
        eigvect[:, i] /= lenth(eigvect[:, i])
    print(eigvect)

print(np.linalg.eig(np.dot(np.copy(A).transpose(), np.copy(A))))
krilov(np.dot(np.copy(A).transpose(), np.copy(A)))