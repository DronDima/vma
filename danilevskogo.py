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

def danilevskogo(A):
    eigvect = np.eye(n)
    matrix = np.eye(n)
    for i in range(n,1,-1):
        B = np.eye(n)
        B[i-2] = -A[i-1]/A[i-1,i-2]
        B[i-2,i-2] = 1/A[i-1,i-2]
        matrix = np.dot(matrix, B)
        A = np.dot(np.dot(np.linalg.inv(B), A), B)
    print("Форма Фробениса:")
    print(A)
    P = np.hstack((np.array([1]), -A[0]))
    x = sy.Symbol("x")
    eigvals = np.array(solve(x**5 + P[1]*x**4 + P[2]*x**3 + P[3]*x**2 + P[4]*x + P[5]))
    print("Y vectors: ")
    for i in range(n):
        y = np.array([eigvals[i]**4, eigvals[i]**3,
                      eigvals[i]**2, eigvals[i], 1])
        print(y.astype(np.double, 4))
        x = np.dot(matrix, y)
        x /= lenth(x)
        eigvect[:,i] = x
    return eigvals, eigvect


#print(np.linalg.eig(np.dot(np.copy(A).transpose(), np.copy(A))))
A = np.dot(np.copy(A).transpose(), np.copy(A))
print(np.linalg.eig(A))
eigvals, eigvect = danilevskogo(A)
print("Собственные значения: ", eigvals.astype(np.double, 4))
print("Собственные векторы:")
print(eigvect)
print("Невязка:")
for i in range(n):
    disc = np.dot(A, eigvect[:, i]) - np.dot(eigvals[i], eigvect[:, i])
    print(disc.astype(np.double, 4))