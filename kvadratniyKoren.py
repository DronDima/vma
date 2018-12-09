import numpy as np
import pandas as pd
n = 5
np.set_printoptions(precision = 4)
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = n, nrows = 1)

def CholeskyMethod(A, b):
    result = {'x': np.zeros(n), 'det': 1}
    y = np.zeros(n)
    G = np.zeros((n,n))
    G[0,0] = np.sqrt(A[0,0])
    G[0, :] = A [0, :]/G[0,0]
    for i in range(1, n):
        G[i, i] = np.sqrt(A[i, i] - np.sum(np.power(G[:i, i], 2)))
        for j in range(2, n):
            if i < j:
                G[i, j] = (A[i, j] - np.sum(G[:j, i] * G[:j, j]))/G[i, i]
    result['det'] = np.power(np.prod(G.diagonal()), 2)
    G = G.transpose()
    y[0] = b[0]/G[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.sum(G[i, :i] * y[:i]))/G[i,i]
    G = G.transpose()
    result['x'][n-1] = y[n-1]/G[n-1,n-1]
    for i in range(n-2, -1, -1):
        result['x'][i] = (y[i] - np.sum(G[i, i+1:] * result['x'][i+1:]))/G[i,i]
    return result

b = np.copy(b)
b = np.dot(A.transpose(), b.reshape(5,1))
A = np.dot(A.transpose(), A)
print("Расширенная матрица:")
print(np.insert(A, n, b.ravel(), axis=1))
L = np.linalg.cholesky(A)
print(L)
data = CholeskyMethod(A, b)
print("Вектор решений: ", data['x'])
print("Невязка: ", np.dot(A, data['x']) - b.ravel())
print("Определитель: ", data['det'])