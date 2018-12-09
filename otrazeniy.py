import numpy as np
import pandas as pd
n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = n, nrows = 1)

def hauseholder(A, b):
    result = {'x': np.zeros(n), 'det': 1, 'A': np.eye(n)}
    L = np.eye(n)
    for i in range(n-1):
        s = np.zeros(i)
        s = np.insert(s, i, A[i:,i])
        l = L[:,i]
        alpha = np.sqrt(np.dot(s, s))
        p = np.sqrt(2*np.square(alpha) + 2*np.abs(alpha)*np.dot(s,l))
        w = (s + alpha*l)/p
        U = np.eye(n) - 2*np.dot(w.reshape(n,1), w.reshape(1,n))
        b = np.dot(U, b)
        A = np.dot(U, A)
    print(A)
    result['A'] = np.copy(A)
    det = 1
    for i in range(n):
        det *= A[i, i]
        result['det'] *= A[i, i]
        b[i] /= A[i, i]
        A[i,:] /= A[i, i]
    for i in range(n-1, -1, -1):
        result['x'][i] = b[i]
        b[:i] -= (A[:i, i:i+1] * result['x'][i]).ravel()
    return result

print("Расширенная матрица после прямого хода:")
data = hauseholder(np.copy(A), np.copy(b).ravel())
print("Вектор решений: ", data['x'])
print("Невязка:", np.dot(A, data['x']) - b)
print("Определитель: ", data['det'])
