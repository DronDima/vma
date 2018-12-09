import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00000001
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)

def t(A):
    tmp = np.copy(A)
    tmp = tmp**2
    return tmp.sum() - np.diag(tmp).sum()

def vrashenia(A):
    count = 0
    V = np.eye(n)
    while(t(A) > eps):
        count += 1
        ma = np.abs(np.triu(A, +1))
        maxElem = ma.max()
        i = ma.argmax() // n
        j = ma.argmax() % n
        angle = np.arctan(2*maxElem/(A[i,i] - A[j,j]))/2
        H = np.eye(n)
        H[i, i] = np.cos(angle)
        H[j, j] = np.cos(angle)
        H[i, j] = -np.sin(angle)
        H[j, i] = np.sin(angle)
        V = np.dot(V, H)
        A = np.dot(np.dot(H.transpose(), A), H)
    print("Собственные значения: ", np.diag(A))
    print("Собственные векторы: ", V)
    print("Количество итераций: ", count)

print(np.linalg.eig(np.dot(np.copy(A).transpose(), np.copy(A))))
vrashenia(np.dot(np.copy(A).transpose(), np.copy(A)))
