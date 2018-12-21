import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00000001
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)

def t(A):
    tmp = np.copy(A)
    tmp = tmp**2
    return tmp.sum() - np.diag(tmp).sum()

def vrashenia(A):
    ACopy = np.copy(A)
    count = 0
    U = np.eye(n)
    while(t(A) > eps):
        count += 1
        ma = np.abs(np.triu(A, +1))
        maxElem = ma.max()
        i = ma.argmax() // n
        j = ma.argmax() % n
        angle = np.arctan(2*maxElem/(A[i,i] - A[j,j]))/2
        T = np.eye(n)
        T[i, i] = np.cos(angle)
        T[j, j] = np.cos(angle)
        T[i, j] = -np.sin(angle)
        T[j, i] = np.sin(angle)
        U = np.dot(U, T)
        A = np.dot(np.dot(T.transpose(), A), T)
    print(A)
    print("Собственные значения: ", np.diag(A))
    print("Собственные векторы: ", U)
    print("Количество итераций: ", count)
    print("Невязка: ")
    for i in range(n):
        disc = np.dot(ACopy, U[:, i]) - np.dot(np.diag(A)[i], U[:, i])
        print(disc.astype(np.double, 4))

A = np.dot(np.copy(A).transpose(), np.copy(A))
vrashenia(A)
