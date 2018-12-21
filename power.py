import numpy as np
import pandas as pd
import random
from math import sqrt

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)

def lenth(x):
    sum = 0
    for el in x:
        sum += el**2
    return sqrt(sum)

def power(A):
    x = np.ones(n)
    X = np.dot(A,x)
    eigvalNext = x[0] / X[0]
    eigvalPrev = eigvalNext + 1
    count = 0
    while(np.abs(eigvalNext - eigvalPrev) > eps):
        count += 1
        randNum = random.randrange(0, n)
        eigvalPrev = eigvalNext
        x = X
        X = np.dot(A,x)
        eigvalNext = X[randNum]/x[randNum]
    X = X/lenth(X)
    print("Количество итераций: ", count)
    return eigvalNext, X

#print(np.linalg.eig(np.dot(np.copy(A).transpose(), np.copy(A))))
A = np.dot(np.copy(A).transpose(), np.copy(A))
eigval, eigvect = power(A)
print("Собственное значения: ", eigval)
print("Собственный вектор: ", eigvect)
disc = np.dot(A, eigvect) - np.dot(eigval, eigvect)
print("Невязка: ", disc)