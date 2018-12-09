import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
stack = []
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = n)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = n, nrows = 1)

def gaussMethod(copy, dCopy):
    result = {'x': np.zeros(n), 'det': 1, 'columns': np.eye(n)}
    invColumns = np.eye(n)
    for k in range(n):
        maxElem = abs(copy).max(axis = 1)
        maxElemIndex = abs(copy).argmax(axis = 1)
        if maxElemIndex[k] != k:
            result['det'] *= -1
        copy[:, k], copy[:, maxElemIndex[k]] = copy[:, maxElemIndex[k]], copy[:, k].copy()
        stack.append(np.array([k, maxElemIndex[k]]))
        for i in range(k, n):
            delitel = copy[i][k]
            if abs(delitel) < eps:
                continue
            result['det'] *= copy[i][k]
            copy[i:i+1, : ] /= delitel
            invColumns[i:i+1, : ] /= delitel
            dCopy[i] /= delitel
            if i == k:
                continue
            copy[i:i+1, : ] -= copy[k:k+1, :]
            invColumns[i:i+1, : ] -= invColumns[k:k+1, :]
            dCopy[i] -= dCopy[k]
    k = n - 1
    while k >= 0:
        result['x'][k] = dCopy[k]
        dCopy[:k] = dCopy[:k] - (copy[:k, k:k+1] * result['x'][k]).ravel()
        k -= 1
    while stack:
        swap = stack.pop()
        result['x'][swap[0]], result['x'][swap[1]] = result['x'][swap[1]], result['x'][swap[0]].copy()
    result['columns'] = invColumns
    return result

def discrepancy(matrix, x, b):
    return np.dot(matrix, x) - b

def discrepancyM(matrix, inverse):
    return np.dot(matrix, inverse) - np.eye(n)

def condition(matrix, inverse):
    return np.linalg.norm(matrix, 'fro') * np.linalg.norm(inverse, 'fro')

def inverse(triangle, columns):
    for i in range(n-1):
        for j in range(n-i-1):
            columns[j,:] -= columns[n-i-1,:] * triangle[j, n-i-1]
    return columns

copy = np.copy(A)
dCopy = np.copy(b).ravel()
print("Расширенная матрица:")
print(np.insert(np.copy(copy), n, dCopy, axis=1))
data = gaussMethod(copy, dCopy)
inverse = inverse(copy, data['columns'])
print("Расширенная верхне-треугольная матрица:")
print(np.insert(np.copy(copy), n, dCopy, axis=1))
print("Вектор решений: ", data['x'])
print("Невязка:")
print(discrepancy(np.copy(A), data['x'], b))
print("Определитель: ", data['det'])
print("Обратная матрица: ")
print(inverse)
print(np.linalg.norm(discrepancyM(np.copy(A), inverse), 1))
print(np.linalg.norm(discrepancy(np.copy(A), data['x'], b), 1))
print("Невязка обратной матрицы: ")
print(discrepancyM(np.copy(A), inverse))
print("Число обусловленности: ")
print(condition(np.copy(A), inverse))
