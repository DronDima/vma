import numpy as np
import pandas as pd
n = 5
np.set_printoptions(precision = 4)
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = 5, nrows = 1)
def ThomasMethod(A, d):
    x = np.zeros(n)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    a = np.append(0, np.diag(A, -1))
    b = np.diag(A, 0)
    c = np.diag(A, 1)
    alpha[n-1] = -a[n-1]/b[n-1]
    beta[n-1] = d[n-1]/b[n-1]
    for i in range(n-2, 0, -1):
        alpha[i] = -a[i]/(b[i] + c[i]*alpha[i+1])
        beta[i] = (d[i] - c[i]*beta[i+1])/(b[i] + c[i]*alpha[i+1])
    x[0] = beta[0] = (d[0] - c[0]*beta[1])/(b[0] + c[0]*alpha[1])
    for i in range(1, n):
        x[i] = alpha[i]*x[i-1] + beta[i]
    return x
b = np.copy(b)
A = np.copy(A)
for i in range(n):
    for j in range(n):
        if i - j > 1 or i - j < -1:
            A[i,j] = 0
print("Расширенная матрица:")
print(np.insert(A, n, b.ravel(), axis=1))
x = ThomasMethod(A, b.reshape(5,1))
print("Вектор решений: ", x)
print("Невязка:", np.dot(A, x) - b)
print("Определитель: ", np.prod(np.diag(A)))
