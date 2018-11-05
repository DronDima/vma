import numpy as np
import pandas as pd

n = 5
np.set_printoptions(precision = 4)
eps = 0.00001
stack = []
A = pd.read_csv('matrix.txt', sep=' ', header = None, nrows = 5)
b = pd.read_csv('matrix.txt', sep=' ', header = None, skiprows = 5, nrows = 1)

def jacobi(A, b):
    B = np.zeros((n,n))
    g = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        B[i,:] = -A[i,:]/A[i,i]
        B[i,i] = 0
        g[i] = b[i]/A[i,i]
    normB = np.linalg.norm(B, np.inf)
    X = np.dot(B,x) + g
    while(np.linalg.norm(X - x, np.inf) > (1*eps - normB*eps)/normB):
        x = X
        X = np.dot(B,x) + g
    print(x)



print("Расширенная матрица:")
print(np.insert(np.copy(A), n, b, axis=1))
jacobi(np.copy(A), np.copy(b).ravel())
