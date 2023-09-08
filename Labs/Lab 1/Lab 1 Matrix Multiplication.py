import numpy as np
import numpy.linalg as la
import math
import time

def matrixmult(A, B):
    
    start = time.time()
    Matrix = []
    for i in range(len(A)):
        N = []
        for j in range(len(B[0])):
            a = 0
            for k in range(len(A[0])):
                a += A[i][k] * B[k][j]
            N.append(a)
        Matrix.append(N)
                
    end = time.time()
    t = end - start
    return Matrix, t

def numpyMM(A, B):
    start = time.time()
    Matrix = np.matmul(A, B)
    end = time.time()
    t = end - start
    
    return Matrix, t
     
A = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     
#driver() 
print('This is without Numpy:', matrixmult(A, B))
print('This is with Numpy:', numpyMM(A, B))