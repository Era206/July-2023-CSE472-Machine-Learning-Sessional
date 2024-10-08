import numpy as np

def rowSumAbsolute(matrix):
    return np.sum(np.abs(matrix), axis=1)

n=int(input("Enter the value of n:"))
A=np.random.randint(-100,100,(n,n))
rowSumMat=rowSumAbsolute(A)
for i in range(n):
    A[i][i]=rowSumMat[i]+5
A=A+A.T
print(A)
eigen_values, eigen_vectors = np.linalg.eig(A)
A1 = np.diag(eigen_values)
A2 = eigen_vectors.dot(A1).dot(eigen_vectors.T)
print(A2)
print(np.allclose(A,A2))

