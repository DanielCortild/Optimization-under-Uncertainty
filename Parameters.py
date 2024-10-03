# File contains all parameters of the problem

import numpy as np

# Problem data
n = 4
k = 3

# Matrix to model constraints
W = np.zeros((n + k, n * k))
for i in range(n):
    for j in range(k * i, k * (i + 1)):
        W[i, j] = 1
for i in range(n, n+k):
    for j in range(i-n, n * k, k):
        W[i, j] = -1

# Problem Parameters, independent of the RV
p_F = np.array([40, 24, 4, 45, 27, 4.5, 32, 19.2, 3.2, 55, 33, 5.5])
A = np.array([[-1, -1, -1, -1], [10, 7, 16, 6]])
c = np.array([10, 7, 16, 6])

# Random matrices and vectors, depending on a RV xi
H = lambda xi: np.concatenate([-np.eye(n), np.zeros((k, n))])
h = lambda xi: np.array([0, 0, 0, 0, -xi[0], -xi[1], -xi[2]])

# Vector, technically dependent on the distribution of the RV, but considered fixed
b = np.array([-14, 120])

# Random Variable xi and its distribution
xis = np.array([[3, 5, 7], [2, 3, 4], [1, 2, 3]])
probs = np.array([0.3, 0.4, 0.3])

# Expected value of xi
xi_exp = np.array([probs.T @ xi for xi in xis])
