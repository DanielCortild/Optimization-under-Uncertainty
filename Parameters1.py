# File contains all parameters of the problem

import numpy as np

## Problem data
# Number of technologies
n = 4
# Number of modes
k = 3
# Production costs
q = np.array([4.0, 4.5, 3.2, 5.5])
# Duration of modes
T = np.array([10, 6, 1])
# Investement costs
c = np.array([10, 7, 16, 6])

# Random Variable xi and its distribution
xis = np.array([[3, 5, 7], [2, 3, 4], [1, 2, 3]])
xis_probs = np.array([0.3, 0.4, 0.3])

# Expected value of xi
xi_exp = np.array([xis_probs.T @ xi for xi in xis])

# Vector, technically dependent on the distribution of the RV, but considered fixed
xi_max = np.array([max(xi) for xi in xis])
b = np.array([-sum(xi_max), 120])

# Matrices to model constraints and objectives
W = np.zeros((n + k, n * k))
for i in range(n):
    for j in range(k * i, k * (i + 1)):
        W[i, j] = 1
for i in range(n, n+k):
    for j in range(i-n, n * k, k):
        W[i, j] = -1
p_F = np.outer(q, T).flatten()
A = np.array([[-1, -1, -1, -1], c])

# Random matrices and vectors, depending on a RV xi
H = lambda xi: np.concatenate([-np.eye(n), np.zeros((k, n))])
h = lambda xi: np.array([0, 0, 0, 0, -xi[0], -xi[1], -xi[2]])
