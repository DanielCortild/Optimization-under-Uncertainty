import numpy as np

# Number of technologies
n = 5
# Number of modes
k = 3
# Production costs
q = np.array([4.0, 4.5, 3.2, 5.5, 10])
# Duration of modes
T = np.array([10, 6, 1])
# Investement costs
c = np.array([10, 7, 16, 6, 0])

# Random Variable xi and its distribution
xis = np.array([[3, 5, 7], [2, 3, 4], [1, 2, 3]])
xis_probs = np.array([0.3, 0.4, 0.3])

# Expected value of xi
xi_exp = np.array([xis_probs.T @ xi for xi in xis])

# Vector, previously dependent on the RV but now considered fixed
b = np.array([120])

# Random variable alpha (uniform over [a, b])
alphas = np.array([[0.6, 0.9], [0.7, 0.8], [0.5, 0.8], [0.9, 1], [1, 1]])

# Matrices to model constraints and objectives
W = np.zeros((n + k, n * k))
for i in range(n):
    for j in range(k * i, k * (i + 1)):
        W[i, j] = 1
for i in range(n, n+k):
    for j in range(i-n, n * k, k):
        W[i, j] = -1
p_F = np.outer(q, T).flatten()
A = np.array([c])

# Random matrices and vectors, depending on a RV xi
H = lambda xi, alpha: np.concatenate([-np.diag(alpha), np.zeros((k, n))])
h = lambda xi, alpha: np.array([0, 0, 0, 0, 0, -xi[0], -xi[1], -xi[2]])
