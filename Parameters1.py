# File contains all parameters of the problem

import numpy as np

## Problem data
# Number of technologies
n = 4
# Number of modes
k = 3

# Investement costs, to model first-stage objective
c = np.array([10, 7, 16, 6])

# Random Variable xi, its distribution, its expected value and maximum value
xis = np.array([[3, 5, 7], [2, 3, 4], [1, 2, 3]])
xis_probs = np.array([0.3, 0.4, 0.3])
xis_flat = [(
    [xis[0, i], xis[1, j], xis[2, l]],
    xis_probs[i] * xis_probs[j] * xis_probs[l]
) for i in range(3) for j in range(3) for l in range(3)]
xi_exp = np.array([xis_probs.T @ xi for xi in xis])
xi_max = np.array([max(xi) for xi in xis])

# Matrix q_T, to model the second-order objective
# Production costs
q = np.array([4.0, 4.5, 3.2, 5.5])
# Duration of modes
T = np.array([10, 6, 1])
q_T = np.outer(q, T)
print(q_T)

# Matrix A and vector b, to model the first-stage constraints
A = np.array([[-1, -1, -1, -1], c])
b = np.array([-sum(xi_max), 120])

# Matrices W, H and h, to model the second-stage constraints
H = lambda xi: np.concatenate([-np.eye(n), np.zeros((k, n))])
h = lambda xi: np.array([0, 0, 0, 0, -xi[0], -xi[1], -xi[2]])
W = np.zeros((n + k, n,  k))
for a in range(n+k):
    for i in range(n):
        for j in range(k):
            if i == a and 1 <= a+1 <= n:
                W[a, i, j] = 1
            if j == a-n and n+1 <= a+1 <= n+k:
                W[a, i, j] = -1
W_apply = lambda y: [W[i].flatten() @ y for i in range(n+k)]
W_T_apply = lambda lamb: sum([lamb[i] * W[i] for i in range(n+k)])
