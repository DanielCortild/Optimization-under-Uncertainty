# File contains all parameters of the problem

import numpy as np

## Problem data
# Number of technologies
n = 4
# Number of modes
k = 3

# Investement costs, to model first-stage objective
c = np.array([10, 7, 16, 6])
cmax = 120

# Random Variable xi, its distribution, its expected value and maximum value
xis = np.array([[3, 5, 7], [2, 3, 4], [1, 2, 3]])
xis_probs = np.array([0.3, 0.4, 0.3])
xis_flat = [(
    [xis[0, i], xis[1, j], xis[2, l]],
    xis_probs[i] * xis_probs[j] * xis_probs[l]
) for i in range(3) for j in range(3) for l in range(3)]
xi_exp = np.array([xis_probs.T @ xi for xi in xis])
xi_max = np.array([max(xi) for xi in xis])

# Random variable alpha (uniform over [a, b])
alphas = np.array([[0.6, 0.9], [0.7, 0.8], [0.5, 0.8], [0.9, 1]])
alpha_exp = np.array([0.5 * (a + b) for a, b in alphas])

# Matrix q_T, to model the second-order objective
# Production costs
q = np.array([4.0, 4.5, 3.2, 5.5])
q_import = np.array([4.0, 4.5, 3.2, 5.5, 10])
# Duration of modes
T = np.array([10, 6, 1])
Ts = np.array([[5, 7.5], [0.5, 1.75]])
Ts_probs = np.array([0.6, 0.4])
Ts_exp = np.array([Ts_probs.T @ Ts for Ts in Ts])

# Matrix q_T, to model the second-order objective
q_T = np.outer(q, T)
q_T_import = np.outer(q_import, T)
q_Ts = np.array([np.outer(q_import, np.array([T[0], Ts[0, i], Ts[1, j]])) 
                for i in range(2) for j in range(2)])
q_Ts_probs = np.array([Ts_probs[i] * Ts_probs[j] for i in range(2) for j in range(2)])
q_Ts_exp = sum([q_Ts_probs[i] * q_Ts[i] for i in range(4)])

# Matrix A and vector b, to model the first-stage constraints
A = np.array([[-1, -1, -1, -1], c])
A_import = np.array([c])
b = np.array([-sum(xi_max), cmax])
b_import = np.array([cmax])

# Matrices W, H and h, to model the second-stage constraints
H = lambda xi, alpha=np.ones(n): np.concatenate([-np.diag(alpha), np.zeros((k, n))])
h = lambda xi, alpha=np.ones(n): np.array([0, 0, 0, 0, -xi[0], -xi[1], -xi[2]])
W = np.zeros((n + k, n,  k))
for a in range(n+k):
    for i in range(n):
        for j in range(k):
            if i == a and 1 <= a+1 <= n:
                W[a, i, j] = 1
            if j == a-n and n+1 <= a+1 <= n+k:
                W[a, i, j] = -1
W_import = np.zeros((n + k, n+1,  k))
for a in range(n+k):
    for i in range(n+1):
        for j in range(k):
            if i == a and 1 <= a+1 <= n:
                W_import[a, i, j] = 1
            if j == a-n and n+1 <= a+1 <= n+k:
                W_import[a, i, j] = -1
W_apply = lambda y: [W[i].flatten() @ y for i in range(n+k)]
W_import_apply = lambda y: [W_import[i].flatten() @ y for i in range(n+k)]
W_T_apply = lambda lamb: sum([lamb[i] * W[i] for i in range(n+k)])
W_import_T_apply = lambda lamb: sum([lamb[i] * W_import[i] for i in range(n+k)])
