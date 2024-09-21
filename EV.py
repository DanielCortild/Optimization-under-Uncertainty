import cvxpy as cp
import numpy as np

# Problem data
n = 4
k = 3
W = np.zeros((n + k, n * k))
for i in range(n):
    for j in range(k * i, k * (i + 1)):
        W[i, j] = 1
for i in range(n, n+k):
    for j in range(i-n, n * k, k):
        W[i, j] = -1
p_F = np.array([40, 24, 4, 25, 27, 4.5, 32, 19.2, 3.2, 55, 33, 5.5])
A = np.array([[-1, -1, -1, -1], [10, 7, 16, 6]])
c = np.array([10, 7, 16, 6])
H = np.concatenate([-np.eye(n), np.zeros((k, n))])

# EV data
h = np.array([0, 0, 0, 0, -5, -3, -2])
b = np.array([-14, 120])

# Construct the problem.
x = cp.Variable(n)
z = cp.Variable(n*k)
objective = cp.Minimize(c @ x + p_F @ z)
constraints = [x >= 0, z >= 0, W @ z <= h - H @ x, A @ x <= b]
prob = cp.Problem(objective, constraints)

# Solve the problem
result = prob.solve(verbose=True)
print(f"X: {x.value}")
print(f"Z: {z.value}")
