import cvxpy as cp
import itertools

# Get all problem parameters
from Parameters import *

# Construct the objective function and constraints
# The optimization problem at hand is min E_xi[g(xi, x)]
def g(xi, x=None):
    # If x is not provided, it is a variable
    if x is None:
        x = cp.Variable(n)
    # Otherwise, it is considered fixed and is a parameter
    else:
        x = cp.Parameter(n, value=x)
    z = cp.Variable(n*k)
    objective = cp.Minimize(c @ x + p_F @ z)
    constraints = [x >= 0, z >= 0, W @ z <= h(xi) - H(xi) @ x, A @ x <= b]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value, z.value, prob.value

# Compute EV
x_bar, z_bar, EV = g(xi_exp)
# print(f"EV Solution X: {x_bar}")
# print(f"EV Solution Z: {z_bar}")
print(f"EV Value: {EV}")

# Compute EEV
EEV = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    _, _, v = g([xis[0, i], xis[1, j], xis[2, r]], x_bar)
    EEV += probs[i]*probs[j]*probs[r] * v
print(f"EEV Value: {EEV}")

# Solve Recourse Model
x = cp.Variable(n)
z = cp.Variable((27, n*k))
objective = cp.Minimize(c @ x + cp.sum([probs[i]*probs[j]*probs[r] * p_F @ z[9*i + 3*j + r] for i, j, r in itertools.product(range(3), range(3), range(3))]))
constraints = [x >= 0, A @ x <= b]
for i, j, r in itertools.product(range(3), range(3), range(3)):
    constraints += [
        z[9*i + 3*j + r] >= 0,
        W @ z[9*i + 3*j + r] <= h([xis[0, i], xis[1, j], xis[2, r]]) - H([xis[0, i], xis[1, j], xis[2, r]]) @ x
    ]
prob = cp.Problem(objective, constraints)
prob.solve()
TS = prob.value
# print(f"Recourse Model Solution X: {x.value}")
print(f"Recourse Value: {TS}")

# Compute the Value of the Stochastic Solution
VSS = EEV - TS
print(f"VSS Value: {VSS}")

# Compute Wait-And-See Solution
WS = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    _, _, v = g([xis[0, i], xis[1, j], xis[2, r]])
    WS += probs[i]*probs[j]*probs[r] * v
print(f"WS Value: {WS}")

# Compute the Expected Value of Perfect Information
EVPI = TS - WS
print(f"EVPI Value: {EVPI}")
