import cvxpy as cp
import itertools

# Get all problem parameters
from Parameters1 import *

# Construct the objective function and constraints
# The optimization problem at hand is min E_xi[g(xi, x)]
def g(xi, x=None):
    # If x is not provided, it is a variable
    if x is None:
        x = cp.Variable(n)
    # Otherwise, it is considered fixed and is a parameter
    else:
        x = cp.Parameter(n, value=x)
    z = cp.Variable(n * k)
    objective = cp.Minimize(c @ x + q_T.flatten() @ z)
    constraints = [
        x >= 0,
        z >= 0,
        cp.vstack(W_apply(z)) <= cp.vstack(h(xi) - H(xi) @ x),
        (A @ x)[1] <= b[1]
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, z.value, prob.value

# Compute EV
x_bar, z_bar, EV = g(xi_exp)
print(f"EV Solution X: {x_bar}")
# print(f"EV Solution Z: {z_bar}")
print(f"EV Value: {EV}")

# Compute EEV
EEV = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    _, _, v = g([xis[0, i], xis[1, j], xis[2, r]], x_bar)
    EEV += xis_probs[i]*xis_probs[j]*xis_probs[r] * v
print(f"EEV Value: {EEV}")

# Solve Recourse Model
x = cp.Variable(n)
z = cp.Variable((27, n*k))
objective = cp.Minimize(c @ x + cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r] * q_T.flatten() @ z[9*i + 3*j + r] for i, j, r in itertools.product(range(3), range(3), range(3))]))
constraints = [x >= 0, A @ x <= b]
for i, j, r in itertools.product(range(3), range(3), range(3)):
    constraints += [
        z[9*i + 3*j + r] >= 0,
        cp.vstack(W_apply(z[9*i + 3*j + r])) <= cp.vstack(h([xis[0, i], xis[1, j], xis[2, r]]) - H([xis[0, i], xis[1, j], xis[2, r]]) @ x),
    ]
prob = cp.Problem(objective, constraints)
prob.solve()
TS = prob.value
# print(f"Recourse Model Solution X: {x.value}")
print(f"Recourse Value: {TS}")

for i, j, r in itertools.product(range(3), range(3), range(3)):
    print(np.round(z[9*i+3*j+r].value, 2).reshape((n, k)))
    print()

# Compute the Value of the Stochastic Solution
VSS = EEV - TS
print(f"VSS Value: {VSS}")

# Compute Wait-And-See Solution
WS = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    _, _, v = g([xis[0, i], xis[1, j], xis[2, r]])
    WS += xis_probs[i]*xis_probs[j]*xis_probs[r] * v
print(f"WS Value: {WS}")

# Compute the Expected Value of Perfect Information
EVPI = TS - WS
print(f"EVPI Value: {EVPI}")
