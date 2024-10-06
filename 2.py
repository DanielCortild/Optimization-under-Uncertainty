import cvxpy as cp
import itertools

# Get all problem parameters
from Parameters2 import *

# First stage variable
x = cp.Variable(n)

# Second stage variables, one per possible realization of the random vector
samples = 50
z = cp.Variable((27*samples, n*k))

# Objective function, with explicit evaluation of the finite expectation
objective = cp.Minimize(c @ x +
    cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r]/samples * p_F @ z[27*a + 9*i + 3*j + r]
        for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples))]))

# First stage constraints
constraints = [x >= 0, A @ x <= b]

# Second stage constraints
for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples)):
    # Retrieve samples
    xi = [xis[0, i], xis[1, j], xis[2, r]]
    alpha = np.array([np.random.uniform(*alp) for alp in alphas])

    # Add constraints
    constraints += [
        z[27*a + 9*i + 3*j + r] >= 0,
        W @ z[27*a + 9*i + 3*j + r] <= h(xi, alpha) - H(xi, alpha) @ x
    ]

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Print results
print(f"Recourse Model Solution X: {x.value}")
print(f"Recourse Value: {prob.value}")
