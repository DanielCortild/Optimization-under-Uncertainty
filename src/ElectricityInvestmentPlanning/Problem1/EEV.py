import cvxpy as cp
import itertools
import time

# Get all problem parameters
from ..Parameters import *

# Construct the objective function and constraints
# The optimization problem at hand is min E_xi[g(xi, x)]
def g(xi, x):
    # First and second stage variables, first stage being fixed
    x = cp.Parameter(n, value=x)
    y = cp.Variable(n * k)

    # Objective function
    objective = cp.Minimize(c @ x + q_T.flatten() @ y)

    # Constraints
    constraints = [
        # First-stage constraints, technically redundant
        x >= 0,
        A @ x <= b,
        # Second-stage constraints
        y >= 0,
        cp.vstack(W_apply(y)) <= cp.vstack(h(xi) - H(xi) @ x),
    ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value

def getEEV(x_bar):
    start = time.time()
    EEV = sum([xis_probs[i]*xis_probs[j]*xis_probs[r] * g([xis[0, i], xis[1, j], xis[2, r]], x_bar) 
            for i, j, r in itertools.product(range(3), range(3), range(3))])  
    return EEV, time.time() - start
