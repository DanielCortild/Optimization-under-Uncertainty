import cvxpy as cp
import itertools
import time

# Get all problem parameters
from ..Parameters import *

# Construct the objective function and constraints
# The optimization problem at hand is min E_xi[g(xi, x)]
def getEVS():
    start = time.time()
    # First and second stage variables
    x = cp.Variable(n)
    y = cp.Variable(n * k)

    # Objective function
    objective = cp.Minimize(c @ x + q_T.flatten() @ y)

    # Constraints
    constraints = [
        # First-stage constraints
        x >= 0,
        A @ x <= b,

        # Second-stage constraints
        y >= 0,
        cp.vstack(W_apply(y)) <= cp.vstack(h(xi_exp) - H(xi_exp) @ x),
    ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value, time.time() - start