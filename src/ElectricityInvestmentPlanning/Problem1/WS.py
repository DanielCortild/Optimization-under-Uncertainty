import cvxpy as cp
import itertools
import time

# Get all problem parameters
from ..Parameters import *

def g(xi):
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
        cp.vstack(W_apply(y)) <= cp.vstack(h(xi) - H(xi) @ x),
    ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value

def getWS():
    start = time.time()
    WS = sum([xis_probs[i]*xis_probs[j]*xis_probs[r] * g([xis[0, i], xis[1, j], xis[2, r]])
                for i, j, r in itertools.product(range(3), range(3), range(3))])
    return WS, time.time() - start
