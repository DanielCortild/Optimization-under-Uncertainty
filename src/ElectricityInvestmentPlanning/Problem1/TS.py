import cvxpy as cp
import itertools
import time

# Get all problem parameters
from ..Parameters import *

def getTS():
    start = time.time()
    # First and second stage variables
    x = cp.Variable(n)
    y = cp.Variable((27, n*k)) # 27 scenarios

    # Objective function
    objective = cp.Minimize(c @ x + \
                 cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r] * q_T.flatten() @ y[9*i + 3*j + r] 
                        for i, j, r in itertools.product(range(3), range(3), range(3))]))

    # First stage constraints
    constraints = [x >= 0, A @ x <= b]

    # Second stage constraints
    for i, j, r in itertools.product(range(3), range(3), range(3)):
        constraints += [
            y[9*i + 3*j + r] >= 0,
            cp.vstack(W_apply(y[9*i + 3*j + r])) <= cp.vstack(h([xis[0, i], xis[1, j], xis[2, r]]) - H([xis[0, i], xis[1, j], xis[2, r]]) @ x),
        ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value, time.time() - start