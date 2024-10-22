import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs
import time

# Get all problem parameters
from ..Parameters import *

def getEVS(fix_xi=False):
    start = time.time()

    # First and second stage variables
    x = cp.Variable(n)
    if fix_xi:
        y = cp.Variable((n+1) * k)
    else:
        y = cp.Variable((27, (n+1) * k))

    # Objective function
    if fix_xi:
        objective = cp.Minimize(c @ x + q_T_import.flatten() @ y)
    else:
        # Objective function, with explicit evaluation of the finite expectation
        objective = cp.Minimize(c @ x +
            q_T_import.flatten() @ cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r] * y[9*i + 3*j + r]
                                    for i, j, r in itertools.product(range(3), range(3), range(3))]))

    # First stage constraints
    constraints = [
        # First-stage constraints
        x >= 0,
        A_import @ x <= b_import,
    ]

    # Second-stage constraints
    if fix_xi:
        constraints += [
            y >= 0,
            cp.vstack(W_import_apply(y)) <= cp.vstack(h(xi_exp) - H(xi_exp) @ x),
        ]
    else:
        # Second stage constraints
        for i, j, r in itertools.product(range(3), range(3), range(3)):
            # Retrieve samples
            xi = [xis[0, i], xis[1, j], xis[2, r]]

            # Add constraints
            constraints += [
                y[9*i + 3*j + r] >= 0,
                cp.vstack(W_import_apply(y[9*i + 3*j + r])) <= cp.vstack(h(xi, alpha_exp) - H(xi, alpha_exp) @ x)
            ]
            

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value, time.time() - start