import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs
import time

# Get all problem parameters
from ..Parameters import *

# We do not fix xi or T in this case
def getEVS(fix_xi_T=False):
    start = time.time()

    # First and second stage variables
    x = cp.Variable(n)
    if fix_xi_T:
        y = cp.Variable((n+1) * k)
    else:    
        y = cp.Variable((27*4, (n+1) * k))

    # Objective function
    if fix_xi_T:
        objective = cp.Minimize(c @ x + q_Ts_exp.flatten() @ y)
    else:
        objective = cp.Minimize(c @ x +
            cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r]*q_Ts_probs[l] * 
                    q_Ts[l].flatten() @ y[27*l + 9*i + 3*j + r]
                                for i, j, r, l in itertools.product(range(3), range(3), range(3), range(4))]))

    # First stage constraints
    constraints = [
        x >= 0,
        A_import @ x <= b_import,
    ]

    # Second-stage constraints
    if fix_xi_T:
        constraints += [
            y >= 0,
            cp.vstack(W_import_apply(y)) <= cp.vstack(h(xi_exp, alpha_exp) - H(xi_exp, alpha_exp) @ x),
        ]
    else:
        for i, j, r, l in itertools.product(range(3), range(3), range(3), range(4)):
            # Retrieve samples
            xi = [xis[0, i], xis[1, j], xis[2, r]]

            # Add constraints
            constraints += [
                y[27*l + 9*i + 3*j + r] >= 0,
                cp.vstack(W_import_apply(y[27*l + 9*i + 3*j + r])) <= cp.vstack(h(xi, alpha_exp) - H(xi, alpha_exp) @ x)
            ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value, time.time() - start