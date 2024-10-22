import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs
import time
import multiprocess as mp

# Get all problem parameters
from ..Parameters import *

def g(xi, alpha):
    start = time.time()

    # First and second stage variables
    x = cp.Variable(n)
    y = cp.Variable((n+1) * k)

    # Objective function
    objective = cp.Minimize(c @ x + q_T_import.flatten() @ y)

    # First stage constraints
    constraints = [
        # First-stage constraints
        x >= 0,
        A_import @ x <= b_import,
    ]

    # Second-stage constraints
    constraints += [
        y >= 0,
        cp.vstack(W_import_apply(y)) <= cp.vstack(h(xi, alpha) - H(xi, alpha) @ x),
    ]
            

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value

def getWS(samples):
    start = time.time()

    # Sample alphas using Latin Hypercube Sampling
    lhs_alphas = lhs(n, samples)

    def task(nbs):
        s, a = nbs
        xi, xi_prob = xis_flat[s]
        alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(n)])
        return g(xi, alpha) * xi_prob / samples

    # Run the tasks in parallel
    start = time.time()
    with mp.Pool() as pool:
        result = list(tqdm(pool.imap(task, itertools.product(range(27), range(samples))), leave=False, total=27*samples))

    return np.mean(result), time.time() - start