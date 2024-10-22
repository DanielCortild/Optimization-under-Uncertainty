import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs
import multiprocess as mp
import time

# Get all problem parameters
from ..Parameters import *

def getSingleEEV(x_bar, alpha):
    # Second stage variables
    y = cp.Variable((27*4, (n+1) * k))

    # Objective function, with explicit evaluation of the finite expectation
    objective = cp.Minimize(c @ x_bar +
        cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r]*q_Ts_probs[l] * 
                q_Ts_exp.flatten() @ y[27*l + 9*i + 3*j + r]
                            for i, j, r, l in itertools.product(range(3), range(3), range(3), range(4))]))

    # Second stage constraints
    constraints = []
    for i, j, r, l in itertools.product(range(3), range(3), range(3), range(4)):
        # Retrieve samples
        xi = [xis[0, i], xis[1, j], xis[2, r]]

        # Add constraints
        constraints += [
            y[27*l + 9*i + 3*j + r] >= 0,
            cp.vstack(W_import_apply(y[27*l + 9*i + 3*j + r])) <= cp.vstack(h(xi, alpha) - H(xi, alpha) @ x_bar)
        ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value

def getEEV(x_bar, samples):
    # Notice the problem is separable when fixing x, so we can run it separately for each alpha
    EEVs = []

    # Sample alphas using Latin Hypercube Sampling
    lhs_alphas = lhs(n, samples)

    def task(a):
        alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(n)])
        return getSingleEEV(x_bar, alpha)

    # Run the tasks in parallel
    start = time.time()
    with mp.Pool() as pool:
        result = pool.map(task, range(samples))

    return np.mean(result), time.time() - start