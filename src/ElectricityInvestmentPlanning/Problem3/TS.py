import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs
import multiprocess as mp
import time

# Get all problem parameters
from ..Parameters import *

def getTS_LShaped(samples, iterations):
    start = time.time()

    # Initial Problem
    x = cp.Variable(n)
    objective = cp.Minimize(c @ x)
    constraints = [x >= 0, A_import @ x <= b_import]
    masterproblem = cp.Problem(objective, constraints)
    masterproblem.solve()

    # Setup of Master Problem
    theta = cp.Variable(1)
    objective = cp.Minimize(c @ x + theta)

    # Collect Objective Values for Plotting
    obj_vals = []

    # Sample values of alpha using LHS
    lhs_alphas = lhs(n, samples)

    for _ in (pbar := tqdm(range(iterations), leave=False)):
        pbar.set_description(f"Objective: {masterproblem.value}")

        # Store value of x
        x_i = x.value

        def task(s, a, l):
            # Get random sample
            xi, xi_prob = xis_flat[s]
            alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(n)])
            alpha_prob = 1 / samples
            q_T = q_Ts[l]
            q_T_prob = q_Ts_probs[l]
            prob = xi_prob * alpha_prob * q_T_prob

            # Setup of Subproblem
            lamb = cp.Variable(n+k)
            hs = h(xi, alpha)
            Hs = H(xi, alpha)
            subobjective = cp.Maximize(lamb.T @ (hs - Hs @ x_i))
            subconstraints = [
                cp.vstack(W_import_T_apply(lamb.T).T.flatten()) <= cp.vstack(q_T.flatten()),
                lamb <= 0
            ]
            subprob = cp.Problem(subobjective, subconstraints)
            subprob.solve()

            return prob * lamb.T.value @ (hs - Hs @ x_i), -prob * lamb.T.value @ Hs

        # Solve Subproblems in parallel
        with mp.Pool() as pool:
            results = pool.starmap(task, itertools.product(range(27), range(samples), range(4)))

        # Update Master Problem
        Qxi = sum([results[0] for results in results])
        ui = sum([results[1] for results in results])
        alpha_i = (Qxi - np.dot(ui, x_i))
        beta_i = ui
        constraints += [theta >= alpha_i + beta_i.reshape((1, n)) @ x]

        # Solve Master Problem
        masterproblem = cp.Problem(objective, constraints)
        masterproblem.solve()

        # Store Objective Value
        obj_vals.append(masterproblem.value)
        print(masterproblem.value)

    return x.value, masterproblem.value, obj_vals, time.time() - start