import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs
import multiprocess as mp
import time

# Get all problem parameters
from ..Parameters import *

def getTS_Naive(samples):
    start = time.time()

    # First and second stage variables
    x = cp.Variable(n)
    y = cp.Variable((27*samples, (n+1)*k))

    # Objective function, with explicit evaluation of the finite expectation
    objective = cp.Minimize(c @ x +
        cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r]/samples * q_T_import.flatten() @ y[27*a + 9*i + 3*j + r]
            for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples))]))

    # First stage constraints
    constraints = [x >= 0, A_import @ x <= b_import]

    # Sample alphas using Latin Hypercube Sampling
    lhs_alphas = lhs(n, samples)

    # Second stage constraints
    for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples)):
        # Retrieve and format samples
        xi = [xis[0, i], xis[1, j], xis[2, r]]
        alpha = np.array([lhs_alphas[a][k] * (alphas[k][1] - alphas[k][0]) + alphas[k][0] for k in range(n)])

        # Add constraints
        constraints += [
            y[27*a + 9*i + 3*j + r] >= 0,
            cp.vstack(W_import_apply(y[27*a + 9*i + 3*j + r])) <= cp.vstack(h(xi, alpha) - H(xi, alpha) @ x)
        ]

    # Formulate and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value, time.time() - start

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

        def task(s, a):
            # Get random sample
            xi, xi_prob = xis_flat[s]
            alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(n)])
            alpha_prob = 1 / samples
            prob = xi_prob * alpha_prob

            # Setup of Subproblem
            lamb = cp.Variable(n+k)
            hs = h(xi, alpha)
            Hs = H(xi, alpha)
            subobjective = cp.Maximize(lamb.T @ (hs - Hs @ x_i))
            subconstraints = [
                cp.vstack(W_import_T_apply(lamb.T).T.flatten()) <= cp.vstack(q_T_import.flatten()),
                lamb <= 0
            ]
            subprob = cp.Problem(subobjective, subconstraints)
            subprob.solve()

            return prob * lamb.T.value @ (hs - Hs @ x_i), -prob * lamb.T.value @ Hs

        # Solve Subproblems in parallel
        with mp.Pool() as pool:
            results = pool.starmap(task, itertools.product(range(27), range(samples)))

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

    return x.value, masterproblem.value, obj_vals, time.time() - start