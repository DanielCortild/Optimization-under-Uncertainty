import cvxpy as cp
import itertools
from pyDOE import lhs
import multiprocess as mp
from tqdm import tqdm

# Get all problem parameters
from Parameters2 import *

# Construct the objective function and constraints
# The optimization problem at hand is min E_xi[g(xi, x)]
def g():
    S = 27
    samples = 1
    lhs_alphas = lhs(n, samples)

    x = cp.Variable(n)
    s = cp.Variable(len(b))
    objective = cp.Minimize(c @ x)
    constraints = [x >= 0, s >= 0, A @ x + s == b]
    masterproblem = cp.Problem(objective, constraints)
    masterproblem.solve()

    theta = cp.Variable(1)
    objective = cp.Minimize(c @ x + theta)

    objectives = []

    for _ in (pbar := tqdm(range(100))):
        pbar.set_description(f"Objective: {masterproblem.value}")
        x_i = x.value
        lamb = cp.Variable(n+k)
        y = cp.Variable((n+1) * k)

        def task(s, a):
            xi, xi_prob = xis_flat[s]
            alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(n)])
            alpha_prob = 1 / samples
            prob = xi_prob * alpha_prob
            hs = h(xi, alpha)
            Hs = H(xi, alpha)
            x_new = cp.Variable(n)
            subobjective = cp.Maximize(lamb.T @ (hs - Hs @ x_i))
            subconstraints = [
                cp.vstack(x_new[:-1]) == cp.vstack(x_i[:-1]),
                cp.vstack(W_T_apply(lamb.T).T.flatten()) <= cp.vstack(q_T.flatten()),
                lamb <= 0
            ]
            subprob = cp.Problem(subobjective, subconstraints)
            subprob.solve(solver='MOSEK')

            return prob * lamb.T.value @ (hs - Hs @ x_i), -prob * lamb.T.value @ Hs

        with mp.Pool() as pool:
            results = pool.starmap(task, itertools.product(range(S), range(samples)))

        Qxi = sum([results[0] for results in results])
        ui = sum([results[1] for results in results])

        alpha_i = (Qxi - np.dot(ui, x_i))
        beta_i = ui

        constraints += [theta >= alpha_i + beta_i.reshape((1, n)) @ x]

        masterproblem = cp.Problem(objective, constraints)
        masterproblem.solve(solver='MOSEK')

        objectives.append(masterproblem.value)

    print(objectives)

g()
