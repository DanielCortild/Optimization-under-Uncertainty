import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyDOE import lhs

# Get all problem parameters
from Parameters2 import *

def solveEVProblem():
    # First stage variable
    x = cp.Variable(n)

    # Second stage variables, one per possible realization of the random vector
    y = cp.Variable(n*k)

    # Objective function, with explicit evaluation of the finite expectation
    objective = cp.Minimize(c @ x + q_T.flatten() @ y)

    # First stage constraints
    constraints = [
        x >= 0,
        y >= 0,
        A @ x <= b,
        cp.vstack(W_apply(y)) <= cp.vstack(h(xi_exp, alpha_exp) - H(xi_exp, alpha_exp) @ x)
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="MOSEK")

    return x.value, y.value, prob.value

# x_star, yse, vra = solveEVProblem()

def computeEEV(x_star):
    # Set up number of samples, depending on whether the random parameter alpha is included or not
    samples = 100
    print("Samples:", samples)

    # Setup the problem variables
    x = cp.Variable(n)
    y = cp.Variable((samples * 27, (n+1) * k))

    # Objective function, with explicit evaluation of the finite expectation
    objective = cp.Minimize(c @ x +
        q_T.flatten() @ cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r]/samples * y[27*a + 9*i + 3*j + r]
            for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples))]))

    constraints = [x >= 0, A @ x <= b]

    # Sample alphas using Latin Hypercube Sampling
    lhs_alphas = lhs(n, samples)

    # Second stage constraints
    for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples)):
        # Retrieve samples
        xi = [xis[0, i], xis[1, j], xis[2, r]]
        alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(n)])

        # Add constraints
        constraints += [
            y[27*a + 9*i + 3*j + r] >= 0,
            cp.vstack(W_apply(y[27*a + 9*i + 3*j + r])) <= cp.vstack(h(xi, alpha) - H(xi, alpha) @ x)
        ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return prob.value

for i in range(10):
    x_star = np.array([2.833, 3, 2.167, 6])
    print("X:", x_star)
    print(computeEEV(x_star))

exit()

def solveProblem():
    # First stage variable
    x = cp.Variable(n)

    # Second stage variables, one per possible realization of the random vector
    samples = 40
    y = cp.Variable((27*samples, n*k))

    # Objective function, with explicit evaluation of the finite expectation
    objective = cp.Minimize(c @ x +
        cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r]/samples * q_T.flatten() @ y[27*a + 9*i + 3*j + r]
            for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples))]))

    # First stage constraints
    constraints = [x >= 0, A @ x <= b]

    # Sample alphas using Latin Hypercube Sampling
    lhs_alphas = lhs(n, samples)

    # Second stage constraints
    for i, j, r, a in itertools.product(range(3), range(3), range(3), range(samples)):
        # Retrieve samples
        xi = [xis[0, i], xis[1, j], xis[2, r]]
        alpha = np.array([lhs_alphas[a][i] * (alphas[i][1] - alphas[i][0]) + alphas[i][0] for i in range(5)])

        # Add constraints
        constraints += [
            y[27*a + 9*i + 3*j + r] >= 0,
            W_apply(y[27*a + 9*i + 3*j + r]) <= h(xi, alpha) - H(xi, alpha) @ x
        ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="MOSEK")

    return x.value, prob.value

# Print results for one run
x, prob = solveProblem()
print(f"Recourse Model Solution X: {x}")
print(f"Recourse Value: {prob}")

# # Plot histogram of x values over 100 runs
# x_values = [solveProblem()[0] for i in tqdm(range(100))]
# xs = lambda i: [x[i] for x in x_values]
# fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True)
# for i, (j, k) in enumerate([[0,0], [0, 1], [0, 2], [1,0], [1,1]]):
#     data = xs(i)
#     axs[j, k].hist(data, bins=np.arange(0, 11, 1), edgecolor='black')
#     axs[j, k].set_title(f'Technology {i+1}')
#     if j == 1:
#         axs[j, k].set_xlabel('Value')
#     if k == 0:
#         axs[j, k].set_ylabel('Count')
# plt.suptitle("Capacities per Technologies")
# plt.tight_layout()
# plt.show()
