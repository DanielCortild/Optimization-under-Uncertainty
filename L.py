import cvxpy as cp
import itertools

# Get all problem parameters
from Parameters1 import *

# Construct the objective function and constraints
# The optimization problem at hand is min E_xi[g(xi, x)]
def g():
    S = 27
    samples = 10

    x = cp.Variable(n)
    s = cp.Variable(len(b))
    objective = cp.Minimize(c @ x)
    constraints = [x >= 0, s >= 0, A @ x + s == b]
    iniprob = cp.Problem(objective, constraints)
    iniprob.solve()

    theta = cp.Variable(1)
    objective = cp.Minimize(c @ x + theta)

    for _ in range(10):
        x_i = x.value
        Qxi = 0
        ui = np.zeros(n)
        unbounded_probs = 0
        lamb = cp.Variable(n+k)
        y = cp.Variable(n*k)

        for s in range(S):
            xi, xi_prob = xis_flat[s]
            hs = h(xi)
            Hs = H(xi)
            # Primal
            # subobjective = cp.Minimize(q_T.flatten() @ y)
            # subconstraints = [
            #     y >= 0, cp.vstack(W_apply(y)) <= cp.vstack(hs - Hs @ x_i)
            # ]
            # Dual
            subobjective = cp.Maximize(lamb.T @ (hs - Hs @ x_i))
            subconstraints = [
                cp.vstack(W_T_apply(lamb.T).T.flatten()) <= cp.vstack(q_T.flatten()),
                lamb <= 0
            ]
            subprob = cp.Problem(subobjective, subconstraints)
            subprob.solve(solver='MOSEK')

            if subprob.status == cp.UNBOUNDED:
                unbounded_probs += xi_prob
            else:
                Qxi += xi_prob * lamb.T.value @ (hs - Hs @ x_i)
                ui -= xi_prob * lamb.T.value @ Hs

        alpha_i = (Qxi - np.dot(ui, x_i)) / (1 - unbounded_probs)
        beta_i = ui / (1 - unbounded_probs)

        constraints += [theta >= alpha_i + beta_i.reshape((1, n)) @ x]

        masterproblem = cp.Problem(objective, constraints)
        masterproblem.solve(solver='MOSEK')

        print(masterproblem.value)


g()

exit()

# Compute EV
x_bar, z_bar, EV = g(xi_exp)
print(f"EV Solution X: {x_bar}")
# print(f"EV Solution Z: {z_bar}")
print(f"EV Value: {EV}")
# Compute EEV
EEV = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    _, _, v = g([xis[0, i], xis[1, j], xis[2, r]], x_bar)
    EEV += xis_probs[i]*xis_probs[j]*xis_probs[r] * v
print(f"EEV Value: {EEV}")

# Solve Recourse Model
x = cp.Variable(n)
y = cp.Variable((27, n*k))
objective = cp.Minimize(c @ x + cp.sum([xis_probs[i]*xis_probs[j]*xis_probs[r] * q_T.flatten() @ z[9*i + 3*j + r] for i, j, r in itertools.product(range(3), range(3), range(3))]))
constraints = [x >= 0, A @ x <= b]
for i, j, r in itertools.product(range(3), range(3), range(3)):
    constraints += [
        y[9*i + 3*j + r] >= 0,
        cp.vstack(W_apply(y[9*i + 3*j + r])) <= cp.vstack(h([xis[0, i], xis[1, j], xis[2, r]]) - H([xis[0, i], xis[1, j], xis[2, r]]) @ x),
    ]
prob = cp.Problem(objective, constraints)
prob.solve()
TS = prob.value
# print(f"Recourse Model Solution X: {x.value}")
print(f"Recourse Value: {TS}")

for i, j, r in itertools.product(range(3), range(3), range(3)):
    print(np.round(z[9*i+3*j+r].value, 2).reshape((n, k)))
    print()

# Compute the Value of the Stochastic Solution
VSS = EEV - TS
print(f"VSS Value: {VSS}")

# Compute Wait-And-See Solution
WS = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    _, _, v = g([xis[0, i], xis[1, j], xis[2, r]])
    WS += xis_probs[i]*xis_probs[j]*xis_probs[r] * v
print(f"WS Value: {WS}")

# Compute the Expected Value of Perfect Information
EVPI = TS - WS
print(f"EVPI Value: {EVPI}")
