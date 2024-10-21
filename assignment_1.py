# %%
import numpy as np
import cvxpy as cp
import numpy as np
import itertools

np.set_printoptions(suppress=True)

n=4
k=3

# %%


# %%
xis=np.array([[3,2,1],[5,3,2],[7,4,3]])
print(xis)
probs=np.array([0.3,0.4,0.3])

expected_xis=np.array([probs@i for i in xis.T])
print(expected_xis)
c_max=120

##Investment costs
c_x=np.array([10,7,16,6])
print(c_x)


# %% [markdown]
# ## EV

# %%
#1st stage
A_1=np.repeat(-1,n)
b_1 = np.array([-sum(expected_xis)])
print(f""" First Restriction: {A_1}""")
print(f""" First Restriction b: {b_1}""")
A_2=c_x
b_2 = np.array([c_max]) 
print(f""" Second Restriction: {A_2}""")
print(f""" Second Restriction b: {b_2}""")

A=np.vstack([A_1,A_2])
print(f"""A: { A} """)
b=np.hstack([b_1,b_2])
print(f"""b: {b}""")

##2nd Stage

A_3=np.repeat(0,k*n)
A_3 = [
    np.array([
        -1 if (j == i) or ((j - n) % n == i) else  # Set `1` at `i-th` x and corresponding `y`
        0  # Otherwise, keep `0`
        for j in range(len(A_3))
    ]) 
    for i in range(n)
]


A_3=np.vstack(A_3)
b_3=np.repeat(0,n)

print(f""" Third Restriction: {A_3}""")
print(f""" Third Restriction b: {b_3}""")

A_4=np.repeat(0,k*n)
A_4 = [
    np.array([
        -1 if (((j) // k == i)) else  # Set `1` at `i-th` x and corresponding `y`
        0  # Otherwise, keep `0`
        for j in range(len(A_4))
    ]) 
    for i in range(k)]

A_4=np.vstack(A_4)
b_4=expected_xis
print(f""" Fourth Restriction: {A_4}""")
print(f""" Fourth Restriction b: {b_4}""")

W=np.vstack([A_3,A_4])
#A=np.vstack([A_1,A_2,A_3,A_4])
#b=np.concatenate([b_1,b_2,b_3,b_4]).reshape(-1,1)

##Modes
T_1=10
T_2=.6*T_1
T_3=.3*T_1

T=np.array([T_1,T_2,T_3])
print(T)
##Investment costs
c=np.array([10,7,16,6])
print(c)
##Production costs

q=[ i*j for i in [4,4.5,3.2,5.5] for j in T]
q=np.array(q)
print(q)
#c=np.hstack((c_x,c_z))
#print(c)
print(f"""A: {A}""")
print(f"""b: {b}""")
#print(f"""c: {c}""")
print(f"""W: {W}""")
H = lambda xi: np.concatenate([-np.eye(n), np.zeros((k, n))])
h = lambda xi: np.array([0, 0, 0, 0, -xi[0], -xi[1], -xi[2]])
def g(xi, x=None):
    # If x is not provided, it is a variable
    if x is None:
        x = cp.Variable(n)
    # Otherwise, it is considered fixed and is a parameter
    else:
        x = cp.Parameter(n, value=x)
    y = cp.Variable(n*k)
    objective = cp.Minimize(c @ x + q @ y)
    constraints = [x >= 0, y >= 0, W @ y <= h(xi) - H(xi) @ x, A @ x <= b]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value, y.value, prob.value


# Compute EV
x_bar, z_bar, EV = g(expected_xis)
print(f"EV Solution X: {x_bar}")
# print(f"EV Solution Z: {z_bar}")
print(f"EV Value: {EV}")



# Compute EEV
EEV = 0
for i, j, r in itertools.product(range(3), range(3), range(3)):
    print(i,j,r)
    _, _, v = g([xis[i, 0], xis[j, 1], xis[r, 2]], x_bar)
    EEV += probs[i]*probs[j]*probs[r] * v
print(f"EEV Value: {EEV}")



