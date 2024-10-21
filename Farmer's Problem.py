#!/usr/bin/env python
# coding: utf-8

# This is a Jupyter Notebook. It allows you to run code step by step (or rather, cell by cell). Select the cell you want to run and press ctrl+enter to run. Press ctrl+shift to add a new cell below. Variables and function defitions carry over between cells, so in principle it is enough to import the correct packages only once. I have opted to add them to the start of each section, so you do not have to always run the first cell to setup the imports.

# # `linprog` example
# We show how to use linprog to solve a simple LP problem:
# 
# ```
# max x1 +  x2 + 2x3
# s.t. x1 + 2x2 + 3x3 <= 4
#      x1 + x2        >= 1
# ```
#      
# <=>
# 
# ```
# min -x1 - x2 - 2x3
# s.t. x1 + 2x2 + 3x3 <= 4
#     -x1 -  x2       <= -1
# ```

# In[9]:


# import functions and packages
from scipy.optimize import linprog
import numpy as np

c = [-1, -1, -2]  # objective function
A = [  # <= constraints
    [1, 2, 3],
    [-1, -1, 0]
]
b = [4, -1]  # right hand side
bounds = [  # bounds per variable
    (0, None),  # the default is (0, None) (so this is redundant)
    (0, None),  # if only one tuple is given (e.g. bounds = (0, None)
    (0, None)   # these will be used for all variables
]
method = 'simplex'  # solver method 
# chose from simplex (old), revised simplex (new) and interior-point (approximate but quick)

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method=method)
# Note that linprog can only handle <= and == constraints. 
# Use keyword parameters ('keyword=...') to define whether a matrix defines the <= or the == constraints
# (The suffix '_ub' means upper bound and represents <= constraints, the suffix '_eq' represents == constraints)

# Display the solution (pay attention to fun (= obj. value), x (=solution) and message (to find errors)
print(res)


# # Deterministic Farmer's Problem
# Here we solve the deterministic farmer's problem. We use the SP notation, where `x` are first-stage variables, `(y, W)` second-stage variables with unit cost vectors `c` and `q`. 
# `A`, `b`, correspond to first-stage constraints, `W` and `h` and the technology matrix `T (=Tech)` to second-stage constraints.
# Because `linprog` can only handle `= ` and `<=` constraints, I convert all inequalities to `<=` (by multiplying the whole constraint by -1, where applicable). 
# Please verify that all matrices of coefficients etc. are correct.

# In[8]:


# import functions and packages
from scipy.optimize import linprog
import numpy as np

# define matrices
A = [1, 1, 1]
Tech = np.vstack((np.diag([-2.5, -3, -20]), [0] * 3))
W = [
    [-1, 1, 0, 0, 0, 0],
    [0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0],
]
b = 500
c = [150, 230, 260]
q = [238, -170, 210, -150, -36, -10]
h = [-200, -240, 0, 6000]

# Combine them in approriate ways to arrive at A, b and c
A_ub = np.hstack((np.vstack((A, Tech)), np.vstack(([0] * len(W[0]), W))))
c = np.hstack((c, q))
b_ub = np.hstack((b, h))

method = 'simplex'

# Solve and print
res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=method)
print(res)


# # Stochastic Farmer's Problem
# Next we solve a stochastic farmer's problem where we have 3 possible values for the yield. We solve this problem using the large-scale deterministic equivalent problem. We set up the code in such a way that it can also be used to solve the LSDE of a SAA (sample average approximation).

# In[25]:


# import functions and packages
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt # for plotting


# ## Options
# Here, there are three options for specifying the distribution of the "yield" `xi` (or rather %dev of avg yield).The first is with three scenarios `xi` = 0.8, 1, 1.2; all with probability 1/3. The second is by drawing a sample of `S` scenarios from the distribution of `xi` (As an example we assume that `xi` is uniformly distributed on [0.8,1.2]) The third is the same as the second, but uses Latin Hypercube Sampling instead of Monte Carlo sampling
# 
# ```
# S  = number of scenarios
# p  = "vector of probabilities"
# xi = "vector of realizations"```
# 
# To select an option, simply run that cell and then run the cell under 'Solving the problem'

# ### Option 1: 3 distinct scenarios

# In[11]:


S = 3
p = [1/S] * S
xi = np.array([0.8, 1, 1.2])


# ### Option 2: 200 random (Monte Carlo) scenarios
# Note that the number of scenarios `linprog` can handle depends on your computer.

# In[57]:


S = 200
p = [1/S] * S
xi = np.random.uniform(0.8, 1.2, S)
#plt.hist(xi, bins = 20, range = (0.8, 1.2)) # display the sampled values


# ### Option 3: 200 random (LHS) scenarios
# Latin Hypercube Sampling. Again, the number of scenarios depends on your computer.

# In[36]:


# Hypercube sampling scenarios
from pyDOE import lhs
S = 200
p = [1/S] * S
xi = lhs(1,S).flatten()*0.4 + 0.8 # create values
#plt.hist(xi, bins = 20, range = (0.8, 1.2)) # display the sampled values


# ## Solving the problem
# We define the coefficient matrices and solve the LP

# In[59]:


Tech = np.vstack((np.diag([-2.5, -3, -20]), [0] * 3))
A = [
    [1,1,1]
]
W = [
    [-1, 1, 0, 0, 0, 0],
    [0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0],
]
b = 500
c = [150, 230, 260]
q = [238, -170, 210, -150, -36, -10] 
h = [-200, -240, 0, 6000]

# Store the dimensions (m = number of constraints, n = number of variables, 1st and 2nd stage)
m1 = np.shape(A)[0]
m2 = np.shape(W)[0]
n1 = np.shape(A)[1]
n2 = np.shape(W)[1]

# What remains is to create the matrix of coefficients for this large-scale problem
# It has the form (if there are 4 scenarios):
#. ( A   0  0  0  0 )
#. ( T1  W  0  0  0 )
#. ( T2  0  W  0  0 )
#. ( T3  0  0  W  0 )
#. ( T4  0  0  0  W )
# We use the kronecker product to quickly create the matrix above. Google it if you do not recall what it is.

Z = np.zeros((m1, n2*S))

LSDE_A = np.hstack((A,Z))
# create first column of matrix above, except first row
col1 = np.kron(np.transpose([xi]), Tech) # the xi matrix should (explicitly) be a matrix with the right dimensions
                                       # so we make it a matrix(=2d array) by wrapping it in [] and then transposing it
#create other columns (except first row)
col2 = np.kron(np.eye(S), W)
# concatenate all to create the final coefficient matrix A
A_ub = np.vstack((LSDE_A, np.hstack((col1, col2))))

# create the objective function using elementwise multiplication for the scenarios
for pi in p:  # then add all the scenarios (weighted) to it
    c = np.append(c, np.multiply(pi, q))
    
# create the rhs
b_ub = [b] + h * S
# solve the problem (we use the revised simplex method because it is quicker, albeit slightly less accurate)
# if you run into problems finding a feasible solution, try lowering the tolerance by adding the keyword argument
# options={'tol':1e-5}
# Note: this might take a while for option 2 and 3. If the title of this cell is `In [*]`, it is running.
result = linprog(c, A_ub=A_ub, b_ub=b_ub, method = 'revised simplex')
# print the useful part of the result
print("fun: " + str(result['fun']) + "\nfirst stage solutions: " + str(result["x"][0:3]))


# # L-shaped algorithm
# Below is the a basic L-shaped algorithm for solving the saming Stochastic Farmer's problem. Again, use the same setup as with the LSDE. Note that you can select which option should be run, just like above. This time, you can do this by commenting and uncommenting the code. (Select the code and press `ctrl + /` to comment/uncomment it)

# In[43]:


# Use one of 3 approaches above
# Scenarios
# S = 3
# p = [1/S] * S
# xi = np.array([0.8, 1, 1.2])

# Uniform random
S = 200
p = [1/S] * S
xi = np.random.uniform(0.8, 1.2, S)

# LHS uniform random
# S = 200
# p = [1/S] * S
# xi = lhs(1,S).flatten()*0.4 + 0.8 # create values

Tech = np.vstack((np.diag([-2.5, -3, -20]), [0] * 3))
A = [
    [1,1,1]
]
W = [
    [-1, 1, 0, 0, 0, 0],
    [0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0],
]
b = 500
c = [150, 230, 260]
q = [238, -170, 210, -150, -36, -10] 
h = [-200, -240, 0, 6000]

# Store the dimensions (m = number of constraints, n = number of variables, 1st and 2nd stage)
m1 = np.shape(A)[0]
m2 = np.shape(W)[0]
n1 = np.shape(A)[1]
n2 = np.shape(W)[1]
# set up the coefficient matrix A, rhs (b) and objective function
m_A = np.hstack((A, np.zeros((m1, 2))))
m_A = np.vstack((m_A, np.concatenate(([0] * (n1+1), [1]))))
m_obj = np.concatenate((c,[1,-1]))
m_b = np.concatenate(([b], [10**10]))

# calculate the initial result (which yields x ~ [0,0,0])
m_result = linprog(m_obj, A_ub = m_A, b_ub = m_b, method='revised simplex')
print("Initial result to master problem")
print('Obj. value ', m_result['fun'], '\nSolution   ', m_result['x'])
print("====================\nStarting algorithm\n====================")

currentX = m_result['x'][0:n1]

# Next initalize the subproblem(s). We will use the same subproblem iteratively
# since only the rhs of these subproblems change. They are given by:
# v(h-T[i]x) = min qy: Wy >= h-T[i]x
# Since linprog does not provide dual solution, we will manually create the dual
# Dual problem: max lambda^T(h-T[i]x): W^T lambda <= q^T
# Recall that we have to change this to a minimization problem
# (And thus that the value of the objective function and coefficient matrix is also modified!)
s_A = -1 * np.transpose(W) # 
s_obj = h - xi[0]*(np.matmul(Tech, currentX))
s_b = q.copy()

Tx = np.matmul(Tech, currentX)

for it in range(100):
    currentX = m_result['x'][0:n1]
    Q = 0
    u = 0
    
    Tx = np.matmul(Tech, currentX)
    i = 0
    # Solve the subproblems
    for i in range(S):
        s_obj = h - xi[i]*Tx
        s_res = linprog(s_obj, A_ub=s_A, b_ub=s_b, method='revised simplex')
        Q = Q + p[i] * s_res['fun']
        u = u - p[i]*xi[i]*s_res['x']
    
    # Calculate u
    u = np.matmul(u, Tech)
    
    # Stopping criterion (epsilon = 10e-5)
    if (np.matmul(c, currentX) - Q < m_result['fun'] + 10**-5):
        break
        
    # Add optimality cut and resolve master problem
    m_A = np.vstack((m_A, np.concatenate((-u, [-1, 1]))))
    m_b = np.concatenate((m_b, [Q - np.matmul(u, currentX)]))
    m_result = linprog(m_obj, A_ub=m_A, b_ub=m_b, method='revised simplex', options = {'tol': 1e-5})
    print('Iteration  ', it)
    print('Obj. value ', m_result['fun'], '\nSolution   ', m_result['x'])
    print('====================')

# print solution
print("Final solution\n====================")
print('Obj. value ', m_result['fun'], '\nSolution   ', m_result['x'])    

