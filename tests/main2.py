###############
## Problem 2 ##
###############

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../src')
from ElectricityInvestmentPlanning import Problem1
from ElectricityInvestmentPlanning import Problem2

print("-------------")
print("| Problem 2 |")
print("-------------")
print()

# Get Expected Value Solution (with xi fixed)
x_EVS_fix, EV_fix, spent_time = Problem2.getEVS(fix_xi=True)
print("EVS Solution (xi fixed) X: ", x_EVS_fix)
print("EVS Value (xi fixed): ", EV_fix, f"({round(spent_time, 2)}s)")
print()

# Get Expected Value Solution (with xi variable)
x_EVS, EV, spent_time = Problem2.getEVS(fix_xi=False)
print("EVS Solution (xi variable) X: ", x_EVS)
print("EVS Value (xi variable): ", EV, f"({round(spent_time, 2)}s)")
print()

# Get Expected Value Solution from Problem 1
x_EVS_Problem1, EV_Problem1, spent_time = Problem1.getEVS()
print("EVS Solution (Problem 1) X: ", x_EVS_Problem1)
print("EVS Value (Problem 1): ", EV_Problem1, f"({round(spent_time, 2)}s)")
print()

# Get Two-Stage Solution from Problem 1
x_TS_Problem1, TS_Problem1, spent_time = Problem1.getTS()
print("TS Solution (Problem 1) X: ", x_TS_Problem1)
print("TS Value (Problem 1): ", TS_Problem1, f"({round(spent_time, 2)}s)")
print()

# Compute Expected Result for each EVS
samples = 10000 # Number of LHS samples. Programs are solved separately for each sample.
EEV_fix, spent_time = Problem2.getEEV(x_EVS_fix, samples)
print("EEV Solution (xi fixed): ", EEV_fix, f"({round(spent_time, 2)}s)")
EEV, spent_time = Problem2.getEEV(x_EVS, samples)
print("EEV Solution (xi variable): ", EEV, f"({round(spent_time, 2)}s)")
EEV_Problem1, spent_time = Problem2.getEEV(x_EVS_Problem1, samples)
print("EEV Solution (Problem 1): ", EEV_Problem1, f"({round(spent_time, 2)}s)")
EEV_TS_Problem1, spent_time = Problem2.getEEV(x_TS_Problem1, samples)
print("EEV Solution (TS Problem 1): ", EEV_TS_Problem1, f"({round(spent_time, 2)}s)")
print(f"Samples: {samples}")
print()

# Compute the TS solution using a naive sampling approach
samples = 100 # Number of LHS samples. Programs are not separate.
counts = 100 # Number of times to run algorithm to observe variety
TS_Naive, time_spent = Problem2.getTSPlot_Naive(samples, counts)
print("TS Naive Solution: ", TS_Naive, f"({round(time_spent, 2)}s)")
print(f"Samples: {samples}, Counts: {counts}")
print()

# Compute the TS solution using an L-shaped algorithm
samples = 1000 # Number of LHS samples. Programs are separate to a certain extent.
iterations = 100 # Number of iterations to run the algorithm
counts = 10 # Number of times to run algorithm to observe variety
TS_L_Shaped, time_spent = Problem2.getTSPlot_LShaped(samples, iterations, counts)
print("TS L-Shaped Solution: ", TS_L_Shaped, f"({round(time_spent, 2)}s)")
print(f"Samples: {samples}, Iterations: {iterations}, Counts: {counts}")
print()

# Plot the convergence of the L-shaped algorithm
samples = 1000 # Number of LHS samples. Programs are separate to a certain extent.
iterations = 100 # Number of iterations to run the algorithm
Problem2.getTSPlot_LShaped_Convergence(samples, iterations)
print(f"Samples: {samples}, Iterations: {iterations}")
print()