###############
## Problem 3 ##
###############

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../src')
from ElectricityInvestmentPlanning import Problem1, Problem2, Problem3

if __name__ == "__main__":
    print("-------------")
    print("| Problem 3 |")
    print("-------------")
    print()                                                 

    # Get Expected Value Solution (with xi and T variable)
    x_EVS_fix, EV_fix, spent_time = Problem3.getEVS()
    print("EVS Solution (xi, T variable) X: ", x_EVS_fix)
    print("EVS Value (xi, T variable): ", EV_fix, f"({round(spent_time, 2)}s)")
    print()

    # Get Expected Value Solution (with xi and T fixed)
    x_EVS, EV, spent_time = Problem3.getEVS(fix_xi_T=True)
    print("EVS Solution (xi, T fixed) X: ", x_EVS)
    print("EVS Value (xi, T fixed): ", EV, f"({round(spent_time, 2)}s)")
    print()

    # Compute Expected Result for each EVS
    samples = 100 # Number of LHS samples. Programs are solved separately for each sample.
    EEV_fix, spent_time = Problem3.getEEV(x_EVS_fix, samples)
    print("EEV Solution (xi, T variable): ", EEV_fix, f"({round(spent_time, 2)}s)")
    EEV, spent_time = Problem3.getEEV(x_EVS, samples)
    print("EEV Solution (xi, T fixed): ", EEV, f"({round(spent_time, 2)}s)")
    print(f"Samples: {samples}")
    print()

    # Compute the TS solution using an L-Shaped algorithm
    samples = 10 # Number of LHS samples. Programs are not separate.
    iterations = 100 # Number of iterations for L-Shaped algorithm
    x_TS_LShaped, TS_LShaped, _, time_spent = Problem3.getTS_LShaped(samples, iterations)