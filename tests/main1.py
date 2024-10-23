###############
## Problem 1 ##
###############

import sys
sys.path.append('../src')
from ElectricityInvestmentPlanning import Problem1

def main():
    print("-------------")
    print("| Problem 1 |")
    print("-------------")
    print()

    # Get Expected Value Solution
    x_EVS, EV, spent_time = Problem1.getEVS()
    print("EVS Solution X: ", x_EVS)
    print("EVS Value: ", EV, f"({round(spent_time, 2)}s)")
    print()
    Problem1.plot_capacity(x_EVS,"EV Problem")

    # Get Expected Result
    EEV, spent_time = Problem1.getEEV(x_EVS)
    print("EEV Value: ", EEV, f"({round(spent_time, 2)}s)")
    print()


    # Get Wait-And-See
    WS, spent_time = Problem1.getWS()
    print("WS Value: ", WS, f"({round(spent_time, 2)}s)")
    print()

    # Get the Two-Stage Solution
    x_TS, TS, spent_time = Problem1.getTS()
    print("TS Solution X: ", x_TS)
    print("TS Value: ", TS, f"({round(spent_time, 2)}s)")
    print()

    # Compute the Expected Value of Perfect Information
    EVPI = TS - WS
    print("EVPI Value: ", EVPI)
    print()

    # Compute the Value of Stochastic Solution
    VSS = EEV - TS
    print("VSS Value: ", VSS)
    print()

if __name__ == "__main__":
    main()