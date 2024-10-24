import sys
sys.path.append('../src')
from ElectricityInvestmentPlanning import Problem1

#Problem 1 - Results
EV=394.66
EEV=399.6
WS=395
TS=397.8
Problem1.plot_solutions(WS=WS,EV=EV,TS=TS,EEV=EEV,prefix="problem_1")


#Extension 1 - Results
EV=395.66 
EEV=419.04
WS=393.87
TS=410.87
Problem1.plot_solutions(WS=WS,EV=EV,TS=TS,EEV=EEV,prefix="problem_2")


#Extension 2 - Results
EV=395.66
EEV=419.04
WS=392.72
TS=410.87
Problem1.plot_solutions(WS=WS,EV=EV,TS=TS,EEV=EEV,prefix="problem_3")
