import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.ticker as mtick

def print_solution(x,c,n):
    print(f"Objective value: {sum([c[i]*x.value[i] for i in range(len(x.value))])}")
    print(f"Capacity of each technology: {x.value[0:n]}")
    precost=sum([c[i]*x.value[i] for i in range(n)])
    prodcost=sum([c[i]*x.value[i] for i in np.arange(n,len(x.value))])
    print(f"Investment costs for phase 1: {precost}")
    print(f"Production costs for phase 2: {prodcost}")
def plot_result(x,c,n):

    
    T = [10, 16, 17]

    x_inv=c[0:n]*x.value[0:n]


    
    n_tech =n
    n = len(x.value[0:n_tech])  

    
    start_points = [0] + T[:-1]  
    end_points = T  

    
    colors = plt.get_cmap('Set2').colors[:n_tech]  

    plt.figure(figsize=(10, 6))

    # Plot each point and color them differently
    colors = plt.cm.viridis(np.linspace(0, 1, n))  # Using a colormap with n different colors

    plt.bar(range(n), x_inv, color=colors)

    # Labeling axes and title
    plt.xlabel('Technology')
    plt.ylabel('Investment ($)')
    plt.title('Investment in Technologies')

    # Set x-ticks to whole numbers and enable grid for y-axis
    plt.xticks(range(n))  # Ensuring only whole numbers on the x-axis
    plt.grid(True, axis='y')  # Grid on y-axis only
    def millions_formatter(x, pos):
        return f'${x:,.1f}M'
    # Format the y-axis to include commas and 'M' for millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(millions_formatter))

    plt.show()