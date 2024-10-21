import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.ticker as mtick

def print_solution(x,obj,c,n=4):
    print(f"Objective value: {obj}")
    print(f"Capacity of each technology: {x}")
    precost=sum([c[i]*x[i] for i in range(n)])
    print(f"Investment costs for phase 1: {precost}")
def plot_capacity(x,title,n=4):
    T = [10, 16, 17]
    x_inv=x[0:n]
    colors = plt.get_cmap('Set2').colors[:n]
    plt.figure(figsize=(10, 6))
    # Plot each point and color them differently
    colors = plt.cm.viridis(np.linspace(0, 1, n))  # Using a colormap with n different colors
    plt.bar(range(n), x_inv, color=colors)
    # Labeling axes and title
    plt.xlabel('Technology')
    plt.ylabel('Capacity')
    plt.title(f'Results {title}')
    # Set x-ticks to whole numbers and enable grid for y-axis
    plt.xticks(range(n))  # Ensuring only whole numbers on the x-axis
    plt.grid(True, axis='y')  # Grid on y-axis only
    def unit_formatter(x, pos):
        return f'{x:,.2f}'
    # Format the y-axis to include commas and 'M' for millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(unit_formatter))
    file_path = f"output/{title}_capacity.png"
    plt.savefig(file_path)
    plt.show()
def plot_solutions(WS,EV,TS,EEV):
    # Sample data for the columns
    columns = ['WS', 'EV', 'TS', 'EEV']
    values = [WS, EV, TS, EEV]

    # Set up the figure
    fig, ax = plt.subplots()

    # Define the color palette from Set2
    colors = plt.get_cmap('Set2')(np.arange(len(columns)))

    # Create the bar chart
    ax.bar(columns, values, color=colors)

    # Add labels and title
    ax.set_ylabel('Cost')
    ax.set_title('Solutions')

    # Show the plot
    plt.show()    