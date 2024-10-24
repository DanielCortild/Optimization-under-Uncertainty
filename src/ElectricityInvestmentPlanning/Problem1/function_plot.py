import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.ticker as mtick
from ..Parameters import c,q
def plot_capacity(x,title,n=4):
    x_inv=x[0:n]
    colors = plt.get_cmap('Set2').colors[:n]
    plt.figure(figsize=(10, 6))
    # Plot each point and color them differently
    colors = plt.cm.viridis(np.linspace(0, 1, n))  # Using a colormap with n different colors
    plt.bar(range(1, n+1), x_inv, color=colors)
    # Setting the y-axis limits
    plt.ylim(0, 7)
    # Labeling axes and title
    plt.xlabel('Technology')
    plt.ylabel('Capacity')
    plt.title(f'Results {title}')
    # Set x-ticks to whole numbers and enable grid for y-axis
    plt.xticks(range(1, n+1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    def unit_formatter(x, pos):
        return f'{x:,.2f}'
    # Format the y-axis to include commas and 'M' for millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(unit_formatter))
    file_path = f"../output/problem_1_{title}_capacity.png"
    plt.savefig(file_path)
def plot_solutions(WS,EV,TS,EEV,prefix="problem_1"):
    # Sample data for the columns
    columns = ['EV','WS', 'TS', 'EEV']
    values = [EV, WS, TS, EEV]

    # Set up the figure
    fig, ax = plt.subplots()

    # Define the color palette from Set2
    colors = plt.get_cmap('Set1')(np.arange(len(columns)))

    # Create the bar chart
    bars=ax.bar(columns, values, color=colors)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:,.2f}', 
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ymin=min(values)-10
    ymax=max(values)+10
    ax.set_ylim(ymin, ymax)
    # Add labels and title
    ax.set_ylabel('Cost')
    ax.set_title('Solution Comparison')
    file_path = f"../output/{prefix}_solutions.png"
    plt.savefig(file_path)
def plot_costs(c=c, q=q, n=4):
    # Ensure the vectors c and c_prod are of the correct size
    c_inv = c
    c_prod_inv = q
    
    # Generate x positions for the bars
    x = np.arange(n)
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Define color map for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    width = 0.2
    # Plot bars for the two sets of values (c and c_prod)
    bars1 = ax.bar(x - width, c_inv, width, label='Investment Costs', color=colors, hatch='//')  # Diagonal stripes
    bars2 = ax.bar(x + width, c_prod_inv, width, label='Production Costs', color=colors, hatch='xx')  # Crosshatch pattern
    
    
    # Set the y-axis limits (you can adjust as needed)
    #ax.set_ylim(0, 7)
    
    # Labeling the axes and setting the title
    ax.set_xlabel('Technology')
    ax.set_ylabel('Cost')
    ax.set_title(f'Cost by Technology')
    ax.tick_params(axis='x', which='major')
    # Set x-ticks with appropriate labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n)])  # Labeling as Tech 1, Tech 2, etc.
    ##Increase font size of labels
    # Increase font size of labels
 
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Add grid lines to the y-axis
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Format the y-axis to include commas and format units
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.2f}'))
    
    # Add a legend to distinguish the two bars
    legend = ax.legend()
    for patch in legend.get_patches():
        patch.set_facecolor('none')
    
    # Save the plot to a file
    file_path = f"../output/problem_1_costs.png"
    plt.savefig(file_path)