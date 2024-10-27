import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.ticker as mtick

from .TS import getTS_Naive, getTS_LShaped
from ..Parameters import c, q_import

def getTSPlot(counts, getTS, title):
    # Initialize arrays to store results
    TSs = []
    xs = []
    spent_times = []

    # Run the process multiple times
    for _ in tqdm(range(counts), leave=False):
        x, TS, spent_time = getTS()
        TSs.append(TS)
        xs.append(x)
        spent_times.append(spent_time)

    # Convert to numpy arrays for easier manipulation
    TSs = np.array(TSs)
    xs = np.array(xs)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Histogram of TS
    axes[0].hist(TSs, bins=10, color='blue', alpha=0.7)
    axes[0].set_xlabel("Expected Cost")
    axes[0].set_ylabel("Count")

    # Right plot: Histogram of x components
    colors = ['r', 'g', 'b', 'c']
    labels = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']

    for i in range(4):
        axes[1].hist(xs[:, i], bins=10, alpha=0.7, color=colors[i], label=labels[i])

    axes[1].set_xlabel("Capacity")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    fig_name = f"plots/TS_{title}_{time.time()}.png"
    plt.savefig(fig_name, bbox_inches='tight')
    print(f"Figure saved as {fig_name}")
    plt.show()

    return np.mean(TS), np.mean(spent_times)

def getTSPlot_Naive(samples, counts):
    return getTSPlot(counts, lambda: getTS_Naive(samples), "Naive")

def getTSPlot_LShaped(samples, iterations, counts):
    def getTS():
        x, obj, obj_lst, tme = getTS_LShaped(samples, iterations)
        return x, obj, tme
    return getTSPlot(counts, getTS, "L-Shaped")

def getTSPlot_LShaped_Convergence(samples, iterations):
    # Retrieve the objective values
    obj_vals = getTS_LShaped(samples, iterations)[2]

    # Plot the convergence
    plt.figure(figsize=(8, 4))
    plt.plot(obj_vals)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Master Problem Objective Values")

    # Save the figure
    fig_name = f"plots/TS_LShaped_Convergence_{time.time()}.png"
    plt.savefig(fig_name, bbox_inches='tight')
    print(f"Figure saved as {fig_name}")

def plot_costs(n=5):
    # Ensure the vectors c and c_prod are of the correct size
    c_inv = np.append(c,0)
    c_prod_inv = q_import

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

    # Labeling the axes and setting the title
    ax.set_xlabel('Technology')
    ax.set_ylabel('Cost')
    ax.set_title(f'Cost by Technology')
    ax.tick_params(axis='x', which='major', labelsize=15)

    # Set x-ticks with appropriate labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n)])  # Labeling as Tech 1, Tech 2, etc.

    # Add grid lines to the y-axis
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Format the y-axis to include commas and format units
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.2f}'))

    # Add a legend to distinguish the two bars
    legend = ax.legend()
    for patch in legend.get_patches():
        patch.set_facecolor('none')

    # Save the plot to a file
    file_path = f"../output/problem_2_costs.png"
    plt.savefig(file_path)
