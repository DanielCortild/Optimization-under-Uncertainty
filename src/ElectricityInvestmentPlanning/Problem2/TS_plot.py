import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from .TS import getTS_Naive, getTS_LShaped

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