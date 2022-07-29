"""
Plot the best results for data with 10 trials
"""

import matplotlib.pyplot as plt
import csv
import numpy as np

n_trials = 10

pure_optimization = []
with open('data/best_costs_dogbox_diff_step_1_least_squares_optimization_best_points_10000_steps_10_trials.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        pure_optimization.append(row_floats)

print(pure_optimization)

standalone_geo = []
with open('data/best_costs_pure_generative_110_steps_10_trails.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        standalone_geo.append(row_floats)

gea = []
with open('data/ega_new_opt.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        gea.append(row_floats)

no_explore = []
with open('data/no_exploration_generative_model.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        no_explore.append(row_floats)

average_geo = [np.mean(trial) for trial in standalone_geo]
#average_geo = [sum(trial)/len(trial) for trial in standalone_geo]
#average_geo = [np.median(trial) for trial in standalone_geo]
average_gea = [np.mean(trial) for trial in gea]
average_no_explore = [np.mean(trial) for trial in no_explore]
for i, x in enumerate(average_gea): 
    print("trail", i)
    print(x)
    print(np.mean(x))

# # Plot pure optimization
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(10)
average_optimzation = [np.mean(trial) for trial in pure_optimization]
pure_optimization = np.array(pure_optimization).T
for i in range(n_trials):
    ax1.scatter(x, pure_optimization[i], s=5, c='k', marker="s")
ax1.scatter(x, average_optimzation, s=50, c='y', marker='*', label="Average Cost")
plt.legend(loc='upper left')
plt.title("10 Trials of Least Squares Optimizer on 10-paramater Levy Function")
plt.xlabel("Trial")
plt.ylabel("Ending Cost")
plt.yscale("log")
plt.savefig("plots/10_trials_pure_optimization.pdf")
plt.show()


# Plot pure generative modle & EGA
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(10)
n_trials = 10

# Change shape so samples from the same trial on are the same column
standalone_geo = np.array(standalone_geo).T
gea = np.array(gea).T
no_explore = np.array(no_explore).T

for i in range(n_trials):
    if i == 0:
        ax1.scatter(x, standalone_geo[i], s=5, c='r', marker="s", label="GEO")
        ax1.scatter(x, gea[i], s=5, c='b', marker="s", label="EGA")
        #ax1.scatter(x, no_explore[i], s=5, c='y', marker="s", label="No Exploration")
    else:
        ax1.scatter(x, standalone_geo[i], s=5, c='r', marker="s")
        ax1.scatter(x, gea[i], s=5, c='b', marker="s")
        #ax1.scatter(x, no_explore[i], s=5, c='y', marker="s")

ax1.scatter(x, average_geo, s=50, c='k', marker='*', label="Average GEO")
ax1.scatter(x, average_gea, s=50, c='y', marker='*', label="Average EGA")
#ax1.scatter(x, average_no_explore, s=50, c='g', marker='*', label="Average no explore")
plt.legend(loc='upper left')
plt.title("10 Best Costs for GEO and EGA on 10-paramater Levy Function")
plt.xlabel("Trial")
plt.ylabel("Costs")
plt.savefig("plots/geo_and_ega.pdf")
plt.show()
