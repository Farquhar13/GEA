from pickletools import optimize
import matplotlib.pyplot as plt
import csv
import numpy as np

pure_optimization = []
with open('data/least_squares_optimization_best_points.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        pure_optimization += row_floats
pure_optimization_starting_costs = pure_optimization[:10] 
pure_optimization_ending_costs = pure_optimization[10:] 

standalone_geo = []
with open('data/pure_generative.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        standalone_geo += row_floats
standalone_geo_starting_costs = standalone_geo[:10]
standalone_geo_ending_costs = standalone_geo[10:]

gea = []
with open('data/generative_evolutionary_algoritm.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        row_floats = [float(x) for x in row]
        gea += row_floats
gea_starting_costs = gea[:10]
gea_ending_costs = gea[10:]

# stopped here 
print(pure_optimization_starting_costs)
print()
print(pure_optimization_ending_costs)

# Plot pure optimization
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.arange(10)
ax1.scatter(x, pure_optimization_starting_costs, s=10, c='r', marker="s", label='starting cost')
ax1.scatter(x, pure_optimization_ending_costs, s=10, c='b', marker="s", label='ending cost')
plt.legend(loc='upper left')
plt.title("Least Squares Optimizer on 10-paramater Levy Function")
plt.xlabel("Trial")
plt.ylabel("Cost")
plt.savefig("plots/pure_optimization.pdf")
plt.show()


# Plot pure generative modle & EGA
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# x = np.arange(10)
# ax1.scatter(x, standalone_geo_starting_costs, s=10, c='r', marker="s", label='GEO starting cost')
# ax1.scatter(x, standalone_geo_ending_costs, s=10, c='b', marker="s", label='GEO ending cost')
# ax1.scatter(x, gea_starting_costs, s=10, c='k', marker="s", label='EGA starting cost')
# ax1.scatter(x, gea_ending_costs, s=10, c='c', marker="s", label='EGA ending cost')
# plt.legend(loc='upper left')
# plt.title("GEO and EGA on 10-paramater Levy Function")
# plt.xlabel("Trial")
# plt.ylabel("Cost")
# plt.savefig("plots/geo_and_ega.pdf")
# plt.show()
