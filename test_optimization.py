# Test pure optimization
import numpy as np
from scipy.optimize import least_squares
import csv
import warnings

def levy(x: np.ndarray) -> float:
    """ From ziess/test_functions/functions.py """
    if np.any(np.abs(x > 10)):
        warnings.warn(
            "The Levy function should be evaluated in the [-10, 10] box", UserWarning
        )
    w = 1 + (x - 1) * 0.25
    return (
        np.sin(np.pi * x[0]) ** 2
        + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1]) ** 2))
        + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    )

n_points = 10
problem_size = 10
bounds = (-10, 10)
objective_function = levy

samples = []
costs = []
n_trials = 10
for i in range(n_trials):
    print("trail", i)
    starting_points = [np.random.uniform(low=-10, high=10, size=(problem_size,)) for _ in range(n_points)]
    starting_costs = [levy(starting_point) for starting_point in starting_points]
    print("staring costs\n", starting_costs)
    trial_costs = []
    for starting_point in starting_points:
        result = least_squares(objective_function, starting_point, bounds=bounds, 
                                ftol=1e-15, xtol=None, gtol=None, max_nfev=1e5, method="dogbox", diff_step=1) # max_nfev could potentially be set to None
        samples.append(result.x) 
        trial_costs.append(result.cost)
    costs.append(trial_costs)

print(costs)
print("best cost", min(costs[0]), "on trial", np.argmin(costs[0]))

with open('data/best_costs_dogbox_diff_step_1_least_squares_optimization_best_points_10000_steps_10_trials.csv', mode='w') as f:
    # create the csv writer
    writer = csv.writer(f)
    for trial_costs in costs:
        writer.writerow(trial_costs)