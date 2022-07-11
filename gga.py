# GGA: Generative Genetic Algorithm, or Generative Evolutionary Algorithm
# Author: Collin Farquhar

# Loop (overproduce and cull)
#### Run generation
#### Select (either here)
#### Run optimization
#### Select (or here)


import warnings
import numpy as np
from scipy.optimize import least_squares
from zqml.qeo.generators import ContinuousVAEGenerator
from math import ciel

# ----------------> Example Objective Functions <----------------
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

def quadratic(x: np.ndarray) -> float:
    return sum(np.power(x, 2))
# -------------------------------------------------------------

class GGA():
    """
    TODO add class attributes for "options" for optimizer and generator
    """
    def __init__(self, objective_function, n_parameters, bounds=(-10, 10)):
        self.objective_function = objective_function
        self.n_parameters = n_parameters
        self.bounds = bounds
        self.optimizer = least_squares
        self.generator = ContinuousVAEGenerator 
        self.n_initial_random_samples = 100
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.n_optimze_steps = 100
        self.n_samples_to_generate = 100
        self.n_selected_samples = 10

    def generate_random_samples(self, n_samples=1):
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], 
                                 size=(n_samples, self.n_parameters))

    def optimize(self, starting_points):
        # the scipy least square optimzer only gives the final result,
        # it would be better to have the entire sample & cost trajectory to
        # give the generator more data to train on.
        # One option would be to use the scipy minimize function, which can be 
        # passed a callback function that is called on every step.

        samples_and_costs = [] # Best point from each in optimization trajectory
        for starting_point in starting_points:
            result = least_squares(objective_function, starting_point, bounds=bounds, 
                                   ftol=1e-15, xtol=None, gtol=None, max_nfev=100) 
            samples_and_costs.append([result.x, result.cost]) 
        return samples_and_costs

    def generate(self):
        """ Generate samples, a subset of which can be selected to be passed to the optimzier.
        Generate samples both randomly and using the generator based on the epsilon parameter. """
        n_exploration_samples = ciel(self.epsilon * self.n_samples_to_generate)
        n_samples_with_generator = self.n_samples_to_generate - n_exploration_samples
        generated_samples = []

        # Exploration: choose some number of samples randomly
        generated_samples += self.generate_random_samples(n_exploration_samples) 

        # Generate samples with the generative model
        generated_samples += self.generator.generate(n_samples_with_generator)

        return generated_samples

        
    def select_samples(self):
        """ Can have more general selection strategies later, but for now, 
        can just pick the 10 generated samples with the best cost. """
        pass

    def train(self):
        """ train the generator after getting new sample and cost data from the optimizer and
        the previously generated samples """
        # _update_history?
        # train

if __name__ == "__main__":
    # Test with optimizer
    #print("starting cost", objective_function(starting_point))
    #result = least_squares(objective_function, starting_point, bounds=(-10, 10), method="dogbox")
    #result = least_squares(objective_function, starting_point, bounds=bounds, ftol=1e-15, xtol=None, gtol=None, max_nfev=1000)
    #result = least_squares(objective_function, starting_point, ftol=1e-15, xtol=None, gtol=None, max_nfev=10)
    # print(result)
    # print("final cost", result.cost)

    # ----------------> Define Objective Functions <----------------
    n_params = 1
    objective_function = quadratic
    bounds = (-10, 10)
    # -------------------------------------------------------------
    gga = GGA(objective_function=quadratic, n_parameters=n_params, bounds=bounds)
    starting_point = gga.generate_random_samples()
    print("starting_point", starting_point, "starting cost", objective_function(starting_point))
    result = gga.optimize(starting_point)
    print(result) 
