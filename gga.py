# GGA: Generative Genetic Algorithm, or Generative Evolutionary Algorithm
# Author: Collin Farquhar

# Loop (overproduce and cull)
#### Run generation
#### Select (either here)
#### Run optimization
#### Select (or here)

# TODO keep a list of terminated optimization points.

import warnings
import numpy as np
from scipy.optimize import least_squares
from zqml.qeo.generators import ContinuousVAEGenerator
from zqml.qeo import QEOStandalone, QEOBooster
from math import ceil
import random
from scipy.special import softmax

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
    def __init__(self, objective_function, n_parameters, bounds=(-10, 10), geo=None):
        self.objective_function = objective_function
        self.n_parameters = n_parameters
        self.bounds = bounds
        self.optimizer = least_squares
        self.generator = None # Either use a generator object or a GEO object
        self.geo = geo
        self.n_initial_random_samples = 100
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.n_optimze_steps = 100
        self.n_samples_to_generate = 100
        self.n_selected_samples = 10
        self.batch_size = 64

        self.samples = [] 
        self.costs = []
        self.probs = []

        self.starter()
    
    def starter(self):
        """ Maybe just for temporary testing purposes """
        samples = self.generate_random_samples(self.batch_size)
        costs = self.evaluate_costs(samples)
        probs = self.evaluate_probabilities(costs)
        self.samples += samples
        self.costs += costs
        print(type(probs))
        self.probs += list(probs)
        # print(samples)
        # print(costs)
        # print(probs)
        for i, cost in enumerate(costs):
            print("cost:", cost, "softmax:", softmax(cost))
            print("cost:", cost, "prob:", self.probs[i])

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
                                   ftol=1e-15, xtol=None, gtol=None, max_nfev=1e3) # max_nfev could potentially be set to None
            samples_and_costs.append([result.x, result.cost]) 
        return samples_and_costs

    def generate(self):
        """ Generate samples, a subset of which can be selected to be passed to the optimizer.
        Generate samples both randomly and using the generator based on the epsilon parameter. """
        n_exploration_samples = ceil(self.epsilon * self.n_samples_to_generate)
        n_samples_with_generator = self.n_samples_to_generate - n_exploration_samples
        generated_samples = []

        # Exploration: choose some number of samples randomly
        generated_samples += self.generate_random_samples(n_exploration_samples) 

        # Generate samples with the generative model
        generated_samples += self.geo._generator.generate_samples(n_samples_with_generator)

        # Update history?
        return generated_samples
    
    def evaluate_costs(self, samples):
        return [self.objective_function(sample) for sample in samples]
    
    def evaluate_probabilities(self, costs):
        return softmax(costs)

    def select_samples_for_optimizer(self, samples, costs):
        """ Can have more general selection strategies later, but for now, 
        can just pick the generated samples with the best cost. """
        best_costs_indices = np.argsort(costs)[:self.n_selected_samples]
        return [(samples[idx], costs[idx]) for idx in best_costs_indices]

    def train(self):
        """ train the generator after getting new sample and cost data from the optimizer and
        the previously generated samples """
        # _update_history?
        # train with jax, see QEOBase.optimize()

        # Add the newest selected samples to batch, randomly select previous samples to fill out the rest of the batch size
        n_previous_samples = len(self.samples) - self.n_selected_samples 
        most_recent_samples_indices = list( range(n_previous_samples, len(self.samples)) )
        past_indices = list(range(n_previous_samples))
        print("n_prev", n_previous_samples)
        print("lpi", len(past_indices))
        print("bs", self.batch_size)
        print("n_selected", self.n_selected_samples)
        batch_indices = most_recent_samples_indices + random.sample(past_indices, self.batch_size - self.n_selected_samples)
        random.shuffle(batch_indices)
        batch = np.array([self.samples[i] for i in batch_indices])
        probs = np.array([self.probs[i] for i in batch_indices])
        
        self.geo._generator.train(
            n_epochs=1,
            xtrain=batch,
            probs=probs,
            batch_size=None,
            random_seed=None,
            learning_rate=None,
            warm_start=True,
        )

    def run(self):
        pass

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

    # ---------------> Test GGA optimize function : good <------------------
    """
    gga = GGA(objective_function=quadratic, n_parameters=n_params, bounds=bounds)
    starting_point = gga.generate_random_samples()
    print("starting_point", starting_point, "starting cost", objective_function(starting_point))
    result = gga.optimize(starting_point)
    print("optimized result", result) 
    """
    # -------------------------------------------------------------


    # ---------------> Test GGA generate function :  <------------------
    generator = ContinuousVAEGenerator(
            sample_size=n_params,
            encoder_widths=[50, 20],
            latents=10,
            decoder_widths=[20, 50, n_params],
            random_seed=1234,
        )
    #print(generator.generate(1))
    initial_bitstrings = np.random.rand(1, n_params)
    print("initial_bitstring", initial_bitstrings)
    geo = QEOStandalone(objective=quadratic, generator=generator, bitstrings=initial_bitstrings) 
    gga = GGA(objective_function=quadratic, n_parameters=n_params, bounds=bounds, geo=geo) 
    gga.train()
    res = gga.generate()
    print(res)