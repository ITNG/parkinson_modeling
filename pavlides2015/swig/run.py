import os
import lib
import torch
import numpy as np
import pylab as plt
from config import *
from time import time
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.analysis import pairplot
from library import (simulator,
                     plot_data,
                     display_time,
                     simulation_wrapper,
                     calculate_summary_statistics)
if not os.path.exists('data'):
    os.makedirs('data')


if __name__ == "__main__":

    # --------------------------------------------------------------#
    def test_example():
        start_time = time()
        fig, ax = plt.subplots(2, figsize=(10, 4.5))
        
        obs1 = simulator(sim_params, true_params)

        sim_params['I_C'] = 100.0 / 1000.0
        sim_params['I_Str'] = 8.0 / 1000.0
        obs2 = simulator(sim_params, true_params)
        
        display_time(time()-start_time)

        plot_data(obs1, ax[0])
        plot_data(obs2, ax[1])
        plt.savefig("data/test.png")
        plt.close()

    # --------------------------------------------------------------#

    def run_single_round(samples_filename="data/samples.pt"):
        start_time = time()

        torch.set_num_threads(num_threads)
        prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                            high=torch.as_tensor(prior_max))
        posterior = infer(simulation_wrapper,
                          prior,
                          method=method,
                          num_simulations=num_simulations,
                          num_workers=num_workers)

        observation_trace = simulator(sim_params, true_params)
        obs_stats = calculate_summary_statistics(observation_trace)
        samples = posterior.sample((num_samples,), x=obs_stats)
        torch.save(samples, f'{samples_filename}')
        display_time(time() - start_time)

    def run_multiple_round(samples_filename="data/samples.pt", n_rounds=2):
        start_time = time()

        torch.set_num_threads(num_threads)
        prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                            high=torch.as_tensor(prior_max))

        observation_trace = simulator(sim_params, true_params)
        obs_stats = calculate_summary_statistics(observation_trace)

        for _ in range(num_rounds):

            posterior = infer(simulation_wrapper,
                          prior,
                          method=method,
                          num_simulations=100,
                          num_workers=num_workers)
            
            prior = posterior.set_default_x(obs_stats)
        
        samples = posterior.sample((num_samples,), x=obs_stats)
        torch.save(samples, f'{samples_filename}')
        display_time(time() - start_time)

    def use_pairplot(samples_filename, output_filename="data/infer.png"):

        try:
            samples = torch.load(samples_filename)
        except:
            print("no input file!")
            exit(0)

        fig, axes = pairplot(samples,
                             labels=['SG', 'GS', 'CS', 'SC', 'GG', 'CC'],
                             figsize=(10, 8),
                             points=true_params,
                             points_offdiag={'markersize': 6},
                             points_colors='r')
        fig.savefig(f"{output_filename}", dpi=150)
        plt.close()

    
    # test_example()

    samples_filename = "data/single_round.pt"
    run_single_round(samples_filename)
    use_pairplot(samples_filename, output_filename="data/single.png")


    samples_filename = "data/multiple_round.pt"
    run_multiple_round(samples_filename)
    use_pairplot(samples_filename, output_filename="data/multiple.png")

    # sol = simulator(sim_params, true_params)
    # t = sol['t']
    # y = sol['data']
    # print(len(t), y.shape)
