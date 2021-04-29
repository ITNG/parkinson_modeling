import os
import lib
import torch
import numpy as np
import pylab as plt
from config import *
from copy import copy
from time import time
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.analysis import pairplot
from library import (simulator,
                     plot_data,
                     display_time,
                     simulation_wrapper,
                     statistics_prop,
                     use_pairplot,
                     welch,
                     fft_1d_real,
                     get_max_probability)
if not os.path.exists('data'):
    os.makedirs('data')


if __name__ == "__main__":

    # --------------------------------------------------------------#
    def sample_curve(par, filename='data/fig.png'):

        fig, ax = plt.subplots(1, figsize=(10, 3.5))
        obs1 = simulator(sim_params, par)
        plot_data(obs1, ax)
        plt.savefig(f"{filename}")
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
        obs_stats = statistics_prop(observation_trace)
        samples = posterior.sample((num_samples,), x=obs_stats)
        torch.save(samples, f'{samples_filename}')
        display_time(time() - start_time)

    def run_multiple_round(samples_filename="data/samples.pt", n_rounds=2):
        start_time = time()

        torch.set_num_threads(num_threads)
        prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                            high=torch.as_tensor(prior_max))

        # observation_trace = simulator(sim_params, true_params)
        # obs_stats = statistics_prop(observation_trace,
        #                             method=sim_params['statistics_method'])
        obs_stats = copy(observation_params)
        # print(obs_stats.shape)
        # print(obs_stats*1000)
        # exit(0)

        for _ in range(num_rounds):

            posterior = infer(simulation_wrapper,
                              prior,
                              method=method,
                              num_workers=num_workers,
                              num_simulations=num_simulations,
                              )

            prior = posterior.set_default_x(obs_stats)

        samples = posterior.sample((num_samples,),
                                   x=obs_stats,
                                   sample_with_mcmc=False)
        torch.save(samples, f'{samples_filename}')
        display_time(time() - start_time)
    
    # -------------------------------------------------------------------------

    # sample_curve(true_params)

    # samples_filename = "data/single_round.pt"
    # run_single_round(samples_filename)
    # use_pairplot(samples_filename, output_filename="data/single.png")

    samples_filename = "data/multiple_round.pt"
    run_multiple_round(samples_filename)
    use_pairplot(samples_filename, output_filename="data/multiple.png")

    par = get_max_probability(samples_filename)
    sample_curve(par, "data/sample.png")
    sample_curve(true_params, "data/true.png")
