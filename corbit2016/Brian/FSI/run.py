import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (clean_directory, simulate_FSI_cell,
                 plot_data,
                 plot_current)
from input_factory import get_step_current

if __name__ == "__main__":

    par = {
        'num': 1,
        'v0': -60*b2.mV,
        'Cm':1 *b2.uF,
        'Iapp': 3.35 * b2.uA,
        'gA': 0.39 *b2.mS,
        'gNa': 112.5 *b2.mS,
        'gK': 225.0 *b2.mS,
        'gL': 0.25 *b2.mS,
        'power_n': 2.0,
    }

    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 1000 * b2.ms,  # 2500 * b2.ms,
        'dt': 0.05 * b2.ms,
    }

    # Figure 1e -----------------------------------------------------

    def plot():
        current_unit = b2.uA
        start_time = time()
        i_stim = [2, 4, 6]
        _, ax = plt.subplots(len(i_stim)+1, figsize=(10, 6), sharex=True)
        for ii in range(len(i_stim)):
            input_current = get_step_current(200,
                                             700,
                                             b2.ms,
                                             i_stim[ii] * current_unit)
            par['i_ext'] = input_current
            st_mon = simulate_FSI_cell(par, par_sim)
            plot_data(st_mon, ax[ii])
            plot_current(st_mon, ax[3], current_unit)
        print("Done in {:.3f}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('data/figure_1.png')
        plt.close()

    plot()
    clean_directory()
