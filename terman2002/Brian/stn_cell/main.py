import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (simulate_STN_cell,
                 plot_data,
                 plot_current)
from input_factory import get_step_current

if __name__ == "__main__":

    par = {
        'num': 1,
        'v0': -60*b2.mV,
        'vl': -60 * b2.mV,
        'vna': 55 * b2.mV,
        'vk': -80 * b2.mV,
        'vca': 140 * b2.mV,
        'gl': 2.25 * b2.msiemens,
        'gna': 37.5 * b2.msiemens,
        'gk': 45 * b2.msiemens,
        'gahp': 9 * b2.msiemens,
        'gca': 0.5 * b2.msiemens,
        'gt': 0.5 * b2.msiemens,
        'thetam': -30,
        'thetas': -39,
        'thetah': -39,
        'thetan': -32,
        'thetab': 0.4,  # Guo 0.25 Terman02 0.4
        'thetaa': -63,
        'thetar': -67,
        'sigmas': 8,
        'sigmah': -3.1,
        'sigman': 8,
        'sigmaa': 7.8,
        'sigmab': -0.1, # Guo 0.07 Terman02 -0.1
        'sigmam': 15,
        'sigmar': -2,
        'sigmaht': -3,
        'sigmant': -26,
        'sigmart': -2.2,
        'taun0': 1 * b2.ms,
        'taun1': 100 * b2.ms,
        'taur0': 40.0 * b2.ms,  # 7.1, Terman02 40.0
        'taur1': 17.5 * b2.ms,
        'tauh0': 1 * b2.ms,
        'tauh1': 500 * b2.ms,
        'eps': 3.75e-05 / b2.ms, # 1/ms Guo 0.00005 Terman02 0.0000375
        'alpha': 5,
        'beta': 1,
        'phir': 0.2, # Guo 0.5 Terman02 0.2
        'phi': 0.75,
        'kca': 22.5,
        'thn': -80,
        'thh': -57,
        'thr': 68,
        'ab': -30,
        'k1': 15,
        # 'i_ext': 0 * b2.uamp,
        'C': 1 * b2.ufarad,
    }

    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 2500 * b2.ms,
        'dt': 0.05 * b2.ms,
    }

    start_time = time()

    # Figure 1e -----------------------------------------------------
    # fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
    # t_stim = [300, 450, 600]
    # for ii in range(len(t_stim)):
    #     input_current = get_step_current(500,
    #                                      500 + t_stim[ii],
    #                                      b2.ms,
    #                                      -25 * b2.uamp)
    #     par['i_ext'] = input_current
    #     st_mon = simulate_STN_cell(par, par_sim)
    #     plot_data(st_mon, ax[ii])
    #     plot_current(st_mon, ax[3])
    # plt.tight_layout()
    # plt.savefig('figure_1e.png')
    # plt.close()

    # Figure 1f -----------------------------------------------------
    fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
    i_stim = [-20, -30, -40]
    for ii in range(len(i_stim)):
        input_current = get_step_current(500,
                                         800,
                                         b2.ms,
                                         i_stim[ii] * b2.uamp)
        par['i_ext'] = input_current
        st_mon = simulate_STN_cell(par, par_sim)
        plot_data(st_mon, ax[ii])
        plot_current(st_mon, ax[3])
    plt.tight_layout()
    plt.savefig('figure_1f.png')
    plt.close()

    
    print("Done in {}".format(time() - start_time))
