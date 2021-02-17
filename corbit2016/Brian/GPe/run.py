import os
import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (clean_directory, simulate_GPe_cell,
                 plot_data, plot_current)
from input_factory import get_step_current

if not os.path.exists("data"):
    os.makedirs("data")

if __name__ == "__main__":

    par = {
        'num': 1,
        'v0': -40*b2.mV,
        'Cm': 1 * b2.uF,
        
        'eNa': 50*b2.mV,
        'eLeak' : -60*b2.mV,
        'eK' : -90.*b2.mV,
        'eCa': 130 *b2.mV,
        'eCat' : -30*b2.mV,
        
        'gnafbar': 50*b2.mS,
        'gnapbar': 0.1*b2.mS,
        'gkv2bar':0.1*b2.mS,
        'gkv3bar':10.*b2.mS,
        'gleak' : 0.068*b2.mS,
        # 'gkvfbar':2.*b2.mS,  # ?
        'gkv4sbar':3.*b2.mS, # Kv4
        'gkcbar':0.15*b2.mS, # KCNQ
        'ghcnbar':0.1*b2.mS,
        'gcahbar':0.3*b2.mS,
        'gkskbar':0.4*b2.mS,
        'Gc_Ca_sat': 5.0,
        'Gcan50' : 0.01**4.6,
    }

    par['record_from'] = ["vg", "Iapp"]

    par_sim = {
        'integration_method': "rk2",
        'simulation_time': 300 * b2.ms,  # 2500 * b2.ms,
        'dt': 0.01 * b2.ms,
    }

    # Figure 1e -----------------------------------------------------

    def plot():
        current_unit = b2.uA
        start_time = time()
        i_stim = [0.]
        _, ax = plt.subplots(len(i_stim), figsize=(10, 3.5))
        for ii in range(len(i_stim)):

            # input_current = b2.TimedArray([0, i_stim[ii],
            #                                0]*b2.uA,
            #                               dt=40*b2.ms)            
            # input_current = get_step_current(50,
            #                                  200,
            #                                  b2.ms,
            #                                  i_stim[ii] * current_unit)
            par['iapp'] = i_stim[ii]*current_unit
            st_mon = simulate_GPe_cell(par, par_sim)
            plot_data(st_mon, ax, lw=1, c='k')
            # plot_current(st_mon, ax[-1], current_unit)
        print("Done in {:.3f}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('data/figure_1.png')
        plt.close()

    plot()
    clean_directory()
