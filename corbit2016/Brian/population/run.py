import os
import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (clean_directory,
                 simulate_circuit,
                 plot_h,
                 plot_m,
                 plot_data,
                 plot_current)
from input_factory import get_step_current

if not os.path.exists("data"):
    os.makedirs("data")

if __name__ == "__main__":

    par_M = {
        'num': 1,
        'v0': -40*b2.mV,
        'Cm': 1 * b2.uF,
        # 'Iapp': 1.67 * b2.uA,
        'eNa': 55*b2.mV,
        'gnabar': 35*b2.mS,
        'eK': -90*b2.mV,
        'gkbar': 6*b2.mS,
        'eLeak': -75*b2.mV,
        'gleak': 0.075*b2.mS,
        'eKir': -90*b2.mV,
        'gkirbar': 0.15*b2.mS,
        'tau_m_Kir': 0.01*b2.ms,
        'eKaf': -73*b2.mV,
        'gkafbar': 0.09 * b2.mS,
        'q10': 2.5,
        'eKas': -85*b2.mV,
        'gkasbar': 0.32*b2.mS,
        'eKrp': -77.5*b2.mV,
        'gkrpbar': 0.42*b2.mS,
        'eNap': 45*b2.mV,
        'gnapbar': 0.02*b2.mS,
        'eNas': 40*b2.mV,
        'gnasbar': 0.11*b2.mS,
        'tadj': 3.952847075210474,  # 2.5**((37-22)/10.)
        'tadj_Nas': 4.3321552697196655,  # 2.5**((37-21)/10.)
    }

    par_G = {
        'num': 1,
        'v0': -40*b2.mV,
        'Cm': 1 * b2.uF,

        'eNa': 50*b2.mV,
        'eLeak': -60*b2.mV,
        'eK': -90.*b2.mV,
        'eCa': 130 * b2.mV,
        'eCat': -30*b2.mV,

        'gnafbar': 50*b2.mS,
        'gnapbar': 0.1*b2.mS,
        'gkv2bar': 0.1*b2.mS,
        'gkv3bar': 10.*b2.mS,
        'gleak': 0.068*b2.mS,
        # 'gkvfbar':2.*b2.mS,  # ?
        'gkv4sbar': 3.*b2.mS,  # Kv4
        'gkcbar': 0.15*b2.mS,  # KCNQ
        'ghcnbar': 0.1*b2.mS,
        'gcahbar': 0.3*b2.mS,
        'gkskbar': 0.4*b2.mS,
        'Gc_Ca_sat': 5.0,
        'Gcan50': 0.01**4.6,
    }

    par_F = {
        'num': 1,
        'v0': -60*b2.mV,
        'Cm': 1 * b2.uF,
        'Iapp': 3.35 * b2.uA,
        'gA': 0.39 * b2.mS,
        'gNa': 112.5 * b2.mS,
        'gK': 225.0 * b2.mS,
        'gL': 0.25 * b2.mS,
        'power_n': 2.0,
    }

    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 1000 * b2.ms,  # 2500 * b2.ms,
        'dt': 0.01 * b2.ms,
    }

    par_M['record_from'] = ["vm", "M_Iapp", "M_h_Kas", "M_m_Kas"]
    par_G['record_from'] = ['vg', 'G_Iapp']
    par_F['record_from'] = ['vf', 'F_Iapp']
    par = {'par_M':par_M, 
           'par_F':par_F,
           'par_G':par_G,
           'par_sim':par_sim,
           }

    # Figure 1e -----------------------------------------------------

    def plot():
        current_unit = b2.uA
        start_time = time()
        i_stim = [1.67]
        _, ax = plt.subplots(len(i_stim)+3, figsize=(10, 7), sharex=True)
        for ii in range(len(i_stim)):

            input_current = b2.TimedArray([0, i_stim[ii],
                                           0, i_stim[ii],
                                           0]*b2.uA,
                                          dt=200*b2.ms)
            # input_current = get_step_current(50,
            #                                  200,
            #                                  b2.ms,
            #                                  i_stim[ii] * current_unit)
            par_M['iapp'] = input_current
            st_mon = simulate_circuit(par)
            plot_data(st_mon, ax[ii], lw=1, c='k')
            plot_h(st_mon, ax[-2])
            plot_m(st_mon, ax[-3])
            plot_current(st_mon, ax[-1], current_unit)
        print("Done in {:.3f}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('data/figure_1.png')
        plt.close()
    


    plot()

    clean_directory()
