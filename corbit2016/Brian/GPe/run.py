import os
import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (clean_directory, simulate_GPe_cell,
                 plot_data, plot_current,
                 plot_channel_currents)
from input_factory import get_step_current

if not os.path.exists("data"):
    os.makedirs("data")

if __name__ == "__main__":

    par = {
        'num': 1,
        'v0': -60*b2.mV,
        'Cm': 1 * b2.uF,

        'eNa': 50*b2.mV,
        'eLeak': -60*b2.mV,
        'eK': -90.*b2.mV,
        'eCa': 130 * b2.mV,
        'eCat': -30*b2.mV,

        'gnafbar': 50*b2.mS,
        'gnapbar': 0.1*b2.mS,
        'gkv2bar': 0.1*b2.mS,
        'gkv3bar': 10*b2.mS,
        'gleak': 0.068*b2.mS,
        # 'gkv4fbar':2.*b2.mS,
        'gkv4sbar': 3*b2.mS,  # Kv4
        'gkcnqbar': 0.15*b2.mS,
        'ghcnbar': 0.1*b2.mS,
        'gcahbar': 0.3*b2.mS,
        'gskbar': 0.4*b2.mS,
        # 'Gc_Ca_sat': 5.0,
        'Gcan50': (0.35*b2.mM)**4.6,
    }

    par_sim = {
        'ADD_SPIKE_MONITOR': False,
        'integration_method': "rk2",
        'simulation_time': 400 * b2.ms,  # 2500 * b2.ms,
        'dt': 0.01 * b2.ms,
    }

    def plot_details(filename='figure_1.png'):
        par['record_from'] = ["vg", "Iapp",
                              "iNap", "iNaf",
                              'iKv2', 'iKv3',
                              'iKv4s', 'iCah',
                              'iSk', 'iKcnq']
        par_sim['simulation_time'] = 100 * b2.ms
        current_unit = b2.uA
        start_time = time()
        i_stim = [2.0]
        _, ax = plt.subplots(3, figsize=(6, 8), sharex=True)
        for ii in range(len(i_stim)):
            input_current = get_step_current(50,
                                             100,
                                             b2.ms,
                                             i_stim[ii] * current_unit)
            par['iapp'] = input_current
            st_mon = simulate_GPe_cell(par, par_sim)
            plot_data(st_mon, ax[ii], lw=1, c='k')
            # plot_current(st_mon, ax[-2], current_unit)
            plot_channel_currents(st_mon, ax[1:], current_unit)

        ax[0].set_xlim(75, 90)
        ax[1].set_xlim(75, 90)
        ax[2].set_xlim(75, 90)
        print("Done in {:.3f}".format(time() - start_time))
        plt.savefig('data/{}'.format(filename))
        plt.close()

    def plot(filename='figure_0.png'):
        par['record_from'] = ["vg", "Iapp"]
        par_sim['simulation_time'] = 400.0 * b2.ms
        current_unit = b2.uA
        start_time = time()
        i_stim = [-1, -2, 1, 2]
        _, ax = plt.subplots(len(i_stim)+1, figsize=(10, 8), sharex=True)
        for ii in range(len(i_stim)):
            input_current = get_step_current(100,
                                             300,
                                             b2.ms,
                                             i_stim[ii] * current_unit)
            par['iapp'] = input_current
            st_mon = simulate_GPe_cell(par, par_sim)
            plot_data(st_mon, ax[ii], lw=1, c='k')
            plot_current(st_mon, ax[-1], current_unit)
        ax[0].set_xlim(0, 400)
        print("Done in {:.3f}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('data/{}'.format(filename))
        plt.close()

    def plot_IF(filename="IF.png"):
        duration = 2 *b2.second
        par_sim['ADD_SPIKE_MONITOR'] = True
        par['record_from'] = ["vg"]
        par_sim['simulation_time'] = duration
        current_unit = b2.uA
        start_time = time()
        i_stim = np.linspace(0, 1, 30) * 10
        par['num'] = len(i_stim)
        firing_rate = []

        _, ax = plt.subplots(1, figsize=(6, 4))
        # par['iapp'] = input_current
        par['iapp'] = "i * ({}) * uA/(N-1)".format(max(i_stim))
        sp_mon, group = simulate_GPe_cell(par, par_sim)
        firing_rate = sp_mon.count / duration

        print("Done in {:.3f}".format(time() - start_time))

        ax.plot(group.Iapp/b2.uA, firing_rate,
                lw=1, marker="o", color='k')
        ax.set_xlim(-2, max(i_stim))
        ax.set_xlabel("I(uA)")
        ax.set_ylabel("Firing rate(sp/s)")
        plt.tight_layout()
        plt.savefig('data/{}'.format(filename))
        plt.close()

    # plot()
    # plot_IF()
    plot_details()
    clean_directory()
