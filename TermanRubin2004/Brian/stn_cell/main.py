import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (simulate_STN_cell,
                 simulate_2_STN_cell_biexp,
                 simulate_2_STN_cell_simpl_biexp,
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
        'gl': 2.25 * b2.nS,
        'gna': 37.5 * b2.nS,
        'gk': 45 * b2.nS,
        'gahp': 9 * b2.nS,
        'gca': 0.5 * b2.nS,
        'gt': 0.5 * b2.nS,
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
        'sigmab': -0.1,  # Guo 0.07 Terman02 -0.1
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
        'eps': 3.75e-05 / b2.ms,  # 1/ms Guo 0.00005 Terman02 0.0000375
        'alpha': 5./b2.ms,
        'beta': 1./b2.ms,
        'phir': 0.2,  # Guo 0.5 Terman02 0.2
        'phi': 0.75,
        'kca': 22.5,
        'thn': -80,
        'thh': -57,
        'thr': 68,
        'ab': -30,
        'k1': 15,
        # 'i_ext': 0 * b2.uamp,
        'C': 1 * b2.pF,
        'v_rev_ss': 0. * b2.mV,
        'w': 1.0,  # *b2.nS
        'w_ss':1.0*b2.nS,
        'thetag':30.,
        'thetagH':-39.,
        'sigmagH':8.,
    }
    tau_1 = 1.0/par['alpha']
    tau_2 = 1.0/par['beta']
    par['scale_f'] = (tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))

    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 2500 * b2.ms,  # 2500 * b2.ms,
        'dt': 0.05 * b2.ms,
    }

    

    # Figure 1e -----------------------------------------------------
    def plot_fig_1e():
        start_time = time()
        fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
        t_stim = [300, 450, 600]
        for ii in range(len(t_stim)):
            input_current = get_step_current(500,
                                             500 + t_stim[ii],
                                             b2.ms,
                                             -25 * b2.pA)
            par['i_ext'] = input_current
            st_mon = simulate_STN_cell(par, par_sim)
            plot_data(st_mon, ax[ii])
            plot_current(st_mon, ax[3])
        print("Done in {}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('figure_1e.png')
        plt.close()

    # Figure 1f -----------------------------------------------------
    def plot_fig_1f():
        current_unit = b2.pA
        start_time = time()
        fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
        i_stim = [-20, -30, -40]
        for ii in range(len(i_stim)):
            input_current = get_step_current(500,
                                             800,
                                             b2.ms,
                                             i_stim[ii] * current_unit)
            par['i_ext'] = input_current
            st_mon = simulate_STN_cell(par, par_sim)
            plot_data(st_mon, ax[ii])
            plot_current(st_mon, ax[3], current_unit)
        print("Done in {}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('figure_1f.png')
        plt.close()

    def two_stn_cell_simpl():
        current_unit = b2.pA
        par_sim['simulation_time'] = 350. * b2.ms
        start_time = time()
        fig, ax = plt.subplots(3, figsize=(10, 5), sharex=True)
        i_stim = [0.]
        for ii in range(len(i_stim)):
            
            st_mon = simulate_2_STN_cell_simpl_biexp(par, par_sim)
            plot_data(st_mon, ax[ii], index=1)
            ax[0].plot(st_mon.t / b2.ms, st_mon.vs[0] /
                       b2.mV, lw=2, c='b', label="0")
            ax[0].plot(st_mon.t / b2.ms, st_mon.vs[1] /
                       b2.mV, lw=2, c='r', label="1")
            ax[1].plot(st_mon.t/b2.ms, st_mon.g_syn_ss[1]/b2.nS, lw=2, c='k')
            ax[0].legend()
            ax[0].set_xlim(0, np.max(st_mon.t / b2.ms))
            ax[0].set_ylabel("v [mV]", fontsize=14)
            ax[1].set_ylabel("g_syn [mV]", fontsize=14)

            ax[-1].plot(st_mon.t / b2.ms,
                        st_mon.i_syn_ss[1] / current_unit, lw=1, c='k', alpha=0.5)
            ax[-1].set_xlabel("t [ms]", fontsize=14)
            ax[-1].set_ylabel("I [{}]".format(str(current_unit)), fontsize=14)
            ax[-1].set_xlim(300, 330)
        
        print("Done in {}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('figure_2_stn_simpl.png')
        plt.show()
        # plt.close()
    
    def two_stn_cell():
        current_unit = b2.pA
        par_sim['simulation_time'] = 2000 * b2.ms
        start_time = time()
        fig, ax = plt.subplots(3, figsize=(10, 5), sharex=True)
        i_stim = [0.]
        for ii in range(len(i_stim)):
            
            st_mon = simulate_2_STN_cell_biexp(par, par_sim)
            plot_data(st_mon, ax[ii], index=1)
            ax[0].plot(st_mon.t / b2.ms, st_mon.vs[0] /
                       b2.mV, lw=2, c='b', label="0")
            ax[0].plot(st_mon.t / b2.ms, st_mon.vs[1] /
                       b2.mV, lw=2, c='r', label="1")
            ax[1].plot(st_mon.t/b2.ms, st_mon.s_ss[0], lw=2, c='b', label="0")
            ax[1].plot(st_mon.t/b2.ms, st_mon.s_ss[1], lw=2, c='r', label="1")
            ax[0].legend()
            ax[1].legend()
            ax[0].set_xlim(0, np.max(st_mon.t / b2.ms))
            ax[0].set_ylabel("v [mV]", fontsize=14)
            ax[1].set_ylabel("s_syn", fontsize=14)

            ax[-1].plot(st_mon.t / b2.ms,
                        st_mon.i_syn_ss[1] / current_unit, lw=1, c='k', alpha=0.5)
            ax[-1].set_xlabel("t [ms]", fontsize=14)
            ax[-1].set_ylabel("I_syn [{}]".format(str(current_unit)), fontsize=14)
            # ax[-1].set_xlim(300, 330)
        
        print("Done in {}".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('figure_2_stn.png')
        plt.show()
        # plt.close()

    # plot_fig_1e()
    # plot_fig_1f()
    # two_stn_cell_simpl()
    two_stn_cell()
