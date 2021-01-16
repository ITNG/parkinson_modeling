import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from os.path import join
from lib import plot_current, simulate_Thl_cell, plot_data, make_grid
from matplotlib.gridspec import GridSpec

par = {
    'v0': -65.*b2.mV,
    'cmthl': 1*b2.pF,
    'glthl': 0.05*b2.nS,
    'gkthl': 5*b2.nS,
    'gtthl': 5*b2.nS,
    'gnathl': 3*b2.nS,

    'vtthl': 0 * b2.mV,
    'vlthl': -70 * b2.mV,
    'vkthl': -90 * b2.mV,
    'vnathl': 50 * b2.mV,

    'sigmthl': 7,
    'sighthl': 4,
    'sigbhthl': 5,
    'sigahthl': 18,
    'sigpthl': 6.2,
    'sigrthl': 4,

    'thtmthl': -37,
    'thththl': -41,
    'thtahthl': -46,
    'thtbhthl': -23,
    'thtpthl': -60,
    'thtrthl': -84,

    'taur0thl': 28*b2.ms,
    'ah0thl': 0.128,
    'bh0thl': 4,
    'phihthl': 1,
    'iext': 0.45*b2.pA,

}
# sensorimotor control
par_SM = {
    'imsmthl': 8,
    'tmsmthl': 25,
    'wsmthl': 5,
    'dsmthl': 80,
    'sigym': 0.001,
}

T_CURRENT = 'fast'

if T_CURRENT == 'fast':
    par['taur1thl'] = 1*b2.ms
    par['thtrtauthl'] = -25.
    par['sigrtauthl'] = 10.5
    par['phirthl'] = 2.5

par_sim = {
    'integration_method': "rk4",
    'simulation_time': 1000 * b2.ms,  # 2500 * b2.ms,
    'dt': 0.05 * b2.ms,
    'num': 1,
}

if __name__ == "__main__":


    def figure2():

        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(6, 1)
        gs.update(left=0.1, right=0.95)
        ax1 = plt.subplot(gs[:2])
        ax2 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[3:5])
        ax4 = plt.subplot(gs[5])
        ax = [ax1, ax2, ax3, ax4]
        
        start_time = time()

        I_sm1 = b2.TimedArray([0.0, 2., 0., 5., 0., 10., 0.]*b2.pA, dt=100*b2.ms)
        par_sim['I_sm'] = I_sm1
        par['i_ext'] = 0.0
        state_monitor = simulate_Thl_cell(par, par_sim)
        print("Done in {}".format(time() - start_time))

        plot_data(state_monitor, ax[0])
        plot_current(state_monitor, ax[1])



        I_sm2 = b2.TimedArray([0.0, -0.5, 0., -1., 0., ]*b2.pA, dt=150*b2.ms)
        par_sim['I_sm'] = I_sm2
        par['i_ext'] = 0.0
        state_monitor = simulate_Thl_cell(par, par_sim)
        print("Done in {}".format(time() - start_time))

        plot_data(state_monitor, ax[2])
        plot_current(state_monitor, ax[3], xlabel="Time [ms]")


        [ax[i].set_xticks([]) for i in range(3)]
        [ax[i].set_xlim([0, 1000]) for i in range(4)]
        plt.savefig(join("data", "figure2.png"))
        plt.show()



    figure2()