import numpy as np
import brian2 as b2
import pylab as plt
from os.path import join


def plot_voltage(monitors, par_s, par_g, par_syn):
    st_mon_s, st_mon_g, sp_mon_s, sp_mon_g = monitors

    fig, ax = plt.subplots(2, figsize=(15, 4), sharex=True)

    # for i in range(par_s['num']):
    for i in range(2):
        ax[0].plot(st_mon_s.t / b2.ms,
                   st_mon_s.vs[i] / b2.mV, lw=1, label="STN", alpha=0.5)

    # for i in range(par_g['num']):
    for i in range(2):    
        ax[1].plot(st_mon_g.t / b2.ms,
                   st_mon_g.vg[i] / b2.mV, lw=1, label='GPe', alpha=0.5)

    ax[00].set_xlim(0, np.max(st_mon_s.t / b2.ms))
    ax[1].set_xlabel("time [ms]", fontsize=14)
    ax[0].set_ylabel("STN, v [mV]", fontsize=14)
    ax[1].set_ylabel("GPE, v [mV]", fontsize=14)
    # ax[0].legend(frameon=False)
    # ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(join("figs", 'voltage.png'))
    # plt.show()


def plot_raster(monitors, par_s, par_g, par_syn):
    sp_mon_s, sp_mon_g = monitors[2:]
    fig, ax = plt.subplots(2, figsize=(9, 4), sharex=True)
    ax[0].plot(sp_mon_s.t / b2.ms,
               sp_mon_s.i, "k.",
               ms=3)
    ax[1].plot(sp_mon_g.t / b2.ms,
               sp_mon_g.i, "k.",
               ms=3)

    ax[1].set_xlabel("time [ms]")
    ax[0].set_ylabel("#neuron index")
    ax[1].set_ylabel("#neuron index")
    plt.tight_layout()
    plt.savefig(join("figs", 'sparse.png'))
    # plt.show()
