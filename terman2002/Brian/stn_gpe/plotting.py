import os
from matplotlib import markers
import numpy as np
import brian2 as b2
import pylab as plt
from os.path import join

if not os.path.exists("data/figs"):
    os.makedirs("data/figs")


def plot_voltage(monitors, indices, filename):
    st_mon_s, st_mon_g = monitors[:2]

    fig, ax = plt.subplots(nrows=2, ncols=len(indices), figsize=(15, 4), sharex=True)

    for i in indices:
        ax[i, 0].plot(st_mon_s.t / b2.ms,
                    st_mon_s.vs[i] / b2.mV, lw=1, 
                    label="STN-{:d}".format(i+1), alpha=0.5)
        ax[i, 0].set_ylabel("STN, v [mV]", fontsize=14)
        ax[i, 0].legend(frameon=False)

    for i in indices:
        ax[i, 1].plot(st_mon_g.t / b2.ms,
                st_mon_g.vg[i] / b2.mV, lw=1, 
                label="GPe-{:d}".format(i+1), alpha=0.5)
        ax[i, 1].set_ylabel("GPe, v [mV]", fontsize=14)
        ax[i, 1].legend(frameon=False)

    
    ax[0, 0].set_xlim(0, np.max(st_mon_s.t / b2.ms))
    ax[-1, 0].set_xlabel("time [ms]", fontsize=14)
    ax[-1, 1].set_xlabel("time [ms]", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(join("data/figs", '{}.png'.format(filename)))
    # plt.show()


def plot_voltage2(monitors, indices, filename, alpha=1):
    st_mon_s, st_mon_g = monitors[:2]


    fig, ax = plt.subplots(3, figsize=(8, 5), sharex=True)

    for i in indices:
        ax[0].plot(st_mon_s.t / b2.ms,
                    st_mon_s.vs[i] / b2.mV, lw=1, 
                    label="STN-{:d}".format(i+1), alpha=alpha)
        ax[0].set_ylabel("STN, v [mV]", fontsize=14)
        ax[0].legend(frameon=False)

    for i in indices:
        ax[1].plot(st_mon_g.t / b2.ms,
                st_mon_g.vg[i] / b2.mV, lw=1, 
                label="GPe-{:d}".format(i+1), alpha=alpha)
        ax[1].set_ylabel("GPe, v [mV]", fontsize=14)
        ax[1].legend(frameon=False)
    
    for i in indices:
        ax[2].plot(st_mon_s.t / b2.ms,
                st_mon_s.i_syn_GtoS[i] / b2.pA, lw=1, 
                label="{:d}".format(i+1), alpha=alpha)
        ax[2].set_ylabel("I_syn [pA]", fontsize=14)
        ax[2].legend(frameon=False)

    
    ax[0].set_xlim(0, np.max(st_mon_s.t / b2.ms))
    ax[1].set_xlabel("time [ms]", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(join("data/figs", '{}.png'.format(filename)))
    plt.show()


def plot_raster(monitors, filename="spikes", markersize=2):
    sp_mon_s, sp_mon_g = monitors[2:]
    fig, ax = plt.subplots(2, figsize=(9, 4), sharex=True)
    ax[0].plot(sp_mon_s.t / b2.ms,
               sp_mon_s.i, "bo",
               ms=markersize)
    ax[1].plot(sp_mon_g.t / b2.ms,
               sp_mon_g.i, "ro",
               ms=markersize)

    ax[1].set_xlabel("time [ms]")
    ax[0].set_ylabel("STN neuron id")
    ax[1].set_ylabel("GPe neuron id")
    plt.tight_layout()
    plt.savefig(join("data/figs", '{}.png'.format(filename)))
    # plt.show()
