import numpy as np
import pylab as plt
from time import time
from numpy import exp
from scipy.integrate import odeint
from lib_stn import *
import utility


def iahp(v, ca): return gahp * ca / (ca + k1) * (v - vk)
def ica(v): return gca * ((sinf(v)) ** 2) * (v - vca)
def it(v, r): return gt * (ainf(v) ** 3) * (binf(r) ** 2) * (v - vca)


def ode(x0, t):

    v, h, n, r, ca = x0

    dv = -(il(v) + ina(v, h) + ik(v, n) +
           iahp(v, ca) + ica(v) + it(v, r)) + Iapp
    dh = phi * (hinf(v) - h) / tauh(v)
    dn = phi * (ninf(v) - n) / taun(v)
    dr = phir * (rinf(v) - r) / taur(v)
    dca = eps * (-ica(v) - it(v, r) - kca * ca) # * phi

    return [dv, dh, dn, dr, dca]


if __name__ == "__main__":

    start = time()

    dt = 0.01
    t = np.arange(0, 1000, dt)
    x0 = init(-55.0)

    I = [0, 180] # np.arange(0, 180 + 1, 10)
    freq = []
    for i in range(len(I)):
        Iapp = I[i]
        sol = odeint(ode, x0, t)

        v = sol[:, 0]
        tspikes = utility.spikeDetection(dt, v, -20.0)
        if len(tspikes) > 1:
            isi = np.diff(tspikes)
            firing_rate = 1.0 / np.mean(isi) * 1000.0
            print("I = {:.1f}, firing rate is {:.2f} Hz".format(Iapp, firing_rate))
            freq.append(firing_rate)
        else:
            freq.append(0)
    freq_gca_gt_0 = []
    gca = 0.0
    for i in range(len(I)):
        Iapp = I[i]
        sol = odeint(ode, x0, t)

        v = sol[:, 0]
        tspikes = utility.spikeDetection(dt, v, -20.0)
        if len(tspikes) > 1:
            isi = np.diff(tspikes)
            firing_rate = 1.0 / np.mean(isi) * 1000.0
            print("I = {:.1f}, firing rate is {:.2f} Hz".format(Iapp, firing_rate))
            freq_gca_gt_0.append(firing_rate)
        else:
            freq_gca_gt_0.append(0)

    utility.display_time(time() - start)

    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.plot(I, freq, c='k', lw=2, label="control")
    ax.plot(I, freq_gca_gt_0, lw=2, label=r"$g_{Na}=0$")

    ax.set_xlabel("t [ms]")
    ax.set_ylabel("frequency [Hz]")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figs/figure1c.png")


