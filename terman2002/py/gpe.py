import numpy as np
import pylab as plt
from numpy import exp
from scipy.integrate import odeint
from lib_gpe import *


if __name__ == "__main__":

    t = np.arange(0, 1000, 0.01)
    x0 = init(-55.0)
    sol = odeint(ode_sys, x0, t)

    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.plot(t, sol[:, 0], c='k', lw=2)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("V [mv]")
    plt.tight_layout()
    plt.savefig("figs/gpe.png")
