import numpy as np
import pylab as plt
from numpy import exp
from scipy.integrate import odeint
from lib_stn import *




if __name__ == "__main__":

    v = np.arange(-90, -21, 0.1)

    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.plot(v, ina(v, hinf(v)), label="Na")
    ax.plot(v, ica(v), label="Ca")
    ax.plot(v, ik(v, ninf(v)), label="K")
    # ax.plot(v, it(v, rinf(v)), label="T")
    # ax.plot(v, il(v), label="L")
    ax.set_xlabel("V [mV]")
    ax.set_ylabel("I [pA]")
    plt.legend()
    plt.ylim(-100, 100)
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig("figs/1a.png")
    plt.show()
