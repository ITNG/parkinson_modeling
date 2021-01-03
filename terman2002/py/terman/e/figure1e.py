import numpy as np
import pylab as plt
from numpy import exp
from scipy.integrate import odeint
import lib_stn as lb



def I_app(t):
    if t < 500:
        return 0.0
    elif t < 800:
        return 25.0
    else:
        return 0

def ode(x0, t):

    v, h, n, r, ca = x0

    dv = -(lb.il(v) + lb.ina(v, h) + lb.ik(v, n) +
           lb.iahp(v, ca) + lb.ica(v) + lb.it(v, r)) + I_app(t)  # - isyn
    dh = lb.phi * (lb.hinf(v) - h) / lb.tauh(v)
    dn = lb.phi * (lb.ninf(v) - n) / lb.taun(v)
    dr = lb.phir * (lb.rinf(v) - r) / lb.taur(v)
    dca = lb.eps * (-lb.ica(v) - lb.it(v, r) - lb.kca * ca) # * phi

    return [dv, dh, dn, dr, dca]


if __name__ == "__main__":

    t = np.arange(0, 2500, 0.01)
    x0 = lb.init(-55.0)
    sol = odeint(ode, x0, t)

    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.plot(t, sol[:, 0], c='k', lw=2)
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("V [mv]")
    plt.tight_layout()
    plt.savefig("figs/figure1e.png")


