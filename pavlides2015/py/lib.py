import numpy as np
import pylab as plt
from numpy import exp
from time import time
from numpy import pi, sin
from numpy.random import normal, rand
from main import *

# -----------------------------------------------------------------------------


def main():
    initial_condition = [0] * num
    max_dalay = np.max(delays)

    lib.set_history(initial_condition, nstart, max_dalay)
    lib.euler(lib.sys_eqs, dt)
# -----------------------------------------------------------------------------


def FS(x): return Ms / (1 + (Ms - Bs) / Bs * exp(-4 * x / Ms))


def FG(x): return Mg / (1 + (Mg - Bg) / Bg * exp(-4 * x / Mg))


def FE(x): return Me / (1 + (Me - Be) / Be * exp(-4 * x / Me))


def FI(x): return Mi / (1 + (Mi - Bi) / Bi * exp(-4 * x / Mi))
# -----------------------------------------------------------------------------


def sys_eqs(t, n):

    dS = 1.0 / tau_S * (FS(wCS * interp_y(t - T_CS, 2, n) - wGS * interp_y(t-T_GS, 1, n)) - y[0, n])
    dG = 1.0 / tau_G * (FG(wSG * interp_y(t - T_SG, 0, n) - wGG * interp_y(t - T_GG, 1, n) - Str) - y[1, n])
    dE = 1.0 / tau_E * (FE(-wSC * interp_y(t - T_SC, 0, n) - wCC * interp_y(t - T_CC, 3, n) + C) - y[2, n])
    dI = 1.0 / tau_I * (FI(wCC * interp_y(t-T_CC, 2, n)) - y[3, n])

    return np.array([dS, dG, dE, dI])
# -----------------------------------------------------------------------------


def interp_y(t0, index, n):
    assert(t0 <= t[n])
    return (np.interp(t0, t[:n], y[index, :n]))


def set_history(hist, nstart, maxdelay):

    for i in range(nstart+1):
        t[i] = -(nstart - i) / float(nstart) * maxdelay

    # x is: N x nstep
    for i in range(num):
        y[i, :(nstart+1)] = hist[i]


def euler(f, h):

    for i in range(nstart, nstart + num_iterarion):
        dy = h * f(t[i], i)
        y[:, i+1] = y[:, i] + dy
        t[i+1] = (i - nstart + 1) * dt

    np.savez("data/euler_interp", t=t, y=y)
# -----------------------------------------------------------------------------


def plot_data():

    fig, ax = plt.subplots(1, figsize=(7, 3.5))
    data = np.load("data/euler_interp.npz")
    t = data["t"]
    y = data["y"]

    ax.plot(t, y[0, :], label="STN", lw=1, c="b")
    ax.plot(t, y[1, :], label="GPE", lw=1, c='g')
    ax.plot(t, y[2, :], label="E", lw=1, c='r')
    ax.plot(t, y[3, :], label="I", lw=1, c='cyan')

    # print t[-1], y[:, -1]
    ax.legend(loc="upper right")
    ax.set_xlabel("time (s)", fontsize=13)
    ax.set_ylabel("Firing rate (spk/s)", fontsize=13)
    ax.margins(x=0)
    plt.tight_layout()
    fig.savefig("data/dde_euler_interp.png", dpi=150)
# -----------------------------------------------------------------------------

# def generate_data(t, par):

#     # t in second
#     b = par['b']    # spk/s
#     A = par['A']    # spk/s
#     f = par['f']    # Hz
#     N = par['N']    
#     dt = par['dt']  # second

#     nstep = int(round(t / dt))
#     theta = np.zeros(nstep)
    
#     for i in range(1, nstep):
#         theta[i] += 2 * pi * f * dt + N * normal(0, dt)
    
#     fr = A * sin(theta) + b

#     spike_train = np.zeros(nstep)
#     # rand_num = rand(nstep)
    
#     for i in range(nstep):
#         if rand() < fr[i]:
#             spike_train[i] = 1
    
    



    
