import lib
import torch
import numpy as np
import pylab as plt
from numpy import exp
from copy import copy
from config import sim_params, true_params
from scipy.integrate import odeint
from scipy.stats import (kurtosis, skew)


# ------------------------------------------------------------------#

def simulation_wrapper(params):
    '''
    Returns summary statistics from conductance values in `params`.
    Summarizes the output of the simulator and converts it 
    to `torch.Tensor`.
    '''
    obs = simulator(sim_params=sim_params, params=params)
    stats = torch.as_tensor(calculate_summary_statistics(obs))
    return stats
# ------------------------------------------------------------------#


def simulator(sim_params, params):

    dt = sim_params['dt']
    num = sim_params['num']
    t_simulation = sim_params['t_simulation']

    I_C = sim_params['I_C']
    I_Str = sim_params['I_Str']

    if torch.is_tensor(params):
        par = np.float64(params.numpy())
    else:
        par = copy(params)

    wSG, wGS, wCS, wSC, wGG, wCC = par

    sol = lib.DDE()
    sol.set_params(t_simulation,
                   dt,
                   I_C,
                   I_Str,
                   wSG,
                   wGS,
                   wCS,
                   wSC,
                   wGG,
                   wCC
                   )
    sol.set_history([0] * num)
    sol.euler_integrator()
    y = sol.y

    nstart = 50
    # drop history:
    for i in range(len(y)):
        y[i] = y[i][nstart+1:]

    return dict(t=sol.t_ar[nstart+1:], data=np.asarray(y))
# ------------------------------------------------------------------


def calculate_summary_statistics(obs):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """

    t = obs["t"]
    dt = sim_params["dt"]

    # initialise array of spike counts
    S = obs["data"][0, :]
    G = obs["data"][1, :]
    E = obs["data"][2, :]
    I = obs["data"][3, :]

    s = np.mean(S), np.std(S), skew(S), kurtosis(S)
    g = np.mean(G), np.std(G), skew(G), kurtosis(G)
    e = np.mean(E), np.std(E), skew(E), kurtosis(E)
    i = np.mean(I), np.std(I), skew(I), kurtosis(I)

    sum_stats_vec = np.concatenate((s, g, e, i))

    return sum_stats_vec


# ------------------------------------------------------------------


def plot_data(obs, ax=None):

    t = obs['t']
    y = obs['data']
    num = y.shape[0]

    save_fig = False
    labels = ["STN", "GPE", "E", "I"]
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 4))
        save_fig = True

    for i in range(num):
        ax.plot(t, y[i, :], label=labels[i])

    ax.margins(x=0)
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Firing Rate (spk/ms")

    if save_fig:
        fig.savefig("data/fig.png")
# ------------------------------------------------------------------


def display_time(time):
    ''' 
    show real time elapsed
    '''
    hour = time//3600
    minute = int(time % 3600) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print("Done in %d hours %d minutes %.4f seconds"
          % (hour, minute, second))

