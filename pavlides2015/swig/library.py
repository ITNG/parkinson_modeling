import lib
import torch
import numpy as np
import pylab as plt
from numpy import exp
from copy import copy
from scipy import fftpack
from scipy import signal
from sbi.analysis import pairplot
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
    results = simulator(sim_params=sim_params, params=params)
    stats = torch.as_tensor(statistics_prop(results,
                                            method=sim_params['statistics_method'])
                            )
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


def statistics_prop(obs, method='moments'):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator
    method : 'moments' or 'firing_rate'

    Returns
    -------
    np.array, summary statistics
    """

    t = obs["t"]
    fs = 1 / (t[-1] - t[-2]) * 1000  # frequency sampling [Hz]
    dt = sim_params["dt"]

    labels = ['S', 'G', 'E', 'I']

    # initialise array of spike counts
    # S = obs["data"][0, :]
    # G = obs["data"][1, :]
    # E = obs["data"][2, :]
    # I = obs["data"][3, :]

    if method == 'moments':
        n0 = 4  # number of features
        stats_vec = np.zeros(4*n0)

        for i in range(4):
            data = np.array([np.mean(obs["data"][i, :]),
                             np.std(obs["data"][i, :]),
                             skew(obs["data"][i, :]),
                             kurtosis(obs["data"][i, :])])
            stats_vec[i*n0:(i+1)*n0] = data

    else:
        n0 = 3  # number of features
        stats_vec = np.zeros(4*n0)
        fmax = np.zeros(4)

        for i in range(4):
            f, a = fft_1d_real(obs['data'][i, :], fs)
            fmax[i] = find_max_frequency(a, f, thr=0.005)

        for i in range(4):
            data = np.array([np.min(obs["data"][i, :]),
                             np.max(obs["data"][i, :]),
                             fmax[i]])
            stats_vec[i*n0:(i+1)*n0] = data

    return stats_vec


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
        ax.plot(t, y[i, :]*1000, label=labels[i])

    ax.margins(x=0)
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Firing Rate (spk/s)")
    plt.tight_layout()

    if save_fig:
        fig.savefig("data/fig.png")
# ------------------------------------------------------------------


def get_max_probability(samples_filename):

    try:
        samples = torch.load(samples_filename)
        samples = samples.numpy()
    except:
        print("no input file!")
        exit(0)

    dim = samples.shape[1]
    max_values = np.zeros(dim)

    for i in range(dim):
        n, bins = np.histogram(samples[:, i], bins=50, density=True)
        max_value = bins[np.argmax(n)]
        max_values[i] = max_value

    return max_values


def display_time(time):
    ''' 
    show real time elapsed
    '''
    hour = time//3600
    minute = int(time % 3600) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print("Done in %d hours %d minutes %.4f seconds"
          % (hour, minute, second))


def fft_1d_real(signal, fs):
    """
    fft from 1 dimensional real signal

    :param signal: [np.array] real signal
    :param fs: [float] frequency sampling in Hz
    :return: [np.array, np.array] frequency, normalized amplitude

    -  example:

    >>> B = 30.0  # max freqeuency to be measured.
    >>> fs = 2 * B
    >>> delta_f = 0.01
    >>> N = int(fs / delta_f)
    >>> T = N / fs
    >>> t = np.linspace(0, T, N)
    >>> nu0, nu1 = 1.5, 22.1
    >>> amp0, amp1, ampNoise = 3.0, 1.0, 1.0
    >>> signal = amp0 * np.sin(2 * np.pi * t * nu0) + amp1 * np.sin(2 * np.pi * t * nu1) +
            ampNoise * np.random.randn(*np.shape(t))
    >>> freq, amp = fft_1d_real(signal, fs)
    >>> pl.plot(freq, amp, lw=2)
    >>> pl.show()

    """

    N = len(signal)
    F = fftpack.fft(signal)
    f = fftpack.fftfreq(N, 1.0 / fs)
    mask = np.where(f >= 0)

    freq = f[mask]
    amplitude = 2.0 * np.abs(F[mask] / N)

    # plt.plot(freq, amplitude)
    # plt.show()
    # exit(0)

    return freq[1:], amplitude[1:]


def welch(sig, fs):

    f, amp = signal.welch(sig, fs, nperseg=256)

    # plt.plot(f, amp)
    # plt.show()
    # exit(0)

    return f, amp

def find_max_frequency(amp, frq, thr=0.005):

    i = np.argmax(amp)
    if amp[i] > thr:
        return frq[i]
    else:
        return 0

def use_pairplot(samples_filename, output_filename="data/infer.png"):

        try:
            samples = torch.load(samples_filename)
        except:
            print("no input file!")
            exit(0)

        fig, axes = pairplot(samples,
                             labels=['SG', 'GS', 'CS', 'SC', 'GG', 'CC'],
                             figsize=(10, 8),
                             points=true_params,
                             points_offdiag={'markersize': 6},
                             points_colors='r')
        fig.savefig(f"{output_filename}", dpi=150)
        plt.close()
