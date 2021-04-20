import numpy as np
import pylab as plt
from numpy import exp
from time import time
from numpy import pi, sin
from scipy import fftpack
from numpy.random import normal, rand
from run import *

# -----------------------------------------------------------------------------


class STN_GPE_CTX_Circuite:
    def __init__(self, parameters):
        par = parameters['par']
        par_simulation = parameters['par_simulation']
        self.dt = par_simulation['dt']
        self.num = par_simulation['num']
        delays = [par['T_CS'],
                  par['T_GS'],
                  par['T_SG'],
                  par['T_GG'],
                  par['T_SC'],
                  par['T_CC']]
        self.max_dalay = np.max(delays)
        self.itau_S = 1/par['tau_S']
        self.itau_G = 1/par['tau_G']
        self.itau_E = 1/par['tau_E']
        self.itau_I = 1/par['tau_I']
        self.Ms = par["Ms"]
        self.Mg = par["Mg"]
        self.Bs = par["Bs"]
        self.Bg = par["Bg"]
        self.T_SG = par["T_SG"]
        self.T_GS = par["T_GS"]
        self.T_GG = par["T_GG"]
        self.T_CS = par["T_CS"]
        self.T_SC = par["T_SC"]
        self.wSG = par["wSG"]
        self.wGS = par["wGS"]
        self.wCS = par["wCS"]
        self.wSC = par["wSC"]
        self.wGG = par["wGG"]
        self.wCC = par["wCC"]
        self.C = par["C"]
        self.Str = par["Str"]
        self.Be = par["Be"]
        self.Bi = par["Bi"]
        self.Me = par["Me"]
        self.Mi = par["Mi"]
        self.T_CC = par["T_CC"]

        self.data_path = "data"
        self.nstart = par_simulation["nstart"]
        self.t_simulation = par_simulation['t_simulation']
        self.num_iterarion = (int)(self.t_simulation/self.dt)
        self.t = np.zeros(self.num_iterarion + self.nstart + 1)  # times
        self.y = np.zeros((self.num, self.num_iterarion + self.nstart + 1))
        self.output_filename = join(self.data_path, par['output_filename'])

    def to_fr(self, x, M, B):
        '''
        convert the activity to firing rate
        '''
        return M / (1 + (M - B) / B * exp(-4 * x / M))

    def simulate(self):
        initial_condition = [0.0] * self.num
        self.set_history(initial_condition, self.nstart, self.max_dalay)
        self.euler_integrator()
    # -------------------------------------------------------------------------

    def sys_eqs(self, t, n):

        dS = self.itau_S * (self.to_fr(self.wCS * self.interp_y(t - self.T_CS, 2, n) -
                                       self.wGS *
                                       self.interp_y(t-self.T_GS, 1, n),
                                       self.Ms, self.Bs) - self.y[0, n])
        dG = self.itau_G * (self.to_fr(self.wSG * self.interp_y(t - self.T_SG, 0, n) -
                                       self.wGG * self.interp_y(
                                           t - self.T_GG, 1, n) - self.Str,
                                       self.Mg, self.Bg) - self.y[1, n])
        dE = self.itau_E * (self.to_fr(-self.wSC * self.interp_y(t - self.T_SC, 0, n) -
                                       self.wCC *
                                       self.interp_y(
                                           t - self.T_CC, 3, n) + self.C,
                                       self.Me, self.Be) - self.y[2, n])
        dI = self.itau_I * \
            (self.to_fr(self.wCC * self.interp_y(t-self.T_CC, 2, n),
                        self.Mi, self.Bi) - self.y[3, n])

        return np.array([dS, dG, dE, dI])
    # -------------------------------------------------------------------------

    def set_history(self, hist, nstart, maxdelay):

        for i in range(nstart+1):
            self.t[i] = -(nstart - i) / float(nstart) * maxdelay

        # x is: num x nstep
        for i in range(self.num):
            self.y[i, :(nstart+1)] = hist[i]
    # -------------------------------------------------------------------------

    def euler_integrator(self):  # f, h

        for i in range(self.nstart, self.nstart + self.num_iterarion):
            dy = self.dt * self.sys_eqs(self.t[i], i)
            self.y[:, i+1] = self.y[:, i] + dy
            self.t[i+1] = (i - self.nstart + 1) * self.dt

        np.savez(self.output_filename,
                 t=self.t[self.nstart:],
                 y=self.y[:, self.nstart:])
    # -------------------------------------------------------------------------

    def interp_y(self, t0, index, n):
        assert(t0 <= self.t[n])
        return (np.interp(t0, self.t[:n], self.y[index, :n]))
    # -------------------------------------------------------------------------


def plot_time_series(filename_data, filename_fig):

    fig, ax = plt.subplots(1, figsize=(7, 3.5))
    data = np.load("data/"+filename_data+".npz")
    t = data["t"]
    y = data["y"]

    labels = ['STN', 'GPe', 'E', 'I']
    colors = ['b', 'g', 'r', 'cyan']

    for i in range(4):
        ax.plot(t, y[i, :], label=labels[i], lw=1, c=colors[i])

    # print t[-1], y[:, -1]
    ax.legend(loc="upper right")
    ax.set_xlabel("time (s)", fontsize=13)
    ax.set_ylabel("Firing rate (spk/s)", fontsize=13)
    ax.margins(x=0)
    plt.tight_layout()
    fig.savefig("data/a-"+filename_fig+".png", dpi=150)

# -------------------------------------------------------------------------


def plot_frequency_spectrum(filename_data, filename_fig, xlim=[0, 100]):

    fig, ax = plt.subplots(1, figsize=(7, 3.5))
    data = np.load("data/"+filename_data+".npz")
    t = data["t"]
    y = data["y"]
    fs = 1000 / (t[1]-t[0])

    labels = ['STN', 'GPe', 'E', 'I']
    colors = ['b', 'g', 'r', 'cyan']

    for i in range(4):
        signal = y[i, :]
        f, a = fft_1d_real(signal, fs)
        ax.plot(f, a, c=colors[i], label=labels[i])

    ax.legend(loc="upper right")
    ax.set_xlabel("frequency (Hz)", fontsize=13)
    ax.set_ylabel("amplitude", fontsize=13)
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.tight_layout()
    fig.savefig("data/f-"+filename_fig+".png", dpi=150)


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

    return freq, amplitude



