import os
import nest
import numpy as np
from numpy.core.records import record
import pylab as plt
from os.path import join
from numpy.random import rand

np.random.seed(5)


class GPe_CELL(object):

    data_path = "../data/"
    BUILT = False       # True, if build() was called
    CONNECTED = False   # True, if connect() was called
    nthreads = 1

    model_name = "terub_gpe_multisyn"
    module_name = '{}_module'.format(model_name)

    record_from = ["V_m", "I_syn_ampa", "I_syn_nmda",
                   "I_syn_gaba_a", "I_syn_gaba_b"]

    def __init__(self, dt):

        self.name = self.__class__.__name__

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        nest.ResetKernel()
        nest.set_verbosity('M_QUIET')
        self.dt = dt

        self.model = "{}_nestml".format(self.model_name)
        nest.Install(self.module_name)

        # parameters = nest.GetDefaults(self.model)
        # for i in parameters:
        #     print(i, parameters[i])

        nest.SetKernelStatus({
            "resolution": dt,
            "print_time": False,
            "overwrite_files": True,
            "data_path": self.data_path,
            "local_num_threads": self.nthreads})

    def set_params(self, **par):

        self.MODULE_LOADED = par['MODULE_LOADED']
        self.t_simulation = par['t_simulation']
        self.dt = par['dt']
        self.I_e = par['I_e']
        self.I_dc = par['I_dc']
        self.TIstop = par['TIstop']
        self.TIstart = par['TIstart']
        self.state = par['state']

    # ---------------------------------------------------------------

    def simulate_single_gpe_cell(self):

        dt = par['dt']

        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": dt})

        t_simulation = self.t_simulation

        if self.state == "Te2002":
            dict_ter2002 = {
                'g_phi_n': 0.05,
                'g_k_Ca': 20.,
            }
            nest.SetDefaults(self.model, dict_ter2002)

        neuron = nest.Create(self.model)
        # nest.SetStatus(neuron, {'I_e': self.I_e})

        dc_gen = nest.Create("dc_generator")
        nest.SetStatus(dc_gen, {"amplitude": self.I_dc,
                                "start": self.TIstart, "stop": self.TIstop})

        multimeter = nest.Create("multimeter")
        nest.SetStatus(multimeter, {"withtime": True,
                                    "record_from": ["V_m"],
                                    "interval": dt})
        # spikedetector = nest.Create("spike_detector",
        #                             params={"withgid": True,
        #                                     "withtime": True})
        nest.Connect(dc_gen, neuron)
        nest.Connect(multimeter, neuron)
        # nest.Connect(neuron, spikedetector)
        nest.Simulate(t_simulation)

        # dSD = nest.GetStatus(spikedetector, keys='events')[0]
        # spikes = dSD['senders']

        # firing_rate = len(spikes) / t_simulation * 1000
        # print("firing rate is ", firing_rate)
        return multimeter
# -------------------------------------------------------------------

    @staticmethod
    def plot_voltages(multimeter, ax, label=None, xlabel=None):
        dmm = nest.GetStatus(multimeter, keys='events')[0]
        Voltages = dmm["V_m"]
        tv = dmm["times"]
        ax.plot(tv, Voltages, lw=1, label=label)
        ax.margins(x=0)
        ax.legend(loc='upper right')
        ax.set_ylabel("voltage")
        if xlabel is not None:
            ax.set_xlabel("time (ms)")


if __name__ == "__main__":

    par = {'I_e': 0.0,
           't_simulation': 1000.0,
           'MODULE_LOADED': False,
           'TIstart': 0.0,
           'TIstop': 1000.0,
           'I_dc': 0.0,
           'dt': 0.01,
           #    "state": "TeRu2004",
           "state": "Te2002",
           }

    def figure_2():

        _, ax = plt.subplots(4, figsize=(8, 5), sharex=True)
        xlabel = None
        sol = GPe_CELL(par['dt'])
        i_stim = [20., 0.0, -0.5, 170.]
        for i in range(len(i_stim)):
            if i == 3:
                xlabel = "Time (ms)"
            par['I_dc'] = i_stim[i]
            sol.set_params(**par)
            mul = sol.simulate_single_gpe_cell()
            sol.plot_voltages(mul, ax=ax[i], label=str(
                i_stim[i]), xlabel=xlabel)
        plt.savefig("../data/gpe2-{}.png".format(par['state']), dpi=150)
        plt.close()

    figure_2()
