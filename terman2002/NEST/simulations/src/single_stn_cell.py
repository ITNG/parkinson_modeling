import os
import nest
import numpy as np
from numpy.core.records import record
import pylab as plt
from os.path import join
from numpy.random import rand

np.random.seed(5)


class STN_CELL(object):

    data_path = "../data/"
    BUILT = False       # True, if build() was called
    CONNECTED = False   # True, if connect() was called
    nthreads = 1

    model_name = "terub_stn_multisyn"
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

    def simulate_single_stn_cell(self):

        dt = par['dt']

        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": dt})

        t_simulation = self.t_simulation

        if self.state == "Te2002":
            dict_ter2002 = {
                'tau_r_0': 40.,
                'theta_b': 0.4,
                'sigma_b': -0.1,
                'phi_r': 0.2,
                'epsilon': 3.75e-5,
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
    def plot_voltages(multimeter, ax, label=None):
        dmm = nest.GetStatus(multimeter, keys='events')[0]
        Voltages = dmm["V_m"]
        tv = dmm["times"]
        ax.plot(tv, Voltages, lw=1, label=label)
        ax.margins(x=0)
        ax.legend(loc='upper right')


if __name__ == "__main__":

    par = {'I_e': 0.0,
           't_simulation': 2000.0,
           'MODULE_LOADED': False,
           'TIstart': 500.0,
           'TIstop': 800.0,
           'I_dc': -25.0,
           'dt': 0.01,
        #    "state": "TeRu2004", 
           "state": "Te2002",
           }

    fig, ax = plt.subplots(3, figsize=(8, 5), sharex=True)
    sol = STN_CELL(par['dt'])
    t_stim = [300.0, 450.0, 600.0]
    for i in range(len(t_stim)):
        par['TIstop'] = t_stim[i] + par['TIstart']
        sol.set_params(**par)
        mul = sol.simulate_single_stn_cell()
        sol.plot_voltages(mul, ax=ax[i], label=str(t_stim[i]))
    plt.savefig("../data/stn1-{}.png".format(par['state']), dpi=150)

    exit(0)

    fig, ax = plt.subplots(3, figsize=(8, 5), sharex=True)
    sol = STN_CELL(par['dt'])
    i_stim = [-20., -30.0, -40.0]
    for i in range(len(i_stim)):
        par['I_dc'] = i_stim[i]
        sol.set_params(**par)
        mul = sol.simulate_single_stn_cell()
        sol.plot_voltages(mul, ax=ax[i], label=str(i_stim[i]))
    plt.savefig("../data/stn2-{}.png".format(par['state']), dpi=150)


    # sol = STN_CELL(par['dt])
    # par['MODULE_LOADED'] = True
    # sol.set_params(**par)
    # spk, mul = sol.simulate_two_stn_cell()
    # sol.plot_data(spk, mul, index=[0, 1], filename="two_stn")


# parameters = nest.GetDefaults(model_name)
# for i in parameters:
#     print(i, parameters[i])
