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
    MODULE_LOADED = False

    model_name = "terub_stn_multisyn"
    module_name = '{}_module'.format(model_name)

    record_from = ["V_m", "I_syn_ampa", "I_syn_nmda",
                   "I_syn_gaba_a", "I_syn_gaba_b"]

    def __init__(self, dt):
        self.name = self.__class__.__name__
        nest.ResetKernel()
        nest.set_verbosity('M_QUIET')
        self.dt = dt

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

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

    # ---------------------------------------------------------------

    def simulate_single_stn_cell(self, **par):

        nest.ResetKernel()
        nest.SetKernelStatus({
            "resolution": self.dt})

        self.set_params(**par)

        dt = par['dt']
        t_simulation = self.t_simulation

        model = "{}_nestml".format(self.model_name)
        if not self.MODULE_LOADED:
            nest.Install(self.module_name)
            self.MODULE_LOADED = True

        neuron = nest.Create(model)

        # parameters = nest.GetDefaults(model_name)
        # for i in parameters:
        #     print(i, parameters[i])

        nest.SetStatus(neuron, {'I_e': self.I_e})

        dc_gen = nest.Create("dc_generator")
        nest.SetStatus(dc_gen, {"amplitude": self.I_dc,
                                "start": self.TIstart, "stop": self.TIstop})

        multimeter = nest.Create("multimeter")
        nest.SetStatus(multimeter, {"withtime": True,
                                    "record_from": ["V_m"],
                                    "interval": dt})
        spikedetector = nest.Create("spike_detector",
                                    params={"withgid": True,
                                            "withtime": True})
        nest.Connect(multimeter, neuron)
        nest.Connect(neuron, spikedetector)
        nest.Simulate(t_simulation)

        dSD = nest.GetStatus(spikedetector, keys='events')[0]
        spikes = dSD['senders']

        firing_rate = len(spikes) / t_simulation * 1000
        print("firing rate is ", firing_rate)
        return spikedetector, multimeter
# -------------------------------------------------------------------

    def simulate_two_stn_cell(self):

        model = "{}_nestml".format(self.model_name)
        if not self.MODULE_LOADED:
            nest.Install(self.module_name)
            self.MODULE_LOADED = True

        neurons = nest.Create(model, 2)

        # parameters = nest.GetDefaults(model_name)
        # for i in parameters:
        #     print(i, parameters[i])

        for neuron in neurons:
            nest.SetStatus([neuron], {'I_e': 0.0 + rand() * 10.0 - 5.5})
            nest.SetStatus([neuron], {'V_m': -65.0 + rand() * 10. - 5.})

        neuron1, neuron2 = neurons
        nest.Connect([neuron1], [neuron2], syn_spec={
                     "receptor_type": 1})  # AMPA
        # nest.Connect([neuron1], [neuron2], syn_spec={"receptor_type": 2})  # NMDA
        nest.Connect([neuron1], [neuron2], syn_spec={
                     "receptor_type": 3})  # GABAA
        # nest.Connect([neuron1], [neuron2], syn_spec={"receptor_type": 4})  # GABAB

        multimeter = nest.Create("multimeter", 2)
        nest.SetStatus(multimeter, {"withtime": True,
                                    "record_from": self.record_from,
                                    "interval": dt})
        spikedetector = nest.Create("spike_detector",
                                    params={"withgid": True,
                                            "withtime": True})
        nest.Connect(multimeter, neurons, "one_to_one")
        nest.Connect(neurons, spikedetector)
        nest.Simulate(self.t_simulation)

        dSD = nest.GetStatus(spikedetector, keys='events')[0]
        spikes = dSD['senders']

        firing_rate = len(spikes) / self.t_simulation * 1000
        print("firing rate is ", firing_rate/2)

        return spikedetector, multimeter

    def plot_data(self, spikedetector, multimeter,
                  index=[0], filename="single_stn"):

        _, ax = plt.subplots(3, figsize=(8, 4), sharex=True)

        for i in index:
            dmm = nest.GetStatus(multimeter, keys='events')[i]
            Voltages = dmm["V_m"]
            tv = dmm["times"]
            ax[0].plot(tv, Voltages, lw=1, label=str(i+1))

        if len(index) > 1:
            labels = ["ampa", "nmda", "gaba_a", "gaba_b"]
            j = 0
            dmm = nest.GetStatus(multimeter)[1]
            tv = dmm['events']["times"]
            for i in self.record_from[1:]:
                g = dmm["events"][i]
                ax[1].plot(tv, g, lw=2, label=labels[j])
                j += 1

        dSD = nest.GetStatus(spikedetector, keys='events')[0]
        spikes = dSD['senders']
        ts = dSD["times"]

        ax[2].plot(ts, spikes, 'ko')
        ax[2].set_xlabel("Time [ms]")
        ax[2].set_xlim(0, self.t_simulation)
        ax[1].set_ylabel("I_syn [uA]")
        ax[2].set_ylabel("Spikes")
        ax[0].set_ylabel("v [ms]")
        ax[0].legend()
        ax[1].legend(frameon=False, loc="upper right")
        # ax[0].set_ylim(-100, 50)

        # for i in ts:
        #     ax[0].axvline(x=i, lw=1., ls="--", color="gray")

        plt.savefig(join("../data/", filename+".png"))
        plt.close()
        # plt.show()
    
    @staticmethod
    def plot_voltages(multimeter, ax, label=None):
        dmm = nest.GetStatus(multimeter, keys='events')[0]
        Voltages = dmm["V_m"]
        tv = dmm["times"]
        ax.plot(tv, Voltages, lw=1, label=label)
        ax.margins(x=0)
        ax.legend()



if __name__ == "__main__":

    par = {'I_e': 0.0,
           't_simulation': 2000.0,
           'MODULE_LOADED': False,
           'TIstart': 500.0,
           'TIstop': 800.0,
           'I_dc': -25.0,
           'dt':0.01
           }


    
    fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
    sol = STN_CELL(par['dt'])
    sol.set_params(**par)
    t_stim = [300., 450, 600.]
    for i in range(len(t_stim)):
        par['TIstop'] = t_stim[i] + par['TIstart']
        spk, mul = sol.simulate_single_stn_cell()
        sol.plot_voltages(mul, ax=ax[i], label=str(t_stim[i]))    
    plt.savefig("../data/stn.png", dpi=150)


    # sol = STN_CELL(par['dt])
    # par['MODULE_LOADED'] = True
    # sol.set_params(**par)
    # spk, mul = sol.simulate_two_stn_cell()
    # sol.plot_data(spk, mul, index=[0, 1], filename="two_stn")
