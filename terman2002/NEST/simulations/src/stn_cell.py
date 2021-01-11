import nest
import numpy as np
from numpy.core.records import record
import pylab as plt
from os.path import join
from numpy.random import rand

np.random.seed(5)


def simulate_single_stn_cell(dt, t_simulation):

    global MODULE_LOADED
    nest.ResetKernel()

    model_name = "terub_stn_multisyn"
    module_name = '{}_module'.format(model_name)
    model = "{}_nestml".format(model_name)
    if not MODULE_LOADED:
        nest.Install(module_name)
        MODULE_LOADED = True

    nest.SetKernelStatus({"resolution": dt})
    neuron = nest.Create(model)

    # parameters = nest.GetDefaults(model_name)
    # for i in parameters:
    #     print(i, parameters[i])

    nest.SetStatus(neuron, {'I_e': 0.0})

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

    dmm = nest.GetStatus(multimeter)[0]
    # Voltages = dmm["events"]["V_m"]
    # tv = dmm["events"]["times"]
    dSD = nest.GetStatus(spikedetector, keys='events')[0]
    spikes = dSD['senders']
    # ts = dSD["times"]

    firing_rate = len(spikes) / t_simulation * 1000
    print("firing rate is ", firing_rate)
    return spikedetector, multimeter
# -------------------------------------------------------------------


def simulate_two_stn_cell(dt, t_simulation):

    global MODULE_LOADED
    nest.ResetKernel()

    model_name = "terub_stn_multisyn"
    module_name = '{}_module'.format(model_name)
    model = "{}_nestml".format(model_name)

    if not MODULE_LOADED:
        nest.Install(module_name)


    nest.SetKernelStatus({"resolution": dt})
    neurons = nest.Create(model, 2)

    # parameters = nest.GetDefaults(model_name)
    # for i in parameters:
    #     print(i, parameters[i])

    for neuron in neurons:
        nest.SetStatus([neuron], {'I_e': 0.0 + rand() * 10.0 - 5.5})
        nest.SetStatus([neuron], {'V_m': -65.0 + rand() * 10. - 5.})

    neuron1, neuron2 = neurons
    nest.Connect([neuron1], [neuron2], syn_spec={"receptor_type": 1})  # AMPA
    # nest.Connect([neuron1], [neuron2], syn_spec={"receptor_type": 2})  # NMDA
    nest.Connect([neuron1], [neuron2], syn_spec={"receptor_type": 3})  # GABAA
    # nest.Connect([neuron1], [neuron2], syn_spec={"receptor_type": 4})  # GABAB

    multimeter = nest.Create("multimeter", 2)
    nest.SetStatus(multimeter, {"withtime": True,
                                "record_from": record_from,
                                "interval": dt})
    spikedetector = nest.Create("spike_detector",
                                params={"withgid": True,
                                        "withtime": True})
    nest.Connect(multimeter, neurons, "one_to_one")
    nest.Connect(neurons, spikedetector)
    nest.Simulate(t_simulation)

    dmm = nest.GetStatus(multimeter)[0]
    # Voltages = dmm["events"]["V_m"]
    # tv = dmm["events"]["times"]
    dSD = nest.GetStatus(spikedetector, keys='events')[0]
    spikes = dSD['senders']
    # ts = dSD["times"]

    firing_rate = len(spikes) / t_simulation * 1000
    print("firing rate is ", firing_rate/2)

    return spikedetector, multimeter


def plot_data(spikedetector, multimeter, index=[0], filename="single_stn"):

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
        for i in record_from[1:]:
            g = dmm["events"][i]
            ax[1].plot(tv, g, lw=2, label=labels[j])
            j += 1

    dSD = nest.GetStatus(spikedetector, keys='events')[0]
    spikes = dSD['senders']
    ts = dSD["times"]

    ax[2].plot(ts, spikes, 'ko')
    ax[2].set_xlabel("Time [ms]")
    ax[2].set_xlim(0, t_simulation)
    ax[1].set_ylabel("I_syn [uA]")
    ax[2].set_ylabel("Spikes")
    ax[0].set_ylabel("v [ms]")
    ax[0].legend()
    ax[1].legend(frameon=False, loc="upper right")
    # ax[0].set_ylim(-100, 50)

    # for i in ts:
    #     ax[0].axvline(x=i, lw=1., ls="--", color="gray")

    plt.savefig(join("../data/", filename+".png"))
    # plt.show()


if __name__ == "__main__":
    
    MODULE_LOADED = False

    dt = 0.01
    t_simulation = 2000.0
    record_from = ["V_m", "I_syn_ampa", "I_syn_nmda",
                   "I_syn_gaba_a", "I_syn_gaba_b"]

    spk, mul = simulate_single_stn_cell(dt, t_simulation)
    plot_data(spk, mul, index=[0], filename="single_stn")

    spk, mul = simulate_two_stn_cell(dt, t_simulation)
    plot_data(spk, mul, index=[0, 1], filename="two_stn")
