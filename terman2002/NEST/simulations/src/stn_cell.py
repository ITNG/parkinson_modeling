import nest
import numpy as np
import pylab as plt
from os.path import join
from numpy.random import rand

# np.random.seed(5)


def simulate_single_stn_cell():
    module_name = 'terub_stn_beta_module'
    model_name = "terub_stn_beta_nestml"
    nest.Install(module_name)

    nest.SetKernelStatus({"resolution": dt})
    neurons = nest.Create(model_name)
    # parameters = nest.GetDefaults(model_name)
    # for i in parameters:
    #     print(i, parameters[i])

    nest.SetStatus(neurons, {'I_e': 0.0})
    
    multimeter = nest.Create("multimeter")
    nest.SetStatus(multimeter, {"withtime": True,
                                "record_from": ["V_m"],
                                "interval": dt})
    spikedetector = nest.Create("spike_detector",
                                params={"withgid": True,
                                        "withtime": True})
    nest.Connect(multimeter, neurons)
    nest.Connect(neurons, spikedetector)
    nest.Simulate(t_simulation)

    dmm = nest.GetStatus(multimeter)[0]
    Voltages = dmm["events"]["V_m"]
    tv = dmm["events"]["times"]
    dSD = nest.GetStatus(spikedetector, keys='events')[0]
    spikes = dSD['senders']
    ts = dSD["times"]

    firing_rate = len(spikes) / t_simulation * 1000
    print("firing rate is ", firing_rate)
    return spikedetector, multimeter


def simulate_two_stn_cell():
    module_name = 'terub_stn_beta_module'
    model_name = "terub_stn_beta_nestml"
    nest.Install(module_name)

    nest.SetKernelStatus({"resolution": dt})
    neurons = nest.Create(model_name, 2)
    # parameters = nest.GetDefaults(model_name)
    # for i in parameters:
    #     print(i, parameters[i])

    for neuron in neurons:
        nest.SetStatus([neuron], {'I_e': 0.0 + rand() * 10.0 - 5.5})
        nest.SetStatus([neuron], {'V_m': -65.0 + rand() * 10. - 5.})
    
    nest.Connect([neurons[0]], [neurons[1]],
                 syn_spec={"weight": 10, "delay": 1.0})


    multimeter = nest.Create("multimeter", 2)
    nest.SetStatus(multimeter, {"withtime": True,
                                "record_from": ["V_m"],
                                "interval": dt})
    spikedetector = nest.Create("spike_detector",
                                params={"withgid": True,
                                        "withtime": True})
    nest.Connect(multimeter, neurons, "one_to_one")
    nest.Connect(neurons, spikedetector)
    nest.Simulate(t_simulation)

    dmm = nest.GetStatus(multimeter)[0]
    Voltages = dmm["events"]["V_m"]
    tv = dmm["events"]["times"]
    dSD = nest.GetStatus(spikedetector, keys='events')[0]
    spikes = dSD['senders']
    ts = dSD["times"]

    firing_rate = len(spikes) / t_simulation * 1000
    print("firing rate is ", firing_rate)

    return spikedetector, multimeter


def plot_data(spikedetector, multimeter, index=[0], filename="single_stn"):

    fig, ax = plt.subplots(2, figsize=(8, 4), sharex=True)

    for i in index:
        dmm = nest.GetStatus(multimeter, keys='events')[i]
        Voltages = dmm["V_m"]
        tv = dmm["times"]
        ax[0].plot(tv, Voltages, lw=1, label=str(i+1))

    dSD = nest.GetStatus(spikedetector, keys='events')[0]
    spikes = dSD['senders']
    ts = dSD["times"]

    ax[1].plot(ts, spikes, 'ko')
    ax[1].set_xlabel("Time [ms]")
    ax[1].set_xlim(0, t_simulation)
    ax[1].set_ylabel("Spikes")
    ax[0].set_ylabel("v [ms]")
    ax[0].legend()
    # ax[0].set_ylim(-100, 50)

    # for i in ts:
    #     ax[0].axvline(x=i, lw=1., ls="--", color="gray")

    plt.savefig(join("../data/", filename+".png"))
    plt.show()


if __name__ == "__main__":
    dt = 0.01
    t_simulation = 2000.0

    # spk, mul = simulate_single_stn_cell()
    # plot_data(spk, mul, index=[0], filename="single_stn")

    spk, mul = simulate_two_stn_cell()
    plot_data(spk, mul, index=[0, 1], filename="two_stn")
