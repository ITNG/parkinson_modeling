"""
terub_gpe_multisyn_test.py
"""

import os
import nest
import unittest
import numpy as np
from pynestml.frontend.pynestml_frontend import to_nest, install_nest
from numpy.random import rand
from os.path import join

np.random.seed(4)

directories = ['../models', 'resources', 'target']
for i in directories:
    if not os.path.exists(i):
        os.makedirs(i)

try:
    import matplotlib.pyplot as plt
    TEST_PLOTS = True
except:
    TEST_PLOTS = False


class NestSTNExpTest(unittest.TestCase):

    def test_terub_gpe_multisyn(self):

        model_name = "terub_gpe_multisyn"

        if not os.path.exists("target"):
            os.makedirs("target")

        input_path = os.path.join(os.path.realpath(os.path.join(
            os.path.dirname(__file__), "../models", "{}.nestml".format(model_name))))
        target_path = "target"
        module_name = '{}_module'.format(model_name)
        nest_path = "/home/abolfazl/prog/nest-build/"
        suffix = '_nestml'

        if 0: #! compile
            to_nest(input_path=input_path,
                    target_path=target_path,
                    logging_level="INFO",
                    suffix=suffix,
                    module_name=module_name)

            install_nest(target_path, nest_path)

        nest.Install(module_name)
        model = "{}_nestml".format(model_name)

        dt = 0.01
        t_simulation = 1500.0
        nest.SetKernelStatus({"resolution": dt})

        neuron1 = nest.Create(model, 1)
        neuron2 = nest.Create(model, 1)
        parameters = nest.GetDefaults(model)

        # if 1:
        #     for i in parameters:
        #         print(i, parameters[i])

        for neuron in neuron1+neuron2:
            nest.SetStatus([neuron], {'I_e': 0.0 + rand() * 20.0 - 20})
            nest.SetStatus([neuron], {'V_m': -65.0 + rand() * 10. - 5.})
        
        # nest.Connect(neuron1, neuron2, syn_spec={"receptor_type": 1})  # AMPA
        # nest.Connect(neuron1, neuron2, syn_spec={"receptor_type": 2})  # NMDA
        nest.Connect(neuron1, neuron2, syn_spec={"receptor_type": 3})  # GABAA
        nest.Connect(neuron2, neuron1, syn_spec={"receptor_type": 4})  # GABAB

        record_from = ["V_m", "I_syn_ampa", "I_syn_nmda",
                       "I_syn_gaba_a", "I_syn_gaba_b"]

        multimeter = nest.Create("multimeter", 2)
        nest.SetStatus(multimeter, {"withtime": True,
                                    "record_from": record_from,
                                    "interval": dt})
        spikedetector = nest.Create("spike_detector",
                                    params={"withgid": True,
                                            "withtime": True})
        nest.Connect(multimeter, neuron1+neuron2, "one_to_one")
        nest.Connect(neuron1+neuron2, spikedetector)
        nest.Simulate(t_simulation)

        dSD = nest.GetStatus(spikedetector, keys='events')[0]
        spikes = dSD['senders']

        firing_rate = len(spikes) / t_simulation * 1000
        print("firing rate is ", firing_rate / 2)

        def plot_data(index=[0]):

            fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
            for i in index:
                dmm = nest.GetStatus(multimeter, keys='events')[i]
                Voltages = dmm["V_m"]
                tv = dmm["times"]
                ax[0].plot(tv, Voltages, lw=1, label=str(i+1))

            labels = ["ampa", "nmda", "gaba_a", "gaba_b"]
            j = 0
            dmm = nest.GetStatus(multimeter) [1]
            tv = dmm['events']["times"]
            for i in record_from[1:]:
                g = dmm["events"][i]
                ax[1].plot(tv, g, lw=2, label=labels[j])
                j += 1
            
            j = 0
            dmm = nest.GetStatus(multimeter) [0]
            tv = dmm['events']["times"]
            for i in record_from[1:]:
                g = dmm["events"][i]
                ax[2].plot(tv, g, lw=2, label=labels[j])
                j += 1
            
            dSD = nest.GetStatus(spikedetector, keys='events')[0]
            spikes = dSD['senders']
            ts = dSD["times"]

            ax[3].plot(ts, spikes, 'ko', ms=3)
            ax[3].set_xlabel("Time [ms]")
            ax[3].set_xlim(0, t_simulation)
            ax[3].set_ylabel("Spikes")
            ax[0].set_title("recording from PSP")
            ax[0].set_ylabel("v [ms]")
            ax[1].set_ylabel("I_syn")
            ax[1].legend(frameon=False, loc="upper right")
            ax[0].legend()


            plt.savefig(join("resources", "terub_gpe_multisyn.png"), dpi=150)
            # plt.show()

        plot_data(index=[0, 1])


if __name__ == "__main__":
    unittest.main()
