"""
wb_cond_exp_test.py
"""

import os
import nest
import unittest
import numpy as np
from pynestml.frontend.pynestml_frontend import to_nest, install_nest
from numpy.random import rand
from os.path import join


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

    def test_terub_stn(self):

        if not os.path.exists("target"):
            os.makedirs("target")

        input_path = os.path.join(os.path.realpath(os.path.join(
            os.path.dirname(__file__), "../models", "terub_stn_beta.nestml")))
        target_path = "target"
        module_name = 'terub_stn_beta_module'
        nest_path = "/home/abolfazl/prog/nest-build/"
        suffix = '_nestml'

        # to_nest(input_path=input_path,
        #         target_path=target_path,
        #         logging_level="INFO",
        #         suffix=suffix,
        #         module_name=module_name)

        # install_nest(target_path, nest_path)

        nest.Install(module_name)
        model = "terub_stn_beta_nestml"

        dt = 0.01
        t_simulation = 2000.0
        nest.SetKernelStatus({"resolution": dt})

        neurons = nest.Create(model, 2)
        # parameters = nest.GetDefaults(model)

        # if 0:
        #     for i in parameters:
        #         print(i, parameters[i])

        for neuron in neurons:
            nest.SetStatus([neuron], {'I_e': 0.0 + rand() * 10.0 - 5.5})
            nest.SetStatus([neuron], {'V_m': -65.0 + rand() * 10. - 5.})

        nest.Connect([neurons[0]], [neurons[1]],
                     syn_spec={"weight": 20, "delay": 1.0})
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

        # dmm = nest.GetStatus(multimeter)[0]
        # Voltages = dmm["events"]["V_m"]
        # tv = dmm["events"]["times"]
        dSD = nest.GetStatus(spikedetector, keys='events')[0]
        spikes = dSD['senders']
        # ts = dSD["times"]

        firing_rate = len(spikes) / t_simulation * 1000
        print("firing rate is ", firing_rate)
        expected_value = np.abs(firing_rate - 14)
        # tolerance_value = 3  # Hz

        # self.assertLessEqual(expected_value, tolerance_value)

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
            plt.savefig(join("resources", "terub_stn_beta.png"))
            plt.show()

        plot_data(spikedetector, multimeter, index=[0, 1])
        # if TEST_PLOTS:

        #     fig, ax = plt.subplots(2, figsize=(8, 4), sharex=True)
        #     ax[0].plot(tv, Voltages, lw=2, color="k")
        #     ax[1].plot(ts, spikes, 'ko')
        #     ax[1].set_xlabel("Time [ms]")
        #     ax[1].set_xlim(0, t_simulation)
        #     ax[1].set_ylabel("Spikes")
        #     ax[0].set_ylabel("v [ms]")
        #     # ax[0].set_ylim(-100, 50)

        # for i in ts:
        #     ax[0].axvline(x=i, lw=1., ls="--", color="gray")

        # plt.savefig("resources/terub_stn_beta.png")
        # # plt.show()


if __name__ == "__main__":
    unittest.main()
