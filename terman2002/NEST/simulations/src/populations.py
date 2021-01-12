import os
import nest
import time
import numpy as np
import pylab as plt
from os.path import join
from numpy.random import rand
from joblib import Parallel, delayed
np.random.seed(5)


def install_modules():
    stn_module_name = 'terub_stn_multisyn_module'
    gpe_module_name = 'terub_gpe_multisyn_module'
    nest.Install(stn_module_name)
    nest.Install(gpe_module_name)


class TER_RUB():

    data_path = "../data/"
    BUILT = False       # True, if build() was called
    CONNECTED = False   # True, if connect() was called
    nthreads = 1
    stn_model_name = "terub_stn_multisyn_nestml"
    gpe_model_name = "terub_gpe_multisyn_nestml"

    # ---------------------------------------------------------------

    def __init__(self, dt):

        self.name = self.__class__.__name__
        nest.ResetKernel()
        nest.set_verbosity('M_QUIET')
        self.dt = dt

        # parameters = nest.GetDefaults(self.stn_model_name)
        # for i in parameters:
        # print(i, parameters[i])

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        nest.SetKernelStatus({
            "resolution": dt,
            "print_time": False,
            "overwrite_files": True,
            "data_path": self.data_path,
            "local_num_threads": self.nthreads})

        np.random.seed(2)

        # Create and seed RNGs
        master_seed = 1000      # master seed
        n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        master_seed_range1 = range(master_seed, master_seed + n_vp)
        self.pyrngs = [np.random.RandomState(s) for s in master_seed_range1]
        master_seed_range2 = range(
            master_seed + n_vp + 1, master_seed + 1 + 2 * n_vp)
        nest.SetKernelStatus({'grng_seed': master_seed + n_vp,
                              'rng_seeds': master_seed_range2})
    # ---------------------------------------------------------------

    def set_params(self, par_stn, par_gpe,
                   par_syn_SG, par_syn_GS, par_syn_GG,
                   par_simulation):

        self.dt = par_simulation['dt']
        self.state = par_simulation['state']
        self.t_simulation = par_simulation['t_simulation']
        self.par_stn = par_stn
        self.par_gpe = par_gpe
        self.par_syn_SG = par_syn_SG
        self.par_syn_GS = par_syn_GS
        self.par_syn_GG = par_syn_GG
        self.par_simulation = par_simulation
    # ---------------------------------------------------------------

    def build(self):
        '''
        Create devices (nodes and monitors) used in the model.
        '''

        n_stn = self.par_simulation['n_stn']
        n_gpe = self.par_simulation['n_gpe']

        if self.state == "Te2002":
            dict_default_stn = {
                'tau_r_0': 40.,
                'theta_b': 0.4,
                'sigma_b': -0.1,
                'phi_r': 0.2,
                'epsilon': 3.75e-5,
            }
            dict_default_gpe = {
                'g_phi_n': 0.05,
                'g_k_Ca': 20.
            }
            nest.SetDefaults(self.stn_model_name, dict_default_stn)
            nest.SetDefaults(self.gpe_model_name, dict_default_gpe)

        nest.CopyModel(self.stn_model_name, "STN_model", self.par_stn)
        nest.CopyModel(self.gpe_model_name, "GPe_model", self.par_gpe)

        self.stn_cells = nest.Create("STN_model", n_stn)
        self.gpe_cells = nest.Create("GPe_model", n_gpe)

        stn_V_thr = nest.GetStatus([self.stn_cells[0]], "V_thr")[0]

        node_info = nest.GetStatus(self.stn_cells+self.gpe_cells)
        local_nodes = [(ni['global_id'], ni['vp'])
                       for ni in node_info if ni['local']]
        for gid, vp in local_nodes:
            nest.SetStatus(
                [gid], {'V_m': self.pyrngs[vp].uniform(-stn_V_thr, stn_V_thr)})

        self.stn_multimeter = nest.Create("multimeter", n_stn)
        self.gpe_multimeter = nest.Create("multimeter", n_gpe)

        stn_record = ['V_m']
        gpe_record = ['V_m']

        nest.SetStatus(self.stn_multimeter, {"withtime": True,
                                             "record_from": stn_record,
                                             "interval": self.dt})
        nest.SetStatus(self.gpe_multimeter, {"withtime": True,
                                             "record_from": gpe_record,
                                             "interval": self.dt})
        self.stn_spikedetector = nest.Create("spike_detector",
                                             params={"withgid": True,
                                                     "withtime": True})
        self.gpe_spikedetector = nest.Create("spike_detector",
                                             params={"withgid": True,
                                                     "withtime": True})
        self.BUILT = True
    # ---------------------------------------------------------------

    def connect(self):

        if not self.BUILT:
            print("network not built")
            exit(0)
        # receptor_types :
        # 1 AMPA
        # 2 NMDA
        # 3 GAPAA
        # 4 GABAB
        conn_dict_GG = {'rule': "all_to_all"}
        conn_dict_SG = {'rule': 'fixed_outdegree',
                        'outdegree': 1, 'autapses': False, 'multapses': False}
        conn_dict_GS = {'rule': 'fixed_outdegree',
                        'outdegree': 3, 'autapses': False, 'multapses': False}

        nest.Connect(self.gpe_cells, self.gpe_cells,
                     conn_spec=conn_dict_GG,
                     syn_spec={"receptor_type": 4,
                               'weight': self.par_syn_GG['weight'],
                               'delay': self.par_syn_GG['delay']})
        nest.Connect(self.stn_cells, self.gpe_cells,
                     conn_spec=conn_dict_SG,
                     syn_spec={"receptor_type": 1,
                               'weight': self.par_syn_SG['weight'],
                               'delay': self.par_syn_SG['delay']})
        nest.Connect(self.gpe_cells, self.stn_cells,
                     conn_spec=conn_dict_GS,
                     syn_spec={"receptor_type": 3,
                               'weight': self.par_syn_GS['weight'],
                               'delay': self.par_syn_GS['delay']})

        nest.Connect(self.stn_multimeter, self.stn_cells, "one_to_one")
        nest.Connect(self.gpe_multimeter, self.gpe_cells, "one_to_one")
        nest.Connect(self.stn_cells, self.stn_spikedetector)
        nest.Connect(self.gpe_cells, self.gpe_spikedetector)

        # printing the connections
        # for i in self.gpe_cells[:2]:
        #     conn = nest.GetConnections([i])
        #     print(nest.GetStatus(conn, ['source', 'target']))
        # print("---------------------------------------------")

        self.CONNECTED = True
    # ---------------------------------------------------------------

    def run(self, VERBOSE=False):

        if not self.BUILT:
            print("network not built.")
            exit(0)
        if not self.CONNECTED:
            print("network not connected.")
            exit(0)

        print("g_SG = {:10.3f}, g_GG = {:10.3f}".format(self.par_syn_SG['weight'],
                                                        self.par_syn_GG['weight']))

        t_simulation = self.par_simulation['t_simulation']
        t_transition = self.par_simulation['t_transition']

        nest.Simulate(t_transition)
        nest.SetStatus(self.stn_spikedetector, {'n_events': 0})
        nest.SetStatus(self.gpe_spikedetector, {'n_events': 0})

        nest.Simulate(t_simulation)

        if VERBOSE:
            events_stn = nest.GetStatus(self.stn_spikedetector, "n_events")[0]
            events_gpe = nest.GetStatus(self.gpe_spikedetector, "n_events")[0]

            rate_stn = events_stn / t_simulation * \
                1000.0 / self.par_simulation['n_stn']
            rate_gpe = events_gpe / t_simulation * \
                1000.0 / self.par_simulation['n_gpe']

            print("stn firing rate: {}".format(rate_stn))
            print("gpe firing rate: {}".format(rate_gpe))
    # ---------------------------------------------------------------

    def plot_voltages(self, filename="v"):
        nrows = 3
        _, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 4), sharex=True)
        for i in range(nrows):
            dmm = nest.GetStatus(self.stn_multimeter, keys='events')[i]
            Voltages = dmm["V_m"]
            tv = dmm["times"]
            ax[i][0].plot(tv, Voltages, lw=1, label=str(i+1))
        for j in range(nrows):
            dmm = nest.GetStatus(self.gpe_multimeter, keys='events')[j]
            Voltages = dmm["V_m"]
            tv = dmm["times"]
            ax[j][1].plot(tv, Voltages, lw=1, label=str(j+1))

        for i in range(2):
            ax[-1][i].set_xlabel("Time [ms]")
        for i in range(nrows):
            ax[i][0].set_ylabel("voltage [mV]")

        plt.savefig(join(self.data_path, "figs", filename+".png"), dpi=100)
        plt.close()
    # ---------------------------------------------------------------

    def plot_raster(self, filename="s"):

        _, ax = plt.subplots(1, figsize=(8, 4), sharex=True)

        dSD = nest.GetStatus(self.stn_spikedetector, keys='events')[0]
        stn_spikes = dSD['senders']
        ts = dSD["times"]
        ax.plot(ts, stn_spikes, 'ro', ms=2)

        dSD = nest.GetStatus(self.gpe_spikedetector, keys='events')[0]
        gpe_spikes = dSD['senders']
        ts = dSD["times"]
        ax.plot(ts, gpe_spikes, 'bo', ms=2)

        ax.set_ylabel("Spikes")
        ax.set_xlabel("Times [ms]")
        plt.savefig(join(self.data_path, "figs", filename+".png"), dpi=100)
        plt.close()


def display_time(time):

    hour = int(time/3600)
    minute = (int(time % 3600)) // 60
    second = time - (3600. * hour + 60. * minute)
    print("Done in %d hours %d minutes %09.6f seconds"
          % (hour, minute, second))


if __name__ == "__main__":

    par_stn = {
        "AMPA_Tau_1": 0.2,
        "AMPA_Tau_2": 1.0,
        "AMPA_E_rev": 0.0,
        "GABA_A_Tau_1": 0.5,
        "GABA_A_Tau_2": 12.5,
        "GABA_A_E_rev": -85.0,
        "GABA_B_Tau_1": 0.5,
        "GABA_B_Tau_2": 12.5,
        "GABA_B_E_rev": -100.0,
    }

    #  I use Gaba_B as Gaba_A, for GG connection
    par_gpe = {
        "AMPA_Tau_1": 0.2,
        "AMPA_Tau_2": 1.0,
        "AMPA_E_rev": 0.0,
        "GABA_A_Tau_1": 0.5,
        "GABA_A_Tau_2": 12.5,
        "GABA_A_E_rev": -85.0,
        "GABA_B_Tau_1": 0.5,
        "GABA_B_Tau_2": 12.5,
        "GABA_B_E_rev": -100.0,
    }

    par_syn_GS = {'delay': 1.0, 'weight': 1.0}
    par_syn_SG = {'delay': 1.0, 'weight': 0.03}
    par_syn_GG = {'delay': 1.0, 'weight': 0.06}
    par_simulation = {
        "dt": 0.1,
        'state': "Te2002",
        't_transition': 100.,
        't_simulation': 2000.,
        'n_stn': 10,
        'n_gpe': 10,
    }

    start_time = time.time()
    install_modules()

    g_SG = np.linspace(0.01, 2.0, 11)
    g_GG = np.linspace(0.01, 2.0, 11)

    for i in range(len(g_SG)):
        for j in range(len(g_GG)):

            sub_name = "{:.6f}-{:.6f}".format(g_SG[i], g_GG[j])

            sol = TER_RUB(par_simulation['dt'])
            par_syn_SG['weight'] = g_SG[i]
            par_syn_GG['weight'] = g_GG[j]
            sol.set_params(par_stn,
                           par_gpe,
                           par_syn_SG,
                           par_syn_GS,
                           par_syn_GG,
                           par_simulation)
            sol.build()
            sol.connect()
            sol.run()
            sol.plot_raster(filename="s-"+sub_name)
            sol.plot_voltages(filename="v-"+sub_name)

    display_time(time.time() - start_time)
