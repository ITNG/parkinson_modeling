import numpy as np
import brian2 as b2
import pylab as plt
import networkx as nx
from copy import deepcopy
from numpy.random import rand
from time import time as wall_time
from joblib import Parallel, delayed
from lib import simulate_STN_GPe_population, to_npz
from plotting import plot_raster, plot_voltage, plot_voltage2

np.random.seed(2)


par_s = {
    'v0': -60*b2.mV, 'vl': -60 * b2.mV, 'vna': 55 * b2.mV,
    'vk': -80 * b2.mV, 'vca': 140 * b2.mV,
    'gl': 2.25 * b2.nS, 'gna': 37.5 * b2.nS, 'gk': 45 * b2.nS,
    'gahp': 9 * b2.nS,  'gca': 0.5 * b2.nS, 'gt': 0.5 * b2.nS,
    'thetam': -30, 'thetas': -39, 'thetah': -39, 'thetan': -32,
    'thetab': 0.4,  # Guo 0.25 Terman02 0.4
    'thetaa': -63, 'thetar': -67, 'sigmas': 8,
    'sigmah': -3.1, 'sigman': 8, 'sigmaa': 7.8,
    'sigmab': -0.1,  # Guo 0.07 Terman02 -0.1
    'sigmam': 15, 'sigmar': -2, 'sigmaht': -3, 'sigmant': -26,
    'sigmart': -2.2,
    'taun0': 1 * b2.ms, 'taun1': 100 * b2.ms, 'taur0': 40.0 * b2.ms,  # 7.1, Terman02 40.0
    'taur1': 17.5 * b2.ms, 'tauh0': 1 * b2.ms, 'tauh1': 500 * b2.ms,
    'eps': 3.75e-05 / b2.ms,  # 1/ms Guo 0.00005 Terman02 0.0000375
    'phi': 0.75, 'phir': 0.2,  # Guo 0.5 Terman02 0.2
    'kca': 22.5,  'thn': -80, 'thh': -57, 'thr': 68, 'ab': -30, 'k1': 15,
    'thetag_s': 30., 'thetagH_s': -39., 'sigmagH_s': 8.,
    'i_ext': -1.2 * b2.pA, 'C': 1 * b2.pF,
}

par_g = {
    'num': 1,
    'v0': -55 * b2.mV, 'vnag': 55. * b2.mV, 'vkg': -80. * b2.mV,
    'vcag': 120. * b2.mV, 'vlg': -55. * b2.mV, 'gnag': 120. * b2.nS,
    'gkg': 30. * b2.nS, 'gahpg': 30. * b2.nS, 'gtg': .5 * b2.nS,
    'gcag': .1 * b2.nS, 'glg': .1 * b2.nS,
    'taurg': 30. * b2.ms, 'taun0g': .05 * b2.ms, 'taun1g': .27*b2.ms,
    'tauh0g': .05*b2.ms, 'tauh1g': .27*b2.ms,

    'sigag': 2., 'sigsg': 2., 'sigrg': -2., 'sigmg': 10.,
    'signg': 14., 'sighg': -12,
    'thetasg': -35., 'thetaag': -57., 'thetarg': -70.,
    'thetamg': -37., 'thetang': -50., 'thetahg': -58,
    'thngt': -40, 'thhgt': -40, 'sng': -12, 'shg': -12,
    'k1g': 30., 'kcag': 20.,  # Report:15,  Terman Rubin 2002: 20.0
    'phig': 1., 'phing': .05,  # Report: 0.1, Terman Rubin 2002: 0.05
    'phihg': .05, 'epsg': 0.0001 / b2.ms,
    'thetag_g': 20.,  'thetagH_g': -57., 'sigmagH_g': 2.,
    'i_ext': -1.2 * b2.pA,  'C': 1 * b2.pF,
}

par_syn = {
    'v_rev_GtoG': -100. * b2.mV,
    'v_rev_StoG': 0. * b2.mV,
    'v_rev_GtoS': -85. * b2.mV,
    'alphas': 5. / b2.ms,
    'betas': 1. / b2.ms,
    'alphag': 2. / b2.ms,
    'betag': 0.08 / b2.ms,
    'g_GtoS': 2.5*b2.nS,
    'g_StoG': 0.03*b2.nS,
    'g_GtoG': 0.06*b2.nS,
    'p_GtoG': 1,
}


def run_command(par):
    start_time = wall_time()
    monitors = simulate_STN_GPe_population(par)
    sub_name = "{:.6f}-{:.6f}".format(par['par_syn']['g_StoG']/b2.nS,
                                      par['par_syn']['g_GtoG']/b2.nS)

    print("{:s} done in {:10.3f}".format(
        sub_name, wall_time() - start_time))
    # to_npz(monitors, subname="d-{}".format(sub_name),
    #        save_voltages=1, width=50*b2.ms)
    plot_voltage(monitors, indices=[0, 1, 2],
                 filename="v-{}".format(sub_name))
    plot_raster(monitors, filename="sp-{}".format(sub_name), par=par_sim)


if __name__ == "__main__":

    par_s['num'] = 10
    par_g['num'] = 10
    par_s['v0'] = (rand(par_s['num']) * 20 - 10 - 70) * b2.mV
    par_g['v0'] = (rand(par_g['num']) * 20 - 10 - 70) * b2.mV
    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 4000 * b2.ms,
        'dt': 0.05 * b2.ms,  # ! dt <= 0.05 ms
        "standalone_mode": 1}

    n_neighbors = 2
    Graph_GtoS = nx.watts_strogatz_graph(par_s['num'], n_neighbors, 0, seed=1)
    A = nx.to_numpy_array(Graph_GtoS, dtype=int)

    par_syn['adj_GtoS'] = A
    params = {"par_sim": par_sim,
              "par_syn": par_syn,
              "par_s": par_s,
              "par_g": par_g}

    g_StoG = np.linspace(0.01, 0.1, 3)
    g_GtoG = np.linspace(0.1, 0.1, 3)
    par_syn['g_GtoS'] = 2.5 * b2.nS
    RUN_IN_SERIAL = False
    RUN_IN_PARALLEL = True
    n_jobs = 4

    # ---------------------------------------------------------------

    if RUN_IN_PARALLEL:
        args = []
        for i in range(len(g_StoG)):
            for j in range(len(g_GtoG)):

                par_syn['g_StoG'] = g_StoG[i] * b2.nS
                par_syn['g_GtoG'] = g_GtoG[j] * b2.nS

                args.append(deepcopy(params))
        Parallel(n_jobs=n_jobs)(map(delayed(run_command), args))

    # ---------------------------------------------------------------

    if RUN_IN_SERIAL:
        for i in range(len(g_StoG)):
            for j in range(len(g_GtoG)):
                start_time = wall_time()
                print("g_StoG = {:10.6f}, g_GtoG = {:10.6f}".format(
                    g_StoG[i], g_GtoG[j]))
                sub_name = "{:.6f}-{:.6f}".format(g_StoG[i], g_GtoG[j])
                par_syn['g_StoG'] = g_StoG[i] * b2.nS
                par_syn['g_GtoG'] = g_GtoG[j] * b2.nS
                monitors = simulate_STN_GPe_population(params)
                print("{:s} Done in {:10.3f}".format(
                    sub_name, wall_time() - start_time))
                # to_npz(monitors, subname="d-{}".format(sub_name),
                #        save_voltages=1, width=50*b2.ms)
                # plot_voltage(monitors, indices=[0, 1, 2],
                #              filename="v-{}".format(sub_name))
                # plot_raster(monitors, filename="sp-{}".format(sub_name), par=par_sim)
