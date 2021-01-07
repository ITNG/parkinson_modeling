import numpy as np
import brian2 as b2
import pylab as plt
from time import time
import networkx as nx
from numpy.random import rand
from lib import simulate_STN_GPe_population
from plotting import plot_raster, plot_voltage

np.random.seed(2)

par_s = {
    # 'v0': -60 * b2.mV,
    'vl': -60 * b2.mV,
    'vna': 55 * b2.mV,
    'vk': -80 * b2.mV,
    'vca': 140 * b2.mV,
    'gl': 2.25 * b2.nS,
    'gna': 37.5 * b2.nS,
    'gk': 45 * b2.nS,
    'gahp': 9 * b2.nS,
    'gca': 0.5 * b2.nS,
    'gt': 0.5 * b2.nS,
    'thetam': -30,
    'thetas': -39,
    'thetah': -39,
    'thetan': -32,
    'thetab': 0.4,  # Guo 0.25 Terman02 0.4
    'thetaa': -63,
    'thetar': -67,
    'sigmas': 8,
    'sigmah': -3.1,
    'sigman': 8,
    'sigmaa': 7.8,
    'sigmab': -0.1,  # Guo 0.07 Terman02 -0.1
    'sigmam': 15,
    'sigmar': -2,
    'sigmaht': -3,
    'sigmant': -26,
    'sigmart': -2.2,
    'taun0': 1 * b2.ms,
    'taun1': 100 * b2.ms,
    'taur0': 40.0 * b2.ms,  # 7.1, Terman02 40.0
    'taur1': 17.5 * b2.ms,
    'tauh0': 1 * b2.ms,
    'tauh1': 500 * b2.ms,
    'eps': 3.75e-05 / b2.ms,  # 1/ms Guo 0.00005 Terman02 0.0000375
    'phir': 0.2,  # Guo 0.5 Terman02 0.2
    'phi': 0.75,
    'kca': 22.5,
    'thn': -80,
    'thh': -57,
    'thr': 68,
    'ab': -30,
    'k1': 15,
    # 'alpha': 5. / b2.ms,
    # 'beta': 1. / b2.ms,
    'i_ext': 0 * b2.pA,
    'C': 1 * b2.pF,
    'thetag_s': 30.,
    'thetagH_s': -39.,
    'sigmagH_s': 8.,
}

par_g = {
    # 'v0': -55 * b2.mV,
    'vnag': 55. * b2.mV,
    'vkg': -80. * b2.mV,
    'vcag': 120. * b2.mV,
    'vlg': -55. * b2.mV,
    'gnag': 120. * b2.nS,
    'gkg': 30. * b2.nS,
    'gahpg': 30. * b2.nS,
    'gtg': .5 * b2.nS,
    'gcag': .1 * b2.nS,
    'glg': .1 * b2.nS,
    'taurg': 30. * b2.ms,
    'taun0g': .05 * b2.ms,
    'taun1g': .27*b2.ms,
    'tauh0g': .05*b2.ms,
    'tauh1g': .27*b2.ms,
    'sigag': 2.,
    'sigsg': 2.,
    'sigrg': -2.,
    'sigmg': 10.,
    'signg': 14.,
    'sighg': -12,
    'thetasg': -35.,
    'thetaag': -57.,
    'thetarg': -70.,
    'thetamg': -37.,
    'thetang': -50.,
    'thetahg': -58,
    'thngt': -40,
    'thhgt': -40,
    'sng': -12,
    'shg': -12,
    'k1g': 30.,
    'kcag': 20.,  # Report:15,  Terman Rubin 2002: 20.0
    'phig': 1.,
    'phing': 0.05,  # Report: 0.1, Terman Rubin 2002: 0.05
    'phihg': 0.05,
    'epsg': 0.0001 / b2.ms,
    'i_ext': -1.2 * b2.pA,
    'C': 1 * b2.pF,
    'thetag_g': 20.,
    'thetagH_g': -57.,
    'sigmagH_g': 2.,
}

par_syn = {
    'v_rev_gg': -100. * b2.mV,
    'v_rev_sg': 0. * b2.mV,
    'v_rev_gs': -85. * b2.mV,
    'alphas': 5. / b2.ms,
    'betas': 1. / b2.ms,
    'alphag': 2. / b2.ms,
    'betag': 0.08 / b2.ms,
}

par_s['num'] = 1
par_g['num'] = 1
par_s['v0'] = (rand(par_s['num']) * 20 - 10 - 70) * b2.mV
par_g['v0'] = (rand(par_g['num']) * 20 - 10 - 70) * b2.mV


g_hat_gs = 2.5*b2.nS
g_hat_sg = 0.03*b2.nS
g_hat_gg = 0.06*b2.nS

# par_syn['p_sg'] =
par_syn['p_gs'] = 3 / par_g['num']
par_syn['p_sg'] = 1 / par_g['num']
par_syn['p_gg'] = 1

par_syn['g_gs'] = g_hat_gs  / (par_syn['p_gs'] * par_g['num'])
par_syn['g_sg'] = g_hat_sg  / (par_syn['p_sg'] * par_s['num'])
par_syn['g_gg'] = g_hat_gg  / (par_syn['p_gg'] * par_g['num'])

par_sim = {
    'integration_method': "rk4",
    'simulation_time': 4000 * b2.ms,
    'dt': 0.1 * b2.ms,
}


if __name__ == "__main__":

    start_time = time()
    state = "sparse"
    K = 2
    # G_gs = nx.watts_strogatz_graph(par_s['num'], K, 0, seed=1)
    # A = nx.to_numpy_array(G_gs, dtype=int)
    # A = np.asarray([[0, 1],[1, 0]])
    # par_syn['adj_gs'] = A

    monitors = simulate_STN_GPe_population(par_s,
                                           par_g,
                                           par_syn,
                                           par_sim)

    print("Done in {}".format(time() - start_time))
    plot_voltage(monitors, indices=[0,0], filename="v_{}".format(state))
    # plot_raster(monitors, filename="sp_{}".format(state))
