import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (simulate_GPe_cell,
                 plot_data,
                 plot_current)
from input_factory import get_step_current                 

if __name__ == "__main__":

    par_g = {
        'num': 1,
        'v0': -55 * b2.mV,
        'vnag': 55.* b2.mV,
        'vkg': -80.* b2.mV,
        'vcag': 120.* b2.mV,
        'vlg': -55.* b2.mV,
        'gnag': 120. * b2.msiemens,
        'gkg': 30.* b2.msiemens,
        'gahpg': 30.* b2.msiemens,
        'gtg': .5* b2.msiemens,
        'gcag': .1* b2.msiemens,
        'glg': .1* b2.msiemens,

        'taurg': 30. *b2.ms,
        'taun0g': .05 *b2.ms,
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
        'kcag': 20., #Report:15,  Terman Rubin 2002: 20.0
        'phig': 1.,
        'phing': .05, # Report: 0.1, Terman Rubin 2002: 0.05
        'phihg': .05,
        'alphag': 2,
        'betag': .08,
        'epsg': 0.0001 / b2.ms,
        # 'iapp': 0.0 * b2.uamp,
        'C': 1 * b2.ufarad,
    }

    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 1000 * b2.ms,
        'dt': 0.01 * b2.ms,
    }

    start_time = time()
    # Figure 2 ------------------------------------------------------
    fig, ax = plt.subplots(4, figsize=(10, 6), sharex=True)
    i_stim = [20, 0, -0.5]
    for ii in range(len(i_stim)):
        input_current = get_step_current(0,
                                         1000,
                                         b2.ms,
                                         i_stim[ii] * b2.uamp)
        par_g['i_ext'] = input_current
        st_mon = simulate_GPe_cell(par_g, par_sim)
        plot_data(st_mon, ax[ii])
        plot_current(st_mon, ax[3], [-1, 22])
    plt.tight_layout()
    plt.savefig('figure_2.png')
    plt.close()

    print("Done in {} seconds".format(time() - start_time))
    
