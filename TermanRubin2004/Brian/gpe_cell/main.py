import numpy as np
import brian2 as b2
import pylab as plt
from time import time
from lib import (simulate_GPe_cell,
                 simulate_two_GPe_cell,
                 plot_data,
                 plot_current)
from input_factory import get_step_current

if __name__ == "__main__":

    par_g = {
        'num': 1,
        'v0': -55 * b2.mV,
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
        'phing': .05,  # Report: 0.1, Terman Rubin 2002: 0.05
        'phihg': .05,
        'alphag': 2,
        'betag': .08,
        'epsg': 0.0001 / b2.ms,
        # 'iapp': 0.0 * b2.uamp,
        'C': 1 * b2.pF,
        'w_gg': 1.0*b2.nS,
        'thetag': 20.,
        'thetagH': -57.,
        'sigmagH': 2.,
        'beta': 0.08 /b2.ms,
        'alpha': 2/b2.ms,
        "v_rev_gg":-100.*b2.mV
    }

    par_sim = {
        'integration_method': "rk4",
        'simulation_time': 1000 * b2.ms,
        'dt': 0.01 * b2.ms,
    }

    def simulate_1_gpe_cell():
        start_time = time()
        current_unit = b2.pA
        # Figure 2 ------------------------------------------------------
        fig, ax = plt.subplots(5, figsize=(10, 8))
        i_stim = [20, 0, -0.5, 170.]
        for ii in range(len(i_stim)):

            input_current = get_step_current(0,
                                             1000,
                                             b2.ms,
                                             i_stim[ii] * current_unit)
            if ii == 3:
                input_current = get_step_current(100,
                                                 250,
                                                 b2.ms,
                                                 i_stim[ii] * current_unit)
                par_sim['simulation_time'] = 350 * b2.ms

            par_g['i_ext'] = input_current
            st_mon = simulate_GPe_cell(par_g, par_sim)
            plot_data(st_mon, ax[ii], title=str(i_stim[ii]))
            plot_current(st_mon, current_unit, ax[-1], [-1, 22])

        print("Done in {} seconds".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('figure_2.png')
        plt.close()

    def simulate_2_gpe_cell():
        start_time = time()
        current_unit = b2.pA
        par_sim['simulation_time'] = 300 * b2.ms
        fig, ax = plt.subplots(3, figsize=(10, 6), sharex=True)

        par_g['iapp'] = -0.5 *b2.pA
        st_mon = simulate_two_GPe_cell(par_g, par_sim)
        plot_data(st_mon, ax[0])
        ax[0].plot(st_mon.t / b2.ms, st_mon.vg[0] /
                   b2.mV, lw=2, c='b', label="0")
        ax[0].plot(st_mon.t / b2.ms, st_mon.vg[1] /
                   b2.mV, lw=2, c='r', label="1")
        ax[1].plot(st_mon.t/b2.ms, st_mon.s_gg[0], lw=2, c='b', label="0")
        ax[1].plot(st_mon.t/b2.ms, st_mon.s_gg[1], lw=2, c='r', label="1")
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlim(0, np.max(st_mon.t / b2.ms))
        ax[0].set_ylabel("v [mV]", fontsize=14)
        ax[1].set_ylabel("s_syn", fontsize=14)

        ax[-1].plot(st_mon.t / b2.ms,
                    st_mon.i_syn_gg[1] / current_unit, lw=1, c='k', alpha=0.5)
        ax[-1].set_xlabel("t [ms]", fontsize=14)
        ax[-1].set_ylabel("I_syn [{}]".format(str(current_unit)), fontsize=14)
        # ax[-1].set_xlim(290, 310)

        print("Done in {} seconds".format(time() - start_time))
        plt.tight_layout()
        plt.savefig('figure_2_gpe.png')
        plt.close()

    simulate_1_gpe_cell()
    # simulate_2_gpe_cell()
