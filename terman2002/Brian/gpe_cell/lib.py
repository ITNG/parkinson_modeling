import numpy as np
import pylab as plt
import brian2 as b2


def simulate_GPe_cell(par_g, par_sim):

    num = par_g['num']
    input_current = par_g['i_ext']

    eqs = '''
    i_extg = input_current(t, i): amp

    ainfg = 1 / (1 + exp(-(vg - thetaag*mV) / (sigag*mV))) : 1
    sinfg = 1 / (1 + exp(-(vg - thetasg*mV) / (sigsg*mV))) : 1
    rinfg = 1 / (1 + exp(-(vg - thetarg*mV) / (sigrg*mV))) : 1
    minfg = 1 / (1 + exp(-(vg - thetamg*mV) / (sigmg*mV))) : 1
    ninfg = 1 / (1 + exp(-(vg - thetang*mV) / (signg*mV))) : 1
    hinfg = 1 / (1 + exp(-(vg - thetahg*mV) / (sighg*mV))) : 1
    taung = taun0g + taun1g / (1 + exp(-(vg - thngt*mV) / (sng*mV))) : second
    tauhg = tauh0g + tauh1g / (1 + exp(-(vg - thhgt*mV) / (shg*mV))) : second

    itg = gtg * (ainfg ** 3) * rg * (vg - vcag) : amp
    inag = gnag * (minfg ** 3) * hg * (vg - vnag) : amp
    ikg = gkg * (ng ** 4) * (vg - vkg) : amp
    iahpg = gahpg * (vg - vkg) * cag / (cag + k1g) : amp
    icag = gcag * (sinfg ** 2) * (vg - vcag) : amp
    ilg = glg * (vg - vlg) : amp

    membrane_Im =  -(itg + inag + ikg + iahpg + icag + ilg) + i_extg : amp
    dhg/dt = phihg*(hinfg-hg)/tauhg : 1
    dng/dt = phing*(ninfg-ng)/taung : 1
    drg/dt = phig*(rinfg-rg)/taurg  : 1
    dcag/dt= epsg*((-icag-itg)/uamp - kcag*cag) : 1

    dvg/dt = membrane_Im / C : volt
    '''

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vg>-20*mV',
                            refractory='vg>-20*mV',
                            namespace=par_g,
                            )

    neuron.vg = par_g['v0']
    neuron.hg = "hinfg"
    neuron.ng = "ninfg"
    neuron.rg = "rinfg"
    neuron.cag = 0
    # neuron.i_extg = par_g['iapp']

    st_mon = b2.StateMonitor(neuron, ["vg", 'i_extg'], record=True)

    net = b2.Network(neuron)
    net.add(st_mon)
    net.run(par_sim['simulation_time'])

    return st_mon


def plot_data(st_mon, ax):
    ax.plot(st_mon.t / b2.ms, st_mon.vg[0] / b2.mV, lw=2, c='k')
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("v [mV]", fontsize=14)


def plot_current(st_mon, ax, ylim=None):
    ax.plot(st_mon.t / b2.ms,
            st_mon.i_extg[0] / b2.uamp, lw=1, c='k', alpha=0.5)
    ax.set_xlabel("t [ms]", fontsize=14)
    ax.set_ylabel("I [micro A]", fontsize=14)
    if ylim is not None:
        ax.set_ylim(ylim)
