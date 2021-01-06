import numpy as np
import pylab as plt
import brian2 as b2


def simulate_two_GPe_cell(par_g, par_sim):

    b2.defaultclock.dt = par_sim['dt']

    eqs = '''
    i_extg : amp

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
    i_syn_gg : amp

    Hinf = 1./(1+exp(-(vg - thetag * mV - thetagH*mV) / (sigmagH*mV))):1
    ds_gg/dt = alpha * Hinf * (1 - s_gg) - beta * s_gg  : 1

    membrane_Im =  -(itg + inag + ikg + iahpg + icag + ilg) + i_extg + i_syn_gg : amp
    dhg/dt = phihg*(hinfg-hg)/tauhg : 1
    dng/dt = phing*(ninfg-ng)/taung : 1
    drg/dt = phig*(rinfg-rg)/taurg  : 1
    dcag/dt= epsg*((-icag-itg)/pA - kcag*cag) : 1  #! pA

    dvg/dt = membrane_Im / C : volt
    '''

    neurons = b2.NeuronGroup(2,
                             eqs,
                             method=par_sim['integration_method'],
                             dt=par_sim['dt'],
                             threshold='vg>-20*mV',
                             refractory='vg>-20*mV',
                             namespace=par_g,
                             )
    eqs_syn_gg = '''  # synapse model, g to g
    i_syn_gg_post = w_gg * s_gg_pre * (v_rev_gg - vg) :amp (summed)
    '''

    syn_gg = b2.Synapses(neurons, neurons,
                         eqs_syn_gg,
                         dt=par_sim['dt'],
                         method=par_sim['integration_method'],
                         namespace=par_g,
                         )
    syn_gg.connect(i=0, j=1)

    neurons.vg = par_g['v0']
    neurons.hg = "hinfg"
    neurons.ng = "ninfg"
    neurons.rg = "rinfg"
    neurons.cag = 0
    neurons.i_extg = par_g['iapp']

    st_mon = b2.StateMonitor(neurons, ["vg", 'i_syn_gg', 's_gg'],
                             record=True)

    net = b2.Network(neurons)
    net.add(syn_gg)
    net.add(st_mon)
    net.run(par_sim['simulation_time'])

    return st_mon


def simulate_GPe_cell(par_g, par_sim):

    num = par_g['num']
    input_current = par_g['i_ext']
    b2.defaultclock.dt = par_sim['dt']

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
    dcag/dt= epsg*((-icag-itg)/pA - kcag*cag) : 1  #! pA 

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


def plot_data(st_mon, ax, title=None):
    ax.plot(st_mon.t / b2.ms, st_mon.vg[0] / b2.mV, lw=2, c='k')
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("v [mV]", fontsize=14)
    if title is not None:
        ax.set_title("I = {:s}".format(title))


def plot_current(st_mon, current_unit=b2.pA, ax=None, ylim=None):
    ax.plot(st_mon.t / b2.ms,
            st_mon.i_extg[0] / current_unit, lw=1, c='k', alpha=0.5)
    ax.set_xlabel("t [ms]", fontsize=14)
    ax.set_ylabel("I [{}]".format(str(current_unit)), fontsize=14)
    if ylim is not None:
        ax.set_ylim(ylim)
