import os
import numpy as np
import pylab as plt
import brian2 as b2
from os.path import join


# -------------------------------------------------------------------

def simulate_FSI_cell(par, par_sim):

    pid = os.getpid()
    b2.set_device('cpp_standalone', directory=join(
        "output", f"standalone-{pid}"))
    b2.get_device().reinit()
    b2.get_device().activate(
        directory=join("output", f"standalone-{pid}"))

    num = par['num']
    input_current = par['i_ext']

    eqs = '''
    Iapp = input_current(t, i): amp 

    m_inf = 1.0/(1.0+exp(-(vf+24*mV)/(11.5*mV))):1
    h_inf = 1.0/(1.0+exp(-(vf+58.3*mV)/(-6.7*mV))):1
    n_inf = 1.0/(1.0+exp(-(vf+12.4*mV)/(6.8*mV))):1
    a_inf = 1.0/(1.0+exp(-(vf+50*mV)/(20.*mV))):1
    b_inf = 1.0/(1.0+exp(-(vf+70*mV)/(-6.*mV))):1
    tau_h=0.5*ms+14.0*ms/(1.0+exp(-(vf+60*mV)/(-12.*mV))):second
    tau_n=(0.087*ms+11.4*ms/(1.0+exp(-(vf+14.6*mV)/(-8.6*mV))))
         *(0.087+11.4/(1.0+exp(-(vf-1.3*mV)/(18.7*mV)))) :second

    membrain_Im = -gNa*m_inf**3 *h*(vf-50*mV)
                  -gK*(n**power_n)*(vf+90*mV)
                  -gL*(vf+70*mV)-gA*a**3*b*(vf+90*mV)+Iapp:amp
    dh/dt=(h_inf-h)/tau_h :1
    dn/dt=(n_inf-n)/tau_n :1
    da/dt=(a_inf-a)/(2.*ms) :1
    db/dt=(b_inf-b)/(150.*ms) :1
    dvf/dt=membrain_Im/Cm :volt
    '''

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vf>-20*mV',
                            refractory='vf>-20*mV',
                            namespace=par,
                            )

    neuron.vf = par['v0']
    neuron.h = "h_inf"
    neuron.n = "n_inf"
    neuron.a = "a_inf"
    neuron.b = "b_inf"

    st_mon = b2.StateMonitor(neuron, ["vf", "Iapp"], record=True)

    net = b2.Network(neuron)
    net.add(st_mon)
    net.run(par_sim['simulation_time'])

    return st_mon
# -------------------------------------------------------------------


def plot_data(st_mon, ax, index=0):
    ax.plot(st_mon.t / b2.ms, st_mon.vf[index] / b2.mV, lw=2, c='k')
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("v [mV]", fontsize=14)
# -------------------------------------------------------------------


def plot_current(st_mon, ax, current_unit=b2.pA):
    ax.plot(st_mon.t / b2.ms,
            st_mon.Iapp[0] / current_unit, lw=1, c='k', alpha=0.5)

    ylabel = "I [{}]".format(str(current_unit))

    ax.set_xlabel("t [ms]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    I = st_mon.Iapp[0] / current_unit
    ax.set_ylim(np.min(I)*1.1, np.max(I)*1.1)


def clean_directory():
    os.system("rm -rf output")
