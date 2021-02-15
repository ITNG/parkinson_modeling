import os
import numpy as np
import pylab as plt
import brian2 as b2
from os.path import join


# -------------------------------------------------------------------


def simulate_MSN_cell(par, par_sim):

    pid = os.getpid()
    b2.set_device('cpp_standalone', directory=join(
        "output", f"standalone-{pid}"))
    b2.get_device().reinit()
    b2.get_device().activate(
        directory=join("output", f"standalone-{pid}"))

    num = par['num']
    input_current = par['iapp']

    eqs = '''
    Iapp = input_current(t): amp 

    am_Na = -0.1*(vm+28*mV)/(1*mV)/(exp(-0.1*(vm+28*mV)/(1*mV))-1) :1
    Bm_Na = 4*exp(-(vm+53*mV)/(18.*mV)) :1
    ah_Na = 0.07*exp(-(vm+51*mV)/(20.*mV)) :1
    Bh_Na =  1/(1+exp(-0.1*(vm+21*mV)/(1.*mV))) :1
    minf_Na = am_Na/(am_Na+Bm_Na) :1
    dh_Na/dt= 5*(ah_Na*(1-h_Na)-Bh_Na*h_Na) * 1/ms :1
    iNa = gnabar*minf_Na**3*h_Na*(vm - eNa) :amp
    
    an_K = (-0.01*(vm+27*mV)/(1*mV)/(exp(-0.1*(vm+27*mV)/(1*mV))-1)) :1
    Bn_K = 0.125*exp(-(vm+37*mV)/(80.*mV)) :1
    dn_K/dt=5*(an_K*(1-n_K)-Bn_K * n_K) * 1/ms :1
    iK = gkbar*n_K**4*(vm - eK) :amp

    iLeak = gleak*(vm - eLeak) :amp

    minf_Kir=1/(1+exp(-(vm+100*mV)/(-10.*mV))) :1
    dm_Kir/dt = (minf_Kir - m_Kir) / tau_m_Kir :1
    iKir = gkirbar*m_Kir*(vm - eKir) :amp    
    
    hinf_Kaf=1/(1+exp(-(vm+70.4*mV)/(-7.6*mV))) :1
    minf_Kaf=1/(1+exp(-(vm+33.1*mV)/(7.5*mV))) :1
    tau_m_Kaf = 1./tadj *1*ms :second
    tau_h_Kaf = 25./tadj *1*ms :second
    dm_Kaf/dt = (minf_Kaf - m_Kaf) / tau_m_Kaf :1
	dh_Kaf/dt = (hinf_Kaf - h_Kaf) / tau_h_Kaf :1
    iKaf = gkafbar * m_Kaf * h_Kaf*(vm - eKaf) :amp

    minf_Kas=1/(1+exp(-(vm+25.6*mV)/(13.3*mV))) :1
	hinf_Kas=1/(1+exp(-(vm+78.8*mV)/(-10.4*mV))) :1
	tau_m_Kas=131.4/(exp(-(vm+37.4*mV)/(27.3*mV))+exp((vm+37.4*mV)/(27.3*mV)))/tadj *1*ms :second
	tau_h_Kas=(1790+2930*exp(-((vm+38.2*mV)/(28*mV))**2)*((vm+38.2*mV)/(28*mV)))/tadj *1*ms :second
    dm_Kas/dt = ( minf_Kas - m_Kas ) / tau_m_Kas :1
	dh_Kas/dt = ( hinf_Kas - h_Kas ) / tau_h_Kas :1
    iKas = gkasbar*m_Kas*h_Kas*(vm - eKas) :amp

    minf_Krp=1/(1+exp(-(vm+13.4*mV)/(12.1*mV))) :1
	hinf_Krp=1/(1+exp(-(vm+55*mV)/(-19*mV))) :1
    tau_m_Krp=206.2/(exp(-(vm+53.9*mV)/(26.5*mV))+exp((vm+53.9*mV)/(26.5*mV)))/tadj *1*ms:second
	tau_h_Krp=3*(1790+2930*exp(-((vm+38.2*mV)/(28*mV))**2)*((vm+38.2*mV)/(28*mV)))/tadj *1*ms :second
    dm_Krp/dt = ( minf_Krp - m_Krp ) / tau_m_Krp :1
	dh_Krp/dt = ( hinf_Krp - h_Krp ) / tau_h_Krp :1
    iKrp = gkrpbar*m_Krp*h_Krp*(vm - eKrp) :amp

    minf_Nap=1/(1+exp(-(vm+47.8*mV)/(3.1*mV))) :1
	tau_m_Nap=1/tadj *1*ms:second
    dm_Nap/dt = (minf_Nap - m_Nap) / tau_m_Nap :1
    iNap = gnapbar*m_Nap*(vm - eNap) :amp
    
    minf_Nas=1/(1+exp(-(vm+16*mV)/(9.4*mV))) :1
	tau_m_Nas=637.8/(exp(-(vm+33.5*mV)/(26.3*mV))+exp((vm+33.5*mV)/(26.3*mV)))/tadj_Nas *1*ms:second
    dm_Nas/dt = ( minf_Nas - m_Nas ) / tau_m_Nas :1
    iNas = gnasbar * m_Nas *(vm - eNas) :amp

    membrain_Im = Iapp-iLeak-iNa-iK-iKir-iKaf-iKrp-iNap-iNas :amp # iKas
    dvm/dt = membrain_Im/Cm :volt
    '''

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vm>-20*mV',
                            refractory='vm>-20*mV',
                            namespace=par,
                            )

    neuron.vm = par['v0']
    neuron.h_Na = 'ah_Na/(ah_Na + Bh_Na)' # 'hinf_Na'
    neuron.n_K = 'an_K/(an_K + Bn_K)'  # ninf_K
    neuron.m_Kir = 'minf_Kir'
    neuron.m_Kaf = 'minf_Kaf'
    neuron.h_Kaf = 'hinf_Kaf'
    neuron.m_Kas = 'minf_Kas'
    neuron.h_Kas = 'hinf_Kas'
    neuron.m_Krp = 'minf_Krp'
    neuron.h_Krp = 'hinf_Krp'
    neuron.m_Nap = 'minf_Nap'
    neuron.m_Nas = 'minf_Nas'

    st_mon = b2.StateMonitor(neuron, par['record_from'], record=True)

    net = b2.Network(neuron)
    net.add(st_mon)
    net.run(par_sim['simulation_time'])

    return st_mon
# -------------------------------------------------------------------


def plot_data(st_mon, ax, index=0, **kwargs):
    ax.plot(st_mon.t / b2.ms, st_mon.vm[index] / b2.mV,**kwargs)
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("v [mV]", fontsize=14)
# -------------------------------------------------------------------
def plot_h(st_mon, ax, index=0):
    ax.plot(st_mon.t / b2.ms, st_mon.h_Kas[index], lw=1, c='k')
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("h", fontsize=14)


def plot_m(st_mon, ax, index=0):
    ax.plot(st_mon.t / b2.ms, st_mon.m_Kas[index], lw=1, c='k')
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("m", fontsize=14)


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

