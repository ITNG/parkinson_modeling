import os
import numpy as np
import pylab as plt
import brian2 as b2
from os.path import join


# -------------------------------------------------------------------


def simulate_GPe_cell(par, par_sim):

    pid = os.getpid()
    b2.set_device('cpp_standalone', directory=join(
        "output", f"standalone-{pid}"))
    b2.get_device().reinit()
    b2.get_device().activate(
        directory=join("output", f"standalone-{pid}"))

    num = par['num']
    input_current = par['iapp']

    eqs = '''
    Iapp = input_current: amp 

    minf_Naf = 1.0 / (1.0 + exp((-39*mV - vg)/(5*mV))) :1
    hinf_Naf = 1.0 / (1.0 + exp(((-48*mV) - vg)/(-2.8*mV))) :1
	sinf_Naf = 0.15 + (1.0 - 0.15)/(1.0 + exp((-40*mV - vg)/(-5.4*mV))) :1
    tauh_Naf = 0.25*ms + (3.75*ms)/(exp((-43*mV - vg)/(10.*mV)) + exp((-43*mV - vg)/(-5*mV))) :second
	taus_Naf = 10*ms + (990*ms)/(exp((-40*mV - vg)/(18.3*mV)) + exp((-40*mV - vg)/(-10*mV))) :second
    dm_Naf/dt = (minf_Naf - m_Naf)/(0.028*ms) :1
    dh_Naf/dt = (hinf_Naf - h_Naf)/tauh_Naf :1
    ds_Naf/dt = (sinf_Naf - s_Naf)/taus_Naf :1
    iNaf = gnafbar*m_Naf**3*h_Naf*s_Naf*(vg-eNa) :amp

    iLeak = gleak * (vg-eLeak) :amp

    minf_Nap = 1.0 / (1.0 + exp((-57.7*mV - vg)/(5.7*mV))) :1
	hinf_Nap = 0.154 + (1.0 - 0.154)/ (1.0 + exp((-57*mV - vg)/(-4.*mV))) :1
	# sinf_Nap = 1.0 / (1.0 + exp((-10*mV - vg)/(-4.9*mV))) :1
	taum_Nap = 0.03*ms + (0.146*ms - 0.03*ms)/(exp((-42.6*mV - vg)/(14.4*mV)) + exp((-42.6*mV - vg)/(-14.4*mV))) :second
    tauh_Nap = 10*ms + (7*ms)/(exp((-34*mV - vg)/(26.*mV)) + exp((-34*mV - vg)/(-31.9*mV))):second
    # alphas_Nap = (-2.88e-6 * vg - 4.9e-5*mV)/mV/(1.0 - exp((vg + 17.014*mV)/(4.63*mV))) :1
	# betas_Nap = 1/mV*(6.94e-6 * vg + 4.47e-4*mV)/(1.0 - exp((vg + (4.47e-4)/(6.94e-6) * 1*mV)/(2.63*mV))) :1
	# taus_Nap = 1.0*ms / (alphas_Nap + betas_Nap) :second
    dm_Nap/dt = (minf_Nap - m_Nap)/taum_Nap :1
    dh_Nap/dt = (hinf_Nap - h_Nap)/tauh_Nap :1
    # ds_Nap/dt = (sinf_Nap - s_Nap)/taus_Nap :1
    # iNap = gnapbar*m_Nap**3*h_Nap*s_Nap*(vg-eNa) :amp
    iNap = gnapbar*m_Nap**3*h_Nap*(vg-eNa) :amp

    minf_Kv2 = 1.0 / (1.0 + exp((-33.2*mV - vg)/(9.1*mV))) :1
	hinf_Kv2 = 0.2 + (0.8) / (1.0 + exp((-20*mV - vg)/(-10.*mV))) :1
	taum_Kv2 = 0.1*ms + (2.9*ms)/(exp((-33.2*mV - vg)/(21.7*mV)) + exp((-33.2*mV - vg)/(-13.9*mV))) :second
    dm_Kv2/dt = (minf_Kv2 - m_Kv2)/taum_Kv2 :1
    dh_Kv2/dt = (hinf_Kv2 - h_Kv2)/(3400.*ms) :1
    iKv2 = gkv2bar*m_Kv2*4*h_Kv2*(vg-eK) :amp

    
    minf_Kv3 = 1.0 / (1.0 + exp((-26*mV - vg)/(7.8*mV))) :1
	hinf_Kv3 = 0.6 + (0.4) / (1.0 + exp((-20*mV - vg)/(-10.*mV))) :1
	taum_Kv3 = 0.1*ms + (13.9*ms)/(exp((-26*mV - vg)/(13.*mV)) + exp((-26*mV - vg)/(-12.*mV))) :second
	tauh_Kv3 = 7*ms + (26*ms)/(exp((-vg)/(10*mV)) + exp((-vg)/(-10*mV))): second
    dm_Kv3/dt = (minf_Kv3 - m_Kv3)/taum_Kv3 :1 
    dh_Kv3/dt = (hinf_Kv3 - h_Kv3)/tauh_Kv3 :1 
    iKv3 = gkv3bar*m_Kv3**4*h_Kv3*(vg-eK) :amp

    # Merged with kv4s
    # minf_Kv4f = 1.0 / (1.0 + exp((-49*mV - vg)/(12.5*mV))) :1
	# hinf_Kv4f = 1.0 / (1.0 + exp((-83*mV - vg)/(-10.*mV))) :1
	# taum_Kv4f = 0.25*ms + (6.75*ms)/(exp((-49*mV - vg)/(29*mV)) + exp((-49*mV - vg)/(-29.*mV))) :second
	# tauh_Kv4f = 7*ms + (14*ms)/(exp((-83*mV - vg)/(10*mV)) + exp((-83*mV - vg)/(-10.*mV))) :second
    # dm_Kv4f/dt = (minf_Kv4f - m_Kv4f)/taum_Kv4f :1
    # dh_Kv4f/dt = (hinf_Kv4f - h_Kv4f)/tauh_Kv4f :1
    # iKv4f = gkv4fbar*m_Kv4f**4*h_Kv4f*(vg-eK) :amp
    

    minf_Kv4s = 1.0 / (1.0 + exp((-49*mV - vg)/(12.5*mV))) :1
	hinf_Kv4s = 1.0 / (1.0 + exp((-83*mV - vg)/(-10.*mV))) :1
	taum_Kv4s = 0.25*ms + (6.75*ms)/(exp((-49*mV - vg)/(29*mV)) + exp((-49*mV - vg)/(-29.*mV))) :second
	tauh_Kv4s = 7*ms + (14*ms)/(exp((-83*mV - vg)/(10*mV)) + exp((-83*mV - vg)/(-10.*mV))) :second
    dm_Kv4s/dt = (minf_Kv4s - m_Kv4s)/taum_Kv4s :1
    dh_Kv4s/dt = (hinf_Kv4s - h_Kv4s)/tauh_Kv4s :1
    iKv4s = gkv4sbar*m_Kv4s**4*h_Kv4s*(vg-eK) :amp
    
    minf_Kc = 1.0 / (1.0 + exp((-61*mV - vg)/(19.5*mV))) :1
	taum_Kc = 6.7*ms + (93.3*ms)/(exp((-61*mV - vg)/(35.*mV)) + exp((-61*mV - vg)/(-25.*mV))) :second
    dm_Kc/dt = (minf_Kc - m_Kc)/taum_Kc :1
    iKc = gkcbar*m_Kc**4*(vg-eK) :amp

    minf_Hcn = 1.0 / (1.0 + exp((-76.4*mV - vg)/(-3.3*mV))) :1
	taum_Hcn = (3625*ms)/(exp((-76.4*mV - vg)/(6.56*mV)) + exp((-76.4*mV - vg)/(-7.48*mV))) :second
    dm_Hcn/dt = (minf_Hcn - m_Hcn)/taum_Hcn :1
    iHcn = ghcnbar*m_Hcn*(vg-eCat) :amp

    minf_Cah = 1.0 / (1.0 + exp((-20*mV - vg)/(7.*mV))) :1
    dm_Cah/dt = (minf_Cah - m_Cah)/(0.2*ms) :1
    iCah  = gcahbar*m_Cah*(vg-eCa) :amp

    # Ca Concentration
    dc_Ca/dt = -iCah*1/uA*3000/(2*96485)*1/ms- 0.4*(c_Ca - 0.01)*1/ms : 1
    Gcan = c_Ca**4.6 :1

    # KSK Current (Ca-dependent)
    minf_ksk = Gcan/(Gcan + Gcan50) :1
    tau_m_ksk = (76*ms-72*ms*c_Ca/5) * int(c_Ca < 5.) + 4*ms * int(c_Ca >= 5) :second
    dm_ksk/dt = (minf_ksk - m_ksk) / tau_m_ksk :1
    iKsk = gkskbar*m_ksk*(vg-eK) :amp


    membrain_Im = Iapp-iNaf-iLeak-iNap-iKv2-iKv3-iKv4s-iCah-iHcn-iKc-iKsk :amp

    dvg/dt = membrain_Im/Cm :volt
    '''

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vg>-20*mV',
                            refractory='vg>-20*mV',
                            namespace=par,
                            )

    neuron.vg = par['v0']
    # neuron.m_Naf = 'minf_Naf'
    # neuron.h_Naf = 'hinf_Naf'
    # neuron.s_Naf = 'sinf_Naf'
    # neuron.m_Nap = 'minf_Nap'
    # neuron.h_Nap = 'hinf_Nap'
    # neuron.m_Cah = 'minf_Cah'
    # neuron.m_ksk = 'minf_ksk'
    # neuron.m_Hcn = 'minf_Hcn'

    st_mon = b2.StateMonitor(neuron, par['record_from'], record=True)

    net = b2.Network(neuron)
    net.add(st_mon)
    net.run(par_sim['simulation_time'])

    return st_mon
# -------------------------------------------------------------------


def plot_data(st_mon, ax, index=0, **kwargs):
    ax.plot(st_mon.t / b2.ms, st_mon.vg[index] / b2.mV, **kwargs)
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
