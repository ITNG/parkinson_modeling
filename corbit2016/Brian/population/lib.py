import os
import numpy as np
import pylab as plt
import brian2 as b2
from os.path import join


# -------------------------------------------------------------------


def simulate_circuit(par):

    par_M = par['par_M']
    par_F = par['par_F']
    par_G = par['par_G']
    par_sim = par['par_sim']

    pid = os.getpid()
    b2.set_device('cpp_standalone', directory=join(
        "output", f"standalone-{pid}"))
    b2.get_device().reinit()
    b2.get_device().activate(
        directory=join("output", f"standalone-{pid}"))

    num_M = par_M['num']
    num_G = par_G['num']
    num_F = par_F['num']

    input_current_M = par_M['iapp']
    input_current_G = par_G['iapp']
    # input_current_F = par_F['iapp']

    #################################################################
    # MSN Neuron
    #################################################################
    eqs_M = '''
    M_Iapp = input_current_M(t): amp 

    M_am_Na = -0.1*(vm+28*mV)/(1*mV)/(exp(-0.1*(vm+28*mV)/(1*mV))-1) :1
    M_Bm_Na = 4*exp(-(vm+53*mV)/(18.*mV)) :1
    M_ah_Na = 0.07*exp(-(vm+51*mV)/(20.*mV)) :1
    M_Bh_Na =  1/(1+exp(-0.1*(vm+21*mV)/(1.*mV))) :1
    M_minf_Na = M_am_Na/(M_am_Na+M_Bm_Na) :1
    dM_h_Na/dt= 5*(M_ah_Na*(1-M_h_Na)-M_Bh_Na*M_h_Na) * 1/ms :1
    M_iNa = gnabar*M_minf_Na**3*M_h_Na*(vm - eNa) :amp
    
    M_an_K = (-0.01*(vm+27*mV)/(1*mV)/(exp(-0.1*(vm+27*mV)/(1*mV))-1)) :1
    M_Bn_K = 0.125*exp(-(vm+37*mV)/(80.*mV)) :1
    dM_n_K/dt=5*(M_an_K*(1-M_n_K)-M_Bn_K * M_n_K) * 1/ms :1
    M_iK = gkbar*M_n_K**4*(vm - eK) :amp

    M_iLeak = gleak*(vm - eLeak) :amp

    M_minf_Kir=1/(1+exp(-(vm+100*mV)/(-10.*mV))) :1
    dM_m_Kir/dt = (M_minf_Kir - M_m_Kir) / tau_m_Kir :1
    M_iKir = gkirbar*M_m_Kir*(vm - eKir) :amp    
    
    M_hinf_Kaf=1/(1+exp(-(vm+70.4*mV)/(-7.6*mV))) :1
    M_minf_Kaf=1/(1+exp(-(vm+33.1*mV)/(7.5*mV))) :1
    M_tau_m_Kaf = 1./tadj *1*ms :second
    M_tau_h_Kaf = 25./tadj *1*ms :second
    dM_m_Kaf/dt = (M_minf_Kaf - M_m_Kaf) / M_tau_m_Kaf :1
	dM_h_Kaf/dt = (M_hinf_Kaf - M_h_Kaf) / M_tau_h_Kaf :1
    M_iKaf = gkafbar * M_m_Kaf * M_h_Kaf*(vm - eKaf) :amp

    M_minf_Kas=1/(1+exp(-(vm+25.6*mV)/(13.3*mV))) :1
	M_hinf_Kas=1/(1+exp(-(vm+78.8*mV)/(-10.4*mV))) :1
	M_tau_m_Kas=131.4/(exp(-(vm+37.4*mV)/(27.3*mV))+exp((vm+37.4*mV)/(27.3*mV)))/tadj *1*ms :second
	M_tau_h_Kas=(1790+2930*exp(-((vm+38.2*mV)/(28*mV))**2)*((vm+38.2*mV)/(28*mV)))/tadj *1*ms :second
    dM_m_Kas/dt = ( M_minf_Kas - M_m_Kas ) / M_tau_m_Kas :1
	dM_h_Kas/dt = ( M_hinf_Kas - M_h_Kas ) / M_tau_h_Kas :1
    M_iKas = gkasbar*M_m_Kas*M_h_Kas*(vm - eKas) :amp

    M_minf_Krp=1/(1+exp(-(vm+13.4*mV)/(12.1*mV))) :1
	M_hinf_Krp=1/(1+exp(-(vm+55*mV)/(-19*mV))) :1
    M_tau_m_Krp=206.2/(exp(-(vm+53.9*mV)/(26.5*mV))+exp((vm+53.9*mV)/(26.5*mV)))/tadj *1*ms:second
	M_tau_h_Krp=3*(1790+2930*exp(-((vm+38.2*mV)/(28*mV))**2)*((vm+38.2*mV)/(28*mV)))/tadj *1*ms :second
    dM_m_Krp/dt = ( M_minf_Krp - M_m_Krp ) / M_tau_m_Krp :1
	dM_h_Krp/dt = ( M_hinf_Krp - M_h_Krp ) / M_tau_h_Krp :1
    M_iKrp = gkrpbar*M_m_Krp*M_h_Krp*(vm - eKrp) :amp

    M_minf_Nap=1/(1+exp(-(vm+47.8*mV)/(3.1*mV))) :1
	M_tau_m_Nap=1/tadj *1*ms:second
    dM_m_Nap/dt = (M_minf_Nap - M_m_Nap) / M_tau_m_Nap :1
    M_iNap = gnapbar*M_m_Nap*(vm - eNap) :amp
    
    M_minf_Nas=1/(1+exp(-(vm+16*mV)/(9.4*mV))) :1
	M_tau_m_Nas=637.8/(exp(-(vm+33.5*mV)/(26.3*mV))+exp((vm+33.5*mV)/(26.3*mV)))/tadj_Nas *1*ms:second
    dM_m_Nas/dt = (M_minf_Nas - M_m_Nas) / M_tau_m_Nas :1
    M_iNas = gnasbar * M_m_Nas *(vm - eNas) :amp

    M_membrain_Im = M_Iapp-M_iLeak-M_iNa-M_iK-M_iKir
                 -M_iKaf-M_iKrp-M_iNap-M_iNas-M_iKas :amp  
    dvm/dt = M_membrain_Im/Cm :volt
    '''

    #################################################################
    # GPe Neuron
    #################################################################

    eqs_G = '''
    Iapp = input_current_G: amp 

    G_minf_Naf = 1.0 / (1.0 + exp((-39*mV - vg)/(5*mV))) :1
    G_hinf_Naf = 1.0 / (1.0 + exp(((-48*mV) - vg)/(-2.8*mV))) :1
	G_sinf_Naf = 0.15 + (1.0 - 0.15)/(1.0 + exp((-40*mV - vg)/(-5.4*mV))) :1
    G_tauh_Naf = 0.25*ms + (3.75*ms)/(exp((-43*mV - vg)/(10.*mV)) + exp((-43*mV - vg)/(-5*mV))) :second
	G_taus_Naf = 10*ms + (990*ms)/(exp((-40*mV - vg)/(18.3*mV)) + exp((-40*mV - vg)/(-10*mV))) :second
    dG_m_Naf/dt = (G_minf_Naf - G_m_Naf)/(0.028*ms) :1
    dG_h_Naf/dt = (G_hinf_Naf - G_h_Naf)/G_tauh_Naf :1
    dG_s_Naf/dt = (G_sinf_Naf - G_s_Naf)/G_taus_Naf :1
    G_iNaf = gnafbar*G_m_Naf**3*G_h_Naf*G_s_Naf*(vg-eNa) :amp

    iLeak = gleak * (vg-eLeak) :amp

    G_minf_Nap = 1.0 / (1.0 + exp((-57.7*mV - vg)/(5.7*mV))) :1
	G_hinf_Nap = 0.154 + (1.0 - 0.154)/ (1.0 + exp((-57*mV - vg)/(-4.*mV))) :1
	# sinf_Nap = 1.0 / (1.0 + exp((-10*mV - vg)/(-4.9*mV))) :1
	G_taum_Nap = 0.03*ms + (0.146*ms - 0.03*ms)/(exp((-42.6*mV - vg)/(14.4*mV)) + exp((-42.6*mV - vg)/(-14.4*mV))) :second
    G_tauh_Nap = 10*ms + (7*ms)/(exp((-34*mV - vg)/(26.*mV)) + exp((-34*mV - vg)/(-31.9*mV))):second
    # G_alphas_Nap = (-2.88e-6 * vg - 4.9e-5*mV)/mV/(1.0 - exp((vg + 17.014*mV)/(4.63*mV))) :1
	# G_betas_Nap = 1/mV*(6.94e-6 * vg + 4.47e-4*mV)/(1.0 - exp((vg + (4.47e-4)/(6.94e-6) * 1*mV)/(2.63*mV))) :1
	# G_taus_Nap = 1.0*ms / (G_alphas_Nap + G_betas_Nap) :second
    dG_m_Nap/dt = (G_minf_Nap - G_m_Nap)/G_taum_Nap :1
    dG_h_Nap/dt = (G_hinf_Nap - G_h_Nap)/G_tauh_Nap :1
    # ds_Nap/dt = (G_sinf_Nap - G_s_Nap)/G_taus_Nap :1
    # G_iNap = gnapbar*G_m_Nap**3*G_h_Nap*G_s_Nap*(vg-eNa) :amp
    G_iNap = gnapbar*G_m_Nap**3*G_h_Nap*(vg-eNa) :amp

    G_minf_Kv2 = 1.0 / (1.0 + exp((-33.2*mV - vg)/(9.1*mV))) :1
	G_hinf_Kv2 = 0.2 + (0.8) / (1.0 + exp((-20*mV - vg)/(-10.*mV))) :1
	G_taum_Kv2 = 0.1*ms + (2.9*ms)/(exp((-33.2*mV - vg)/(21.7*mV)) + exp((-33.2*mV - vg)/(-13.9*mV))) :second
    dG_m_Kv2/dt = (G_minf_Kv2 - G_m_Kv2)/G_taum_Kv2 :1
    dG_h_Kv2/dt = (G_hinf_Kv2 - G_h_Kv2)/(3400.*ms) :1
    G_iKv2 = gkv2bar*G_m_Kv2*4*G_h_Kv2*(vg-eK) :amp

    
    G_minf_Kv3 = 1.0 / (1.0 + exp((-26*mV - vg)/(7.8*mV))) :1
	G_hinf_Kv3 = 0.6 + (0.4) / (1.0 + exp((-20*mV - vg)/(-10.*mV))) :1
	G_taum_Kv3 = 0.1*ms + (13.9*ms)/(exp((-26*mV - vg)/(13.*mV)) + exp((-26*mV - vg)/(-12.*mV))) :second
	G_tauh_Kv3 = 7*ms + (26*ms)/(exp((-vg)/(10*mV)) + exp((-vg)/(-10*mV))): second
    dG_m_Kv3/dt = (G_minf_Kv3 - G_m_Kv3)/G_taum_Kv3 :1 
    dG_h_Kv3/dt = (G_hinf_Kv3 - G_h_Kv3)/G_tauh_Kv3 :1 
    G_iKv3 = gkv3bar*G_m_Kv3**4*G_h_Kv3*(vg-eK) :amp

    # Merged with kv4s
    # minf_Kv4f = 1.0 / (1.0 + exp((-49*mV - vg)/(12.5*mV))) :1
	# hinf_Kv4f = 1.0 / (1.0 + exp((-83*mV - vg)/(-10.*mV))) :1
	# taum_Kv4f = 0.25*ms + (6.75*ms)/(exp((-49*mV - vg)/(29*mV)) + exp((-49*mV - vg)/(-29.*mV))) :second
	# tauh_Kv4f = 7*ms + (14*ms)/(exp((-83*mV - vg)/(10*mV)) + exp((-83*mV - vg)/(-10.*mV))) :second
    # dm_Kv4f/dt = (minf_Kv4f - m_Kv4f)/taum_Kv4f :1
    # dh_Kv4f/dt = (hinf_Kv4f - h_Kv4f)/tauh_Kv4f :1
    # iKv4f = gkv4fbar*m_Kv4f**4*h_Kv4f*(vg-eK) :amp
    

    G_minf_Kv4s = 1.0 / (1.0 + exp((-49*mV - vg)/(12.5*mV))) :1
	G_hinf_Kv4s = 1.0 / (1.0 + exp((-83*mV - vg)/(-10.*mV))) :1
	G_taum_Kv4s = 0.25*ms + (6.75*ms)/(exp((-49*mV - vg)/(29*mV)) + exp((-49*mV - vg)/(-29.*mV))) :second
	G_tauh_Kv4s = 7*ms + (14*ms)/(exp((-83*mV - vg)/(10*mV)) + exp((-83*mV - vg)/(-10.*mV))) :second
    dG_m_Kv4s/dt = (G_minf_Kv4s - G_m_Kv4s)/G_taum_Kv4s :1
    dG_h_Kv4s/dt = (G_hinf_Kv4s - G_h_Kv4s)/G_tauh_Kv4s :1
    G_iKv4s = gkv4sbar*G_m_Kv4s**4*G_h_Kv4s*(vg-eK) :amp
    
    G_minf_Kc = 1.0 / (1.0 + exp((-61*mV - vg)/(19.5*mV))) :1
	G_taum_Kc = 6.7*ms + (93.3*ms)/(exp((-61*mV - vg)/(35.*mV)) + exp((-61*mV - vg)/(-25.*mV))) :second
    dG_m_Kc/dt = (G_minf_Kc - G_m_Kc)/G_taum_Kc :1
    G_iKc = gkcbar*G_m_Kc**4*(vg-eK) :amp

    G_minf_Hcn = 1.0 / (1.0 + exp((-76.4*mV - vg)/(-3.3*mV))) :1
	G_taum_Hcn = (3625*ms)/(exp((-76.4*mV - vg)/(6.56*mV)) + exp((-76.4*mV - vg)/(-7.48*mV))) :second
    dG_m_Hcn/dt = (G_minf_Hcn - G_m_Hcn)/G_taum_Hcn :1
    G_iHcn = ghcnbar*G_m_Hcn*(vg-eCat) :amp

    G_minf_Cah = 1.0 / (1.0 + exp((-20*mV - vg)/(7.*mV))) :1
    dG_m_Cah/dt = (G_minf_Cah - G_m_Cah)/(0.2*ms) :1
    G_iCah  = gcahbar*G_m_Cah*(vg-eCa) :amp

    # Ca Concentration
    dG_c_Ca/dt = -G_iCah*1/uA*3000/(2*96485)*1/ms- 0.4*(G_c_Ca - 0.01)*1/ms : 1
    Gcan = c_Ca**4.6 :1

    # KSK Current (Ca-dependent)
    G_minf_ksk = Gcan/(Gcan + Gcan50) :1
    G_tau_m_ksk = (76*ms-72*ms*G_c_Ca/5) * int(G_c_Ca < 5.) + 4*ms * int(G_c_Ca >= 5) :second
    dG_m_ksk/dt = (G_minf_ksk - G_m_ksk) / G_tau_m_ksk :1
    G_iKsk = gkskbar*G_m_ksk*(vg-eK) :amp


    G_membrain_Im = G_Iapp-G_iNaf-G_iLeak-G_iNap-G_iKv2
                 -G_iKv3-G_iKv4s-G_iCah-G_iHcn-G_iKc-G_iKsk :amp

    dvg/dt = G_membrain_Im/Cm :volt
    '''

    #################################################################
    # FSI Neuron
    #################################################################

    eqs_F = '''
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

    #################################################################

    neurons_M = b2.NeuronGroup(num_M,
                               eqs_M,
                               method=par_sim['integration_method'],
                               dt=par_sim['dt'],
                               threshold='vm>-20*mV',
                               refractory='vm>-20*mV',
                               namespace=par_M,
                               )

    neurons_M.vm = par_M['v0']
    neurons_M.M_h_Na = 'M_ah_Na/(M_ah_Na + M_Bh_Na)'  # 'hinf_Na'
    neurons_M.M_n_K = 'M_an_K/(M_an_K + M_Bn_K)'  # ninf_K
    neurons_M.M_m_Kir = 'M_minf_Kir'
    neurons_M.M_m_Kaf = 'M_minf_Kaf'
    neurons_M.M_h_Kaf = 'M_hinf_Kaf'
    neurons_M.M_m_Kas = 'M_minf_Kas'
    neurons_M.M_h_Kas = 'M_hinf_Kas'
    neurons_M.M_m_Krp = 'M_minf_Krp'
    neurons_M.M_h_Krp = 'M_hinf_Krp'
    neurons_M.M_m_Nap = 'M_minf_Nap'
    neurons_M.M_m_Nas = 'M_minf_Nas'

    st_mon_M = b2.StateMonitor(neurons_M, par_M['record_from'], record=True)

    neurons_G = b2.NeuronGroup(num_G,
                               eqs_G,
                               method=par_sim['integration_method'],
                               dt=par_sim['dt'],
                               threshold='vg>-20*mV',
                               refractory='vg>-20*mV',
                               namespace=par_G,
                               )
    st_mon_M = b2.StateMonitor(neurons_M, par_G['record_from'], record=True)
    


    # net = b2.Network(neurons_M)
    # net.add(st_mon_M)
    net = b2.Network(neurons_G)
    net.add(st_mon_G)

    net.run(par_sim['simulation_time'])

    return st_mon_M
# -------------------------------------------------------------------


def plot_data(st_mon, ax, index=0, **kwargs):
    ax.plot(st_mon.t / b2.ms, st_mon.vm[index] / b2.mV, **kwargs)
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
            st_mon.M_Iapp[0] / current_unit, lw=1, c='k', alpha=0.5)

    ylabel = "I [{}]".format(str(current_unit))

    ax.set_xlabel("t [ms]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    I = st_mon.M_Iapp[0] / current_unit
    ax.set_ylim(np.min(I)*1.1, np.max(I)*1.1)


def clean_directory():
    os.system("rm -rf output")
