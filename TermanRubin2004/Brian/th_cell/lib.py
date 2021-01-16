import numpy as np
import pylab as plt
import brian2 as b2


def simulate_Th_cell(par, par_sim):
    num = par['num']
    input_current = par['i_ext']

    eqs = '''

    minftc  =  1./(1+exp(-(vt+37.*mV)/(7.*mV))) : 1
    pinftc  = 1./(1.+exp(-(vt+60.*mV)/(6.2*mV))) : 1
    ahtc =  0.128*exp(-(46.*mV+vt)/(18.*mV)) : 1
    bhtc =  4./(1+exp(-(23.*mV+vt)/(5.*mV))) : 1
    hinftc  =  1./(1.+exp((vt+41.*mV)/(4.*mV))) : 1
    rinftc  = 1./(1.+exp((vt+84.*mV)/(4.*mV))) : 1
    tauhtc = 1./(ahtc+bhtc) *ms                    : second
    taurtc = (28.+ exp((vt+25.*mV)/(-10.5*mV)))*ms : second
    
    iltc = glbar*(vt-eleak) : amp 
    inatc = gnabar*minftc*minftc*minftc*htc*(vt-ena) : amp
    n2tc = .75*(1-htc) * 0.75*(1-htc) : 1
    n4tc = n2tc * n2tc : 1
    iktc = gkbar*n4tc*(vt-ek) : amp
    ittc= gtbar*pinftc*pinftc*htc*vt : amp 

    membrane_Itc = -(iltc+inatc+iktc+ittc) + iexttc : amp

    dhtc/dt = (hinftc - htc) / tauhtc : 1
    drtc/dt = (rinftc - rtc) / taurtc
    dvt/dt = membrane_Itc / Cm : volt

    '''

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vs>-20*mV',
                            refractory='vs>-20*mV',
                            namespace=par,
                            )

    neuron.vs = par['v0']
    neuron.h = "hinf"
    neuron.n = "ninf"
    neuron.r = "rinf"
    neuron.ca = 0
