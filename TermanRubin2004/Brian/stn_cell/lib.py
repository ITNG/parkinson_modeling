from brian2.equations import refractory
import numpy as np
import pylab as plt
import brian2 as b2

# -------------------------------------------------------------------


def simulate_2_STN_cell_simpl_biexp(par, par_sim):

    """
    without considering Hinf in synapse equation
    """

    num = par['num']

    eqs = '''
    I_ext : amp 

    minf = 1/(1+exp(-(vs-thetam*mV)/(sigmam*mV))) : 1
    hinf = 1/(1+exp(-(vs-thetah*mV)/(sigmah*mV))) : 1 
    ninf = 1/(1+exp(-(vs-thetan*mV)/(sigman*mV))) : 1
    ainf = 1/(1+exp(-(vs-thetaa*mV)/(sigmaa*mV))) : 1
    binf = 1/(1+exp((r-thetab)/sigmab))-1/(1+exp(-thetab/sigmab)) : 1
    rinf = 1/(1+exp(-(vs-thetar*mV)/(sigmar*mV))) : 1
    sinf = 1/(1+exp(-(vs-thetas*mV)/(sigmas*mV))) : 1
    taun = taun0+taun1/(1+exp(-(vs-thn*mV)/(sigmant*mV))) : second
    tauh = tauh0+tauh1/(1+exp(-(vs-thh*mV)/(sigmaht*mV))) : second
    taur = taur0+taur1/(1+exp(-(vs-thr*mV)/(sigmart*mV))) : second

    il = gl * (vs - vl) : amp
    ina = gna * minf ** 3 * h * (vs - vna) : amp
    ik = gk * n ** 4 * (vs - vk) : amp
    iahp = gahp * ca / (ca + k1) * (vs - vk) : amp 
    ica = gca * sinf ** 2 * (vs - vca) : amp 
    it = gt * ainf ** 3 * binf ** 2 * (vs - vca) : amp 
    i_syn_ss = g_syn_ss * (v_rev_ss - vs)  :amp

    ds_ss/dt = - s_ss * beta : siemens
    dg_syn_ss/dt = scale_f * (s_ss - g_syn_ss) * alpha : siemens

    dh/dt  = phi * (hinf - h) / tauh  : 1
    dn/dt  = phi * (ninf - n) / taun  : 1
    dr/dt  = phir * (rinf - r) / taur : 1
    dca/dt = eps * ((-ica - it)/pA - kca* ca) : 1  #! pA
    membrane_Im = -(il + ina + ik + it + ica + iahp) + I_ext + i_syn_ss : amp
        
    dvs/dt = membrane_Im/C : volt
    '''

    neurons = b2.NeuronGroup(2,
                             eqs,
                             method=par_sim['integration_method'],
                             dt=par_sim['dt'],
                             threshold='vs>-20*mV',
                             refractory='vs>-20*mV',
                             namespace=par,
                             )

    syn_ss = b2.Synapses(neurons, neurons,
                         on_pre="s_ss +={}*nS".format(par["w"]),
                         dt=par_sim['dt'],
                         method=par_sim['integration_method'],
                         namespace=par
                         )
    syn_ss.connect(i=0, j=1)

    neurons.vs = par['v0']
    neurons.h = "hinf"
    neurons.n = "ninf"
    neurons.r = "rinf"
    neurons.ca = 0
    neurons.I_ext = 0*b2.pA

    st_mon = b2.StateMonitor(neurons,
                             ["vs", "i_syn_ss", "g_syn_ss"],
                             record=True)

    net = b2.Network(neurons)
    net.add(st_mon)
    net.add(syn_ss)
    net.run(par_sim['simulation_time'])

    return st_mon
# -------------------------------------------------------------------


def simulate_2_STN_cell_biexp(par, par_sim):

    """
    considering Hinf in synapse equation
    """

    num = par['num']
    b2.defaultclock.dt = par_sim['dt']


    eqs = '''
    I_ext : amp 

    minf = 1/(1+exp(-(vs-thetam*mV)/(sigmam*mV))) : 1
    hinf = 1/(1+exp(-(vs-thetah*mV)/(sigmah*mV))) : 1 
    ninf = 1/(1+exp(-(vs-thetan*mV)/(sigman*mV))) : 1
    ainf = 1/(1+exp(-(vs-thetaa*mV)/(sigmaa*mV))) : 1
    binf = 1/(1+exp((r-thetab)/sigmab))-1/(1+exp(-thetab/sigmab)) : 1
    rinf = 1/(1+exp(-(vs-thetar*mV)/(sigmar*mV))) : 1
    sinf = 1/(1+exp(-(vs-thetas*mV)/(sigmas*mV))) : 1
    taun = taun0+taun1/(1+exp(-(vs-thn*mV)/(sigmant*mV))) : second
    tauh = tauh0+tauh1/(1+exp(-(vs-thh*mV)/(sigmaht*mV))) : second
    taur = taur0+taur1/(1+exp(-(vs-thr*mV)/(sigmart*mV))) : second

    il = gl * (vs - vl) : amp
    ina = gna * minf ** 3 * h * (vs - vna) : amp
    ik = gk * n ** 4 * (vs - vk) : amp
    iahp = gahp * ca / (ca + k1) * (vs - vk) : amp 
    ica = gca * sinf ** 2 * (vs - vca) : amp 
    it = gt * ainf ** 3 * binf ** 2 * (vs - vca) : amp 
    i_syn_ss : amp


    tmp =  vs - thetag * mV : volt
    Hinf = 1./(1+exp(-(tmp - thetagH*mV) / (sigmagH*mV))):1
    ds_ss/dt = alpha * Hinf * (1 - s_ss) - beta * s_ss  : 1
    # ds_ss/dt = 0.5 * (1 + tanh(0.1*vs/mV)) * alpha * (1 - s_ss) - beta * s_ss  : 1

    dh/dt  = phi * (hinf - h) / tauh  : 1
    dn/dt  = phi * (ninf - n) / taun  : 1
    dr/dt  = phir * (rinf - r) / taur : 1
    dca/dt = eps * ((-ica - it)/pA - kca* ca) : 1  #! pA
    membrane_Im = -(il + ina + ik + it + ica + iahp) + I_ext + i_syn_ss : amp
        
    dvs/dt = membrane_Im/C : volt
    '''

    eqs_syn_ss = '''  # synapse model, s to s
    i_syn_ss_post = w_ss * s_ss_pre * (v_rev_ss - vs) :amp (summed)
    '''
    neurons = b2.NeuronGroup(2,
                             eqs,
                             method=par_sim['integration_method'],
                             dt=par_sim['dt'],
                             threshold='vs>-20*mV',
                             refractory='vs>-20*mV',
                             namespace=par,
                             )

    syn_ss = b2.Synapses(neurons, neurons,
                        eqs_syn_ss,
                         dt=par_sim['dt'],
                         method=par_sim['integration_method'],
                         namespace=par,
                         )
    syn_ss.connect(i=0, j=1)

    neurons.vs = par['v0']
    neurons.h = "hinf"
    neurons.n = "ninf"
    neurons.r = "rinf"
    neurons.ca = 0
    neurons.I_ext = 0*b2.pA

    st_mon = b2.StateMonitor(neurons,
                             ["vs", "i_syn_ss", "s_ss"],
                             record=[0, 1])

    net = b2.Network(neurons)
    net.add(st_mon)
    net.add(syn_ss)
    net.run(par_sim['simulation_time'])

    return st_mon
# -------------------------------------------------------------------


def simulate_STN_cell(par, par_sim):

    num = par['num']
    input_current = par['i_ext']

    eqs = '''
    I_ext = input_current(t, i): amp 

    minf = 1/(1+exp(-(vs-thetam*mV)/(sigmam*mV))) : 1
    hinf = 1/(1+exp(-(vs-thetah*mV)/(sigmah*mV))) : 1 
    ninf = 1/(1+exp(-(vs-thetan*mV)/(sigman*mV))) : 1
    ainf = 1/(1+exp(-(vs-thetaa*mV)/(sigmaa*mV))) : 1
    binf = 1/(1+exp((r-thetab)/sigmab))-1/(1+exp(-thetab/sigmab)) : 1
    rinf = 1/(1+exp(-(vs-thetar*mV)/(sigmar*mV))) : 1
    sinf = 1/(1+exp(-(vs-thetas*mV)/(sigmas*mV))) : 1
    taun = taun0+taun1/(1+exp(-(vs-thn*mV)/(sigmant*mV))) : second
    tauh = tauh0+tauh1/(1+exp(-(vs-thh*mV)/(sigmaht*mV))) : second
    taur = taur0+taur1/(1+exp(-(vs-thr*mV)/(sigmart*mV))) : second

    il = gl * (vs - vl) : amp
    ina = gna * minf ** 3 * h * (vs - vna) : amp
    ik = gk * n ** 4 * (vs - vk) : amp
    iahp = gahp * ca / (ca + k1) * (vs - vk) : amp 
    ica = gca * sinf ** 2 * (vs - vca) : amp 
    it = gt * ainf ** 3 * binf ** 2 * (vs - vca) : amp 

    dh/dt  = phi * (hinf - h) / tauh  : 1
    dn/dt  = phi * (ninf - n) / taun  : 1
    dr/dt  = phir * (rinf - r) / taur : 1
    dca/dt = eps * ((-ica - it)/pA - kca* ca) : 1    #! pA
    membrane_Im = -(il + ina + ik + it + ica + iahp) + I_ext  : amp
        
    dvs/dt = membrane_Im/C : volt
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

    st_mon = b2.StateMonitor(neuron, ["vs", "I_ext"], record=True)

    net = b2.Network(neuron)
    net.add(st_mon)
    net.run(par_sim['simulation_time'])

    return st_mon
# -------------------------------------------------------------------


def plot_data(st_mon, ax, index=0):
    ax.plot(st_mon.t / b2.ms, st_mon.vs[index] / b2.mV, lw=2, c='k')
    ax.set_xlim(0, np.max(st_mon.t / b2.ms))
    # ax.set_xlabel("time [ms]", fontsize=14)
    ax.set_ylabel("v [mV]", fontsize=14)
# -------------------------------------------------------------------


def plot_current(st_mon, ax, current_unit=b2.pA):
    ax.plot(st_mon.t / b2.ms,
            st_mon.I_ext[0] / current_unit, lw=1, c='k', alpha=0.5)

    ylabel = "I [{}]".format(str(current_unit))

    ax.set_xlabel("t [ms]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    I = st_mon.I_ext[0] / current_unit
    ax.set_ylim(np.min(I)*1.1, np.max(I)*1.1)
