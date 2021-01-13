import numpy as np
import pylab as plt
import brian2 as b2
from os.path import join


if 1:
    b2.set_device('cpp_standalone',
                  build_on_run=False,
                  directory="output")


def simulate_STN_GPe_population(par_s, par_g, par_syn, par_sim):

    b2.start_scope()

    if par_sim['standalone_mode']:
        b2.get_device().reinit()
        b2.get_device().activate(build_on_run=False,
                                 directory="output")
    # b2.prefs.codegen.target = 'numpy'
    b2.prefs.devices.cpp_standalone.openmp_threads = 1

    b2.defaultclock.dt = par_sim['dt']

    if par_sim['state'] == "sparse":
        g_hat_gs = 2.5*b2.nS
        g_hat_sg = 0.03*b2.nS
        g_hat_gg = 0.06*b2.nS

    elif par_sim['state'] == "episodic":
        g_hat_gs = 2.5*b2.nS
        g_hat_sg = 0.016*b2.nS
        g_hat_gg = 0.0*b2.nS

    elif par_sim['state'] == "continuous":
        g_hat_gs = 2.5*b2.nS
        g_hat_sg = 0.1*b2.nS
        g_hat_gg = 0.02*b2.nS

    else:
        print("unknown state of the network")
        exit(0)

    par_syn['p_gs'] = 3 / par_g['num']
    par_syn['p_sg'] = 1 / par_g['num']
    par_syn['p_gg'] = 1.0

    par_syn['g_gs'] = g_hat_gs # / (par_syn['p_gs'] * par_g['num'])
    par_syn['g_sg'] = g_hat_sg # / (par_syn['p_sg'] * par_s['num'])
    par_syn['g_gg'] = g_hat_gg # / (par_syn['p_gg'] * par_g['num'])

    eqs_s = '''

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
    i_exts : amp 
    i_syn_gs : amp

    # tmp2 = vs-thetag_s*mV : volt
    # Hinf_s = 1./(1+exp(-(tmp2-thetagH_s*mV)/(sigmagH_s*mV))):1
    # ds_sg/dt = alphas * Hinf_s * (1 - s_sg) - betas * s_sg : 1  #!

    dh/dt  = phi * (hinf - h) / tauh  : 1
    dn/dt  = phi * (ninf - n) / taun  : 1
    dr/dt  = phir * (rinf - r) / taur : 1
    dca/dt = eps * ((-ica - it)/pA - kca* ca) : 1 
    membrane_Im = -(il + ina + ik + it + ica + iahp)+i_exts+i_syn_gs:amp
        
    dvs/dt = membrane_Im/C : volt
    '''
    eqs_g = '''
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
    i_syn_sg : amp
    i_syn_gg : amp

    tmp1 = vg-thetag_g*mV : volt
    Hinf_g = 1./(1+exp(-(tmp1-thetagH_g*mV)/(sigmagH_g*mV))):1
    ds_gs/dt = alphag * Hinf_g * (1 - s_gs) - betag * s_gs : 1 #!
    ds_gg/dt = alphas * Hinf_g * (1 - s_gg) - betas * s_gg : 1 #!

    membrane_Im =-(itg+inag+ikg+iahpg+icag+ilg)+i_extg+i_syn_sg+i_syn_gg : amp
    dhg/dt = phihg*(hinfg-hg)/tauhg : 1
    dng/dt = phing*(ninfg-ng)/taung : 1
    drg/dt = phig*(rinfg-rg)/taurg  : 1
    dcag/dt= epsg*((-icag-itg)/pA - kcag*cag) : 1 

    dvg/dt = membrane_Im / C : volt
    '''

    eqs_syn_gs = '''
    i_syn_gs_post = g_gs * s_gs_pre * (v_rev_gs - vs):amp (summed)
    '''
    eqs_syn_sg = '''
    tmp2 = vs-thetag_s*mV : volt
    Hinf_s = 1./(1+exp(-(tmp2-thetagH_s*mV)/(sigmagH_s*mV))):1
    ds_sg/dt = alphag * Hinf_s * (1 - s_sg) - betag * s_sg : 1
    i_syn_sg_post = g_sg * s_sg_pre * (v_rev_sg - vg) : amp (summed)
    '''
    eqs_syn_gg = '''
    i_syn_gg_post = g_gg * s_gg_pre * (v_rev_gg - vg):amp (summed)
    '''

    #---------------------------------------------------------------#
    neurons_s = b2.NeuronGroup(par_s['num'],
                               eqs_s,
                               method=par_sim['integration_method'],
                               dt=par_sim['dt'],
                               threshold='vs>-20*mV',
                               refractory='vs>-20*mV',
                               namespace={**par_s, **par_syn},
                               )
    #---------------------------------------------------------------#
    neurons_g = b2.NeuronGroup(par_g['num'],
                               eqs_g,
                               method=par_sim['integration_method'],
                               dt=par_sim['dt'],
                               threshold='vg>-20*mV',
                               refractory='vg>-20*mV',
                               namespace={**par_g, **par_syn},
                               )

    syn_gs = b2.Synapses(neurons_g, neurons_s, eqs_syn_gs,
                         method=par_sim['integration_method'],
                         dt=par_sim['dt'],
                         namespace=par_syn)

    cols, rows = np.nonzero(par_syn['adj_gs'])
    syn_gs.connect(i=rows, j=cols)
    # syn_gs.connect(j='i')

    syn_sg = b2.Synapses(neurons_s, neurons_g, eqs_syn_sg,
                         method=par_sim['integration_method'],
                         dt=par_sim['dt'],
                         namespace=par_syn)
    syn_sg.connect(j='i')

    syn_gg = b2.Synapses(neurons_g, neurons_g, eqs_syn_gg,
                         method=par_sim['integration_method'],
                         dt=par_sim['dt'],
                         namespace=par_syn)
    syn_gg.connect(p=par_syn['p_gg'], condition='i != j')

    neurons_s.vs = par_s['v0']
    neurons_s.h = "hinf"
    neurons_s.n = "ninf"
    neurons_s.r = "rinf"
    neurons_s.ca = 0
    neurons_s.i_exts = par_s['i_ext']

    neurons_g.vg = par_g['v0']
    neurons_g.hg = "hinfg"
    neurons_g.ng = "ninfg"
    neurons_g.rg = "rinfg"
    neurons_g.cag = 0
    neurons_g.i_extg = par_g['i_ext']

    #---------------------------------------------------------------#

    st_mon_s = b2.StateMonitor(neurons_s, "vs", record=True)
    st_mon_g = b2.StateMonitor(neurons_g, "vg", record=True)
    sp_mon_s = b2.SpikeMonitor(neurons_s)
    sp_mon_g = b2.SpikeMonitor(neurons_g)

    net = b2.Network(neurons_s)
    net.add(neurons_g)
    net.add(syn_gs)
    net.add(syn_sg)
    net.add(syn_gg)
    net.add(st_mon_s)
    net.add(st_mon_g)
    net.add(sp_mon_s)
    net.add(sp_mon_g)

    net.run(par_sim['simulation_time'])

    if par_sim['standalone_mode']:
        b2.get_device().build(directory="output",
                              compile=True,
                              run=True,
                              debug=False)

    return st_mon_s, st_mon_g, sp_mon_s, sp_mon_g


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def visualise_connectivity(S, file_name):
    Ns = len(S.source)
    Nt = len(S.target)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    ax[0].plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)

    for i, j in zip(S.i, S.j):
        ax[0].plot([0, 1], [i, j], '-k')
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['Source', 'Target'])
    ax[0].set_ylabel('Neuron index')
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_ylim(-1, max(Ns, Nt))

    ax[1].plot(S.i, S.j, 'ok')
    ax[1].set_xlim(-1, Ns)
    ax[1].set_ylim(-1, Nt)
    ax[1].set_xlabel('Source neuron index')
    ax[1].set_ylabel('Target neuron index')

    plt.savefig(file_name)
    plt.close()


#Synapse equations----------------------------------------------#
# syn_sg_eqs = '''
# w : 1
# Hinfg = 1/(1+exp(-(vg-thetag_s*mV - thetagH_s*mV) / (sigmagH_s*mV))) : 1
# dss/dt = alpha * Hinfg * (1 - ss) - beta * ss : 1 (clock-driven)
# i_syn_sg_post = w * g_gs * ss * (v_rev_sg - vg) : amp (summed)
# '''
# syn_gs_eqs = '''
# w : 1
# Hinfs = 1/(1+exp(-(vs-thetag_g*mV - thetagH_g*mV)/(sigmagH_g*mV))) : 1
# dsg/dt = alphag * Hinfs * (1 - sg) - betag * sg : 1 (clock-driven)
# i_syn_gs_post = w * g_sg * sg * (v_rev_gs - vs) : amp (summed)
# '''
# syn_gg_eqs = '''
# w : 1
# Hinfgg = 1/(1+exp(-(vg-thetag_g*mV - thetagH_g*mV)/(sigmagH_g*mV))) : 1
# dsg/dt = alphag * Hinfgg * (1 - sg) - betag * sg : 1 (clock-driven)
# i_syn_gg_post = w * g_gg * sg * (v_rev_gg - vg) : amp (summed)
# '''
# visualise_connectivity(S_gg, join("figs", "S_gg.png"))
# visualise_connectivity(S_gs, join("figs", "S_gs.png"))
# visualise_connectivity(S_sg, join("figs", "S_sg.png"))
