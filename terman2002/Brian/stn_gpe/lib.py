import os
import numpy as np
import pylab as plt
import brian2 as b2
from os.path import join
from copy import deepcopy

# b2.prefs.codegen.target = 'numpy'

# if 1:
# b2.prefs.devices.cpp_standalone.openmp_threads = 1

for d in ["data", "output"]:
    if not os.path.exists(d):
        os.makedirs(d)


def simulate_STN_GPe_population(params):

    pid = os.getpid()
    b2.set_device('cpp_standalone',
                #   build_on_run=False,
                  directory=join("output", f"standalone-{pid}"))

    # b2.start_scope()

    par_s = params['par_s']
    par_g = params['par_g']
    par_syn = params['par_syn']
    par_sim = params['par_sim']

    if par_sim['standalone_mode']:
        b2.get_device().reinit()
        b2.get_device().activate(
            # build_on_run=False,
            directory=join("output", f"standalone-{pid}"))

    b2.defaultclock.dt = par_sim['dt']

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
    i_syn_GtoS : amp
    s_GtoS_sum : 1

    tmp2 = vs - thetag_s *mV : volt
    # Hinf_s = 1 / (1 + exp(-(tmp2 - thetas*mV) / (sigmas*mV))) : 1
    Hinf_s = 1 / (1 + exp(-(tmp2 - thetagH_s*mV) / (sigmagH_s*mV))) : 1
    ds_StoG/dt = alphas * Hinf_s * (1 - s_StoG) - betas * s_StoG : 1

    dh/dt  = phi * (hinf - h) / tauh  : 1
    dn/dt  = phi * (ninf - n) / taun  : 1
    dr/dt  = phir * (rinf - r) / taur : 1
    dca/dt = eps * ((-ica - it)/pA - kca* ca) : 1 
    membrane_Im = -(il + ina + ik + it + ica + iahp)+i_exts+i_syn_GtoS:amp
        
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

    dhg/dt = phihg*(hinfg-hg)/tauhg : 1
    dng/dt = phing*(ninfg-ng)/taung : 1
    drg/dt = phig*(rinfg-rg)/taurg  : 1
    dcag/dt= epsg*((-icag-itg)/pA - kcag*cag) : 1 
    dvg/dt = membrane_Im / C : volt

    tmp1 = vg - thetag_g *mV : volt
    # Hinf_g = 1 / (1 + exp(-(tmp1 - thetasg*mV) / (sigsg*mV))) : 1
    Hinf_g = 1 / (1 + exp(-(tmp1 - thetagH_g*mV) / (sigmagH_g*mV))) : 1
    ds_GtoS/dt = alphag * (1 - s_GtoS) * Hinf_g - betag * s_GtoS : 1

    itg = gtg * (ainfg ** 3) * rg * (vg - vcag) : amp
    inag = gnag * (minfg ** 3) * hg * (vg - vnag) : amp
    ikg = gkg * (ng ** 4) * (vg - vkg) : amp
    iahpg = gahpg * (vg - vkg) * cag / (cag + k1g) : amp
    icag = gcag * (sinfg ** 2) * (vg - vcag) : amp
    ilg = glg * (vg - vlg) : amp

    s_StoG_sum : 1
    i_syn_StoG : amp
    i_syn_GtoG : amp

    membrane_Im =-(itg+inag+ikg+iahpg+icag+ilg)+i_extg+i_syn_StoG+i_syn_GtoG : amp 
    '''

    eqs_syn_GtoS = '''
    i_syn_GtoS_post = g_GtoS*s_GtoS_pre*(v_rev_GtoS-vs):amp (summed)
    '''
    eqs_syn_StoG = '''
    i_syn_StoG_post = g_StoG*s_StoG_pre*(v_rev_StoG-vg):amp (summed)
    '''
    eqs_syn_GtoG = '''
    i_syn_GtoG_post = g_GtoG*s_GtoS_pre*(v_rev_GtoG-vg):amp (summed)
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

    syn_GtoS = b2.Synapses(neurons_g, neurons_s, eqs_syn_GtoS,
                           method=par_sim['integration_method'],
                           dt=par_sim['dt'],
                           namespace=par_syn)

    cols, rows = np.nonzero(par_syn['adj_GtoS'])
    syn_GtoS.connect(i=rows, j=cols)
    syn_GtoS.connect(j='i')

    syn_StoG = b2.Synapses(neurons_s, neurons_g, eqs_syn_StoG,
                           method=par_sim['integration_method'],
                           dt=par_sim['dt'],
                           namespace=par_syn)
    syn_StoG.connect(j='i')
    # syn_StoG.connect(i=0, j=0)

    syn_GtoG = b2.Synapses(neurons_g, neurons_g, eqs_syn_GtoG,
                           method=par_sim['integration_method'],
                           dt=par_sim['dt'],
                           namespace=par_syn)
    syn_GtoG.connect(p=par_syn['p_GtoG'], condition='i != j')
    # syn_GtoG.connect(i=0, j=0)

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

    state_mon_s = b2.StateMonitor(
        neurons_s, ["vs", "i_syn_GtoS", "ca"], record=True)
    state_mon_g = b2.StateMonitor(
        neurons_g, ["vg", "i_syn_StoG", "cag"], record=True)
    spike_mon_s = b2.SpikeMonitor(neurons_s)
    spike_mon_g = b2.SpikeMonitor(neurons_g)

    lfp_stn = b2.PopulationRateMonitor(neurons_s)
    lfp_gpe = b2.PopulationRateMonitor(neurons_g)

    net = b2.Network(neurons_s, neurons_g,
                     state_mon_g, spike_mon_s,
                     state_mon_s, spike_mon_g)

    net.add(syn_GtoS)
    net.add(syn_StoG)
    net.add(syn_GtoG)
    net.add(lfp_gpe)
    net.add(lfp_stn)

    net.run(par_sim['simulation_time'])

    # if par_sim['standalone_mode']:
    #     b2.get_device().build(directory="output",
    #                           compile=True,
    #                           run=True,
    #                           debug=False)
    monitors = {
        "state_stn": state_mon_s,
        "state_gpe": state_mon_g,
        "spike_stn": spike_mon_s,
        "spike_gpe": spike_mon_g,
        "lfp_stn": lfp_stn,
        "lfp_gpe": lfp_gpe,
    }

    return monitors


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
# -------------------------------------------------------------------


def to_npz(monitors, subname, indices=[0], save_voltages=False,
           width=1*b2.ms):

    spike_stn = monitors['spike_stn']
    spike_gpe = monitors['spike_gpe']
    state_stn = monitors['state_stn']
    state_gpe = monitors['state_gpe']
    rate_stn = monitors['rate_stn']
    rate_gpe = monitors['rate_gpe']

    stn = [spike_stn, rate_stn, state_stn]
    gpe = [spike_gpe, rate_gpe, state_gpe]
    labels = ['stn', 'gpe']

    counter = 0
    for mon in [stn, gpe]:
        file_name = "{:s}-{:s}".format(labels[counter], subname)
        spikes_id = mon[0].i
        spike_times = mon[0].t / b2.ms
        rate_times = mon[1].i
        rate_amp = mon[1].smooth_rate(width=width) / b2.Hz

        voltages = []
        times = mon[2].t / b2.ms,
        if mon is stn:
            for i in indices:
                voltages.append(mon[2].vs[i] / b2.mV)
        else:
            for i in indices:
                voltages.append(mon[2].vg[i] / b2.mV)

        np.savez(file_name,
                 spikes_time=spike_times,
                 spikes_id=spikes_id,
                 rate_amp=rate_amp,
                 rate_times=rate_times,
                 voltage_times=times,
                 voltages=voltages,
                 )
        counter += 1


# spikes_id = spike_monitor.i
# spike_times = spike_monitor.t/b2.ms

# rate_times = rate_monitor.t/b2.ms
# rate_amp = rate_monitor.smooth_rate(width=width) / b2.Hz

#Synapse equations----------------------------------------------#
# syn_StoG_eqs = '''
# w : 1
# Hinfg = 1/(1+exp(-(vg-thetag_s*mV - thetagH_s*mV) / (sigmagH_s*mV))) : 1
# dss/dt = alpha * Hinfg * (1 - ss) - beta * ss : 1 (clock-driven)
# i_syn_StoG_post = w * g_GtoS * ss * (v_rev_sg - vg) : amp (summed)
# '''
# syn_GtoS_eqs = '''
# w : 1
# Hinfs = 1/(1+exp(-(vs-thetag_g*mV - thetagH_g*mV)/(sigmagH_g*mV))) : 1
# dsg/dt = alphag * Hinfs * (1 - sg) - betag * sg : 1 (clock-driven)
# i_syn_GtoS_post = w * g_StoG * sg * (v_rev_gs - vs) : amp (summed)
# '''
# syn_GtoG_eqs = '''
# w : 1
# Hinfgg = 1/(1+exp(-(vg-thetag_g*mV - thetagH_g*mV)/(sigmagH_g*mV))) : 1
# dsg/dt = alphag * Hinfgg * (1 - sg) - betag * sg : 1 (clock-driven)
# i_syn_GtoG_post = w * g_GtoG * sg * (v_rev_gg - vg) : amp (summed)
# '''
# visualise_connectivity(S_gg, join("figs", "S_gg.png"))
# visualise_connectivity(S_gs, join("figs", "S_gs.png"))
# visualise_connectivity(S_sg, join("figs", "S_sg.png"))


# if par_sim['state'] == "sparse":
    #     g_hat_GtoS = 2.5*b2.nS
    #     g_hat_StoG = 0.03*b2.nS
    #     g_hat_GtoG = 0.06*b2.nS

    # elif par_sim['state'] == "episodic":
    #     g_hat_GtoS = 2.5*b2.nS
    #     g_hat_StoG = 0.016*b2.nS
    #     g_hat_GtoG = 0.0*b2.nS

    # elif par_sim['state'] == "continuous":
    #     g_hat_GtoS = 2.5*b2.nS
    #     g_hat_StoG = 0.1*b2.nS
    #     g_hat_GtoG = 0.02*b2.nS

    # else:
    #     print("unknown state of the network")
    #     exit(0)

    # par_syn['p_GtoS'] = 3 / par_g['num']
    # par_syn['p_StoG'] = 1 / par_g['num']

    # par_syn['g_GtoS'] = g_hat_GtoS # / (par_syn['p_GtoS'] * par_g['num'])
    # par_syn['g_StoG'] = g_hat_StoG # / (par_syn['p_StoG'] * par_s['num'])
    # par_syn['g_GtoG'] = g_hat_GtoG # / (par_syn['p_GtoG'] * par_g['num'])
