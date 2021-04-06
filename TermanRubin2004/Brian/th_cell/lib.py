import numpy as np
import pylab as plt
import brian2 as b2
from matplotlib.gridspec import GridSpec
import os

if not os.path.exists("data"):
    os.makedirs("data")


def simulate_Thl_cell_fig2(par, par_sim):
    num = par_sim['num']
    i_sensory_motor = par_sim['I_sm']

    # b2.set_device('cpp_standalone')
    b2.start_scope()

    eqs = b2.Equations('''    
    minfthl = 1/(1+exp(-(vt-thtmthl*mV)/(sigmthl*mV))): 1
    hinfthl =  1/(1+exp((vt-thththl*mV)/(sighthl*mV))): 1
    pinfthl = 1/(1+exp(-(vt-thtpthl*mV)/(sigpthl*mV))) :1
    rinfthl = 1/(1+exp((vt-thtrthl*mV)/(sigrthl*mV))) :1
    ahthl =  ah0thl*exp(-(vt-thtahthl*mV)/(sigahthl*mV)) :1
    bhthl =  bh0thl/(1+exp(-(vt-thtbhthl*mV)/(sigbhthl*mV))) :1
    tauhthl =  1/(ahthl+bhthl) *ms :second
    taurthl = taur0thl+taur1thl*exp(-(vt-thtrtauthl*mV)/(sigrtauthl*mV)) :second

    ilthl=glthl*(vt-vlthl):amp
    inathl=gnathl*minfthl*minfthl*minfthl*hthl*(vt-vnathl) :amp
    ikthl=gkthl*((0.75*(1-hthl))**4)*(vt-vkthl) :amp
    itthl=gtthl*pinfthl*pinfthl*rthl*(vt-vtthl) :amp
    iextthl:amp
    ithl_sm = i_sensory_motor(t) : amp

    membrane_Ithl = -(ilthl+inathl+ikthl+itthl)+iextthl+ithl_sm:amp
    drthl/dt=phirthl*(rinfthl-rthl)/taurthl :1
    dhthl/dt=phihthl*(hinfthl-hthl)/tauhthl :1 
    dvt/dt = membrane_Ithl/cmthl : volt
    '''
    )

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vt>-55*mV',
                            refractory='vt>-55*mV',
                            namespace=par,
                            )

    neuron.vt = par['v0']
    neuron.hthl = "hinfthl"
    neuron.rthl = "rinfthl"
    neuron.iextthl = par['iext']

    state_monitor = b2.StateMonitor(neuron, ["vt", "ithl_sm"], record=True)

    net = b2.Network(neuron)
    net.add(state_monitor)
    net.run(par_sim['simulation_time'])

    return state_monitor

# -------------------------------------------------------------------


def simulate_Thl_cell_fig3(par, par_sim):
    num = par_sim['num']
    i_sensory_motor = par_sim['I_sm']

    # b2.set_device('cpp_standalone')
    b2.start_scope()

    eqs = b2.Equations('''    
    minfthl = 1/(1+exp(-(vt-thtmthl*mV)/(sigmthl*mV))): 1
    hinfthl =  1/(1+exp((vt-thththl*mV)/(sighthl*mV))): 1
    pinfthl = 1/(1+exp(-(vt-thtpthl*mV)/(sigpthl*mV))) :1
    rinfthl = 1/(1+exp((vt-thtrthl*mV)/(sigrthl*mV))) :1
    ahthl =  ah0thl*exp(-(vt-thtahthl*mV)/(sigahthl*mV)) :1
    bhthl =  bh0thl/(1+exp(-(vt-thtbhthl*mV)/(sigbhthl*mV))) :1
    tauhthl =  1/(ahthl+bhthl) *ms :second
    taurthl = taur0thl+taur1thl*exp(-(vt-thtrtauthl*mV)/(sigrtauthl*mV)) :second

    ilthl=glthl*(vt-vlthl):amp
    inathl=gnathl*minfthl*minfthl*minfthl*hthl*(vt-vnathl) :amp
    ikthl=gkthl*((0.75*(1-hthl))**4)*(vt-vkthl) :amp
    itthl=gtthl*pinfthl*pinfthl*rthl*(vt-vtthl) :amp
    iextthl:amp

    tmp_thl1 = sin(2*pi*(t-dsmthl)/tmsmthl) :1
    tmp_thl2 = sin(2*pi*(t-dsmthl+wsmthl)/tmsmthl) :1
    ym_thl1=1/(1+exp(-tmp_thl1/sigym)):1
    ym_thl2=1/(1+exp(-tmp_thl2/sigym)):1
    ithl_sm=imsmthl*ym_thl1*(1-ym_thl2) :amp
    
    membrane_Ithl = -(ilthl+inathl+ikthl+itthl)+iextthl+ithl_sm:amp    
    drthl/dt=phirthl*(rinfthl-rthl)/taurthl :1
    dhthl/dt=phihthl*(hinfthl-hthl)/tauhthl :1 
    dvt/dt = membrane_Ithl/cmthl : volt
    '''
    )

    neuron = b2.NeuronGroup(num,
                            eqs,
                            method=par_sim['integration_method'],
                            dt=par_sim['dt'],
                            threshold='vt>-55*mV',
                            refractory='vt>-55*mV',
                            namespace=par,
                            )

    neuron.vt = par['v0']
    neuron.hthl = "hinfthl"
    neuron.rthl = "rinfthl"
    neuron.iextthl = par['iext']

    state_monitor = b2.StateMonitor(neuron, ["vt", "ithl_sm"], record=True)

    net = b2.Network(neuron)
    net.add(state_monitor)
    net.run(par_sim['simulation_time'])

    return state_monitor


def plot_data(state_monitor, ax, index=0, xlabel=None):
    ax.plot(state_monitor.t / b2.ms, state_monitor.vt[index] / b2.mV,
            lw=1, c='k')
    ax.set_xlim(0, np.max(state_monitor.t / b2.ms))
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("v [mV]", fontsize=12)
    # plt.show()


def plot_current(state_monitor, ax, current_unit=b2.pA, xlabel=None):
    ax.plot(state_monitor.t / b2.ms,
            state_monitor.ithl_sm[0] / current_unit, lw=1, c='k', alpha=0.5)

    ylabel = "I [{}]".format(str(current_unit))

    ax.set_ylabel(ylabel, fontsize=12)
    I = state_monitor.ithl_sm[0] / current_unit
    # ax.set_ylim(np.min(I)*1.1, np.max(I)*1.1)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def make_grid(nrows, ncols, left=0.05, right=0.9,
              bottom=0.05, top=0.95, hspace=0.2,
              wspace=0.2):

    gs = GridSpec(nrows, ncols)
    gs.update(left=left, right=right,
              hspace=hspace, wspace=wspace,
              bottom=bottom, top=top)
    ax = []
    if nrows > 1:
        for i in range(nrows):
            ax_row = []
            for j in range(ncols):
                ax_row.append(plt.subplot(gs[i, j]))
            ax.append(ax_row)
    else:
        for j in range(ncols):
            ax.append(plt.subplot(gs[j]))

    return [ax, gs]
