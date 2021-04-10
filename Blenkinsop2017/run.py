import os
import math
import numpy as np
import pylab as plt
from time import time

if not os.path.exists('data/'):
    os.makedirs('data/')

np.random.seed(2)


def set_history(hist, nstart, maxdelay):

    for i in range(nstart+1):
        t[i] = -(nstart - i) / float(nstart) * maxdelay

    # x: [N x nstep]
    for i in range(N):
        y[i, :(nstart+1)] = hist[i]


def euler_integrator(f, h):

    k0 = np.zeros(N)
    for i in range(nstart, nstart + n_iteration):
        dy = h * f(t[i], k0, i)
        y[:, i+1] = y[:, i] + dy
        t[i+1] = (i - nstart + 1) * dt


def rungeKutta_integrator(f, h):

    for i in range(nstart, nstart + n_iteration):

        k0 = np.zeros(N)
        k1 = h * f(t[i], k0, i)
        k2 = h * f(t[i] + 0.5 * h, 0.5 * k1, i)
        k3 = h * f(t[i] + 0.5 * h, 0.5 * k2, i)
        k4 = h * f(t[i] + h, k3, i)

        y[:, i + 1] = y[:, i] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1] = (i - nstart + 1) * dt

    np.savez(f"data/{output_filename}", t=t, y=y)


def gompertz(x, M, B):
    inv_M = 1.0 / M
    return (M*(inv_M*B)**(np.exp(-inv_M*math.e*x)))


def sys_eqns(t, k, n):
    '''
    [s1, s2, stn, ge, gi, mc]
    yPs1  : 0     Ss1 :  12
    dyPs1 : 1     dSs1 : 13
    yPs2  : 2     Ss2 :  14
    dyPs2 : 3     dSs2 : 15
    Pstn  : 4     Sstn:  16
    dPstn : 5     dSstn: 17
    Pge   : 6     Sge :  18
    dPge  : 7     dSge : 19
    Pgi   : 8     Sgi :  20
    dPgi  : 9     dSgi : 21
    Pmc   : 10    Smc :  22
    dPmc  : 11    dSmc : 23
    '''

    yPs1 = y[0, n] + k[0]
    dyPs1 = y[1, n] + k[1]
    yPs2 = y[2, n] + k[2]
    dyPs2 = y[3, n] + k[3]
    yPstn = y[4, n] + k[4]
    dyPstn = y[5, n] + k[5]
    yPge = y[6, n] + k[6]
    dyPge = y[7, n] + k[7]
    yPgi = y[8, n] + k[8]
    dyPgi = y[9, n] + k[9]
    yPmc = y[10, n] + k[10]
    dyPmc = y[11, n] + k[11]
    ySs1 = y[12, n] + k[12]
    dySs1 = y[13, n] + k[13]
    ySs2 = y[14, n] + k[14]
    dySs2 = y[15, n] + k[15]
    ySstn = y[16, n] + k[16]
    dySstn = y[17, n] + k[17]
    ySge = y[18, n] + k[18]
    dySge = y[19, n] + k[19]
    ySgi = y[20, n] + k[20]
    dySgi = y[21, n] + k[21]
    ySmc = y[22, n] + k[22]
    dySmc = y[23, n] + k[23]

    INp = IN(t, gp, a, b, coeff_ab, 0, 0.3)
    INs = IN(t, gs, a, b, coeff_ab, 0, 0.3)

    # channel I
    ddyPs1 = inv_tau2 * (- W_s_s*gompertz(ySs1, M_str, B_str)
                         + W_sc_s * (1 + da) * INp
                         + W_mc_s * (1 + da) * gompertz(
        interp_y(t-T_ctx_stn, 10, n), M_ctx, B_ctx)  # yPmc
        - W_ge_s*gompertz(
        interp_y(t-T_gpe_stn, 18, n), M_GP, B_GP)  # ySge
    ) - 2*inv_tau*dyPs1-inv_tau2*yPs1

    # print(gompertz(interp_y(t-T_gpe_stn, 18, n), M_GP, B_GP))
    # print(interp_y(t-T_gpe_stn, 18, n))
    # exit(0)

    ddyPs2 = inv_tau2*(-W_s_s*gompertz(ySs2, M_str, B_str)
                       + W_sc_s*(1-da)*INp
                       + W_mc_s*(1-da)*gompertz(
                           interp_y(t-T_ctx_stn, 10, n), M_ctx, B_ctx)  # yPmc
                       - W_ge_s*gompertz(
                           interp_y(t-T_gpe_stn, 18, n), M_GP, B_GP)  # ySge
                       ) - 2*inv_tau*dyPs2-inv_tau2*yPs2

    ddyPstn = inv_tau2*(-W_ge_stn*gompertz(
        interp_y(t-T_gpe_stn, 6, n), M_GP, B_GP)  # yPge
        + W_mc_stn*gompertz(
        interp_y(t-T_ctx_stn, 10, n), M_ctx, B_ctx)  # yPmc
        + W_sc_stn*INp) - 2*inv_tau*dyPstn - inv_tau2*yPstn

    ddyPge = inv_tau2*(-W_s2_ge*gompertz(
        interp_y(t-T_stn_gpe, 2, n), M_stn, B_stn)  # yPs2
        + W_stn_ge * gompertz(interp_y(t-T_stn_gpe, 4, n),
                              M_stn, B_stn)  # yPstn
        + W_stn_ge * gompertz(interp_y(t-T_stn_gpe, 16, n),
                              M_stn, B_stn)  # ySstn
        - W_ge_ge * gompertz(interp_y(t-T_gpe_gpe, 18, n), M_GP, B_GP)  # ySge
        - W_ge_R * gompertz(yPge, M_GP, B_GP)
    ) - 2*inv_tau * dyPge - inv_tau2 * yPge

    ddyPgi = inv_tau2*(-W_s1_gi*gompertz(
        interp_y(t-T_stn_gpi, 0, n), M_stn, B_stn)  # yPs1
        + W_stn_gi * gompertz(interp_y(t-T_stn_gpi, 4, n),
                              M_stn, B_stn)  # yPstn
        + W_stn_gi * gompertz(interp_y(t-T_stn_gpi, 16, n),
                              M_stn, B_stn)  # ySstn
        - W_ge_gi * gompertz(interp_y(t-T_gpe_gpi, 18, n), M_GP, B_GP)  # ySge
    ) - 2*inv_tau * dyPgi - inv_tau2 * yPgi

    ddyPmc = inv_tau2*(
        - W_gi_mc*gompertz(interp_y(t-T_gpi_mctx, 8, n), M_GP, B_GP)  # yPgi
        + W_sc_mc * INp
    ) - 2*inv_tau * dyPmc - inv_tau2 * yPmc

    # channel II
    ddySs1 = inv_tau2*(- W_s_s * gompertz(yPs1, M_str, B_str)
                       + W_sc_s*(1+da)*INs
                       + W_mc_s*(1+da)*gompertz(
                           interp_y(t-T_ctx_stn, 22, n), M_ctx, B_ctx)  # ySmc
                       - W_ge_s * gompertz(
                           interp_y(t-T_gpe_stn, 6, n), M_GP, B_GP)  # yPge
                       ) - 2*inv_tau*dySs1-inv_tau2*ySs1

    ddySs2 = inv_tau2 * (-W_s_s*gompertz(yPs2, M_str, B_str)
                         + W_sc_s*(1-da)*INs
                         + W_mc_s*(1-da)*gompertz(
        interp_y(t-T_ctx_stn, 22, n), M_ctx, B_ctx)  # ySmc
        - W_ge_s*gompertz(
        interp_y(t-T_gpe_stn, 6, n), M_GP, B_GP)  # yPge
    ) - 2*inv_tau*dySs2-inv_tau2*ySs2
    ddySstn = inv_tau2 * (- W_ge_stn*gompertz(
        interp_y(t-T_gpe_stn, 18, n), M_GP, B_GP)  # ySge
        + W_mc_stn*gompertz(
        interp_y(t-T_ctx_stn, 22, n), M_ctx, B_ctx)  # ySmc
        + W_sc_stn*INs) - 2*inv_tau*dySstn - inv_tau2*ySstn

    ddySge = inv_tau2 * (- W_s2_ge * gompertz(
        interp_y(t-T_stn_gpe, 14, n), M_stn, B_stn)  # ySs2
        + W_stn_ge * gompertz(interp_y(t-T_stn_gpe, 4, n),
                              M_stn, B_stn)  # yPstn
        + W_stn_ge * gompertz(interp_y(t-T_stn_gpe, 16, n),
                              M_stn, B_stn)  # ySstn
        - W_ge_ge * gompertz(interp_y(t-T_gpe_gpe, 6, n), M_GP, B_GP)  # yPge
        - W_ge_R * gompertz(ySge, M_GP, B_GP)
    ) - 2*inv_tau * dySge - inv_tau2 * ySge

    ddySgi = inv_tau2*(- W_s1_gi * gompertz(
        interp_y(t-T_stn_gpi, 12, n), M_stn, B_stn)  # ySs1
        + W_stn_gi * gompertz(interp_y(t-T_stn_gpi, 4, n),
                              M_stn, B_stn)  # yPstn
        + W_stn_gi * gompertz(interp_y(t-T_stn_gpi, 16, n),
                              M_stn, B_stn)  # ySstn
        - W_ge_gi * gompertz(interp_y(t-T_gpe_gpi, 6, n), M_GP, B_GP)  # yPge
    ) - 2*inv_tau * dySgi - inv_tau2 * ySgi

    ddySmc = inv_tau2*(
        - W_gi_mc * gompertz(interp_y(t-T_gpi_mctx, 20, n), M_GP, B_GP)  # ySgi
        + W_sc_mc * INs
    ) - 2*inv_tau * dySmc - inv_tau2 * ySmc

    return np.asarray([dyPs1, ddyPs1,
                       dyPs2, ddyPs2,
                       dyPstn, ddyPstn,
                       dyPge, ddyPge,
                       dyPgi, ddyPgi,
                       dyPmc, ddyPmc,
                       dySs1, ddySs1,
                       dySs2, ddySs2,
                       dySstn, ddySstn,
                       dySge, ddySge,
                       dySgi, ddySgi,
                       dySmc, ddySmc
                       ])


def IN(t, g, a, b, coeff, t_start, t_end):
    if (t >= t_start) and (t <= t_end):
        return B_ctx + g * coeff * (np.exp(-b*t) - np.exp(-a*t))
    else:
        return B_ctx


def interp_y(t0, index, n):
    assert(t0 <= t[n])
    return (np.interp(t0, t[:n], y[index, :n]))


if __name__ == "__main__":

    start = time()
    output_filename = "data"
    nucleoli = ['Str', 'STN', 'GPe', 'GPi', 'MC']
    n_nucleoli = len(nucleoli)
    N = 24
    dt = 0.1
    t_sim = 150
    y0 = np.random.rand(N)*10
    # delays = np.zeros((n_nucleoli, n_nucleoli), dtype=float)
    # delays = np.array([
    #     [0.0, 0.0, 0.0, 0.0, 2.5],
    #     [0.0, 0.0, 1.0, 0.0, 2.5],
    #     [7.0, 2.5, 1.0, 0.0, 0.0],
    #     [12., 2.5, 1.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 3.0, 0.0]
    # ])

    # parameters
    tau = 2.
    M_str = 90*1000.
    M_stn = 250*1000.
    M_ctx = 22*1000.
    M_GP = 300*1000.
    B_str = 0.1*1000
    B_stn = 50*1000.
    B_ctx = 4*1000.
    B_GP = 150*1000
    da = 0.3
    a = 100.0
    b = 1000.0
    gp = 0.25
    gs = 0.17

    W_mc_stn = 20.
    W_ge_stn = 3.
    W_s2_ge = 40.
    W_stn_ge = 0.72
    W_ge_ge = 1.37
    W_ge_gi = 0.8
    W_s1_gi = 4.
    W_stn_gi = 0.2
    W_s_s = 0.3
    W_gi_mc = 0.25
    W_sc_s = 4.
    W_sc_stn = 20.
    W_mc_s = 0.65
    W_sc_mc = 1.
    W_ge_s = 0.1
    W_ge_R = 0.3

    # W_mc_stn = 0.
    # W_ge_stn = 0.
    # W_s2_ge = 0.
    # W_stn_ge = 0.
    # W_ge_ge = 0
    # W_ge_gi = 0.
    # W_s1_gi = .0
    # W_stn_gi = 0.
    # W_s_s = 0.
    # W_gi_mc = 0.
    # W_sc_s = .0
    # W_sc_stn = 0.
    # W_mc_s = 0.
    # W_sc_mc = .0
    # W_ge_s = 0.
    # W_ge_R = 0.

    T_ctx_str = 2.5
    T_ctx_stn = 2.5
    T_stn_gpe = 2.5
    T_stn_gpi = 2.5
    T_gpe_stn = 1.0
    T_str_gpe = 7.0
    T_str_gpi = 12.
    T_gpe_gpe = 1.0
    T_gpe_gpi = 1.0
    T_gpi_mctx = 3.0

    # Auxilary parameters
    inv_tau = 1 / tau
    inv_tau2 = inv_tau * inv_tau
    coeff_ab = a*b/(a-b)
    maxdelay = 12.  # np.max(delays)
    nstart = 50
    n_iteration = (int)((t_sim)/dt)

    y = np.zeros((N, n_iteration+nstart+1))
    t = np.zeros(n_iteration+nstart+1)
    set_history(y0, nstart, maxdelay)
    # rungeKutta_integrator(sys_eqns, dt)
    euler_integrator(sys_eqns, dt)

    print("Done in %g seconds" % (time()-start))
    labels = ['str1', 'str2', 'stn', 'gpe', 'gpi']
    for i in range(0, 10, 2):
        plt.plot(t, y[i, :]/1000, label=labels[i//2])

    plt.legend()
    plt.savefig(f"data/{output_filename}.png", dpi=150)
    plt.show()

    # y = np.arange(0, 100, 1)
    # f = gompertz(y, M_GP, B_GP)
    # f = gompertz(y, M_ctx, B_ctx)

    # plt.plot(y, f)
    # plt.show()

    # t = np.arange(-50,150, 0.1)/1000.
    # INp = np.zeros(len(t))
    # INs = np.zeros_like(INp)
    # for i in range(len(t)):
    #     INp[i] = IN(t[i], gp, a, b, coeff_ab)
    #     INs[i] = IN(t[i], gs, a, b, coeff_ab)

    # plt.plot(t, INp, label="INp")
    # plt.plot(t, INs, label="INs")
    # plt.legend()
    # plt.show()
