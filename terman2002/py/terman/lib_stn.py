import numpy as np
import pylab as plt
from numpy import exp
from scipy.integrate import odeint


def sinf(v): return 1./(1.+exp(-(v-thetas)/sigmas))
def minf(v): return 1./(1.+exp(-(v-thetam)/sigmam))
def hinf(v): return 1./(1.+exp(-(v-thetah)/sigmah))
def ninf(v): return 1./(1.+exp(-(v-thetan)/sigman))
def ainf(v): return 1/(1+exp(-(v-thetaa)/sigmaa))
def binf(r): return 1/(1+exp((r-thetab)/sigmab))-1/(1+exp(-thetab/sigmab))
def rinf(v): return 1/(1+exp(-(v-thetar)/sigmar))
def taun(v): return taun0+taun1/(1+exp(-(v-thn)/sigmant))
def tauh(v): return tauh0+tauh1/(1+exp(-(v-thh)/sigmaht))
def taur(v): return taur0+taur1/(1+exp(-(v-thr)/sigmart))

def il(v): return gl * (v - vl)
def ina(v, h): return gna * (minf(v)) ** 3 * h * (v - vna)
def ik(v, n): return gk * n ** 4 * (v - vk)
def iahp(v, ca): return gahp * ca / (ca + k1) * (v - vk)
def ica(v): return gca * ((sinf(v)) ** 2) * (v - vca)
def it(v, r): return gt * (ainf(v) ** 3) * (binf(r) ** 2) * (v - vca)
def I_app(t): return 0.0

def ode_sys(x0, t):

    v, h, n, r, ca = x0

    dv = -(il(v) + ina(v, h) + ik(v, n) +
           iahp(v, ca) + ica(v) + it(v, r)) + I_app(t)  # - isyn
    dh = phi * (hinf(v) - h) / tauh(v)
    dn = phi * (ninf(v) - n) / taun(v)
    dr = phir * (rinf(v) - r) / taur(v)
    dca = eps * (-ica(v) - it(v, r) - kca * ca) # * phi
    # ds = alpha * (1 - s) * sinf(v + ab) - beta * s

    return [dv, dh, dn, dr, dca]


def init(v):

    h = hinf(v)
    n = ninf(v)
    r = rinf(v)
    ca = 0
    # s = 0

    return [v, h, n, r, ca]


vl = -60.
vna = 55.
vk = -80.
thetam = -30
sigmam = 15
gl = 2.25
gna = 37.5
gk = 45.
# tn = 1.
# th = 0.05
gahp = 9.
gca = .5
vca = 140.
k1 = 15.
eps = 5e-05
kca = 22.5
thetas = -39.
sigmas = 8.
# xp = 1.
# i = 0.
thetah = -39
sigmah = -3.1
thetan = -32.
sigman = 8.
taun0 = 1
taun1 = 100.
thn = -80.
sigmant = -26.
tauh0 = 1
tauh1 = 500
thh = -57.
sigmaht = -3.
phi = .75
thetaa = -63.
sigmaa = 7.8
gt = .5
phir = .5
thetar = -67
sigmar = -2.
taur0 = 7.1
taur1 = 17.5
thr = 68.
sigmart = -2.2

alpha = 5
beta = 1.
ab = -30.
gGtoS = 5
vGtoS = -100
thetab = .25
sigmab = -.07


if __name__ == "__main__":

    pass