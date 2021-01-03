import numpy as np
import pylab as plt
from numpy import exp
from scipy.integrate import odeint


def ainfg(v): return 1/(1+exp(-(v-thag)/sigag))
def sinfg(v): return 1/(1+exp(-(v-thsg)/sigsg))
def rinfg(v): return 1/(1+exp(-(v-thrg)/sigrg))
def minfg(v): return 1./(1.+exp(-(v-thmg)/sigmg))
def ninfg(v): return 1./(1.+exp(-(v-thng)/signg))
def taung(v): return taun0g+taun1g/(1+exp(-(v-thngt)/sng))
def hinfg(v): return 1./(1.+exp(-(v-thhg)/sighg))
def tauhg(v): return tauh0g+tauh1g/(1+exp(-(v-thhgt)/shg))

def itg(vg,rg):return gtg*(ainfg(vg)**3)*rg*(vg-vcag)
def inag(vg,hg):return gnag*(minfg(vg)**3)*hg*(vg-vnag)
def ikg(vg,ng):return gkg*(ng**4)*(vg-vkg)
def iahpg(vg,cag):return gahpg*(vg-vkg)*cag/(cag+k1g)
def icag(vg):return gcag*((sinfg(vg))**2)*(vg-vcag)
def ilg(vg):return glg*(vg-vlg)


def ode_sys(x0, t):

    vg, hg, ng, rg, cag = x0
    dvg= -(itg(vg,rg)+inag(vg,hg)+ikg(vg,ng)+iahpg(vg,cag)+icag(vg)+ilg(vg)) +iapp # -isyngg(vg,stot)-isyng
    dhg= phihg*(hinfg(vg)-hg)/tauhg(vg)
    dng= phing*(ninfg(vg)-ng)/taung(vg)
    drg= phig*(rinfg(vg)-rg)/taurg
    dcag=epsg*(-icag(vg)-itg(vg,rg) - kcag*cag)
    # dsg=alphag*(1-sg)*sinfg(vg+abg)-betag*sg

    return [dvg, dhg, dng, drg, dcag]



def init(vg):

    hg = hinfg(vg)
    ng = ninfg(vg)
    rg = rinfg(vg)
    cag = 0
    # sg = 0

    return [vg, hg, ng, rg, cag]


gnag = 120.
gkg = 30.
gahpg = 30.
gtg = .5
gcag = .1
glg = .1
vnag = 55.
vkg = -80.
vcag = 120.
vlg = -55.
thag = -57.
sigag = 2.
thsg = -35.
sigsg = 2.
thrg = -70.
sigrg = -2.
taurg = 30.
thmg = -37.
sigmg = 10.
thng = -50.
signg = 14.
taun0g = .05
taun1g = .27
thngt = -40
sng = -12
thhg = -58
sighg = -12
tauh0g = .05
tauh1g = .27
thhgt = -40
shg = -12
k1g = 30.
kcag = 20.
epsg = 0.0001
phig = 1.
phing = .05
phihg = .05
iapp = -0.5
gGtoG = 1
vGtoG = -100.
gStoG = 0.03
vStoG = 0
alphag = 2
betag = .08
abg = -20


if __name__ == "__main__":

    pass