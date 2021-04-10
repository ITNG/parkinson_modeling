import os
import lib
import numpy as np
import pylab as plt
from time import time

if not os.path.exists('data'):
    os.makedirs('data')


# CONSTANTS----------------------------------------------------------
T_SG = 6.0        # ms
T_GS = 6.0
T_GG = 4.0
T_CS = 5.5
T_SC = 21.5
tau_S = 12.8            # ms
tau_G = 20.0
Ms = 300/1000    # spk/s
Mg = 400/1000
Bs = 10/1000
Bg = 20/1000
# T_CC = [1.0, 10.0]
# tau_E = [10.0, 20.0]
# tau_I = [10.0, 20.0]
# Me = [50, 80]
# Mi = [200, 330]
# Be = [0, 20]
# Bi = [0, 20]

# PARAMETERS---------------------------------------------------------
wSG = 4.87
wGS = 1.33
wCS = 9.98
wSC = 8.93
wGG = 0.53
wCC = 6.17
C = 172.18/1000
Str = 8.46/1000
Be = 17.85/1000
Bi = 9.87/1000
Me = 75.77/1000
Mi = 205.72/1000
T_CC = 4.65
tau_E = 11.59
tau_I = 13.02

num = 4
dt = 5e-3
nstart = 50
t_simulation = 200.0
num_iterarion = (int)(t_simulation / dt)
delays = [T_CS, T_GS, T_SG, T_GG, T_SC, T_CC]

y = np.zeros((num, num_iterarion + nstart + 1))      # coordinates
t = np.zeros(num_iterarion + nstart + 1)  # times

par_rate_STN = {
    'b': 65,
    'A': 60,
    'f': 13.7,
    'N': 3.3
}
par_rate_GPe = {
    'b': 100,
    'A': 55,
    'f': 14.6,
    'N': 3.9
}

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    start = time()
    lib.main()
    print("Done in %g seconds" % (time() - start))

    lib.plot_data()
