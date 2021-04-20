import os
import lib
import numpy as np
import pylab as plt
from time import time
from os.path import join

if not os.path.exists('data'):
    os.makedirs('data')


par_feedback = {
    "T_SG": 6.0,
    "T_GS": 6.0,
    "T_GG": 4.0,
    "T_CS": 5.5,
    "T_SC": 21.5,
    "tau_S": 12.8,
    "tau_G": 20.0,
    "Ms": 300/1000,
    "Mg": 400/1000,
    "Bs": 10/1000,
    "Bg": 20/1000,
    "wSG": 4.87,
    "wGS": 1.33,
    "wCS": 9.98,
    "wSC": 8.93,
    "wGG": 0.53,
    "wCC": 6.17,
    "C": 172.18/1000,
    "Str": 8.46/1000,
    "Be": 17.85/1000,
    "Bi": 9.87/1000,
    "Me": 75.77/1000,
    "Mi": 205.72/1000,
    "T_CC": 4.65,
    "tau_E": 11.59,
    "tau_I": 13.02,
    "output_filename": "resonance",
}

par_resonance = {
    "T_SG": 6.0,
    "T_GS": 6.0,
    "T_GG": 4.0,
    "T_CS": 5.5,
    "T_SC": 21.5,
    "tau_S": 12.8,
    "tau_G": 20.0,
    "Ms": 300/1000,
    "Mg": 400/1000,
    "Bs": 10/1000,
    "Bg": 20/1000,
    "wSG": 2.56,
    "wGS": 3.22, #0
    "wCS": 6.60,
    "wSC": 0.0,
    "wGG": 0.9,
    "wCC": 3.08,
    "C": 277/1000,
    "Str": 40.51/1000,
    "Be": 3.62/1000,
    "Bi": 7.18/1000,
    "Me": 71.77/1000,
    "Mi": 276.39/1000,
    "T_CC": 7.74,
    "tau_E": 11.69,
    "tau_I": 10.45,
    "output_filename": "feedback",
}

par_simulation = {"num": 4,
                  "dt": 1e-2,
                  "nstart": 50,
                  "t_simulation": 1000.0,
                  }


# -----------------------------------------------------------------------------

if __name__ == "__main__":

    parameters = {}
    parameters["par_simulation"] = par_simulation
    for p in [par_resonance]:
        start = time()
        parameters["par"] = p
        sol = lib.STN_GPE_CTX_Circuite(parameters)
        sol.simulate()
        print("Done in %g seconds" % (time() - start))

        # plotting
        filename = parameters['par']['output_filename']
        lib.plot_time_series(filename_data=filename,
                             filename_fig=filename)
        lib.plot_frequency_spectrum(filename_data=filename,
                                    filename_fig=filename,
                                    xlim=[1, 60])


# par_rate_STN = {
#     'b': 65,
#     'A': 60,
#     'f': 13.7,
#     'N': 3.3
# }
# par_rate_GPe = {
#     'b': 100,
#     'A': 55,
#     'f': 14.6,
#     'N': 3.9
# }
# T_CC = [1.0, 10.0]
# tau_E = [10.0, 20.0]
# tau_I = [10.0, 20.0]
# Me = [50, 80]
# Mi = [200, 330]
# Be = [0, 20]
# Bi = [0, 20]
