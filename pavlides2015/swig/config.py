import numpy as np

sim_params = {
    'num': 4,
    'dt': 0.005,
    't_simulation': 1500.0,
    'I_C': 172.18 / 1000.0,
    'I_Str': 8.46 / 1000.0,
    "statistics_method": "firing_rate",  # moments
}


# wSG, wGS, wCS, wSC, wGG, wCC
# wSG = 4.87
# wGS = 1.33
# wCS = 9.98
# wSC = 8.93
# wGG = 0.53
# wCC = 6.17
true_params = np.array([4.87, 1.33, 9.98, 8.93, 0.53, 6.17])  # weights

num_workers = 8
num_simulations = 100
method = 'SNPE'
num_rounds = 1

prior_min = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prior_max = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
num_samples = 10000
num_threads = 1

# observation_params = np.array([0.01, 190,      # S
#                                0.01, 262,      # G
#                                0.01, 53,       # E
#                                0.01, 133,      # I
#                                ]) / 1000
observation_params = np.array([3.90625000e-03, 1.89927824e+02, 14,
                               4.61310964e-03, 2.61592497e+02, 14, 
                               3.26756986e-02, 5.31832408e+01, 14, 
                               3.79032258e-03, 1.32794478e+02, 14])/1000
# sim_params['I_C'] = 100.0 / 1000.0
# sim_params['I_Str'] = 8.0 / 1000.0
